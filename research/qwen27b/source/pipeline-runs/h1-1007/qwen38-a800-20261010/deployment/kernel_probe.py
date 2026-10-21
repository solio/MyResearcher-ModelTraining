"""Check target-runtime CUDA/Triton and the group's asymmetric INT4 kernel.

These are kernel checks with synthetic matrices, not a model quality test.
The Marlin quantization helper ships with the pinned vLLM distribution.
"""
import argparse
import datetime
import json
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Synthetic kernels; does not validate 27B checkpoint loading or semantic accuracy.',
              'tests': []}
    try:
        import torch
        import triton
        import triton.language as tl
        import vllm
        import transformers
        from vllm import _custom_ops as ops
        from vllm.scalar_type import scalar_types
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize
        report['versions'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                  triton=triton.__version__, vllm=vllm.__version__,
                                  transformers=transformers.__version__)

        @triton.jit
        def add_kernel(x, y, z, n: tl.constexpr, block: tl.constexpr):
            offset = tl.program_id(0)*block + tl.arange(0, block)
            mask = offset < n
            tl.store(z + offset, tl.load(x + offset, mask=mask) + tl.load(y + offset, mask=mask), mask=mask)

        for gpu in (0, 1):
            torch.cuda.set_device(gpu)
            device = torch.device('cuda', gpu)
            try:
                x = torch.randn(4096, device=device)
                y = torch.randn_like(x)
                z = torch.empty_like(x)
                start = time.perf_counter()
                add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, z, x.numel(), 256)
                torch.cuda.synchronize()
                assert torch.equal(z, x+y)
                report['tests'].append(dict(name='triton_jit_add', gpu=gpu, passed=True,
                    compile_and_run_seconds=time.perf_counter()-start))
            except Exception:
                report['tests'].append(dict(name='triton_jit_add', gpu=gpu, passed=False,
                                          error=traceback.format_exc()))
            try:
                k, n = 1024, 2048
                torch.manual_seed(127)
                weight = torch.randn((k, n), dtype=torch.bfloat16, device=device)
                reference, packed, scales, zero_points = awq_marlin_quantize(weight, scalar_types.uint4, 32)
                for m in (1, 32):
                    activation = torch.randn((m, k), dtype=torch.bfloat16, device=device)
                    workspace = marlin_make_workspace_new(device, 4)
                    actual = ops.marlin_gemm(activation, None, packed, None, scales, None, None,
                                            zero_points, workspace, scalar_types.uint4, m, n, k,
                                            use_fp32_reduce=True)
                    expected = activation @ reference
                    error = ((actual.float()-expected.float()).norm()/expected.float().norm()).item()
                    assert torch.isfinite(actual).all().item() and error < .02, error
                    report['tests'].append(dict(name='asymmetric_int4_marlin', gpu=gpu,
                        activation='bfloat16', group_size=32, m=m, k=k, n=n,
                        relative_l2_error=error, passed=True))
            except Exception:
                report['tests'].append(dict(name='asymmetric_int4_marlin', gpu=gpu, passed=False,
                                          error=traceback.format_exc()))
    except Exception:
        report['setup_error'] = traceback.format_exc()
    report['passed'] = 'setup_error' not in report and len(report['tests']) == 6 and all(t['passed'] for t in report['tests'])
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
