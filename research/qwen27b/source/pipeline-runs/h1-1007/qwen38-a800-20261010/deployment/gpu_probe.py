import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def collective(rank, world, port, out):
    torch.cuda.set_device(rank)
    try:
        dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{port}',
                                rank=rank, world_size=world,
                                timeout=datetime.timedelta(seconds=45))
        x = torch.full((8 * 1024 * 1024,), rank + 1., device=f'cuda:{rank}')
        dist.all_reduce(x)
        torch.cuda.synchronize()
        assert torch.all(x == 3).item(), 'all-reduce correctness failed'
        x.zero_()
        for _ in range(3):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / 20
        report = dict(rank=rank, passed=True, payload_bytes=x.numel()*x.element_size(),
                      seconds_per_all_reduce=seconds,
                      algorithm_GB_per_s=x.numel()*x.element_size()/seconds/1e9)
    except Exception:
        report = dict(rank=rank, passed=False, error=traceback.format_exc())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    Path(out, f'nccl-rank-{rank}.json').write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  gpu_count=torch.cuda.device_count(), gpus=[],
                  note='Synthetic capacity/kernel checks; not model throughput or model accuracy.')
    assert report['gpu_count'] == 2
    torch.backends.cuda.matmul.allow_tf32 = False
    for gpu in range(2):
        torch.cuda.set_device(gpu)
        info = dict(index=gpu, name=torch.cuda.get_device_name(gpu),
                    capability=torch.cuda.get_device_capability(gpu),
                    total_memory_bytes=torch.cuda.get_device_properties(gpu).total_memory)
        try:
            a = torch.randn((512, 512), device=f'cuda:{gpu}', dtype=torch.bfloat16)
            b = torch.randn_like(a)
            reference = a.float() @ b.float()
            result = a @ b
            error = ((result.float()-reference).norm()/reference.norm()).item()
            assert error < .02 and torch.isfinite(result).all().item()
            info['bf16_relative_l2_error'] = error
            del a, b, result, reference
            a = torch.randn((4096, 4096), device=f'cuda:{gpu}', dtype=torch.bfloat16)
            b = torch.randn_like(a)
            for _ in range(5):
                result = a @ b
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(30):
                result = a @ b
            torch.cuda.synchronize()
            elapsed = (time.perf_counter()-start)/30
            info['bf16_4096_square_matmul_seconds'] = elapsed
            info['bf16_matmul_TFLOP_per_s'] = 2*4096**3/elapsed/1e12
            del a, b, result
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            block = torch.empty(30*1024**3, dtype=torch.uint8, device=f'cuda:{gpu}')
            block.fill_(123)
            torch.cuda.synchronize()
            assert torch.all(block[::1024*1024] == 123).item()
            info['allocation_write_read_verified_bytes'] = block.numel()
            info['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            del block
            torch.cuda.empty_cache()
            info['passed'] = True
        except Exception:
            info['passed'] = False
            info['error'] = traceback.format_exc()
        report['gpus'].append(info)
        print(json.dumps(info), flush=True)
    report['peer_access_0_to_1'] = torch.cuda.can_device_access_peer(0, 1)
    report['peer_access_1_to_0'] = torch.cuda.can_device_access_peer(1, 0)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    try:
        mp.spawn(collective, args=(2, port, str(out)), nprocs=2, join=True)
        report['nccl'] = [json.loads((out/f'nccl-rank-{i}.json').read_text()) for i in range(2)]
    except Exception:
        report['nccl_error'] = traceback.format_exc()
    report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'gpu-probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
