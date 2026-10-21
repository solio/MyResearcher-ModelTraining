#!/usr/bin/env python3
"""Launch ONE explicitly selected arm on an existing inference server. No GPU rental."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['baseline', 'structure', 'semantic'], required=True)
    parser.add_argument('--base-url', required=True, help='Existing vLLM service; never start or rent a GPU')
    parser.add_argument('--model', required=True, help='Use the SAME model alias in all arms')
    parser.add_argument('--model-revision', required=True, help='Same pinned INT4 weights in all arms')
    parser.add_argument('--deadline-epoch', required=True, type=float)
    args = parser.parse_args()
    frozen = json.loads((ROOT / 'experiment.json').read_text())
    if {'model': args.model, 'model_revision': args.model_revision} != frozen['frozen_model']:
        raise SystemExit('This experiment requires the original pinned INT4 model; BF16 is a separate comparison.')
    original = json.loads((ROOT / 'baseline-manifest.json').read_text())
    for name, expected in original['baseline_files'].items():
        if digest(ROOT / 'baseline' / name) != expected:
            raise SystemExit(f'Frozen baseline code/prompt was changed: {name}')
    for name, expected in frozen['data_sha256'].items():
        if digest(DATA / name) != expected:
            raise SystemExit(f'Frozen input/reference was changed: {name}')
    lock = ROOT / 'results/comparison-model.json'
    lock.parent.mkdir(exist_ok=True)
    identity = {'model': args.model, 'model_revision': args.model_revision,
                'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 2048, 'thinking': 'off',
                'concurrency': 2, 'input_sha256': frozen['data_sha256']['stage1-inputs.json']}
    if lock.exists():
        if json.loads(lock.read_text()) != identity:
            raise SystemExit('Comparison model/settings changed; create a separate experiment.')
    else:
        lock.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n')
    runner = ROOT / ('baseline' if args.arm == 'baseline' else 'runner')
    prompt = runner / ('prompt-semantic.txt' if args.arm == 'semantic' else 'prompt.txt')
    out = ROOT / 'results' / args.arm
    if out.exists() and (out / 'manifest.json').exists():
        print('Existing run: runner will verify hashes and only resume unattempted inputs.', flush=True)
    command = [sys.executable, str(runner / 'run_batch.py'), '--input', str(DATA / 'stage1-inputs.json'),
               '--out', str(out), '--prompt', str(prompt), '--base-url', args.base_url,
               '--model', args.model, '--model-revision', args.model_revision,
               '--deadline-epoch', str(args.deadline_epoch), '--thinking', 'off',
               '--reasoning-effort', 'omit', '--temperature', '0', '--top-p', '1',
               '--max-tokens', '2048', '--concurrency', '2', '--timeout', '120', '--stage', 'stage1']
    result = subprocess.run(command)
    if (out / 'manifest.json').exists() and (out / 'predictions.jsonl').exists():
        # Original reference is frozen. A false gate is not hidden and does not rewrite labels.
        subprocess.run([sys.executable, str(runner / 'evaluate.py'), '--input', str(DATA / 'stage1-inputs.json'),
                        '--reference', str(DATA / 'stage1-reference.json'), '--run', str(out)])
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
