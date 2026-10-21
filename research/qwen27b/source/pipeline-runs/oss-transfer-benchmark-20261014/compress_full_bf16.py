"""Create standard, lossless per-file Zstandard archives; never remove sources."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

source = Path('/root/model-cache/Qwen3.8-27B-BF16')
target = Path('/root/model-cache/Qwen3.8-27B-BF16-zstd3')
target.mkdir(exist_ok=True)
report = Path('/root/oss-transfer-benchmark-20261014/weight-compression-full.json')
verification = json.loads((source/'download-verification.json').read_text())
items = sorted(p for p in source.iterdir() if p.is_file())
state = dict(started_epoch=time.time(), source=str(source), target=str(target), method='zstd -3 -T1', original_bytes=sum(p.stat().st_size for p in items), source_verification='download-verification.json (historical)', files=[], complete=False)

def save():
    state['updated_epoch'] = time.time()
    state['compressed_bytes'] = sum(x['compressed_bytes'] for x in state['files'])
    report.write_text(json.dumps(state, ensure_ascii=False, indent=2))

for path in items:
    final = target/(path.name+'.zst')
    partial = target/(path.name+'.zst.partial')
    if final.exists() or partial.exists():
        raise RuntimeError('Refusing to overwrite existing archive: '+str(final))
    if shutil.disk_usage(target).free < path.stat().st_size + 4*1024**3:
        raise RuntimeError('Insufficient safety margin on disk')
    started = time.perf_counter()
    subprocess.run(['zstd', '-q', '-3', '-T1', '--check', str(path), '-o', str(partial)], check=True, timeout=150)
    partial.rename(final)
    state['files'].append(dict(file=path.name, original_bytes=path.stat().st_size, compressed_bytes=final.stat().st_size, compression_seconds=time.perf_counter()-started, archive=str(final)))
    save()
    if path.suffix == '.safetensors':
        print(json.dumps(dict(file=path.name, original_bytes=path.stat().st_size, compressed_bytes=final.stat().st_size, seconds=state['files'][-1]['compression_seconds'], total_compressed_bytes=state['compressed_bytes'])), flush=True)
state['complete'] = True
state['elapsed_seconds'] = time.time()-state['started_epoch']
save()
print(json.dumps(dict(complete=True, original_bytes=state['original_bytes'], compressed_bytes=state['compressed_bytes'], elapsed_seconds=state['elapsed_seconds'], files=len(state['files']))), flush=True)
