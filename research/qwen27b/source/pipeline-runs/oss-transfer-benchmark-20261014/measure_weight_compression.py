"""Read-only, stratified compression measurements of actual BF16 shard bytes."""
import gzip
import hashlib
import json
import struct
import subprocess
import time
from pathlib import Path

root = Path('/root/model-cache/Qwen3.8-27B-BF16')
out = Path('/root/oss-transfer-benchmark-20261014/weight-compression-sample.json')
block = 8 * 1024 * 1024
methods = ['zstd-3', 'zstd-9', 'gzip-6']
records = []
started = time.time()
for path in sorted(root.glob('*.safetensors')):
    size = path.stat().st_size
    with path.open('rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
        dtype_bytes = {}
        for name, item in header.items():
            if name != '__metadata__':
                dtype_bytes[item['dtype']] = dtype_bytes.get(item['dtype'], 0) + item['data_offsets'][1] - item['data_offsets'][0]
        row = dict(file=path.name, bytes=size, dtype_bytes=dtype_bytes, samples=[])
        for fraction in [1/6, 1/2, 5/6]:
            offset = min(size-block, max(header_size+8, int(size*fraction)))
            f.seek(offset)
            raw = f.read(block)
            sample = dict(offset=offset, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), methods={})
            for method in methods:
                t = time.perf_counter()
                if method.startswith('zstd'):
                    packed = subprocess.run(['zstd', '-q', '-T1', '-' + method.split('-')[1], '-c'], input=raw, capture_output=True, check=True).stdout
                else:
                    packed = gzip.compress(raw, compresslevel=6, mtime=0)
                seconds = time.perf_counter()-t
                if method.startswith('zstd'):
                    restored = subprocess.run(['zstd', '-q', '-d', '-c'], input=packed, capture_output=True, check=True).stdout
                else:
                    restored = gzip.decompress(packed)
                assert restored == raw, (path.name, offset, method)
                sample['methods'][method] = dict(compressed_bytes=len(packed), seconds=seconds, roundtrip_identical=True)
            row['samples'].append(sample)
    records.append(row)
    totals = {}
    for method in methods:
        estimate = sum(r['bytes'] * sum(s['methods'][method]['compressed_bytes'] for s in r['samples']) / sum(s['bytes'] for s in r['samples']) for r in records)
        sampled = sum(s['bytes'] for r in records for s in r['samples'])
        seconds = sum(s['methods'][method]['seconds'] for r in records for s in r['samples'])
        totals[method] = dict(estimated_compressed_bytes=round(estimate), original_bytes_covered=sum(r['bytes'] for r in records), sampled_bytes=sampled, compression_seconds=seconds)
    result = dict(measurement='stratified_samples_not_full_archive', started_epoch=started, updated_epoch=time.time(), block_bytes=block, records=records, totals=totals)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(dict(completed_shards=len(records), totals=totals)), flush=True)
