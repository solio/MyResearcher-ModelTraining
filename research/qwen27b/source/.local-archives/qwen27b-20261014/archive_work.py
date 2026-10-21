"""Archive existing local 27B work and verify frozen result snapshots. No remote mutation."""
import hashlib
import json
import os
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path('/Users/mac/Documents/trae_projects/MyResearcher')
OUT = Path(__file__).resolve().parent
BASE = ROOT/'pipeline-runs/h1-1007/qwen38-single-20261010'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]

audit = dict(checked_epoch=time.time(), remote_status='SSH master absent; new connection closed by host before authentication. Final live remote inventory unavailable.', git_status='pipeline-runs is outside all five project Git repositories; local preservation is not a Git commit or push.', batches=[], snapshots=[], raw_files=[])
for path in sorted(BASE.rglob('raw.jsonl')):
    if 'snapshots' in path.parts:
        continue
    records = rows(path)
    ids = [r.get('task_id') or (r.get('arm'), r['id']) for r in records]
    audit['raw_files'].append(dict(path=str(path.relative_to(ROOT)), rows=len(records), unique_requests=len(set(ids)), duplicate_requests=len(ids)-len(set(ids)), sha256=sha(path)))

for name, count in [('k3-bf16-388-v1',388), ('k3-top5-bf16-1808-v1',1808), ('k3-same-days-unhit-v1',2588)]:
    d = BASE/'trials'/name
    incoming = json.loads((d/'data/model-inputs.json').read_text())
    inputs = incoming['rows'] if isinstance(incoming,dict) else incoming
    records = rows(d/'results/main/raw.jsonl')
    expected = {r['id'] for r in inputs}; actual = {r['id'] for r in records}
    manifest = json.loads((d/'results/main/manifest.json').read_text())
    source = d/'runner/run_batch.py'
    row = dict(batch=name, expected_requests=count, input_rows=len(inputs), receipts=len(records), unique_receipts=len(actual), missing=sorted(expected-actual), extra=sorted(actual-expected), runner_sha256=sha(source), recorded_runner_sha256=manifest['runner_sha256'])
    row['runner_matches_executed_hash'] = row['runner_sha256']==row['recorded_runner_sha256']
    assert len(records)==len(actual)==len(inputs)==count and expected==actual, row
    audit['batches'].append(row)

old = json.loads((BASE/'trials/k3-full-days-analysis-v1/verification.json').read_text())
for batch in old['batches']:
    name=batch['batch']; d=BASE/'trials'/name
    snapshot = d/'snapshots'/('20261013-complete' if '1808' in name else '20261013-final')/'remote-snapshot.tgz'
    archive_sha=sha(snapshot)
    assert archive_sha==batch['archive_sha256']
    checked=[]
    with tarfile.open(snapshot) as tar:
        manifest=json.load(tar.extractfile('snapshot-manifest.json'))
        for name_in_archive,expected_hash in manifest['files_sha256'].items():
            actual_hash=hashlib.sha256(tar.extractfile(name_in_archive).read()).hexdigest()
            assert actual_hash==expected_hash, name_in_archive
            checked.append(name_in_archive)
    assert sha(d/'results/main/raw.jsonl')==batch['raw_sha256']
    audit['snapshots'].append(dict(batch=name, sha256=archive_sha, files_verified=len(checked), all_embedded_hashes_match=True, current_raw_matches_snapshot=True))

for rel, expected_count in [('k3-top5-bf16-1808-v1/exports/k3-full-raw-return.jsonl',1808), ('k3-same-days-unhit-v1/exports/k3-return-with-reuse.jsonl',2594), ('k3-full-days-analysis-v1/all-4402-original-answers.jsonl',4402)]:
    path=BASE/'trials'/rel
    assert len(rows(path))==expected_count
audit['combined_export_rows']=4402
audit['unhit_reused_rows']=6
audit['not_backed_up_here']=[
    {'item':'BF16 model directory', 'bytes':55586123662, 'location':'/root/model-cache/Qwen3.8-27B-BF16', 'status':'Previously observed on remote; no full local or OSS copy verified.'},
    {'item':'Lossless BF16 zstd archives', 'bytes':43274159025, 'location':'/root/model-cache/Qwen3.8-27B-BF16-zstd3', 'status':'Compression report local; actual compressed weight files remain remote.'},
    {'item':'AWQ INT4 weights', 'weight_bytes':21018000928, 'location':'/root/model-cache/Qwen3.8-27B-AWQ-INT4', 'status':'Previously observed on remote; no full local copy found in project.'},
    {'item':'Remote installed venv, system CUDA libraries, package caches', 'status':'Deployment scripts, runtime lock and wheel manifest local; no complete environment image archived.'},
    {'item':'Remote-only files or changes since last snapshots', 'status':'Cannot exclude omissions while node is inaccessible. This is not authorization or confirmation to erase entire disk.'}
]
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))

folders=[ROOT/'pipeline-runs/h1-1007'/name for name in [
    'qwen38-single-20261010','qwen38-staged-20261010','qwen38-a800-20261010',
    'qwen38-bf16-download-20261010','mirror-speedtest-20261010','ready-semantic-comparison-20261010']]
folders.append(ROOT/'pipeline-runs/oss-transfer-benchmark-20261014')
skip_dirs={'.git','.venv','venv','__pycache__','node_modules','.pytest_cache','.cache','model'}
files=[]
for folder in folders:
    for directory,dirs,names in os.walk(folder):
        dirs[:]=[d for d in dirs if d not in skip_dirs]
        for name in names:
            path=Path(directory)/name
            if path.is_symlink() or name=='.DS_Store' or path.suffix in {'.pyc','.pid','.safetensors','.gguf','.whl'}:
                continue
            if path.is_file():files.append(path)
exports=ROOT/'worktrees/dashboard-k3-experiments/exports'
for pattern in ['gpt-compare-sample-20261012*.json*','top5-extreme-day-hit-sentences-20261012*.json*']:
    files.extend(exports.glob(pattern))
files += [OUT/'audit.json',OUT/'archive_work.py',OUT/'README.txt']
files=sorted(set(files))
archive=OUT/'qwen27b-work-with-results.zip'
manifest=[]
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
    for path in files:
        relative=str(path.relative_to(ROOT))
        h=hashlib.sha256(); size=0
        info=zipfile.ZipInfo.from_file(path,arcname=relative)
        info.compress_type=zipfile.ZIP_DEFLATED
        with path.open('rb') as src,z.open(info,'w',force_zip64=True) as dst:
            for block in iter(lambda:src.read(1024*1024),b''):
                dst.write(block);h.update(block);size+=len(block)
        manifest.append(dict(path=relative,bytes=size,sha256=h.hexdigest()))
    z.writestr('archive-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(archive) as z:
    for record in manifest:
        h=hashlib.sha256()
        with z.open(record['path']) as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        assert h.hexdigest()==record['sha256'],record['path']
digest=sha(archive)
(OUT/'archive-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(OUT/'SHA256SUMS.txt').write_text(f'{digest}  {archive.name}\n')
result=dict(archive=str(archive),archive_bytes=archive.stat().st_size,files=len(manifest),uncompressed_bytes=sum(x['bytes'] for x in manifest),sha256=digest,all_archive_members_verified=True,audit=str(OUT/'audit.json'),scripts=sum(Path(x['path']).suffix in {'.py','.sh','.command'} for x in manifest))
(OUT/'archive-complete.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False),flush=True)
