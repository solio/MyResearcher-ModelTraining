#!/usr/bin/env python3
"""Pinned, resumable download. File sizes alone never count as verification."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request, urlopen

TASK = Path('/root/model-cache/qwen38-bf16-download')
MODEL = Path('/root/model-cache/Qwen3.8-27B-BF16')
REPO = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
PORT = 16803


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    with temp.open('rb') as stream:
        os.fsync(stream.fileno())
    temp.replace(path)


def fetch_metadata():
    path = TASK / 'repo_metadata.json'
    if path.exists():
        metadata = json.loads(path.read_text())
    else:
        url = f'https://hf-mirror.com/api/models/{REPO}/revision/{REVISION}?blobs=true'
        with urlopen(url, timeout=45) as response:
            metadata = json.load(response)
        save(path, metadata)
    assert metadata['sha'] == REVISION
    for item in metadata['siblings']:
        relative = Path(item['rfilename'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert item.get('size', item.get('lfs', {}).get('size')) is not None
        assert item.get('lfs', {}).get('sha256') or item.get('blobId')
    return metadata


def rpc(method, params):
    body = json.dumps(dict(jsonrpc='2.0', id=1, method='aria2.' + method, params=params)).encode()
    request = Request(f'http://127.0.0.1:{PORT}/jsonrpc', data=body,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=3) as response:
        return json.load(response)['result']


def snapshot(metadata, running):
    tasks = []
    speed = 0
    if running:
        try:
            keys = ['gid', 'status', 'completedLength', 'totalLength', 'downloadSpeed', 'files', 'errorMessage']
            tasks = rpc('tellActive', [keys]) + rpc('tellWaiting', [0, 1000, keys]) + rpc('tellStopped', [0, 1000, keys])
            speed = int(rpc('getGlobalStat', [])['downloadSpeed'])
        except Exception:
            pass
    by_path = {entry['path']: task for task in tasks for entry in task['files']}
    previous_path = TASK / 'latest-status.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    old = {row['file']: row for row in previous.get('files', [])}
    rows = []
    for item in metadata['siblings']:
        relative = item['rfilename']
        path = MODEL / relative
        total = int(item.get('size', item.get('lfs', {}).get('size')))
        task = by_path.get(str(path))
        sidecar = Path(str(path) + '.aria2').exists()
        if task:
            completed = int(task['completedLength'])
            state = task['status']
        elif path.is_file() and path.stat().st_size == total and not sidecar:
            completed, state = total, 'present_unverified'
        else:
            completed = old.get(relative, {}).get('completed_bytes', 0) if sidecar else 0
            state = 'checkpoint_last_observed' if sidecar else 'pending'
        rows.append(dict(file=relative, total_bytes=total, completed_bytes=completed,
                         state=state, aria2_checkpoint=sidecar))
    report = dict(time=now(), repo=REPO, revision=REVISION, model_path=str(MODEL),
                  node='223.109.239.36:14024 (A800 compute disk; endpoint may change)',
                  downloader_running=running, speed_bytes_per_sec=speed,
                  completed_bytes=sum(r['completed_bytes'] for r in rows),
                  total_bytes=sum(r['total_bytes'] for r in rows), files=rows,
                  progress_note='RPC bytes when available; offline partial counts are the last observed checkpoint, never sparse file lengths. Completion requires verification.json.')
    save(TASK / 'latest-status.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, ensure_ascii=False), flush=True)


def verify(metadata):
    rows = []
    for item in metadata['siblings']:
        path = MODEL / item['rfilename']
        lfs = item.get('lfs')
        expected_size = int(item.get('size', (lfs or {}).get('size')))
        expected_hash = lfs['sha256'] if lfs else item['blobId']
        valid = path.is_file() and path.stat().st_size == expected_size and not Path(str(path) + '.aria2').exists()
        actual_hash = None
        if valid:
            digest = hashlib.sha256() if lfs else hashlib.sha1()
            if not lfs:
                digest.update(f'blob {expected_size}\0'.encode())
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            actual_hash = digest.hexdigest()
            valid = actual_hash == expected_hash
        rows.append(dict(file=item['rfilename'], expected_size=expected_size,
                         expected_hash=expected_hash, actual_hash=actual_hash, valid=valid))
        print(json.dumps(rows[-1]), flush=True)
    result = dict(repo=REPO, revision=REVISION, checked_at=now(),
                  complete=bool(rows) and all(r['valid'] for r in rows), files=rows)
    save(TASK / 'verification.json', result)
    save(MODEL / 'download-verification.json', result)
    return result['complete']


def main():
    TASK.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    with (TASK / 'download.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Download already running.', flush=True)
            return
        (TASK / 'download.pid').write_text(str(os.getpid()) + '\n')
        while True:
            try:
                metadata = fetch_metadata()
                break
            except Exception as error:
                print(now(), 'metadata_retry', repr(error), flush=True)
                time.sleep(15)
        records = []
        mirror_path = TASK / 'mirror-metadata.json'
        mirrors = {item['Path']: item for item in json.loads(mirror_path.read_text())['Data']['Files']} if mirror_path.exists() else {}
        for item in sorted(metadata['siblings'], key=lambda row: (not row['rfilename'].endswith('.safetensors'), row['rfilename'])):
            relative = item['rfilename']
            path = MODEL / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            urls = [f'https://hf-mirror.com/{REPO}/resolve/{REVISION}/{quote(relative, safe="/")}']
            mirror = mirrors.get(relative)
            if item.get('lfs') and mirror and mirror['Size'] == item['size'] and mirror['Sha256'] == item['lfs']['sha256']:
                urls.insert(0, f'https://modelscope.cn/models/{REPO}/resolve/{mirror["Revision"]}/{quote(relative, safe="/")}')
            record = ['\t'.join(urls), f'  dir={path.parent}', f'  out={path.name}']
            if item.get('lfs'):
                record.append('  checksum=sha-256=' + item['lfs']['sha256'])
            records.append('\n'.join(record))
        (TASK / 'urls.txt').write_text('\n'.join(records) + '\n')
        command = ['/usr/bin/aria2c', '--continue=true', '--auto-file-renaming=false',
                   '--allow-overwrite=false', '--file-allocation=none', '--check-integrity=true',
                   '--auto-save-interval=5', '--save-session-interval=5',
                   f'--save-session={TASK}/aria2.session', '--force-save=true',
                   '--max-connection-per-server=8', '--split=8', '--max-concurrent-downloads=1',
                   '--min-split-size=4M', '--connect-timeout=20', '--timeout=60',
                   '--max-tries=10', '--retry-wait=5', '--enable-rpc=true',
                   '--rpc-listen-all=false', f'--rpc-listen-port={PORT}',
                   '--summary-interval=30', '--console-log-level=warn',
                   f'--log={TASK}/aria2.log', '--log-level=notice', f'--input-file={TASK}/urls.txt']
        save(TASK / 'download-plan.json', dict(repo=REPO, revision=REVISION, started_at=now(),
              model_path=str(MODEL), command=command, checkpoint_seconds=5,
              total_bytes=sum(int(i.get('size', i.get('lfs', {}).get('size'))) for i in metadata['siblings']),
              files=len(metadata['siblings']), compute_node_at_start='223.109.239.36:14024',
              resume_command=f'nohup python3 {TASK}/download.py >> {TASK}/supervisor.log 2>&1 < /dev/null &'))
        while True:
            with (TASK / 'aria2-console.log').open('a') as output:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                (TASK / 'aria2.pid').write_text(str(process.pid) + '\n')
                while process.poll() is None:
                    snapshot(metadata, True)
                    time.sleep(15)
            snapshot(metadata, False)
            print(now(), 'aria2_exit', process.returncode, flush=True)
            if process.returncode == 0:
                if verify(metadata):
                    print(now(), 'ALL_FILES_HASH_VERIFIED', flush=True)
                    return
                print(now(), 'VERIFICATION_FAILED: preserve files; inspect before retry', flush=True)
                raise SystemExit(1)
            time.sleep(15)


if __name__ == '__main__':
    main()
