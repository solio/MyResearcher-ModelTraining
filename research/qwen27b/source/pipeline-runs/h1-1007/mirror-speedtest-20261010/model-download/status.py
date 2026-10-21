import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

def rpc(method, params):
    request = urllib.request.Request('http://127.0.0.1:16801/jsonrpc',
        data=json.dumps(dict(jsonrpc='2.0', id=1, method='aria2.'+method, params=params)).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=5) as response:
        result = json.load(response)
    return result['result']

root = Path('/root/model-cache/Qwen3.8-27B-AWQ-INT4')
keys = ['gid', 'status', 'totalLength', 'completedLength', 'downloadSpeed', 'connections', 'files', 'errorMessage']
try:
    tasks = rpc('tellActive', [keys]) + rpc('tellWaiting', [0, 1000, keys]) + rpc('tellStopped', [0, 1000, keys])
    speed = int(rpc('getGlobalStat', [])['downloadSpeed'])
    downloader_running = True
except (urllib.error.URLError, TimeoutError, ConnectionError):
    tasks = []
    speed = 0
    downloader_running = False
per_file = {}
for task in tasks:
    for entry in task.get('files', []):
        path = Path(entry['path'])
        if not path.is_absolute():
            path = root / path
        completed = task['completedLength'] if len(task['files']) == 1 else entry.get('completedLength', 0)
        per_file[str(path)] = dict(file=path.name,
            completed=int(completed), total=int(entry.get('length', 0)),
            status=task['status'], speed=int(task['downloadSpeed']), connections=int(task.get('connections', 0)))
metadata = json.loads((root / '.hfd/repo_metadata.json').read_text())
verification_path = root / 'download-verification.json'
verified = {}
verified_at = None
if verification_path.is_file():
    verification = json.loads(verification_path.read_text())
    if verification.get('revision') == metadata['sha']:
        verified = {item['file']: item for item in verification['files']}
        verified_at = datetime.fromisoformat(verification['checked_at']).timestamp()
rows = []
for item in metadata['siblings']:
    file = root / item['rfilename']
    total = int(item.get('size', item.get('lfs', {}).get('size', 0)))
    row = per_file.get(str(file))
    if row is None:
        complete = file.is_file() and file.stat().st_size == total and not Path(str(file)+'.aria2').exists()
        row = dict(file=item['rfilename'], total=total, completed=total if complete else 0,
                   status='present_unverified' if complete else 'pending', speed=0, connections=0)
    row['total'] = total
    previous = verified.get(item['rfilename'], {})
    if (previous.get('valid') and file.is_file() and file.stat().st_size == total
            and not Path(str(file)+'.aria2').exists() and file.stat().st_mtime <= verified_at):
        row['status'] = 'verified'
        row['completed'] = total
    rows.append(row)
report = dict(time=datetime.now(timezone.utc).isoformat(), model_path=str(root),
    completed_bytes=sum(row['completed'] for row in rows), total_bytes=sum(row['total'] for row in rows),
    speed_bytes_per_sec=speed, downloader_running=downloader_running,
    hash_verified_complete=bool(rows) and all(row['status'] == 'verified' for row in rows),
    progress_note='RPC progress when running; offline partial files are excluded from completed_bytes.', files=rows)
print(json.dumps(report, indent=2))
