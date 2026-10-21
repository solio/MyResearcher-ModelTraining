import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
revision = '6e134bae811fb5adac50ee042ae5f029ac6779aa'
metadata = json.loads((root / '.hfd/repo_metadata.json').read_text())
assert metadata['sha'] == revision, 'Unexpected model revision'
results = []
for item in metadata['siblings']:
    path = root / item['rfilename']
    lfs = item.get('lfs')
    expected_size = item.get('size', (lfs or {}).get('size'))
    expected_hash = lfs['sha256'] if lfs else item.get('blobId')
    actual_size = path.stat().st_size if path.is_file() else None
    valid = actual_size == expected_size and not Path(str(path) + '.aria2').exists()
    actual_hash = None
    if valid:
        digest = hashlib.sha256() if lfs else hashlib.sha1()
        if not lfs:
            digest.update(f'blob {actual_size}\0'.encode())
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        valid = actual_hash == expected_hash
    row = dict(file=item['rfilename'], size=actual_size, expected_size=expected_size,
               expected_hash=expected_hash, actual_hash=actual_hash, valid=valid)
    results.append(row)
    print(json.dumps(row), flush=True)
report = dict(repo='cyankiwi/Qwen3.8-27B-AWQ-INT4', revision=revision,
              checked_at=datetime.now(timezone.utc).isoformat(),
              complete=bool(results) and all(row['valid'] for row in results), files=results)
(root / 'download-verification.json').write_text(json.dumps(report, indent=2) + '\n')
sys.exit(0 if report['complete'] else 1)
