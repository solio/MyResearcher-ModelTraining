import hashlib
import json
from pathlib import Path
import datetime

base = Path('/root/qwen38-a800-20261010')
root = base / 'models/Qwen3.5-0.8B'
metadata = json.loads((base / 'Qwen--Qwen3.5-0.8B-metadata.json').read_text())
assert metadata['sha'] == '2fc06364715b967f1860aea9cf38778875588b17'
results = []
for item in metadata['siblings']:
    path = root / item['rfilename']
    lfs = item.get('lfs')
    expected = lfs['sha256'] if lfs else item['blobId']
    valid = path.is_file() and path.stat().st_size == item['size'] and not Path(str(path)+'.aria2').exists()
    actual = None
    if valid:
        digest = hashlib.sha256() if lfs else hashlib.sha1()
        if not lfs:
            digest.update(f'blob {item["size"]}\0'.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        actual = digest.hexdigest()
        valid = actual == expected
    results.append(dict(file=item['rfilename'], valid=valid, expected=expected, actual=actual))
report = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), repo=metadata['id'],
              revision=metadata['sha'], complete=all(x['valid'] for x in results), files=results)
(base/'results').mkdir(exist_ok=True)
(base/'results/small-model-verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['complete'] else 2)
