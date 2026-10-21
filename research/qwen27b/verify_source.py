"""Verify archived source bytes without executing historical jobs or using the network."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    entries = json.loads((root / 'source-manifest.json').read_text())['files']
    failures = []
    for entry in entries:
        path = root / 'source' / entry['path']
        if not path.is_file():
            failures.append({'path': entry['path'], 'problem': 'missing'})
            continue
        content = path.read_bytes()
        if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
            failures.append({'path': entry['path'], 'problem': 'content changed'})
    print(json.dumps({'files': len(entries), 'failures': failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
