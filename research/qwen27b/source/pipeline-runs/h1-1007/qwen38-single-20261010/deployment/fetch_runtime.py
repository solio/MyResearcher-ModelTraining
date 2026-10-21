"""Download resolved wheels in parallel with aria2, then install offline."""
import concurrent.futures
import hashlib
import html
import json
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import unquote, urlparse, urljoin
from urllib.request import Request, urlopen
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename

ROOT = Path('/root/qwen38-single-20261010')
WHEELS = ROOT / 'wheels'
TAGS = {tag: index for index, tag in enumerate(sys_tags())}


def get(url):
    cache = ROOT / 'metadata-cache' / hashlib.sha256(url.encode()).hexdigest()
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache.read_bytes()
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'pip/24.2'}), timeout=15) as response:
                data = response.read()
                cache.write_bytes(data)
                return data
        except Exception as error:
            print('METADATA_RETRY', url, repr(error), flush=True)
            if attempt == 3:
                raise
            time.sleep(1)


def rank(filename):
    try:
        _, _, _, tags = parse_wheel_filename(filename)
        return min((TAGS[tag] for tag in tags if tag in TAGS), default=999999)
    except Exception:
        return 999999


def choose(item):
    name, version = item
    if name in ('torch', 'torchvision', 'torchaudio', 'torchcodec'):
        page = get(f'https://download.pytorch.org/whl/cu129/{name}/').decode()
        candidates = []
        for raw_url in re.findall(r'href="([^"]+)"', page):
            url = urljoin(f'https://download.pytorch.org/whl/cu129/{name}/', html.unescape(raw_url))
            filename = unquote(urlparse(url).path.rsplit('/', 1)[-1])
            try:
                _, v, _, _ = parse_wheel_filename(filename)
            except Exception:
                continue
            if str(v) == version and rank(filename) < 999999:
                candidates.append(dict(filename=filename, url=url.split('#')[0],
                                       sha256=url.split('#sha256=')[-1] if '#sha256=' in url else None))
    else:
        meta = json.loads(get(f'https://pypi.org/pypi/{name}/{version}/json'))
        candidates = [dict(filename=x['filename'], url=x['url'], sha256=x['digests']['sha256'], size=x['size'])
                      for x in meta['urls'] if x['packagetype'] == 'bdist_wheel' and rank(x['filename']) < 999999]
    if not candidates:
        raise RuntimeError(f'No compatible wheel for {name}=={version}')
    selected = min(candidates, key=lambda x: rank(x['filename']))
    if not selected.get('sha256') and urlparse(selected['url']).hostname not in ('download.pytorch.org', 'download-r2.pytorch.org'):
        raise RuntimeError(f'No source hash for {selected["filename"]}')
    selected.update(name=name, version=version)
    print('RESOLVED', selected['filename'], flush=True)
    return selected


def main():
    lock = ROOT / 'runtime-locked-cu129.txt'
    for _ in range(180):
        if lock.exists():
            break
        time.sleep(5)
    else:
        raise RuntimeError('Dependency resolution did not finish')
    items = []
    for line in lock.read_text().splitlines():
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^ ;]+)', line.strip())
        if match and match[1] != 'vllm':
            items.append((match[1], match[2]))
    assert len(items) > 30
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        manifest = list(pool.map(choose, items))
    manifest.sort(key=lambda row: -row.get('size', 1000000000))
    (ROOT / 'wheel-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    records = []
    for row in manifest:
        entry = [row['url'], f'  dir={WHEELS}', f'  out={row["filename"]}']
        if row.get('sha256'):
            entry.append(f'  checksum=sha-256={row["sha256"]}')
        records.append('\n'.join(entry))
    (ROOT / 'wheel-urls.txt').write_text('\n'.join(records) + '\n')
    cmd = ['aria2c', '-c', '-x', '8', '-s', '8', '-j', '4', '--file-allocation=none',
           '--auto-save-interval=5', '--auto-file-renaming=false', '--check-integrity=true',
           '--max-tries=10', '--retry-wait=3', '--connect-timeout=15', '--timeout=30',
           '--summary-interval=30', '--console-log-level=warn', '--enable-rpc=true',
           '--rpc-listen-all=false', '--rpc-listen-port=16805',
           '--input-file=' + str(ROOT / 'wheel-urls.txt')]
    for attempt in range(3):
        with (ROOT / 'logs/wheels-download.log').open('a') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(3)
                try:
                    body = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'aria2.getGlobalStat', 'params': []}).encode()
                    with urlopen(Request('http://127.0.0.1:16805/jsonrpc', data=body, headers={'Content-Type': 'application/json'}), timeout=2) as response:
                        stat = json.load(response)['result']
                    if int(stat['numStoppedTotal']) > 0 and int(stat['numActive']) == 0 and int(stat['numWaiting']) == 0:
                        body = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'aria2.shutdown', 'params': []}).encode()
                        with urlopen(Request('http://127.0.0.1:16805/jsonrpc', data=body, headers={'Content-Type': 'application/json'}), timeout=2) as response:
                            response.read()
                except Exception:
                    pass
            rc = process.returncode
            if any(not (WHEELS / row['filename']).exists() or Path(str(WHEELS / row['filename']) + '.aria2').exists() for row in manifest):
                rc = 1
        if rc == 0:
            break
    else:
        raise RuntimeError('Wheel download incomplete; checkpoints retained')
    vllm = WHEELS / 'vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
    for _ in range(180):
        if vllm.exists() and not Path(str(vllm) + '.aria2').exists() and vllm.stat().st_size == 553357469:
            break
        time.sleep(5)
    else:
        raise RuntimeError('vLLM wheel incomplete')
    digest = hashlib.sha256()
    with vllm.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == '5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f'
    wheels = [str(WHEELS / row['filename']) for row in manifest] + [str(vllm)]
    subprocess.run([str(ROOT / 'venv/bin/uv'), 'pip', 'install', '--python', str(ROOT / 'venv/bin/python'),
                    '--no-index', '--no-deps', *wheels], check=True)
    subprocess.run([str(ROOT / 'venv/bin/uv'), 'pip', 'check', '--python', str(ROOT / 'venv/bin/python')], check=True)
    (ROOT / 'runtime-ready.json').write_text(json.dumps({'ready': True, 'time_epoch': time.time(), 'packages': len(wheels)}))
    print('RUNTIME_INSTALLED', flush=True)


if __name__ == '__main__':
    main()
