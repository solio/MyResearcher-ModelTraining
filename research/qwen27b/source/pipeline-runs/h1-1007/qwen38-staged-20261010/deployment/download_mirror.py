"""Download official Qwen mirror files, verifying against pinned HF contents."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

REPO = 'Qwen/Qwen3.8-27B'
REV = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
ROOT = Path('/root/qwen38-pilot-20261010')
DEST = ROOT / 'models/Qwen3.8-27B'

def fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)

def digest(path, lfs):
    h = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    hf = fetch_json(f'https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true')
    ms = fetch_json(f'https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true')
    mirror = {x['Path']: x for x in ms['Data']['Files']}
    wanted = [x for x in hf['siblings'] if x['rfilename'].endswith(('.json','.safetensors','.jinja','.txt','.model'))]
    (ROOT/'logs/mirror-manifest.json').write_text(json.dumps({'hf': hf, 'modelscope': ms}, indent=2))
    def one(item):
        name = item['rfilename']; path = DEST/name
        lfs = item.get('lfs'); expected = lfs['sha256'] if lfs else item['blobId']
        if path.exists() and path.stat().st_size == item['size'] and digest(path, lfs) == expected:
            print('VERIFIED existing', name, flush=True); return
        m = mirror.get(name)
        if m is None or m['Size'] != item['size'] or (lfs and m['Sha256'] != expected):
            raise ValueError('Mirror does not match pinned HF file: '+name)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix+'.mirror-partial')
        url = f'https://modelscope.cn/models/{REPO}/resolve/{m["Revision"]}/{name}'
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                req = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
                started = time.time()
                with urllib.request.urlopen(req, timeout=45) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise ValueError('Bad resume range')
                    with partial.open('ab' if append else 'wb') as out:
                        while True:
                            chunk = response.read(4*1024*1024)
                            if not chunk: break
                            out.write(chunk)
                if partial.stat().st_size != item['size'] or digest(partial, lfs) != expected:
                    raise ValueError('File size or digest mismatch: '+name)
                partial.replace(path)
                print('VERIFIED downloaded', name, item['size'], round(time.time()-started,1), flush=True)
                return
            except Exception as exc:
                print('RETRY', name, attempt+1, repr(exc), flush=True)
                if attempt == 2: raise
                time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=18) as pool:
        list(pool.map(one,wanted))
    (ROOT/'logs/model-ready.json').write_text(json.dumps({'repo':REPO,'revision':REV,'files':len(wanted),'all_content_verified':True,'completed_epoch':time.time()}, indent=2))
    print('DOWNLOAD_COMPLETE_ALL_HASHES_VERIFIED', flush=True)

if __name__ == '__main__':
    main()
