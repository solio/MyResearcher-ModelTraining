"""Resumable parallel-range download; every final file verified to pinned HF."""
import concurrent.futures as cf
import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

REPO='cyankiwi/Qwen3.8-27B-AWQ-INT4'
REV='6e134bae811fb5adac50ee042ae5f029ac6779aa'
ROOT=Path('/root/qwen38-pilot-20261010')
DEST=ROOT/'models/Qwen3.8-27B-AWQ-INT4'
CHUNK=32*1024*1024
LOCK=threading.Lock()

def get_json(url):
    with urllib.request.urlopen(url,timeout=30) as r: return json.load(r)

def file_hash(path, lfs):
    h=hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs: h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    hf=get_json(f'https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true')
    ms=get_json(f'https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true')
    mirrors={x['Path']:x for x in ms['Data']['Files']}
    (ROOT/'logs/quantized-manifest.json').write_text(json.dumps({'hf':hf,'modelscope':ms},indent=2))
    files=[x for x in hf['siblings'] if x['rfilename'].endswith(('.safetensors','.json','.jinja','.model','.txt'))]
    jobs=[]; handles={}; targets=[]
    for x in files:
        name=x['rfilename']; path=DEST/name; lfs=x.get('lfs')
        expected=lfs['sha256'] if lfs else x['blobId']
        if path.exists() and path.stat().st_size==x['size'] and file_hash(path,lfs)==expected:
            print('VERIFIED existing',name,flush=True); continue
        m=mirrors[name]
        if m['Size']!=x['size'] or (lfs and m['Sha256']!=expected): raise ValueError('Mirror differs: '+name)
        partial=path.with_suffix(path.suffix+'.part')
        partial.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(partial,os.O_RDWR|os.O_CREAT,0o600)
        os.ftruncate(fd,x['size']); handles[name]=fd
        markers=DEST/'.chunks'/name; markers.mkdir(parents=True,exist_ok=True)
        url=f'https://modelscope.cn/models/{REPO}/resolve/{m["Revision"]}/{name}'
        for start in range(0,x['size'],CHUNK):
            end=min(start+CHUNK,x['size'])-1
            marker=markers/str(start)
            if marker.exists():
                expected_chunk=marker.read_text().strip()
                if hashlib.sha256(os.pread(fd,end-start+1,start)).hexdigest()==expected_chunk: continue
            jobs.append((name,fd,url,start,end,marker))
        targets.append((x,path,partial,expected,lfs))
    completed=0; started=time.time()
    def one(job):
        nonlocal completed
        name,fd,url,start,end,marker=job
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=45) as r:
                    if r.status==206:
                        if not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'): raise ValueError('Bad Content-Range')
                    elif start!=0 or end+1>CHUNK: raise ValueError('Range request ignored')
                    position=start; h=hashlib.sha256()
                    while position<=end:
                        b=r.read(min(1024*1024,end-position+1))
                        if not b: raise ValueError('Premature EOF')
                        offset=0
                        while offset<len(b): offset+=os.pwrite(fd,b[offset:],position+offset)
                        position+=len(b);h.update(b)
                os.fsync(fd)
                marker.write_text(h.hexdigest())
                with LOCK:
                    completed+=end-start+1
                    print('CHUNK_OK',name,start,end,'new_MB',round(completed/1e6),'MBps',round(completed/1e6/(time.time()-started),2),flush=True)
                return
            except Exception as exc:
                print('RETRY',name,start,attempt+1,repr(exc),flush=True)
                if attempt==2: raise
                time.sleep(2)
    print('DOWNLOAD_START',REPO,REV,'chunks',len(jobs),flush=True)
    with cf.ThreadPoolExecutor(max_workers=32) as pool: list(pool.map(one,jobs))
    for fd in handles.values(): os.close(fd)
    for x,path,partial,expected,lfs in targets:
        if file_hash(partial,lfs)!=expected: raise ValueError('FINAL HASH MISMATCH: '+x['rfilename'])
        partial.replace(path);print('FILE_VERIFIED',x['rfilename'],flush=True)
    (ROOT/'logs/quantized-ready.json').write_text(json.dumps({'repo':REPO,'revision':REV,'all_content_verified':True,'completed_epoch':time.time(),'files':len(files)},indent=2))
    print('DOWNLOAD_COMPLETE_ALL_HASHES_VERIFIED',flush=True)

if __name__=='__main__': main()
