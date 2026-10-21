"""Bounded sequential comparisons of identical public model byte ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request

REPO='Qwen/Qwen3.8-27B'
HF_REV='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
MS_REV='1098534ab5d7220ea0f4a6b9f07bb03729a79c1d'
FILE='model-00001-of-00018.safetensors'
SOURCES={
 'hf_origin':f'https://huggingface.co/{REPO}/resolve/{HF_REV}/{FILE}',
 'hf_mirror':f'https://hf-mirror.com/{REPO}/resolve/{HF_REV}/{FILE}',
 'modelscope_resolve':f'https://modelscope.cn/models/{REPO}/resolve/{MS_REV}/{FILE}',
 'modelscope_api':f'https://modelscope.cn/api/v1/models/{REPO}/repo?Revision={MS_REV}&FilePath={FILE}',
}

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

def one(args):
    url=SOURCES[args.source];path=Path(args.out);start=time.monotonic()
    size=args.mib*1024*1024; offset=args.offset_mib*1024*1024
    d={'source':args.source,'url':url,'location':'rented_CPU_instance_223.109.239.36','started_epoch':time.time(),
       'requested_bytes':size,'offset':offset,'bytes_received':0,'complete':False,'status':'connecting'}
    save(path,d);h=hashlib.sha256();prefix=hashlib.sha256();last_save=start
    try:
        req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{offset+size-1}','User-Agent':'model-download-benchmark/1.0'})
        with urllib.request.urlopen(req,timeout=12) as response:
            d.update(http_status=response.status,final_host=urllib.parse.urlparse(response.geturl()).hostname,
                     content_range=response.headers.get('Content-Range'),header_seconds=time.monotonic()-start)
            if response.status!=206 or not d['content_range'].startswith(f'bytes {offset}-{offset+size-1}/'):
                raise ValueError('Range response differs from requested byte interval')
            d['status']='receiving';save(path,d)
            while d['bytes_received']<size:
                chunk=response.read(min(64*1024,size-d['bytes_received']))
                if not chunk: raise ValueError('Premature end of response')
                if d['bytes_received']==0:d['first_data_seconds']=time.monotonic()-start
                left=max(0,2*1024*1024-d['bytes_received'])
                if left:prefix.update(chunk[:left])
                h.update(chunk);d['bytes_received']+=len(chunk)
                now=time.monotonic();d['elapsed_seconds']=now-start
                if now-last_save>=.5:save(path,d);last_save=now
            d.update(complete=True,status='complete',sha256=h.hexdigest(),first_2MiB_sha256=prefix.hexdigest())
    except Exception as exc:
        d.update(status='error',error=repr(exc))
    d['elapsed_seconds']=time.monotonic()-start
    d['total_MiB_per_second']=d['bytes_received']/1048576/d['elapsed_seconds']
    if d.get('first_data_seconds') is not None:
        d['body_MiB_per_second']=d['bytes_received']/1048576/max(.000001,d['elapsed_seconds']-d['first_data_seconds'])
    save(path,d);print(json.dumps(d),flush=True)

def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);results=[]
    order=args.sources.split(',') if args.sources else list(SOURCES)
    for source in order:
        path=out/f'{source}-{args.mib}MiB-offset{args.offset_mib}.json'
        cmd=[sys.executable,__file__,'--one','--source',source,'--mib',str(args.mib),'--offset-mib',str(args.offset_mib),'--out',str(path)]
        try:subprocess.run(cmd,timeout=args.wall_timeout,check=True)
        except subprocess.TimeoutExpired:
            d=json.loads(path.read_text());d.update(status='wall_timeout',complete=False,wall_timeout_seconds=args.wall_timeout,
                 total_MiB_per_second=d['bytes_received']/1048576/args.wall_timeout);save(path,d);print(json.dumps(d),flush=True)
        results.append(json.loads(path.read_text()));save(out/'summary.json',results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--one',action='store_true');p.add_argument('--source',choices=list(SOURCES));p.add_argument('--sources');p.add_argument('--mib',type=int,default=8);p.add_argument('--offset-mib',type=int,default=0);p.add_argument('--out',required=True);p.add_argument('--wall-timeout',type=float,default=45);a=p.parse_args()
    one(a) if a.one else main(a)
