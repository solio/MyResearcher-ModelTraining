"""Measure real hfd/aria2 transfers with bounded runtime and resumable files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT=Path('/root/hf-mirror-speedtest-20261010')
HF_REV='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
MS_REV='1098534ab5d7220ea0f4a6b9f07bb03729a79c1d'
NAME='model-00001-of-00018.safetensors'
EXPECTED='ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c'

def rpc(method):
    data=json.dumps({'jsonrpc':'2.0','id':'bench','method':method,'params':[]}).encode()
    req=urllib.request.Request('http://127.0.0.1:16800/jsonrpc',data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=2) as r:return json.load(r).get('result',[])

def main(a):
    out=ROOT/a.name;out.mkdir(parents=True,exist_ok=True)
    wrapper=out/'bin';wrapper.mkdir(exist_ok=True)
    (wrapper/'aria2c').write_text('#!/bin/sh\nexec /usr/bin/aria2c --enable-rpc=true --rpc-listen-all=false --rpc-listen-port=16800 --auto-save-interval=5 "$@"\n')
    (wrapper/'aria2c').chmod(0o700)
    env=os.environ.copy();env['PATH']=str(wrapper)+':'+env.get('PATH','');env['HF_ENDPOINT']='https://hf-mirror.com'
    if a.mode=='hf_mirror_hfd':
        cmd=['bash',str(ROOT/'hfd.sh'),'Qwen/Qwen3.8-27B','--revision',HF_REV,'--include',NAME,
             '--tool','aria2c','-x',str(a.connections),'-j','1','--local-dir',str(out/'files')]
    else:
        if a.mode=='modelscope_aria2':url=f'https://modelscope.cn/models/Qwen/Qwen3.8-27B/resolve/{MS_REV}/{NAME}'
        else:url=f'https://huggingface.co/Qwen/Qwen3.8-27B/resolve/{HF_REV}/{NAME}'
        cmd=[str(wrapper/'aria2c'),'--file-allocation=none','--summary-interval=5','--console-log-level=notice',
             '-x',str(a.connections),'-s',str(a.connections),'-j','1','-k','1M','-c','--auto-file-renaming=false',
             '--dir',str(out/'files'),'--out',NAME,url]
    started=time.monotonic();start_epoch=time.time();samples=[];first_transfer=None
    print('START',a.name,flush=True)
    with (out/'console.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while time.monotonic()-started<a.seconds:
                if proc.poll() is not None:break
                try:
                    items=rpc('aria2.tellActive')
                    if items:
                        d=items[0];sample={k:d.get(k) for k in ['gid','status','totalLength','completedLength','downloadSpeed','connections']}
                        sample['elapsed_seconds']=round(time.monotonic()-started,3);samples.append(sample)
                        if int(d.get('completedLength',0))>0 and first_transfer is None:first_transfer=sample
                        (out/'progress.json').write_text(json.dumps(samples,indent=2))
                        print(json.dumps(sample),flush=True)
                except Exception:pass
                time.sleep(5)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
    elapsed=time.monotonic()-started
    result={'mode':a.mode,'connections_requested':a.connections,'start_epoch':start_epoch,'elapsed_seconds':elapsed,
            'returncode':proc.returncode,'samples':samples,'model_complete':False,
            'measurement':'aria2 RPC completedLength, not sparse file apparent size'}
    if samples:
        result['last_completed_bytes']=int(samples[-1]['completedLength'])
        result['end_to_end_MiB_per_second']=result['last_completed_bytes']/1048576/samples[-1]['elapsed_seconds']
        if first_transfer and samples[-1]['elapsed_seconds']>first_transfer['elapsed_seconds']:
            result['observed_transfer_MiB_per_second']=(result['last_completed_bytes']-int(first_transfer['completedLength']))/1048576/(samples[-1]['elapsed_seconds']-first_transfer['elapsed_seconds'])
    f=out/'files'/NAME
    if f.exists() and not f.with_name(f.name+'.aria2').exists() and f.stat().st_size==3966730552:
        h=hashlib.sha256()
        with f.open('rb') as s:
            for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
        result.update(full_file_sha256=h.hexdigest(),model_shard_complete=h.hexdigest()==EXPECTED)
    (out/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['hf_mirror_hfd','modelscope_aria2','hf_origin_aria2'],required=True);p.add_argument('--name',required=True);p.add_argument('--seconds',type=int,default=90);p.add_argument('--connections',type=int,default=16);main(p.parse_args())
