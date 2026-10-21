"""Back up this explicitly authorized trial while its SSH connection is available."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1] / 'trials/bf16-400-v1'
REMOTE = '''
import json
from pathlib import Path
p=Path('/root/qwen38-single-20261010/trials/bf16-400-v1')
names=['session-status.json','session.json','serve-bf16.sh','control-status.json','history-index-verification.json','build_history_index.py']
for batch in ['startup','natural','controls']:
 names += ['results/'+batch+'/'+n for n in ['manifest.json','resolved-prompt.txt','predictions.jsonl','raw.jsonl','runtime.json']]
files={}
for name in names:
 f=p/name
 if f.exists():
  text=f.read_text()
  if name.endswith('.jsonl') and text and not text.endswith('\\n'):
   text=text.rsplit('\\n',1)[0]+'\\n' if '\\n' in text else ''
  files[name]=text
print(json.dumps(files))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--socket',required=True)
    parser.add_argument('--port',required=True)
    parser.add_argument('--host',required=True)
    parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args()
    failures=0
    while time.time() < args.deadline+120:
        try:
            result=subprocess.run(['ssh','-S',args.socket,'-p',args.port,'-o','BatchMode=yes',
                '-o','ConnectTimeout=8','root@'+args.host,'python3 -'],input=REMOTE,text=True,
                capture_output=True,timeout=20)
            if result.returncode: raise RuntimeError('SSH unavailable')
            files=json.loads(result.stdout)
            for name,text in files.items():
                path=ROOT/name
                path.parent.mkdir(parents=True,exist_ok=True)
                temp=path.with_suffix(path.suffix+'.sync-tmp');temp.write_text(text);temp.replace(path)
            session=json.loads(files.get('session-status.json','{}'))
            rows=[json.loads(x) for x in files.get('results/natural/predictions.jsonl','').splitlines()]
            print(json.dumps({'at':time.time(),'state':session.get('state'),'natural_backed_up':len(rows),
                'valid':sum(r.get('status')=='ok' for r in rows)},ensure_ascii=False),flush=True)
            failures=0
            if session.get('state') == 'stopped':return
            control=json.loads(files.get('control-status.json','{}'))
            if session.get('state') == 'complete' and control.get('state') in ['complete','stopped','skipped']:return
        except Exception:
            failures+=1
            print('Snapshot unavailable; keeping existing local backup.',flush=True)
            if failures>=5:return
        time.sleep(20)


if __name__=='__main__':
    main()
