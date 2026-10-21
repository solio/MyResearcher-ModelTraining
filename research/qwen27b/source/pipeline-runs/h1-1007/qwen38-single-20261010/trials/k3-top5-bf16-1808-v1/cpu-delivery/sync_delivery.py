"""Back up the independent CPU inspector; join private selection metadata only locally."""
import base64, fcntl, gzip, hashlib, json, shlex, subprocess, time
from pathlib import Path

D = Path(__file__).resolve().parent
T = D.parent
S = json.loads((T/'session-launch.json').read_text())
REMOTE = '/root/qwen38-single-20261010/trials/k3-top5-bf16-1808-v1/cpu-delivery'
BASE = ['ssh','-S',S['socket'],'-p',str(S['port']),'-o','BatchMode=yes','-o','ConnectTimeout=8','root@'+S['host']]
SCRIPT = '''import base64,fcntl,json
from pathlib import Path
d=Path(DEST);o=d/'output'
with (o/'snapshot.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 m=json.loads((o/'manifest.json').read_text())
 data={n:base64.b64encode((o/n).read_bytes()).decode() for n in m['files_sha256']}
 print(json.dumps({'manifest':m,'files':data}))
'''.replace('DEST',repr(REMOTE))

def dump(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode()

def atomic(path, data):
    temp = path.with_name(path.name+'.tmp');temp.write_bytes(data);temp.replace(path)

def main():
    lock = (D/'sync.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    private = {r['id']:r for r in map(json.loads,(T/'private/evaluation-sidecar.jsonl').read_text().splitlines())}
    out = D/'output';out.mkdir(exist_ok=True)
    failures = 0
    while time.time() < S['deadline_epoch']+90:
        try:
            p = subprocess.run(BASE+['python3 -c '+shlex.quote(SCRIPT)],capture_output=True,timeout=20,check=True)
            snap = json.loads(p.stdout); manifest=snap['manifest']
            blobs = {n:base64.b64decode(b) for n,b in snap['files'].items()}
            for name,h in manifest['files_sha256'].items():
                assert hashlib.sha256(blobs[name]).hexdigest()==h,'CPU delivery transfer hash mismatch'
            for name,b in blobs.items():atomic(out/name,b)
            atomic(out/'manifest.json',dump(manifest))
            answers=gzip.decompress(blobs['model-answers.jsonl.gz'])
            atomic(D/'model-answers.jsonl',answers)
            joined=[]
            for line in answers.splitlines():
                r=json.loads(line)
                source=private.get(r['id'])
                if source is None:raise ValueError('Unknown request ID in CPU delivery: '+str(r['id']))
                joined.append({**source,'qwen_model_output':r['model_output'],
                    'qwen_model_content':r['model_content'],
                    'original_contract_status':r['original_contract_status'],
                    'original_contract_error':r['original_contract_error'],
                    'mechanical_audit':{k:r[k] for k in ('request_input_matches_frozen','span_checks','mechanical_issues','raw_receipt')},
                    'semantic_quality':'not_assessed'})
            data=b''.join(dump(r)+b'\n' for r in joined)
            atomic(D/'k3-raw-return.jsonl',data)
            summary=json.loads(blobs['summary.json'])
            record={'synced_epoch':time.time(),'rows':len(joined),'expected':1808,
                    'state':summary['state'],'remote_artifact_hashes_verified':True,
                    'private_metadata_join':'local_only','sha256':{
                        'k3-raw-return.jsonl':hashlib.sha256(data).hexdigest(),
                        'model-answers.jsonl':hashlib.sha256(answers).hexdigest()},
                    'notice':'Completed responses only; contains original answers including contract errors. Not semantic approval or repaired results.'}
            atomic(D/'delivery-manifest.json',dump(record))
            failures=0
            print(json.dumps({'state':summary['state'],'rows':len(joined),'synced_epoch':time.time()}),flush=True)
            if summary['state']!='running':break
        except Exception as e:
            failures+=1;print('CPU delivery backup retry: '+str(e)[:300],flush=True)
            atomic(D/'sync-error.json',dump({'epoch':time.time(),'consecutive_failures':failures,'error':str(e)[:300]}))
            if failures>=5:raise
        time.sleep(30)

if __name__=='__main__':main()
