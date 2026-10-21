from pathlib import Path
import hashlib,json,tarfile,time
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=D/'plan.json';plan=json.loads(p.read_text())
for n,h in plan['files_sha256'].items():assert sha(D/n)==h,n
plan.update(status='queued_after_k3',execution='User authorized continuing immediately after K3 within the current session. Reuse loaded BF16 service; fixed deadline1791795380.876541. No renewal.',remaining_before_execution=[])
plan['reference_sha256']=sha(D/'reference-pre-output.json')
plan['reference_status']='80 provisional references authored before any A/B inference, after both prompts were frozen; local-only, not human gold'
plan['transport_note']='Both arms preserve JSON schema property order on wire. A=id/checks/observations, B=id/observations/checks; same allowed response values, input IDs and text, decoding and model.'
p.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
names=['heldout-inputs.json','sampling-manifest.json','prompt-a.txt','prompt-b.txt','meaning_schema.json','plan.json','run_ab.py','runner/runner_core.py','runner/output_contract.py']
ready={'created_epoch':time.time(),'files_sha256':{n:sha(D/n) for n in names},'reference_sha256':sha(D/'reference-pre-output.json'),'reference_not_uploaded':True,'deadline_epoch':1791795380.876541}
(D/'queue-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
with tarfile.open(D/'queue-bundle.tar.gz','w:gz') as t:
 for n in names+['queue-ready.json']:t.add(D/n,arcname=n)
print(json.dumps({'bytes':(D/'queue-bundle.tar.gz').stat().st_size,'sha256':sha(D/'queue-bundle.tar.gz'),'frozen_prompts_unchanged':True,'reference_rows':80}))
