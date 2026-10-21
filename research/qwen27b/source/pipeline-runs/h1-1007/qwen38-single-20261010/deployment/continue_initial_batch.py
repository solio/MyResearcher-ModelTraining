"""Complete the authorized 80 examples if prepaid instance remains available. Never renews billing."""
import json,subprocess,time
from pathlib import Path
R=Path('/root/qwen38-single-20261010')
plan=json.loads((R/'session-plan.json').read_text()) if (R/'session-plan.json').exists() else {'boot_epoch':1791646405.7904878}
limit=plan['boot_epoch']+7200
pid=int((R/'structured.pid').read_text())
while time.time()<limit:
    stat=Path('/proc')/str(pid)/'stat'
    if not stat.exists() or stat.read_text().split()[2]=='Z': break
    time.sleep(3)
rt=R/'results/stage1-structured/runtime.json'
if rt.exists():
    state=json.loads(rt.read_text())
    if state['pending_ids'] and not state['errors'] and state['stop_reason']=='deadline_or_signal' and time.time()<limit-5:
        rt.rename(rt.with_name('runtime-before-extension.json'))
        (R/'run-deadline.txt').write_text(str(limit)+'\n')
        cmd=['python3',str(R/'runner-structured/run_batch.py'),'--input',str(R/'data/stage1-remaining.json'),'--out',str(R/'results/stage1-structured'),'--model','qwen38-27b-int4','--model-revision','6e134bae811fb5adac50ee042ae5f029ac6779aa','--thinking','off','--reasoning-effort','omit','--max-tokens','2048','--concurrency','2','--smoke-count','1','--deadline-epoch',str(limit)]
        with (R/'logs/stage1-structured.log').open('a') as f:
            p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            (R/'structured.pid').write_text(str(p.pid)+'\n');p.wait()
subprocess.call(['python3',str(R/'finish_repair.py')])
