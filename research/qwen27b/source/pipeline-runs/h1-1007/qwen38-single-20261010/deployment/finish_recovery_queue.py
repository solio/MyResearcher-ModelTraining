"""Finish only the already authorized 31 remaining requests; retain format failures."""
import json,subprocess,time
from pathlib import Path
R=Path('/root/qwen38-single-20261010')
pid=int((R/'structured.pid').read_text())
while True:
    stat=Path('/proc')/str(pid)/'stat'
    if not stat.exists() or stat.read_text().split()[2]=='Z':break
    time.sleep(2)
path=R/'results/stage1-recovery/runtime.json'
if path.exists():
    state=json.loads(path.read_text())
    if state['pending_ids'] and state['stop_reason']=='deadline_or_signal':
        # The user explicitly prioritized keeping this finite queue running.
        # No new dataset, model, or purchase; resume only unattempted IDs.
        deadline=time.time()+1800
        (R/'run-deadline.txt').write_text(str(deadline)+'\n')
        path.rename(path.with_name('runtime-before-final-resume.json'))
        cmd=['python3',str(R/'runner-tolerant/run_batch.py'),'--input',str(R/'data/stage1-recovery.json'),'--out',str(R/'results/stage1-recovery'),'--model','qwen38-27b-int4','--model-revision','6e134bae811fb5adac50ee042ae5f029ac6779aa','--thinking','off','--reasoning-effort','omit','--max-tokens','2048','--concurrency','2','--smoke-count','1','--deadline-epoch',str(deadline)]
        with (R/'logs/stage1-recovery.log').open('a') as out:
            p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            (R/'structured.pid').write_text(str(p.pid)+'\n');p.wait()
subprocess.call(['python3',str(R/'finish_repair.py')])
