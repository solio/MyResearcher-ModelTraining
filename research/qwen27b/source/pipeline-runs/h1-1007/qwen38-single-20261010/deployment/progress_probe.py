import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/root/qwen38-single-20261010')
report = {'checked_epoch': time.time(), 'node': '单张 A800 40GB', 'model': 'Qwen3.8-27B · INT4',
          'phase': 'environment', 'completed': 0, 'expected': 80, 'valid_responses': 0}


def rpc(method, params):
    request = urllib.request.Request('http://127.0.0.1:16805/jsonrpc',
        data=json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'aria2.' + method, 'params': params}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.load(response)['result']


manifest = ROOT / 'wheel-manifest.json'
if manifest.exists():
    rows = json.loads(manifest.read_text())
    tasks = []
    try:
        keys = ['totalLength', 'completedLength', 'files', 'status']
        tasks = rpc('tellActive', [keys]) + rpc('tellWaiting', [0, 1000, keys]) + rpc('tellStopped', [0, 1000, keys])
        report['speed_bytes_per_second'] = int(rpc('getGlobalStat', [])['downloadSpeed'])
    except Exception:
        report['speed_bytes_per_second'] = 0
    by_name = {Path(f['path']).name: t for t in tasks for f in t['files']}
    total = completed = finished = 0
    for row in rows:
        path = ROOT / 'wheels' / row['filename']
        task = by_name.get(row['filename'], {})
        partial = Path(str(path) + '.aria2').exists()
        size = int(int(task.get('totalLength') or 0) or row.get('size') or (path.stat().st_size if path.exists() and not partial else 0))
        done = int(task.get('completedLength') or (size if path.exists() and not partial else 0))
        total += size
        completed += done
        finished += int(size > 0 and done == size and not partial)
    report.update(download_bytes=completed, download_total_bytes=total, packages_completed=finished, packages_total=len(rows))
if (ROOT / 'runtime-ready.json').exists():
    report['phase'] = 'loading'
    pid_file = ROOT / 'server.pid'
    if pid_file.exists():
        process = Path('/proc') / pid_file.read_text().strip() / 'stat'
        if not process.exists() or process.read_text().split()[2] == 'Z':
            report['phase'] = 'needs_check'
            report['last_error'] = '模型服务已退出，尚未确认跑通。正在检查启动日志。'
try:
    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1) as response:
        if response.status == 200:
            report['phase'] = 'model_ready'
except Exception:
    pass
repair = (ROOT / 'repair-plan.json').exists()
paths = [ROOT / 'results/stage1/predictions.jsonl']
if repair:
    paths.append(ROOT / 'results/stage1-structured/predictions.jsonl')
continuation = (ROOT / 'continuation-plan.json').exists()
if continuation:
    paths.append(ROOT / 'results/stage1-continuation/predictions.jsonl')
recovery = (ROOT / 'recovery-plan.json').exists()
if recovery:
    paths.append(ROOT / 'results/stage1-recovery/predictions.jsonl')
by_id = {}
for preds in paths:
    if not preds.exists():
        continue
    for line in preds.read_text().splitlines():
        try:
            row = json.loads(line)
            by_id[row['id']] = row
        except json.JSONDecodeError:
            pass
records = list(by_id.values())
ok = [row for row in records if row.get('status') == 'ok']
report['completed'] = len(ok)
report['valid_responses'] = len(ok)
report['attempted_unique'] = len(records)
report['format_error_count'] = len(records)-len(ok)
if records:
    report['phase'] = 'labelling'
    report['mean_seconds'] = sum(x['elapsed_seconds'] for x in records) / len(records)
    report['last_error'] = next((x.get('error') for x in records if x.get('status') != 'ok'), None)
if ok:
    first = ok[0]
    inputs = json.loads((ROOT / 'data/stage1-inputs.json').read_text())['rows']
    original = next(x for x in inputs if x['id'] == first['id'])
    labels = {x['id']: x['label'] for x in json.loads((ROOT / 'runner/meaning_schema.json').read_text())['core_candidates']}
    report['first_result'] = {'id': first['id'], 'text': original['text'], 'seconds': first['elapsed_seconds'],
        'supported': [{'label': labels[k], 'quote': v.get('quote'), 'subject': v.get('subject'), 'time': v.get('time')}
                      for k, v in first['judgments'].items() if v.get('decision') == 'supported']}
active = 'stage1-continuation' if continuation else ('stage1-structured' if repair else 'stage1')
if recovery:
    active = 'stage1-recovery'
runtime = ROOT / 'results' / active / 'runtime.json'
if runtime.exists():
    report['run'] = json.loads(runtime.read_text())
    report['phase'] = ('complete' if len(ok) == 80 else 'complete_with_errors') if len(records) == 80 else 'needs_check'
if repair:
    report['repair_note'] = '第 19 条出现字段错误；保留首批 18 条成功结果，增加 JSON 格式约束后继续剩余 62 条。标签定义与语义提示词相同，两个批次的原始记录分别保存。'
    if not runtime.exists():
        pid_file = ROOT / 'structured.pid'
        proc = Path('/proc') / pid_file.read_text().strip() / 'stat' if pid_file.exists() else None
        report['phase'] = 'labelling' if proc and proc.exists() and proc.read_text().split()[2] != 'Z' else 'needs_check'
if continuation:
    report['repair_note'] = '先前两条请求被总截止时间缩短到 1 秒和 6 秒而超时，已保留原记录并续跑未完成样本；35 条已完成结果不重跑。格式有效不代表判断正确。'
if recovery:
    report['repair_note'] = '已恢复推理；单条格式错误保留原始结果并隔离，其他帖子继续执行。已有的 49 条有效结果不重跑。'
quality_path = ROOT / 'results/completed-review.json'
if quality_path.exists():
    report['completed_review'] = json.loads(quality_path.read_text())
try:
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=3).strip().split(',')
    report['gpu'] = dict(zip(['used_mib', 'total_mib', 'utilization_percent'], map(int, gpu)))
except Exception:
    pass
memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
report['host_memory_available_mib'] = int(memory['MemAvailable'].split()[0]) // 1024
for filename in (['combined-evaluation.json'] if repair else ['evaluation.json', 'stage1-evaluation.json']):
    p = ROOT / 'results' / filename
    if p.exists() and len(records) == 80:
        report['evaluation'] = json.loads(p.read_text())
bf16_root = Path('/root/model-cache/qwen38-bf16-download')
if (bf16_root / 'latest-status.json').exists():
    report['bf16'] = json.loads((bf16_root / 'latest-status.json').read_text())
    pid_file = bf16_root / 'download.pid'
    report['bf16']['supervisor_alive'] = False
    if pid_file.exists():
        process = Path('/proc') / pid_file.read_text().strip() / 'stat'
        report['bf16']['supervisor_alive'] = process.exists() and process.read_text().split()[2] != 'Z'
    if (bf16_root / 'verification.json').exists():
        verification = json.loads((bf16_root / 'verification.json').read_text())
        report['bf16']['verified'] = verification.get('complete', False)
        report['bf16']['verification'] = verification
print(json.dumps(report, ensure_ascii=False))
