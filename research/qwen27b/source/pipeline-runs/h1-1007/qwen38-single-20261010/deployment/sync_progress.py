import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
deadline = time.time() + 86400
last_backup = 0
while time.time() < deadline:
    try:
        probe = (ROOT / 'deployment/progress_probe.py').read_text()
        completed = subprocess.run(['ssh', '-S', '/tmp/q38-cpu-24452.sock', '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=5', '-p', '24452', 'root@223.109.239.11', 'python3', '-'],
            input=probe, text=True, capture_output=True, timeout=12, check=True)
        report = json.loads(completed.stdout)
        report['connected'] = True
        report['node'] = '无卡节点 · 223.109.239.11:24452'
        report['inference_historical'] = True
        report.pop('gpu', None)
        if report.get('bf16'):
            report['bf16']['node'] = report['node']
        if report.get('attempted_unique') == 80:
            report['repair_note'] = '80 条推理已在原 A800 节点执行完成；当前无卡节点仅续传 BF16 权重。2 条输出矛盾已保留，不代表请求未执行。'
        temp = ROOT / 'progress.json.tmp'
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temp.replace(ROOT / 'progress.json')
        if report.get('bf16'):
            (ROOT / 'bf16/latest-status.json').write_text(json.dumps(report['bf16'], ensure_ascii=False, indent=2) + '\n')
        if report.get('completed', 0) and time.time() - last_backup >= 60:
            copied = subprocess.run(['scp', '-q', '-r', '-o', 'ControlPath=/tmp/q38-cpu-24452.sock',
                '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-P', '24452',
                'root@223.109.239.11:/root/qwen38-single-20261010/results/.', str(ROOT / 'results')],
                capture_output=True, timeout=30)
            if copied.returncode == 0:
                last_backup = time.time()
        if report.get('phase') in ('complete', 'complete_with_errors') and report.get('evaluation') and report.get('bf16', {}).get('verified'):
            break
    except Exception as error:
        print(time.time(), type(error).__name__, flush=True)
    time.sleep(15)
