"""Fixed task adapter, sent over SSH stdin. No credentials or user commands."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

TASK = Path('/root/model-cache/qwen38-bf16-download')
MODEL = Path('/root/model-cache/Qwen3.8-27B-BF16')
RUN = Path('/root/qwen38-single-20261010')
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'


def read(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback


def alive(task):
    try:
        pid = (task / 'download.pid').read_text().strip()
        if not pid.isdigit():
            return False
        proc = Path('/proc') / pid
        command = (proc / 'cmdline').read_bytes().split(b'\0')
        return (str(task / 'download.py').encode() in command
                and (proc / 'stat').read_text().split()[2] != 'Z')
    except (OSError, IndexError):
        return False


def rpc(method, params):
    request = urllib.request.Request('http://127.0.0.1:16803/jsonrpc',
        data=json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'aria2.' + method, 'params': params}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.load(response)['result']


def verified_on_disk(task, model, metadata):
    verification = read(task / 'verification.json', {})
    if not verification.get('complete') or verification.get('revision') != REVISION:
        return False
    entries = {x['file']: x for x in verification.get('files', [])}
    try:
        stamp = (task / 'verification.json').stat().st_mtime
        for item in metadata['siblings']:
            name = item['rfilename']; path = model / name
            expected = entries.get(name, {})
            if (not expected.get('valid') or not path.is_file()
                    or path.stat().st_size != item['size']
                    or path.stat().st_mtime > stamp + 1
                    or Path(str(path) + '.aria2').exists()):
                return False
        return bool(metadata['siblings'])
    except (OSError, KeyError, TypeError):
        return False


def download_snapshot(resume=False, task=TASK, model=MODEL, starter=None):
    metadata = read(task / 'repo_metadata.json', {})
    status = read(task / 'latest-status.json', {})
    result = {k: v for k, v in status.items() if k not in ('node', 'progress_note')}
    result.update(checked_epoch=time.time(), found=task.is_dir() and model.is_dir(),
                  supervisor_alive=alive(task), verified=False, resume_action='not_requested')
    if not result['found'] or not (task / 'download.py').is_file():
        result.update(state='not_found', message='没有找到原下载任务目录，未启动新下载。')
        return result
    siblings = metadata.get('siblings', [])
    valid_paths = all(isinstance(x.get('rfilename'), str) and not Path(x['rfilename']).is_absolute()
                      and '..' not in Path(x['rfilename']).parts for x in siblings)
    if metadata.get('sha') != REVISION or not siblings or not valid_paths:
        result.update(state='blocked', message='任务版本清单不匹配，已保留文件并停止自动恢复。')
        return result
    result['total_bytes'] = sum(x['size'] for x in siblings)
    result['completed_bytes'] = min(result.get('completed_bytes', 0), result['total_bytes'])
    result['free_bytes'] = shutil.disk_usage(model).free
    result['verified'] = verified_on_disk(task, model, metadata)
    if result['verified']:
        result.update(state='complete', completed_bytes=result['total_bytes'], speed_bytes_per_sec=0,
                      resume_action='already_complete' if resume else 'not_requested',
                      message='全部文件已通过哈希校验；无需重复下载。')
        return result
    if resume and not result['supervisor_alive']:
        if not Path('/usr/bin/aria2c').is_file() and starter is None:
            result.update(state='blocked', message='节点缺少 aria2 下载程序，断点已保留。')
            return result
        if result['free_bytes'] < 256 * 1024 * 1024:
            result.update(state='blocked', message='磁盘剩余空间不足，未启动下载。')
            return result
        verification = read(task / 'verification.json', {})
        if verification and verification.get('complete') is False:
            result.update(state='blocked', message='已有文件校验失败记录，需检查对应文件；没有覆盖或删除断点。')
            return result
        if starter:
            starter(task)
        else:
            with (task / 'supervisor.log').open('ab', buffering=0) as log:
                subprocess.Popen([sys.executable, str(task / 'download.py')], stdout=log,
                                 stderr=log, stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        # The downloader owns an exclusive file lock: concurrent launches cannot duplicate transfers.
        result.update(state='starting', resume_action='started',
                      message='已发出断点恢复请求，正在等待下载进程与实时字节确认。')
        return result
    if result['supervisor_alive']:
        if resume:
            result['resume_action'] = 'already_running'
        result.update(state='running', message='下载程序运行中，正在等待网络或校验。')
        try:
            stats = rpc('getGlobalStat', [])
            active = rpc('tellActive', [['gid', 'status', 'completedLength', 'totalLength',
                                        'downloadSpeed', 'verifiedLength']])
            speed = int(stats['downloadSpeed'])
            result['speed_bytes_per_sec'] = speed
            result['active_transfers'] = active
            if any(int(x.get('verifiedLength', 0)) > 0 for x in active):
                result.update(state='checking', message='正在校验已有分片，校验后继续缺失部分。')
            elif speed > 0:
                result.update(state='downloading', message='正在下载缺失部分，断点持续保存。')
            elif result['completed_bytes'] == result['total_bytes']:
                result.update(state='checking', message='传输字节已齐，正在等待最终校验结果。')
        except Exception:
            if result['completed_bytes'] == result['total_bytes']:
                result.update(state='checking', message='传输字节已齐，等待最终哈希校验完成。')
    else:
        result.update(state='paused', speed_bytes_per_sec=0, message='任务未运行，磁盘中的文件与断点保留。')
    return result


def inference_snapshot(run=RUN):
    rows, broken = {}, 0
    for batch in ('stage1', 'stage1-structured', 'stage1-continuation', 'stage1-recovery'):
        path = run / 'results' / batch / 'predictions.jsonl'
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line); rows[row['id']] = row
            except (ValueError, KeyError):
                broken += 1
    inputs = read(run / 'data/stage1-inputs.json', {})
    expected = len(inputs.get('rows', [])) or 80
    valid = sum(x.get('status') == 'ok' for x in rows.values())
    return {'found': run.is_dir(), 'attempted_unique': len(rows), 'expected': expected,
            'valid_responses': valid, 'format_error_count': len(rows)-valid,
            'complete': len(rows) >= expected, 'malformed_log_lines': broken,
            'notice': '请求完成数不是语义正确数；连接不会重跑已完成样本，也不会启动新实验。'}


def trial_snapshot(run=RUN):
    trial = run / 'trials/complete-extraction-v1'
    session = read(trial / 'session-status.json', {})
    rows = {}
    path = trial / 'results/semantic/predictions.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line); rows[row['id']] = row
            except (ValueError, KeyError):
                pass
    valid = sum(x.get('status') == 'ok' for x in rows.values())
    times = [x['elapsed_seconds'] for x in rows.values() if x.get('elapsed_seconds') is not None]
    out = {**session, 'expected': 80, 'attempted': len(rows), 'valid': valid,
           'errors': len(rows) - valid, 'mean_seconds': sum(times)/len(times) if times else None}
    pid = read_pid = (trial / 'session.pid')
    try:
        pid = int(read_pid.read_text())
        proc = Path('/proc') / str(pid)
        out['alive'] = b'run_semantic_session.py' in (proc / 'cmdline').read_bytes() and (proc / 'stat').read_text().split()[2] != 'Z'
    except (OSError, ValueError, IndexError):
        out['alive'] = False
    if session.get('state') in ('running', 'waiting_server') and not out['alive']:
        out['state'] = 'interrupted'
    if rows:
        latest = list(rows.values())[-1]
        out['latest'] = {k: latest.get(k) for k in ['id', 'status', 'error', 'elapsed_seconds']}
    return out


def bf16_trial_snapshot(run=RUN):
    trial = run / 'trials/bf16-400-v1'
    session = read(trial / 'session-status.json', {})
    plan = read(trial / 'session.json', {})
    out = {**session, 'deadline_epoch': plan.get('deadline_epoch'),
           'controls': read(trial / 'control-status.json', {}), 'batches': {}}
    for name, expected in [('startup', 8), ('natural', 400), ('controls', 72)]:
        rows = {}
        path = trial / 'results' / name / 'predictions.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line); rows[row['id']] = row
                except (ValueError, KeyError):
                    pass
        valid = sum(x.get('status') == 'ok' for x in rows.values())
        out['batches'][name] = {'expected': expected, 'attempted': len(rows),
                              'valid': valid, 'errors': len(rows)-valid}
    try:
        pid = int((trial / 'supervisor.pid').read_text())
        proc = Path('/proc') / str(pid)
        out['alive'] = b'run_session.py' in (proc / 'cmdline').read_bytes() and (proc / 'stat').read_text().split()[2] != 'Z'
    except (OSError, ValueError, IndexError):
        out['alive'] = False
    if session.get('state') in ('running', 'waiting_server') and not out['alive']:
        out['state'] = 'interrupted'
    return out


def main(resume=False):
    report = {'checked_epoch': time.time(), 'bf16': download_snapshot(resume),
              'inference': inference_snapshot(), 'trial': trial_snapshot(),
              'bf16_trial': bf16_trial_snapshot(), 'gpu': None}
    try:
        text = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL, timeout=3)
        report['gpu'] = [{'name': x[0].strip(), 'total_mib': int(x[1]), 'used_mib': int(x[2]), 'utilization': int(x[3])}
                         for x in (line.split(',') for line in text.strip().splitlines())]
    except Exception:
        pass
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main(len(sys.argv) > 1 and sys.argv[1] == 'resume')
