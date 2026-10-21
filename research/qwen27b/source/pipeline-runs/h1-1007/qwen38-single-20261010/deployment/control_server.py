#!/usr/bin/env python3
"""Loopback-only SSH task controller. Credentials never enter files or logs."""
import argparse
from collections import deque
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import threading
import time
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'control'
SCRIPT = Path(__file__).with_name('control_remote.py')


def load(path, fallback):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback


def save(path, obj):
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix('.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
    tmp.replace(path)


def endpoint(body):
    host = str(body.get('host', '')).strip()
    user = str(body.get('username', 'root')).strip()
    try:
        port = int(body.get('port', 22))
    except (ValueError, TypeError):
        raise ValueError('端口须为 1～65535 的整数。')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]{0,252}', host):
        raise ValueError('请填写 IP 地址或主机名，不要包含 ssh 命令。')
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]{0,63}', user) or not 1 <= port <= 65535:
        raise ValueError('用户名或端口格式不正确。')
    password = body.get('password', '')
    if not isinstance(password, str) or len(password) > 1024 or '\n' in password or '\r' in password:
        raise ValueError('密码格式不正确。')
    return {'host': host, 'port': port, 'username': user}, password


class ConnectionFailure(Exception):
    pass


class AuthenticationFailure(ConnectionFailure):
    pass


class RemoteFailure(Exception):
    pass


class SSH:
    def __init__(self, target):
        self.target = target
        self.socket = '/tmp/q38-page-' + secrets.token_hex(8) + '.sock'
        self.destination = target['username'] + '@' + target['host']

    def connect(self, password):
        import pexpect
        args = ['-M', '-N', '-f', '-S', self.socket, '-p', str(self.target['port']),
                '-o', 'ControlPersist=600', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                '-o', 'ConnectTimeout=8', '-o', 'NumberOfPasswordPrompts=1',
                '-o', 'StrictHostKeyChecking=accept-new']
        args += ['-o', 'PreferredAuthentications=password,keyboard-interactive'] if password else ['-o', 'BatchMode=yes']
        child = pexpect.spawn('/usr/bin/ssh', args + [self.destination], encoding='utf-8',
                              timeout=15, echo=False)
        # Deliberately no logfile, argv/env password, or exception text containing transcripts.
        try:
            hit = child.expect([r'(?i)password:\s*$', pexpect.EOF, pexpect.TIMEOUT])
            if hit == 0:
                if not password:
                    raise AuthenticationFailure('需要 SSH 密码或已配置的密钥，请重新填写连接信息。')
                child.sendline(password)
                hit = child.expect([r'(?i)password:\s*$', pexpect.EOF, pexpect.TIMEOUT])
                if hit == 0:
                    raise AuthenticationFailure('SSH 认证失败，请检查用户名和密码。')
            if hit == 2:
                raise ConnectionFailure('SSH 连接超时，正在等待节点或网络恢复。')
            transcript = child.before or ''
            child.close()
            if child.exitstatus != 0:
                if 'Permission denied' in transcript:
                    raise AuthenticationFailure('SSH 认证失败，请检查用户名和密码。')
                if 'REMOTE HOST IDENTIFICATION HAS CHANGED' in transcript:
                    raise AuthenticationFailure('该地址的主机身份与已保存记录不同，请核对实例连接信息。')
                raise ConnectionFailure('SSH 连接未建立，请检查实例是否运行及地址、端口是否更新。')
        finally:
            if child.isalive():
                child.close(force=True)

    def probe(self, resume):
        cmd = ['/usr/bin/ssh', '-S', self.socket, '-p', str(self.target['port']),
               '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', self.destination,
               'python3 - ' + ('resume' if resume else 'inspect')]
        try:
            r = subprocess.run(cmd, input=SCRIPT.read_text(), text=True,
                               capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            raise ConnectionFailure('读取远端状态超时，正在重新连接。')
        if r.returncode == 255:
            raise ConnectionFailure('SSH 连接中断，正在自动重连。')
        if r.returncode:
            raise RemoteFailure('SSH 已连接，但远端任务读取失败；请检查 Python 或任务目录权限。')
        try:
            report = json.loads(r.stdout)
            if not isinstance(report, dict) or 'bf16' not in report:
                raise ValueError()
            return report
        except ValueError:
            raise RemoteFailure('SSH 已连接，但远端没有返回有效的任务状态。')

    def close(self):
        try:
            subprocess.run(['/usr/bin/ssh', '-S', self.socket, '-O', 'exit', '-p', str(self.target['port']),
                            self.destination], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            pass


class Controller:
    def __init__(self, root=ROOT, transport=SSH, poll_seconds=15):
        self.root, self.transport, self.poll_seconds = root, transport, poll_seconds
        self.lock = threading.RLock()
        self.generation = 0
        self.wake = threading.Event()
        self.events = deque(maxlen=30)
        self.target = load(root / 'control/endpoint.json', {'host': '223.109.239.11', 'port': 24452, 'username': 'root'})
        cached = load(root / 'progress.json', {})
        if 'inference' not in cached and cached:
            cached['inference'] = {k: cached.get(k) for k in ['attempted_unique', 'valid_responses', 'expected', 'format_error_count']}
            cached['inference']['complete'] = cached.get('attempted_unique') == cached.get('expected', 80)
        self.report = cached or None
        self.phase, self.message = 'idle', '填写 SSH 信息后，连接并恢复未完成的下载。'
        self.resume_requested = False

    def event(self, message):
        self.events.appendleft({'time': time.time(), 'message': message})

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps({'phase': self.phase, 'message': self.message, 'endpoint': self.target,
                'report': self.report, 'events': list(self.events), 'poll_seconds': self.poll_seconds}))

    def connect(self, body):
        target, password = endpoint(body)
        automatic = body.get('resume', True) is True
        with self.lock:
            self.generation += 1
            gen = self.generation
            self.wake.set(); self.wake = threading.Event()
            wake = self.wake
            self.target = target
            self.phase, self.message = 'connecting', '正在建立 SSH 连接，随后检查原任务与断点。'
            self.resume_requested = automatic
            self.event('开始连接 ' + target['host'] + ':' + str(target['port']))
            save(self.root / 'control/endpoint.json', target)
        threading.Thread(target=self.work, args=(gen, target, password, automatic, wake), daemon=True).start()

    def current(self, gen):
        return gen == self.generation

    def work(self, gen, target, password, automatic, wake):
        failures, last_task_state = 0, None
        client = None
        try:
            while self.current(gen):
                if client is None:
                    client = self.transport(target)
                    try:
                        client.connect(password)
                        with self.lock:
                            if not self.current(gen):
                                return
                            self.phase, self.message = 'connecting', 'SSH 已连接，正在读取任务状态。'
                            self.resume_requested = automatic
                            self.event('SSH 连接成功。')
                    except AuthenticationFailure as error:
                        with self.lock:
                            if self.current(gen):
                                self.phase, self.message = 'auth_error', str(error)
                                self.event(str(error))
                        return
                    except ConnectionFailure as error:
                        client.close(); client = None
                        failures += 1
                        with self.lock:
                            if self.current(gen):
                                self.phase, self.message = 'reconnecting', str(error)
                        wake.wait(min(60, 5 * 2 ** min(failures-1, 4))); wake.clear()
                        continue
                try:
                    with self.lock:
                        if not self.current(gen):
                            return
                        resume = self.resume_requested
                        self.resume_requested = False
                    report = client.probe(resume)
                    report['endpoint'] = target
                    with self.lock:
                        if not self.current(gen):
                            return
                        self.report = report
                        self.phase, self.message = 'connected', '已连接；任务状态持续更新。'
                        failures = 0
                        task = report['bf16']
                        state = (task.get('state'), task.get('resume_action'))
                        if state != last_task_state:
                            self.event(task.get('message', '任务状态已更新。'))
                            last_task_state = state
                        save(self.root / 'progress.json', report)
                        if task.get('found'):
                            save(self.root / 'bf16/latest-status.json', task)
                except ConnectionFailure as error:
                    client.close(); client = None
                    with self.lock:
                        if self.current(gen):
                            self.phase, self.message = 'reconnecting', str(error)
                    continue
                except RemoteFailure as error:
                    with self.lock:
                        if self.current(gen):
                            self.phase, self.message = 'remote_error', str(error)
                wake.wait(self.poll_seconds); wake.clear()
        except Exception:
            # Never leak a transport transcript or leave a dead worker marked live.
            with self.lock:
                if self.current(gen):
                    self.phase = 'local_error'
                    self.message = '本地跟踪发生异常，已停止本次连接；远端文件保留，请重新连接。'
                    self.event(self.message)
        finally:
            password = None  # Session-only credential; never serialized.
            if client:
                client.close()

    def resume(self):
        with self.lock:
            if self.phase not in ('connected', 'remote_error'):
                raise ValueError('请先连接 SSH。')
            self.resume_requested = True
            self.wake.set()
            self.event('已请求检查并恢复 BF16 下载；已运行或已完成的任务不会重复启动。')

    def disconnect(self):
        with self.lock:
            self.generation += 1
            self.wake.set()
            self.phase, self.message = 'idle', '本地跟踪已断开，远端下载继续运行。'
            self.event(self.message)


def handler_for(controller, token, root=ROOT):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, *args):
            pass  # Do not log requests, bodies, credentials, or remote transcripts.

        def allowed(self, mutation=False):
            port = self.server.server_port
            hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
            if self.headers.get('Host') not in hosts:
                return False
            origin = self.headers.get('Origin')
            if origin and origin not in {'http://' + h for h in hosts}:
                return False
            return not mutation or secrets.compare_digest(self.headers.get('X-CSRF-Token', '').encode(), token.encode())

        def respond(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if not self.allowed():
                return self.respond(403, {'error': '仅允许本机页面访问。'})
            path = urlsplit(self.path).path
            if path == '/api/session':
                return self.respond(200, {'token': token, 'service': 'q38-task-controller-v1'})
            if path == '/api/status':
                return self.respond(200, controller.snapshot())
            if path == '/':
                self.path = '/progress.html'; path = self.path
            local = (root / unquote(path).lstrip('/')).resolve()
            if not local.is_relative_to(root.resolve()) or 'control' in local.relative_to(root.resolve()).parts or local.is_dir():
                return self.respond(404, {'error': '文件不存在。'})
            super().do_GET()

        def do_HEAD(self):
            if not self.allowed():
                self.send_error(403)
                return
            path = (root / unquote(urlsplit(self.path).path).lstrip('/')).resolve()
            if not path.is_relative_to(root.resolve()) or 'control' in path.relative_to(root.resolve()).parts or path.is_dir():
                self.send_error(404)
                return
            super().do_HEAD()

        def do_POST(self):
            if not self.allowed(mutation=True):
                return self.respond(403, {'error': '请求来自其他页面或页面已过期，请刷新。'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 8192:
                    raise ValueError('请求大小不正确。')
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('请求格式须为 JSON。')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('请求格式不正确。')
                path = urlsplit(self.path).path
                if path == '/api/connect':
                    controller.connect(body)
                elif path == '/api/resume':
                    controller.resume()
                elif path == '/api/disconnect':
                    controller.disconnect()
                else:
                    return self.respond(404, {'error': '接口不存在。'})
                return self.respond(202, {'accepted': True})
            except (ValueError, TypeError) as error:
                # Only validation messages are exposed, never body/traceback/transcript.
                message = str(error) if type(error) is ValueError else '请求格式不正确。'
                return self.respond(400, {'error': message})
            except Exception:
                return self.respond(500, {'error': '本地服务未完成请求，任务文件未删除。'})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=57710)
    args = parser.parse_args()
    CONTROL.mkdir(exist_ok=True, mode=0o700)
    controller = Controller()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(controller, secrets.token_urlsafe(32)))
    print(f'Local task controller: http://127.0.0.1:{server.server_port}/progress.html', flush=True)
    try:
        server.serve_forever()
    finally:
        controller.disconnect(); server.server_close()


if __name__ == '__main__':
    main()
