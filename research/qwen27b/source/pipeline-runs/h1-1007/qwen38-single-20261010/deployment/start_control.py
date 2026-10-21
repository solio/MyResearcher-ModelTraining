"""Double-click launcher: start the loopback controller once and open its page."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def running(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/session', timeout=1) as response:
            return json.load(response).get('service') == 'q38-task-controller-v1'
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-open', action='store_true')
    parser.add_argument('--port', type=int, default=57710)
    args = parser.parse_args()
    port = args.port
    if not running(port):
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                print('端口已被其他程序占用；没有停止其他程序。请检查原控制服务。', file=sys.stderr)
                return 1
        directory = ROOT / 'control'; directory.mkdir(exist_ok=True, mode=0o700)
        with (directory / 'service.log').open('ab', buffering=0) as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name('control_server.py')),
                '--port', str(port)], cwd=ROOT, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                start_new_session=True, close_fds=True)
        (directory / 'service.pid').write_text(str(proc.pid) + '\n')
        for _ in range(40):
            if running(port):
                break
            if proc.poll() is not None:
                print('本地服务启动失败，请查看 control/service.log。', file=sys.stderr)
                return 1
            time.sleep(.25)
        else:
            print('本地服务尚未响应。', file=sys.stderr)
            return 1
    url = f'http://127.0.0.1:{port}/progress.html'
    print(url)
    if not args.no_open:
        webbrowser.open(url)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
