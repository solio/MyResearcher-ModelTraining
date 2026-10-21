"""Task recovery and local API tests with no paid or remote requests."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import control_remote as remote
from control_server import Controller, ThreadingHTTPServer, endpoint, handler_for


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.task = self.root / 'task'; self.model = self.root / 'model'
        self.task.mkdir(); self.model.mkdir(); (self.task/'download.py').write_text('# fixture')
        self.meta = {'sha': remote.REVISION, 'siblings': [{'rfilename': 'weight', 'size': 3}]}
        (self.task/'repo_metadata.json').write_text(json.dumps(self.meta))
        self.started = []

    def tearDown(self):
        self.temp.cleanup()

    def snapshot(self, resume=True):
        return remote.download_snapshot(resume, self.task, self.model, self.started.append)

    def test_missing_disk_never_starts_fresh_download(self):
        (self.task/'repo_metadata.json').unlink(); self.assertEqual(self.snapshot()['state'], 'blocked')
        self.assertEqual(self.started, [])

    def test_partial_task_starts_resume(self):
        with patch.object(remote, 'alive', return_value=False):
            result = self.snapshot()
        self.assertEqual(result['resume_action'], 'started'); self.assertEqual(len(self.started), 1)

    def test_running_task_not_started_again(self):
        with patch.object(remote, 'alive', return_value=True), patch.object(remote, 'rpc', side_effect=OSError):
            result = self.snapshot()
        self.assertEqual(result['resume_action'], 'already_running'); self.assertEqual(self.started, [])

    def test_verified_completion_skips_resume(self):
        (self.model/'weight').write_bytes(b'abc')
        (self.task/'verification.json').write_text(json.dumps({'complete': True, 'revision': remote.REVISION,
            'files': [{'file': 'weight', 'valid': True}]}))
        result = self.snapshot(); self.assertEqual(result['state'], 'complete'); self.assertEqual(self.started, [])
        (self.model/'weight').unlink()
        self.assertFalse(remote.verified_on_disk(self.task, self.model, self.meta))

    def test_full_bytes_without_verification_not_complete(self):
        (self.task/'latest-status.json').write_text(json.dumps({'completed_bytes': 3}))
        with patch.object(remote, 'alive', return_value=True), patch.object(remote, 'rpc', side_effect=OSError):
            result = self.snapshot()
        self.assertEqual(result['state'], 'checking'); self.assertFalse(result['verified'])

    def test_read_only_connect_does_not_start(self):
        result = self.snapshot(False)
        self.assertEqual(result['state'], 'paused'); self.assertEqual(self.started, [])

    def test_metadata_path_escape_is_blocked(self):
        self.meta['siblings'][0]['rfilename'] = '../weight'
        (self.task/'repo_metadata.json').write_text(json.dumps(self.meta))
        self.assertEqual(self.snapshot()['state'], 'blocked'); self.assertEqual(self.started, [])


class FakeSSH:
    calls = []
    def __init__(self, target): self.target = target
    def connect(self, password): self.calls.append(('connect', bool(password)))
    def probe(self, resume):
        self.calls.append(('probe', resume))
        return {'checked_epoch': time.time(), 'bf16': {'found': True, 'state': 'downloading',
                'message': '续传中', 'resume_action': 'already_running' if resume else 'not_requested',
                'completed_bytes': 1, 'total_bytes': 10}, 'inference': {'complete': True}}
    def close(self): self.calls.append(('disconnect_ssh_only', True))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root/'bf16').mkdir(); (self.root/'progress.html').write_text('test page')
        FakeSSH.calls = []
        self.controller = Controller(self.root, FakeSSH, .03)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.controller, 'test-token', self.root))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.controller.disconnect(); time.sleep(.05)
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.temp.cleanup()

    def request(self, path, body=None, **headers):
        data = json.dumps(body).encode() if body is not None else None
        if data is not None: headers.setdefault('Content-Type', 'application/json')
        request = urllib.request.Request(self.url+path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def test_cross_origin_and_missing_token_cannot_connect(self):
        body = {'host': '127.0.0.1', 'port': 22, 'username': 'root', 'password': 'fake'}
        self.assertEqual(self.request('/api/connect', body)[0], 403)
        self.assertEqual(self.request('/api/connect', body, **{'X-CSRF-Token':'test-token','Origin':'https://example.com'})[0], 403)
        self.assertEqual(FakeSSH.calls, [])

    def test_connection_tracks_without_persisting_password(self):
        secret = 'TEST_CREDENTIAL_NOT_FOR_DISK'
        body = {'host': '127.0.0.1', 'port': 12345, 'username': 'root', 'password': secret, 'resume': True}
        self.assertEqual(self.request('/api/connect', body, **{'X-CSRF-Token':'test-token'})[0], 202)
        for _ in range(100):
            if self.controller.snapshot()['report']: break
            time.sleep(.01)
        status, response = self.request('/api/status'); self.assertEqual(status, 200)
        self.assertNotIn(secret, response.decode())
        self.assertIn(('probe', True), FakeSSH.calls)
        for f in self.root.rglob('*'):
            if f.is_file(): self.assertNotIn(secret, f.read_text())
        self.assertEqual(self.request('/control/endpoint.json')[0], 404)
        self.assertEqual(self.request('/api/disconnect', {}, **{'X-CSRF-Token':'test-token'})[0], 202)
        self.assertEqual(self.controller.snapshot()['phase'], 'idle')

    def test_bad_host_never_reaches_ssh(self):
        for host in ['-oProxyCommand=bad', 'host; echo hi', '$(date)', 'user@host']:
            with self.subTest(host=host):
                self.assertEqual(self.request('/api/connect', {'host':host}, **{'X-CSRF-Token':'test-token'})[0], 400)
        self.assertEqual(FakeSSH.calls, [])

    def test_dns_rebinding_host_rejected(self):
        self.assertEqual(self.request('/api/session', Host='evil.example:123')[0], 403)

    def test_connected_only_after_first_successful_probe(self):
        entered, release = threading.Event(), threading.Event()
        class SlowSSH(FakeSSH):
            def probe(self, resume):
                entered.set(); release.wait(2)
                return super().probe(resume)
        self.controller.transport = SlowSSH
        self.controller.connect({'host':'127.0.0.1', 'password':'fake'})
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(self.controller.snapshot()['phase'], 'connecting')
        finally:
            release.set()

    def test_unexpected_worker_error_is_sanitized_and_not_live(self):
        class BrokenSSH(FakeSSH):
            def connect(self, password):
                raise OSError('sensitive transcript ' + password)
        self.controller.transport = BrokenSSH
        self.controller.connect({'host':'127.0.0.1', 'password':'SECRET_TEST'})
        for _ in range(100):
            if self.controller.snapshot()['phase'] == 'local_error': break
            time.sleep(.01)
        state = self.controller.snapshot()
        self.assertEqual(state['phase'], 'local_error')
        self.assertNotIn('SECRET_TEST', json.dumps(state))


if __name__ == '__main__':
    unittest.main(verbosity=2)
