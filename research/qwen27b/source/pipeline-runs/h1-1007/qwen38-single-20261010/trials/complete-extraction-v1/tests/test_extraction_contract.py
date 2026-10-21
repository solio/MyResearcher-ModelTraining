"""Protocol/queue regression tests. These are not model-quality evaluations."""
import copy
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runner'))
from output_contract import normalize, response_schema

SCHEMA = json.loads((ROOT / 'runner/meaning_schema.json').read_text())


def empty(ident='test'):
    return {'id': ident, 'checks': {x['id']: {'state': 'not_expressed'}
                                  for x in SCHEMA['core_candidates']}, 'observations': []}


def claim(quote, state='expressed', time_text='', target='unknown', when='current'):
    return {'state': state, 'quote': quote, 'subject': 'self', 'target': target,
            'time': when, 'time_text': time_text, 'intensity': None if state == 'explicitly_negated' else 0.8,
            'support': 0.9, 'reason': '信息不足' if state == 'ambiguous' else '仅为协议测试提供的非空说明，不代表语义正确。'}


def parse(obj, text='原文'):
    return normalize(json.dumps(obj, ensure_ascii=False), {'id': 'test', 'text': text}, SCHEMA)


class ContractTests(unittest.TestCase):
    def test_missing_is_not_defaulted_to_negative(self):
        obj = empty()
        del obj['checks']['emo_fear']
        with self.assertRaisesRegex(ValueError, 'missing=.*emo_fear'):
            parse(obj)

    def test_absence_does_not_gain_zero_scores(self):
        p = parse(empty())
        self.assertIsNone(p['judgments']['emo_fear']['intensity'])
        self.assertTrue(p['judgments']['emo_fear']['explicit_coverage'])

    def test_duplicate_json_key_rejected_before_overwrite(self):
        text = json.dumps(empty()).replace('"state": "not_expressed"',
                 '"state": "expressed", "state": "not_expressed"', 1)
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            normalize(text, {'id': 'test', 'text': '原文'}, SCHEMA)

    def test_same_evidence_cannot_be_in_both_states(self):
        obj = empty()
        obj['checks']['action_buy_completed'] = {'claims': [claim('买了'), claim('买了', 'explicitly_negated')]}
        with self.assertRaisesRegex(ValueError, 'duplicate/conflicting'):
            parse(obj, '买了')

    def test_different_times_keep_both_states_and_scores(self):
        obj = empty()
        a = claim('昨天很害怕', time_text='昨天', when='past')
        b = claim('今天不怕', 'explicitly_negated', time_text='今天')
        obj['checks']['emo_fear'] = {'claims': [a, b]}
        p = parse(obj, '昨天很害怕，今天不怕')
        self.assertEqual(p['judgments']['emo_fear']['claims'], [a, b])
        self.assertFalse(p['needs_scope_review'])
        self.assertTrue(p['judgments']['emo_fear']['projection_only'])
        self.assertIsNone(p['judgments']['emo_fear']['intensity'])

    def test_different_targets_are_not_merged(self):
        obj = empty()
        a, b = claim('买了甲', target='甲'), claim('没买乙', 'explicitly_negated', target='乙')
        obj['checks']['action_buy_completed'] = {'claims': [a, b]}
        p = parse(obj, '买了甲，没买乙')
        self.assertFalse(p['needs_scope_review'])
        self.assertEqual(len(p['judgments']['action_buy_completed']['claims']), 2)

    def test_possible_scope_conflict_is_preserved_and_flagged(self):
        obj = empty()
        obj['checks']['emo_fear'] = {'claims': [claim('害怕'), claim('不怕', 'explicitly_negated')]}
        p = parse(obj, '说害怕，又说不怕')
        self.assertTrue(p['needs_scope_review'])
        self.assertEqual(len(p['checks']['emo_fear']['claims']), 2)

    def test_fabricated_quote_is_rejected(self):
        obj = empty()
        obj['checks']['emo_fear'] = {'claims': [claim('十分恐慌')]}
        with self.assertRaisesRegex(ValueError, 'exact nonempty'):
            parse(obj, '没说过这句话')

    def test_fabricated_time_anchor_is_rejected(self):
        obj = empty()
        obj['checks']['emo_fear'] = {'claims': [claim('害怕', time_text='昨天')]}
        with self.assertRaisesRegex(ValueError, 'time_text'):
            parse(obj, '害怕')

    def test_invalid_score_and_denial_intensity_are_rejected(self):
        for value in [True, float('nan'), 1.5]:
            with self.subTest(value=value):
                obj = empty(); c = claim('害怕'); c['support'] = value
                obj['checks']['emo_fear'] = {'claims': [c]}
                with self.assertRaises(ValueError):
                    parse(obj, '害怕')
        obj = empty(); c = claim('不怕', 'explicitly_negated'); c['intensity'] = 0.9
        obj['checks']['emo_fear'] = {'claims': [c]}
        with self.assertRaisesRegex(ValueError, 'denial'):
            parse(obj, '不怕')

    def test_ambiguous_is_not_a_negative(self):
        obj = empty(); obj['checks']['emo_fear'] = {'claims': [claim('这个', 'ambiguous')]}
        p = parse(obj, '这个')
        self.assertEqual(p['judgments']['emo_fear']['decision'], 'abstain')

    def test_extra_meanings_and_their_scores_are_preserved(self):
        obj = empty(); obs = claim('累了'); obs.pop('state'); obs['meaning'] = '疲惫'
        obj['observations'] = [obs]
        self.assertEqual(parse(obj, '累了')['observations'], [obs])

    def test_empty_reason_rejected_for_every_claim_state(self):
        for state in ['expressed', 'explicitly_negated', 'ambiguous']:
            for value in ['', '  ', None]:
                with self.subTest(state=state, value=value):
                    obj = empty(); c = claim('害怕', state); c['reason'] = value
                    obj['checks']['emo_fear'] = {'claims': [c]}
                    with self.assertRaisesRegex(ValueError, 'nonempty reason'):
                        parse(obj, '害怕')

    def test_empty_extra_observation_reason_rejected(self):
        obj = empty(); obs = claim('累了'); obs.pop('state'); obs['meaning'] = '疲惫'; obs['reason'] = ''
        obj['observations'] = [obs]
        with self.assertRaisesRegex(ValueError, 'nonempty reason'):
            parse(obj, '累了')

    def test_structurally_valid_can_still_be_semantically_wrong(self):
        obj = empty(); obj['checks']['action_clear'] = {'claims': [claim('不要加仓')]}
        p = parse(obj, '不要加仓')
        self.assertEqual(p['status'], 'ok')
        self.assertIn('not certified', p['quality_notice'])
        # Intentional: the validator must not pretend it can adjudicate language.

    def test_original_meaning_definitions_are_unchanged(self):
        old = json.loads((ROOT / 'baseline/meaning_schema.json').read_text())
        self.assertEqual(SCHEMA['core_candidates'], old['core_candidates'])

    def test_json_schema_examples(self):
        import jsonschema
        s = response_schema(SCHEMA)
        jsonschema.Draft202012Validator.check_schema(s)
        jsonschema.validate(empty(), s)
        obj = empty(); obj['checks']['emo_fear'] = {'claims': [claim('害怕')]}
        jsonschema.validate(obj, s)
        del obj['checks']['emo_regret']
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(obj, s)


class QueueTests(unittest.TestCase):
    def test_invalid_first_record_does_not_stop_queue_and_resume_does_not_repeat(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                row = json.loads(request['messages'][1]['content'])
                received.append(row['id'])
                answer = empty(row['id'])
                if row['id'] == 'bad-first':
                    del answer['checks']['emo_fear']
                response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}],
                            'usage': {'completion_tokens': 1, 'prompt_tokens': 1, 'total_tokens': 2}}
                body = json.dumps(response).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                t = Path(tmp); inp = t / 'inputs.json'; out = t / 'out'
                inp.write_text(json.dumps([{'id': x, 'text': '用于协议测试，不评价模型。'}
                                          for x in ['bad-first', 'next', 'last']]))
                cmd = [sys.executable, str(ROOT / 'runner/run_batch.py'), '--input', str(inp), '--out', str(out),
                       '--base-url', f'http://127.0.0.1:{server.server_port}/v1', '--model', 'offline-fixture',
                       '--model-revision', 'offline-fixture', '--deadline-epoch', str(time.time()+300),
                       '--timeout', '10', '--smoke-count', '1', '--thinking', 'off', '--reasoning-effort', 'omit']
                first = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                self.assertEqual(first.returncode, 2, first.stderr)
                runtime = json.loads((out / 'runtime.json').read_text())
                self.assertTrue(runtime['all_attempted'])
                self.assertEqual(runtime['successful_rows'], 2)
                self.assertEqual(runtime['stop_reason'], 'complete')
                self.assertEqual(runtime['pending_ids'], [])
                self.assertCountEqual(received, ['bad-first', 'next', 'last'])
                raw_before = (out / 'raw.jsonl').read_bytes()
                second = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                self.assertEqual(second.returncode, 2, second.stderr)
                self.assertEqual(len(received), 3)
                self.assertEqual((out / 'raw.jsonl').read_bytes(), raw_before)
                prompt = t / 'changed-prompt.txt'
                prompt.write_text((ROOT / 'runner/prompt.txt').read_text() + '\n修改版本\n')
                changed = subprocess.run(cmd + ['--prompt', str(prompt)], capture_output=True, text=True, timeout=20)
                self.assertNotEqual(changed.returncode, 0)
                self.assertIn('Resume refused', changed.stderr)
                self.assertEqual(len(received), 3)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
