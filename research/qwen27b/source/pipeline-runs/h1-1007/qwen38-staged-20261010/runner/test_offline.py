#!/usr/bin/env python3
"""Small safety-relevant tests: missing/invalid ≠ neutral; subject/negation gate."""
import copy
import json
import unittest
import tempfile
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
from run_batch import HERE, normalize, build_prompt, request_one, load_rows
from evaluate import evaluate

SCHEMA = json.loads((HERE / 'meaning_schema.json').read_text())
ORDER = [x['id'] for x in SCHEMA['core_candidates']]


class OfflineChecks(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'test', 'text': '我今天没有买入，一股也没买。'}
        self.obj = {'id': 'test', 'supported': [], 'uncertain': [], 'negated': [
            {'id': 'action_buy_completed', 'quote': '没有买入', 'reason': '作者否认已买'}], 'observations': []}

    def test_explicit_negative_not_neutral_and_absence_distinct(self):
        p = normalize(json.dumps(self.obj, ensure_ascii=False), self.row, SCHEMA)
        self.assertEqual(p['judgments']['action_buy_completed']['state'], 'explicitly_negated')
        self.assertEqual(p['judgments']['emo_fear']['state'], 'not_expressed')
        self.assertIsNone(p['judgments']['emo_fear']['support'])

    def test_nonexact_quote_rejected(self):
        self.obj['negated'][0]['quote'] = '今天没买'
        with self.assertRaises(ValueError):
            normalize(json.dumps(self.obj), self.row, SCHEMA)

    def test_wrong_id_or_duplicate_rejected(self):
        self.obj['id'] = 'some_other_row'
        with self.assertRaises(ValueError):
            normalize(json.dumps(self.obj), self.row, SCHEMA)
        self.obj['id'] = 'test'
        self.obj['negated'].append(copy.deepcopy(self.obj['negated'][0]))
        with self.assertRaises(ValueError):
            normalize(json.dumps(self.obj), self.row, SCHEMA)

    def test_ambiguous_not_no_emotion(self):
        self.obj['uncertain'] = [{'id': 'emo_fear', 'quote': '一股也没买', 'reason': '无法确认情绪', 'support': .2}]
        p = normalize(json.dumps(self.obj), self.row, SCHEMA)
        self.assertEqual(p['judgments']['emo_fear']['decision'], 'abstain')

    def test_invalid_prediction_cannot_pass(self):
        ref = {'id': 'test', 'judgments': {k: 'N' for k in ORDER},
               'critical': [{'id': 'action_buy_completed', 'expected': 'N'}]}
        ref['judgments']['emo_fear'] = 'Y'
        result = evaluate([self.row], [ref], [{'id': 'test', 'status': 'error', 'judgments': None}], SCHEMA)
        self.assertFalse(result['gate']['passed'])
        self.assertEqual(result['per_class']['emo_fear']['FN'], 1)
        self.assertEqual(len(result['critical_failures']), 1)

    def test_basic_reversal_is_visible(self):
        self.obj['negated'] = []
        self.obj['supported'] = [{'id': 'action_buy_completed', 'quote': '买入', 'subject': 'self',
                                  'time': 'completed', 'target': 'unknown', 'intensity': None, 'support': .98}]
        pred = normalize(json.dumps(self.obj), self.row, SCHEMA)
        ref = {'id': 'test', 'judgments': {k: 'N' for k in ORDER},
               'critical': [{'id': 'action_buy_completed', 'expected': 'N', 'reason': '否定反转'}]}
        result = evaluate([self.row], [ref], [pred], SCHEMA)
        self.assertEqual(result['per_class']['action_buy_completed']['FP'], 1)
        self.assertEqual(len(result['critical_failures']), 1)
        self.assertFalse(result['gate']['passed'])

    def test_full_prompt_has_all_meanings(self):
        prompt = build_prompt(SCHEMA, HERE / 'prompt.txt')
        self.assertNotIn('{{MEANINGS}}', prompt)
        for key in ORDER:
            self.assertIn(key, prompt)

    def test_reasoning_effort_reaches_qwen_chat_template(self):
        args = Namespace(model='test', temperature=0, top_p=1, max_tokens=4096,
                         thinking='on', reasoning_effort='medium', timeout=2,
                         deadline_epoch=0, api_key_env='OFFLINE_EMPTY_TEST_KEY', base_url='http://127.0.0.1:1/v1')
        with patch('run_batch.urllib.request.urlopen', side_effect=RuntimeError('offline mock')):
            raw, pred = request_one(self.row, args, 'test prompt', SCHEMA, 'test-run')
        self.assertEqual(raw['request']['chat_template_kwargs'],
                         {'enable_thinking': True, 'reasoning_effort': 'medium'})
        self.assertNotIn('reasoning_effort', {k:v for k,v in raw['request'].items() if k != 'chat_template_kwargs'})
        self.assertEqual(pred['status'], 'error')

    def test_single_row_jsonl_is_loadable_for_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'predictions.jsonl'
            path.write_text(json.dumps({'id': 'first', 'status': 'error'}) + '\n')
            self.assertEqual(load_rows(path), [{'id': 'first', 'status': 'error'}])


if __name__ == '__main__':
    unittest.main()
