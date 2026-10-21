"""Local protocol/failure tests only. Mock answers are NEVER quality evidence."""
import copy
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'runner'))
from output_contract import normalize,response_schema
SCHEMA=json.loads((D/'runner/meaning_schema.json').read_text())
ROWS=json.loads((D/'data/model-inputs.json').read_text())['rows']
class Handler(BaseHTTPRequestHandler):
 calls=[];mode='malformed';errors=[]
 def log_message(self,*args):pass
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  r=json.loads(body['messages'][1]['content']);self.calls.append(body)
  if set(r)!={'id','text','stock_code','stock_name','published_at'}:self.errors.append('input allowlist violated')
  for token in ('rule_hits','day_context','day_z','day_density','hit_type','bearw_v2'):
   if token in json.dumps(body,ensure_ascii=False):self.errors.append('leak '+token)
  if list(body['structured_outputs']['json']['properties'])!=['id','observations','checks']:self.errors.append('schema ordering mismatch')
  obj={'id':r['id'],'observations':[],'checks':{k['id']:{'state':'not_expressed'} for k in SCHEMA['core_candidates']}}
  obs={'kind':'emotion','meaning':'模拟结构记录，不是真实标注','state':'expressed','quote':r['text'], 'subject':'self','target':'unknown','time':'unknown','time_text':'','intensity':0.2,'support':0.31,'reason':'仅用于本地传输和字段测试。'}
  obj['observations']=[obs]
  content=json.dumps(obj,ensure_ascii=False);finish='stop'
  if self.mode=='malformed':
   if r['id']==ROWS[0]['id']:content='{broken'
   elif r['id']==ROWS[1]['id']:obj['id']='wrong-id';content=json.dumps(obj,ensure_ascii=False)
   elif r['id']==ROWS[2]['id']:obj['observations'][0]['quote']='THIS IS NOT AN ORIGINAL QUOTE';content=json.dumps(obj,ensure_ascii=False)
   elif r['id']==ROWS[3]['id']:finish='length'
  if self.mode=='service_failure':
   self.send_response(503);self.end_headers();self.wfile.write(b'temporary service failure');return
  result={'choices':[{'finish_reason':finish,'message':{'content':content}}],'usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20}}
  raw=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)

class Pipeline(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
  threading.Thread(target=cls.server.serve_forever,daemon=True).start()
 @classmethod
 def tearDownClass(cls):cls.server.shutdown();cls.server.server_close()
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.work=Path(self.tmp.name);Handler.calls=[];Handler.errors=[];Handler.mode='malformed'
 def tearDown(self):self.tmp.cleanup()
 def run_batch(self,extra=(),input_path=None,out=None):
  cmd=[sys.executable,str(D/'runner/run_batch.py'),'--input',str(input_path or D/'data/model-inputs.json'),'--out',str(out or self.work/'out'),'--prompt',str(D/'runner/prompt.txt'),'--schema',str(D/'runner/meaning_schema.json'),'--model','local-mock-not-qwen','--model-revision','local-mock-only','--stage','k3_same_days_unhit','--base-url',f'http://127.0.0.1:{self.server.server_port}/v1','--thinking','off','--reasoning-effort','omit','--timeout','10','--smoke-count','1','--deadline-epoch',str(time.time()+120),'--max-tokens','2048',*extra]
  return subprocess.run(cmd,capture_output=True,text=True,timeout=60)
 def test_all2588_bad_outputs_continue_and_export_roundtrip(self):
  result=self.run_batch();self.assertEqual(result.returncode,2,result.stderr)
  runtime=json.loads((self.work/'out/runtime.json').read_text())
  self.assertEqual(runtime['completed_rows'],2588);self.assertEqual(runtime['successful_rows'],2584)
  self.assertEqual(len(Handler.calls),2588);self.assertFalse(Handler.errors)
  # Lost materialized prediction and torn final writes recover from durable receipts without requests.
  (self.work/'out/predictions.jsonl').unlink()
  with (self.work/'out/raw.jsonl').open('a') as f:f.write('{"unfinished":')
  result=self.run_batch();self.assertEqual(result.returncode,2,result.stderr)
  self.assertEqual(len(Handler.calls),2588)
  self.assertEqual(len((self.work/'out/predictions.jsonl').read_text().splitlines()),2588)
 def test_changed_prompt_resume_is_refused(self):
  f=self.work/'one.json';f.write_text(json.dumps({'rows':ROWS[:1]}));Handler.mode='ok'
  self.assertEqual(self.run_batch(input_path=f).returncode,0)
  changed=self.work/'changed.txt';changed.write_text((D/'runner/prompt.txt').read_text()+'\nchanged')
  result=self.run_batch(['--prompt',str(changed)],input_path=f)
  self.assertNotEqual(result.returncode,0);self.assertIn('Resume refused',result.stderr);self.assertEqual(len(Handler.calls),1)
 def test_lock_and_deadline_do_not_submit_requests(self):
  out=self.work/'locked';out.mkdir();lock=(out/'runner.lock').open('a')
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.assertNotEqual(self.run_batch(out=out).returncode,0);self.assertEqual(len(Handler.calls),0);lock.close()
  result=self.run_batch(['--deadline-epoch',str(time.time()+3)])
  self.assertEqual(result.returncode,2);self.assertEqual(len(Handler.calls),0)
 def test_service_failure_resume_does_not_stick_at_old_smoke(self):
  f=self.work/'two.json';f.write_text(json.dumps({'rows':ROWS[:2]}));Handler.mode='service_failure'
  self.assertEqual(self.run_batch(input_path=f).returncode,2)
  Handler.mode='ok';self.assertEqual(self.run_batch(input_path=f).returncode,2)
  runtime=json.loads((self.work/'out/runtime.json').read_text())
  self.assertTrue(runtime['all_attempted']);self.assertEqual(runtime['successful_rows'],1)
  self.assertEqual(len(Handler.calls),2)
 def test_new_batch_cap_does_not_expand_to_corpus(self):
  rows=[{**ROWS[0],'id':f'extra{i}'} for i in range(2589)]
  f=self.work/'too-many.json';f.write_text(json.dumps({'rows':rows}))
  result=self.run_batch(input_path=f)
  self.assertNotEqual(result.returncode,0);self.assertIn('cannot exceed 2588',result.stderr)
  self.assertEqual(len(Handler.calls),0)
 def test_negation_and_subjects_survive(self):
  row={'id':'test','text':'我没有卖，他吓坏了'}
  obs={'kind':'emotion','meaning':'他人恐惧','state':'expressed','quote':'他吓坏了','subject':'other_retail','target':'unknown','time':'current','time_text':'','intensity':0.8,'support':0.7,'reason':'结构测试'}
  claim={'state':'explicitly_negated','quote':'我没有卖','subject':'self','target':'unknown','time':'past','time_text':'','intensity':None,'support':None,'reason':'结构测试'}
  obj={'id':'test','observations':[obs],'checks':{r['id']:{'state':'not_expressed'} for r in SCHEMA['core_candidates']}}
  obj['checks']['sell_declaration']={'claims':[claim]}
  pred=normalize(json.dumps(obj,ensure_ascii=False),row,SCHEMA)
  self.assertEqual(pred['judgments']['sell_declaration']['decision'],'not_supported')
  self.assertEqual(pred['observations'][0]['subject'],'other_retail')
  self.assertIsNone(pred['judgments']['sell_declaration']['claims'][0]['intensity'])

if __name__=='__main__':unittest.main(verbosity=2)
