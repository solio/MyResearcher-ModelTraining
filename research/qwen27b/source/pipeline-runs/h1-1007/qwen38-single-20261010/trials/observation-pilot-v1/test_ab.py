"""Protocol tests, not actual semantic-model results."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json,subprocess,sys,tempfile,threading,time,unittest
D=Path(__file__).resolve().parent
class H(BaseHTTPRequestHandler):
 calls=[]
 def log_message(self,*a):pass
 def do_POST(self):
  p=json.loads(self.rfile.read(int(self.headers['Content-Length'])));H.calls.append(p)
  r=json.loads(p['messages'][1]['content']);shape=p['structured_outputs']['json'];arm='B' if list(shape['properties'])[1]=='observations' else 'A'
  obj={'id':r['id'],'observations':[],'checks':{k:{'state':'not_expressed'} for k in shape['properties']['checks']['properties']}}
  content=json.dumps(obj)
  if r['id']=='O1H007' and arm=='B':content='{bad'
  raw=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':content}}],'usage':{'prompt_tokens':1,'completion_tokens':1}}).encode()
  self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
class T(unittest.TestCase):
 def test_paired_inputs_blindness_bad_output_resume(self):
  server=ThreadingHTTPServer(('127.0.0.1',0),H);threading.Thread(target=server.serve_forever,daemon=True).start()
  try:
   with tempfile.TemporaryDirectory() as temp:
    cmd=[sys.executable,str(D/'run_ab.py'),'--deadline-epoch',str(time.time()+90),'--timeout','5','--base-url',f'http://127.0.0.1:{server.server_port}/v1','--out',temp]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=45);self.assertEqual(r.returncode,0,r.stderr)
    runtime=json.loads((Path(temp)/'runtime.json').read_text());self.assertEqual(runtime['attempted_requests'],160);self.assertEqual(runtime['paired_samples'],80)
    self.assertEqual(runtime['arms']['A']['format_valid'],80);self.assertEqual(runtime['arms']['B']['format_valid'],79)
    pairs={}
    for p in H.calls:
     inp=json.loads(p['messages'][1]['content']);pairs.setdefault(inp['id'],[]).append(p)
     self.assertNotIn('arm',inp);self.assertNotIn('reference',inp);self.assertEqual(p['temperature'],0);self.assertFalse(p['chat_template_kwargs']['enable_thinking'])
    self.assertEqual(len(pairs),80)
    for pair in pairs.values():
     self.assertEqual(len(pair),2);self.assertEqual(pair[0]['messages'][1],pair[1]['messages'][1]);self.assertNotEqual(pair[0]['messages'][0],pair[1]['messages'][0])
     shapes=[p['structured_outputs']['json'] for p in pair]
     self.assertEqual({tuple(x['properties']) for x in shapes},{('id','checks','observations'),('id','observations','checks')})
     # Both allow exactly the same response fields/values; only root generation order changes.
     for x in shapes:x['required']=sorted(x['required'])
     self.assertEqual(shapes[0],shapes[1])
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=20);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(len(H.calls),160)
  finally:server.shutdown();server.server_close()
if __name__=='__main__':unittest.main(verbosity=2)
