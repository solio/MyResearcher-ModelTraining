from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent
f=json.load(open(D/'data-freeze.json'))
for n,h in f['files'].items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h,n
x=json.load(open(D/'stage1-inputs.json'))['rows']; y=json.load(open(D/'stage1-reference.json'))['rows']
assert len(x)==len(y)==80
for a,b in zip(x,y):
 assert a['id']==b['id'] and a['text']==b['text']
 assert a['text_hash']==hashlib.sha256(a['text'].encode()).hexdigest()
 for j in b['judgments'].values():
  assert j['decision'] in ['supported','not_supported','abstain']
  assert not j['quote'] or j['quote'] in a['text']
  if j['decision']=='supported':assert j['quote']
print('PASS: frozen hashes, 80 ID/text/hash pairs, all judgment quote spans')
