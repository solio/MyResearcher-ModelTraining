"""Mechanical export of complete original answers, with explicit reused-receipt provenance."""
from pathlib import Path
import collections,hashlib,json,time
D=Path(__file__).resolve().parent

def atomic(p,b):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(b);q.replace(p)

def main():
 source=[json.loads(l) for l in (D/'private/evaluation-sidecar.jsonl').read_text().splitlines()]
 reused={r['target_id']:r for r in map(json.loads,(D/'private/reused-prior.jsonl').read_text().splitlines())}
 path=D/'results/main/raw.jsonl';new={}
 if path.exists():
  for line in path.read_text().splitlines():
   r=json.loads(line)
   if r['id'] in new:raise ValueError('Duplicate new receipt')
   new[r['id']]=r
 output=[];pending=[];days=collections.defaultdict(lambda:{'population':0,'received':0,'contract_ok':0,'reused':0})
 for row in source:
  rid=row['id'];receipt=new.get(rid);provenance='new_request'
  if rid in reused:
   if receipt is not None:raise ValueError('Previously reused row was also rerun')
   receipt=reused[rid]['original_raw_receipt'];provenance='reused_prior388'
  cell=days[(row['stock_code'],row['day'])];cell['population']+=1
  if receipt is None:pending.append(rid);continue
  content=None;parsed=None
  try:content=receipt['response']['choices'][0]['message']['content']
  except (KeyError,IndexError,TypeError):pass
  try:
   def unique(pairs):
    d={}
    for k,v in pairs:
     if k in d:raise ValueError('Duplicate JSON key')
     d[k]=v
    return d
   parsed=json.loads(content,object_pairs_hook=unique)
  except (ValueError,TypeError):pass
  pred=receipt.get('prediction') or {};cell['received']+=1;cell['contract_ok']+=pred.get('status')=='ok';cell['reused']+=provenance=='reused_prior388'
  output.append({**row,'qwen_model_output':parsed,'qwen_model_content':content,'original_contract_status':pred.get('status'),'original_contract_error':pred.get('error'),'semantic_quality':'not_assessed','receipt_provenance':{'source':provenance,'original_id':receipt['id'],'target_id':rid,'model_output_id_left_unchanged':True}})
 data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output).encode();atomic(D/'exports/k3-return-with-reuse.jsonl',data)
 atomic(D/'exports/missing-ids.json',json.dumps(pending).encode())
 summary={'updated_epoch':time.time(),'new_requests_returned':len(new),'new_requests_expected':2588,'reused_prior':len(reused),'complement_received':len(output),'complement_expected':2594,'missing':len(pending),'unique_new_ids':len(new),'sha256':hashlib.sha256(data).hexdigest(),'semantic_quality':'not_assessed','day_coverage':[{'stock_code':c,'day':d,**v} for (c,d),v in sorted(days.items())],'scope':'Only original selected stock-date cells; rule non-hit is not a calm label. Received count is not semantic correctness.'}
 atomic(D/'exports/summary.json',json.dumps(summary,ensure_ascii=False,indent=2).encode())
if __name__=='__main__':main()
