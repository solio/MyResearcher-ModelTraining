"""Join unchanged local-only source metadata to model results after inference."""
from pathlib import Path
import argparse,json,hashlib
D=Path(__file__).resolve().parent
read=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
def export(out,results):
 m=D/'evaluation-manifest.json'
 if m.exists():
  for n,h in json.loads(m.read_text())['files_sha256'].items():
   if hashlib.sha256((D/n).read_bytes()).hexdigest()!=h:raise ValueError('Frozen local file changed: '+n)
 side=read(D/'private/evaluation-sidecar.jsonl');preds=read(results/'predictions.jsonl');by={r['id']:r for r in preds}
 assert len(by)==len(preds) and not(set(by)-{r['id'] for r in side})
 out.mkdir(parents=True,exist_ok=True);returned=[]
 for src in side:
  p=by.get(src['id'],{});valid=p.get('status')=='ok';obs=p.get('observations',[]) if valid else None
  categories={kind:[o['meaning'] for o in (obs or []) if o['kind']==kind and o['state']=='expressed' and (o['subject']=='self' or kind=='style' and o['subject']=='text')] if valid else None for kind in ['emotion','stance','action','style']}
  returned.append({**src,'inference_status':p.get('status','pending'),'error':p.get('error'),'emotions':categories['emotion'],'stances':categories['stance'],'actions':categories['action'],'styles':categories['style'],'confidence':None,'confidence_note':'Scoped support is uncalibrated. No fabricated aggregate confidence or score filtering.','observations':obs,'checks':p.get('checks') if valid else None,'judgments':p.get('judgments') if valid else None,'day_extreme_vote':None,'day_extreme_status':'not_estimable_from_selected_hit_titles_only','raw_receipt_file':'results/main/raw.jsonl','raw_receipt_id':src['id'],'semantic_review_status':'unreviewed'})
 (out/'k3-return.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in returned))
 summary={'expected':len(side),'attempted':len(preds),'structure_valid':sum(r.get('status')=='ok' for r in preds),'pending':len(side)-len(preds),'all_attempted':len(side)==len(preds),'semantic_quality':'unreviewed','day_extreme_vote_available':False,'notice':'Format failures preserve raw replies; missing/negated/ambiguous are distinct. Source selection is not semantic truth. All1808source rows retained.'}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=D/'exports');p.add_argument('--results',type=Path,default=D/'results/main');a=p.parse_args();print(json.dumps(export(a.out,a.results)))
