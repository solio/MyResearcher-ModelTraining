"""Probability sampling within deduplicated unseen stock-month short-text strata."""
from pathlib import Path
import json,hashlib,collections,random,math,re
D=Path(__file__).resolve().parent; ROOT=D.parents[3]
norm=lambda t:re.sub(r'\s+','',str(t))
hashtext=lambda t:hashlib.sha256(norm(t).encode()).hexdigest()
old=set(); files=[]; errors=[]
textkeys={'text','title','content','clean_text','raw_text','post_text','body','full_text','input_text'}
def visit(x):
 if isinstance(x,dict):
  for k,v in x.items():
   if k in textkeys and isinstance(v,str) and v.strip():old.add(hashtext(v))
   elif isinstance(v,(dict,list)):visit(v)
 elif isinstance(x,list):
  for v in x:visit(v)
for p in sorted((ROOT/'pipeline-runs').rglob('*')):
 if not p.is_file() or p.suffix not in ('.json','.jsonl') or 'qwen38-staged-20261010' in str(p):continue
 if not any(s in p.name.lower() for s in ('input','sample','dataset','candidate','reference','review')):continue
 try:
  if p.suffix=='.jsonl':
   with p.open() as f:
    for l in f:
     if l.strip():visit(json.loads(l))
  else:visit(json.load(p.open()))
  files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size})
 except Exception as e:errors.append({'path':str(p),'error':str(e)})
source=json.load(open(D/'source-snapshot.json')); stage1=json.load(open(D/'stage1-inputs.json'))['rows']; old.update(hashtext(r['text']) for r in stage1)
# reserve independent unused easy-candidate identities, no outputs/prompts consulted
chosen_ids={r.get('candidate_id') for r in stage1};cand=json.load(open(D/'candidates-stage1.json'))
reserve=[r for r in cand if r['candidate_id'] in [4,5,6,11,16,23,25,30,31,39,45,49,52,57,59,62,70,80,84,86,17,22,28,32,36,40,42,43,48,54,60,63,72,74,76] and r['candidate_id'] not in chosen_ids and hashtext(r['text']) not in old][:20]
for i,r in enumerate(reserve,1):r['id']=f'S1H{i:03}';r['source_kind']='real_directed_retry_reserve';old.add(hashtext(r['text']))
(D/'retry-reserve-inputs.json').write_text(json.dumps({'schema_version':'qwen38-staged-v1','purpose':'Held out for same-kind retry; not prompt examples; semantic reference not yet authored','rows':reserve},ensure_ascii=False,indent=2))
pop=collections.defaultdict(list);seen=set();counts=collections.Counter();rawstrata=collections.Counter()
for r in source:
 s=(r['stock_code'],r['published_at'][:7]);rawstrata[s]+=1
 t=r['text'];h=hashtext(t)
 if not t.strip():counts['empty']+=1;continue
 if len(t)>200:counts['over_200_chars']+=1;continue
 if h in old:counts['prior_experiment_or_stage1_or_reserve_text']+=1;continue
 key=s+(h,)
 if key in seen:counts['within_stock_month_duplicate_text']+=1;continue
 seen.add(key);pop[s].append(r)
assert len({s[0] for s in pop})==20
rng=random.Random(2026101002);sample=[];strata=[]
for stock in sorted({s[0] for s in pop}):
 ss=sorted(s for s in pop if s[0]==stock);alloc={s:1 for s in ss};remain=20-len(ss);capacity=sum(len(pop[s])-1 for s in ss)
 frac={s:remain*(len(pop[s])-1)/capacity if capacity else 0 for s in ss}
 for s in ss:alloc[s]+=math.floor(frac[s])
 left=20-sum(alloc.values())
 for s in sorted(ss,key=lambda s:(-(frac[s]-math.floor(frac[s])),s))[:left]:alloc[s]+=1
 for s in ss:
  n=alloc[s];N=len(pop[s]); selected=rng.sample(pop[s],n);p=n/N
  strata.append({'stock_code':s[0],'month':s[1],'source_records':rawstrata[s],'eligible_unique_short_unseen_texts':N,'sample_n':n,'inclusion_probability':p,'inverse_probability_weight':1/p})
  for r in selected:
   d={k:r[k] for k in ['source','source_item_id','stock_code','stock_name','published_at','url','text','text_hash']};d.update(id=f'S2N{len(sample)+1:03}',source_kind='stratified_natural_unseen_short_text',sampling_probability=p,inverse_probability_weight=1/p,sampling_stratum=f'{s[0]}:{s[1]}');sample.append(d)
assert len(sample)==400
assert not {hashtext(r['text']) for r in sample}&{hashtext(r['text']) for r in stage1}
(D/'stage2-inputs.json').write_text(json.dumps({'schema_version':'qwen38-staged-v1','reference_status':'unlabeled','rows':sample},ensure_ascii=False,indent=2))
manifest={'seed':2026101002,'collector_read_mode':'sqlite mode=ro','source_record_count':len(source),'source_date_min':min(r['published_at'] for r in source),'source_date_max':max(r['published_at'] for r in source),'source_stock_count':20,'text_build_policy':'content if title substring of content; otherwise title if no body or body substring of title; else title+newline+content. No truncation or inferred missing context.','prior_exclusion_normalization':'remove all unicode whitespace then sha256','prior_file_discovery':'pipeline-runs/**/*.json and .jsonl with input/sample/dataset/candidate/reference/review in filename, excluding current run; recursively collect text,title,content,clean_text,raw_text,post_text,body,full_text,input_text strings','prior_text_hash_count_including_stage1_and_reserve':len(old),'prior_files':files,'prior_parse_errors':errors,'coverage_limitation':'Known discoverable old experiment inputs under pipeline-runs only; not a claim of excluding every source ever used in historical training outside these artifacts.','exclusion_counts':dict(counts),'stage1':{'real':50,'synthetic':30,'selection':'keyword-retrieved candidates followed by full semantic reading; directed easy examples, NOT probability sample','reference':'single assistant read; all 17 predicates read per row; not human gold','real_month_counts':dict(collections.Counter(r['published_at'][:7] for r in stage1 if r.get('published_at')))},'stage2':{'n':400,'population':'available nonempty <=200-character unseen records, one representative of identical whitespace-normalized text per stock-month. No sentiment/keyword/model score filter.','allocation':'20 per stock; minimum one per eligible stock-month, remaining allocated proportional to residual population via largest remainder; uniform random draw without replacement within each stratum','reporting':'Raw averages reflect this balanced design, not whole-corpus prevalence; use recorded inverse probability weights to estimate the eligible unique-short-text population, not excluded long/old texts','strata':strata},'retry_reserve_count':len(reserve),'files_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['stage1-inputs.json','stage1-reference.json','stage2-inputs.json','retry-reserve-inputs.json']}}
(D/'sampling-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('prior files',len(files),'errors',errors,'known hashes',len(old),'stage2',len(sample),'strata',len(strata),'reserve',len(reserve),'eligible',sum(len(x) for x in pop.values()),'excluded',counts)
