"""Reproducible local comparison. Original inputs, receipts and semantic contracts remain unchanged."""
from pathlib import Path
import collections, copy, hashlib, importlib.util, json, math, statistics, tarfile, time

D = Path(__file__).resolve().parent
T = D.parent / 'k3-top5-bf16-1808-v1'
N = D.parent / 'k3-same-days-unhit-v1'
OLD = D.parent / 'k3-bf16-388-v1'
CORE = ['bearish_expression', 'sell_declaration', 'high_arousal_expression', 'buy_declaration']
LABELS = ['看跌判断／对下跌或弱势的负面评价', '作者自己的卖出／减仓表达', '明显强烈的情绪或表达（含正负两向）', '作者自己的买入／加仓表达']

def read(p): return json.loads(p.read_text())
def lines(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, value): (D / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))
def identity(row): return tuple(row[k] for k in ('stock_code', 'published_at', 'title'))

spec = importlib.util.spec_from_file_location('frozen_contract', N / 'runner/output_contract.py')
contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)
schema = read(N / 'runner/meaning_schema.json')

def field_decision(row, key):
    """Check each original slot independently; errors elsewhere cannot turn this slot into a negative."""
    obj = row.get('qwen_model_output')
    if not isinstance(obj, dict): return {'decision':'unknown', 'reason':'原回答未能完整解析', 'claims':[]}
    check = obj.get('checks', {}).get(key)
    candidate = {'id':obj.get('id'), 'observations':[], 'checks':{k:({'state':'not_expressed'} if k != key else check) for k in CORE}}
    expected_id = row.get('receipt_provenance', {}).get('original_id', row['id'])
    try:
        checked = contract.normalize(json.dumps(candidate,ensure_ascii=False), {'id':expected_id,'text':row['title']}, schema)
        claims = check.get('claims', [])
    except (ValueError, TypeError, KeyError) as exc:
        return {'decision':'unknown', 'reason':str(exc), 'claims':check.get('claims',[]) if isinstance(check,dict) else []}
    positive = [c for c in claims if c['state']=='expressed' and c['subject']=='self']
    ambiguous = [c for c in claims if c['state']=='ambiguous' and c['subject'] in ('self','unknown') or c['state']=='expressed' and c['subject']=='unknown']
    # A same-scope contradiction is unknown, never resolved by choosing a convenient claim.
    scopes = collections.defaultdict(set)
    for c in claims:
        if c['subject']=='self': scopes[tuple(c[k] for k in ('subject','target','time','time_text'))].add(c['state'])
    conflicts = any({'expressed','explicitly_negated'} <= states for states in scopes.values())
    decision = 'unknown' if conflicts else 'positive' if positive else 'unknown' if ambiguous else 'not_positive'
    return {'decision':decision, 'reason':'同一语境肯否冲突' if conflicts else '保留原命题、主体与肯否；未另设支持度阈值', 'claims':claims,
            'other_subject_positive':any(c['state']=='expressed' and c['subject']!='self' for c in claims),
            'negative_kind':None if decision!='not_positive' else 'not_expressed' if not claims else 'negated_or_other_subject',
            'original_record_contract_ok':row['original_contract_status']=='ok'}

def main():
    D.mkdir(exist_ok=True)
    verification=[]
    for p, snap, result in [(T,'20261013-complete','k3-full-raw-return.jsonl'),(N,'20261013-final','k3-return-with-reuse.jsonl')]:
        archive=p/'snapshots'/snap/'remote-snapshot.tgz'
        with tarfile.open(archive) as tf:
            manifest=json.load(tf.extractfile('snapshot-manifest.json'))
            for f,h in manifest['files_sha256'].items(): assert hashlib.sha256(tf.extractfile(f).read()).hexdigest()==h,(p.name,f)
            raw_hash=manifest['files_sha256']['results/main/raw.jsonl']
            assert sha(p/'results/main/raw.jsonl')==raw_hash
        inputs=read(p/'data/model-inputs.json')['rows']; receipts=lines(p/'results/main/raw.jsonl')
        assert len(receipts)==len({x['id'] for x in receipts})==len(inputs)
        assert {x['id'] for x in inputs}=={x['id'] for x in receipts}
        assert sha(p/'exports'/result)==sha(p/'snapshots'/snap/('k3-raw-return.jsonl' if p==T else result))
        verification.append({'batch':p.name,'new_receipts':len(receipts),'missing_ids':[], 'archive_verified_files':len(manifest['files_sha256']), 'archive_sha256':sha(archive),'raw_sha256':raw_hash,'export_sha256':sha(p/'exports'/result)})
    for file in ['prompt.txt','meaning_schema.json','output_contract.py']:
        assert sha(T/'runner'/file)==sha(N/'runner'/file)
    reuses=lines(N/'private/reused-prior.jsonl'); old_raw={x['id']:x for x in lines(OLD/'results/main/raw.jsonl')}
    next_inputs={x['id']:x for x in lines(N/'private/evaluation-sidecar.jsonl')}
    for reuse in reuses:
        assert reuse['original_raw_receipt']==old_raw[reuse['prior_receipt_id']]
        a=reuse['original_input']; b=next_inputs[reuse['target_id']]
        assert a['text']==b['title'] and all(a[k]==b[k] for k in ['stock_code','stock_name','published_at'])
    originals=lines(T/'exports/k3-full-raw-return.jsonl'); complementary=lines(N/'exports/k3-return-with-reuse.jsonl')
    combined=originals+complementary; assert len(combined)==len({x['id'] for x in combined})==4402
    population=lines(N/'private/full-day-source.jsonl')
    assert collections.Counter(identity(x) for x in combined)==collections.Counter(identity(x) for x in population)
    for group,flag in [(originals,True),(complementary,False)]:
        assert collections.Counter(identity(x) for x in group)==collections.Counter(identity(x) for x in population if x['rule_hit']==flag)
    source_lookup=collections.defaultdict(list)
    for source in population: source_lookup[identity(source)].append(source)
    derived=[]
    for row in combined:
        src=source_lookup[identity(row)].pop()
        decisions={k:field_decision(row,k) for k in CORE}
        targets={}
        for key, decision in decisions.items():
            claims=[c for c in decision['claims'] if decision['decision']=='positive' and c['subject']=='self' and c['state']=='expressed']
            targets[key]='not_positive' if not claims else 'explicit_stock_string' if any(row['stock_name'] in c['target'] or row['stock_code'] in c['target'] for c in claims) else 'unknown_target' if any(c['target']=='unknown' for c in claims) else 'other_target_string_requires_review'
        derived.append({**row, 'source':src['source'],'source_item_id':src['source_item_id'],'rule_hit':src['rule_hit'], 'bearw_v2_hit':src['bearw_v2_hit'],'nick_hit':src['nick_hit'], 'analysis_checks':decisions,'target_field_accounting':targets})
    cells=collections.defaultdict(list)
    for row in derived: cells[(row['stock_code'],row['day'])].append(row)
    assert len(cells)==40
    days=[]
    for (stock,day),group in sorted(cells.items()):
        matched=[x for x in group if x['rule_hit']]
        expected={x['day_posts'] for x in matched}; assert expected=={len(group)},(stock,day,expected,len(group))
        n=len(group); hit=len(matched)
        c={'stock_code':stock,'stock_name':group[0]['stock_name'],'day':day,'posts':n,'received':n,'rule_positive':hit,'rule_density':hit/n,'original_rule_z':matched[0]['day_z'],'model_checks':{}}
        for k in CORE:
            counts=collections.Counter(x['analysis_checks'][k]['decision'] for x in group)
            pp=sum(x['rule_hit'] and x['analysis_checks'][k]['decision']=='positive' for x in group)
            np=sum(not x['rule_hit'] and x['analysis_checks'][k]['decision']=='positive' for x in group)
            c['model_checks'][k]={'positive':counts['positive'],'not_positive':counts['not_positive'],'unknown':counts['unknown'],'rule_hit_and_model_positive':pp,'rule_unhit_and_model_positive':np,'density_lower':counts['positive']/n,'density_upper':(counts['positive']+counts['unknown'])/n,'target_field_accounting':dict(collections.Counter(x['target_field_accounting'][k] for x in group if x['analysis_checks'][k]['decision']=='positive'))}
        days.append(c)
    matrices={}
    for k,label in zip(CORE,LABELS):
        groups={}
        for flag,name in [(True,'rule_hit'),(False,'rule_unhit')]:
            group=[x for x in derived if x['rule_hit']==flag]
            counts=collections.Counter(x['analysis_checks'][k]['decision'] for x in group)
            strict=collections.Counter(x['analysis_checks'][k]['decision'] for x in group if x['original_contract_status']=='ok')
            groups[name]={'total':len(group),'positive':counts['positive'],'not_positive':counts['not_positive'],'unknown':counts['unknown'],'original_contract_ok_only':dict(strict)}
        selfclaims=[c for x in derived for c in x['analysis_checks'][k]['claims'] if x['analysis_checks'][k]['decision']=='positive' and c['subject']=='self' and c['state']=='expressed']
        def distribution(field):
            values=[c[field] for c in selfclaims if c.get(field) is not None]
            return {'n_claims':len(values),'median':statistics.median(values) if values else None,'counts':dict(sorted(collections.Counter(values).items()))}
        matrices[k]={'label':label,**groups,'intensity':distribution('intensity'),'support':distribution('support'),'target_field_accounting':dict(collections.Counter(x['target_field_accounting'][k] for x in derived if x['analysis_checks'][k]['decision']=='positive'))}
    # Freeze an equal-sized diagnostic sample from every stock and every comparison cell.
    # Not prevalence-weighted and not a held-out benchmark; selection uses a fixed SHA key.
    selected=[]
    for stock in sorted({x['stock_code'] for x in derived}):
        for rule in [True,False]:
            for state in ['positive','not_positive']:
                candidates=[x for x in derived if x['stock_code']==stock and x['rule_hit']==rule and x['analysis_checks']['bearish_expression']['decision']==state]
                candidates.sort(key=lambda x:hashlib.sha256(('review-20261013-v1|'+x['id']).encode()).hexdigest())
                for row in candidates[:4]:
                    selected.append({'id':row['id'],'stock':stock,'cell':('rule_hit' if rule else 'rule_unhit')+'__'+state,'title':row['title'],'model_bearish':row['analysis_checks']['bearish_expression'],'observations':(row.get('qwen_model_output') or {}).get('observations',[]),'original_contract_status':row['original_contract_status']})
    errors=collections.Counter(x['original_contract_status'] for x in derived)
    summary={'created_epoch':time.time(),'received':4402,'rule_hit':1808,'rule_unhit':2594,'new_receipts':4396,'reused_receipts':6,'missing':0,'stock_date_cells':40,'original_contract_counts':dict(errors),'complete_json_outputs':sum(isinstance(x.get('qwen_model_output'),dict) for x in derived),'truncated_titles_possible':sum(len(x['title'])==40 for x in derived),'comparison':matrices,'diagnostic_sample_size':len(selected),
             'definitions':{'positive':'原命题存在 expressed 且 subject=self 的有效声明；保留所有支持度，不新增阈值。','not_positive':'模型未表达该命题，或只是否认／其他主体；不代表平静。','unknown':'不完整、该项格式不成立、主体不明、歧义或肯否冲突；不当作0。','field_recovery':'只对原四项逐项运行原校验器；其他观察的引用错误不会连带删除有效项；不改原文或模型回答。','daily_density':'作者自身肯定表达帖子数 / 当日全部帖子；未知项给上下界；不是恐惧分数。','selection':'5股40日均来自K3规则挑出的极端日，而且股票按既往t值筛选；不能估计全市场泛化或验证收益提高。','score':'强度与支持度原样保留，均未校准；不拟合盈利权重，不把强度当仓位。'},'verification':verification,'automation':'No existing k3-10-13 automation was found by the app when pausing; no new automation created.'}
    save('verification.json',{'created_epoch':time.time(),'rows':4402,'missing':0,'days':40,'reused_exactly_verified':6,'batches':verification})
    save('analysis-summary.json',summary); save('day-summary.json',days);save('diagnostic-sample.json',selected)
    (D/'all-4402-original-answers.jsonl').write_text(''.join(json.dumps({**x,'analysis_checks':x['analysis_checks']},ensure_ascii=False)+'\n' for x in derived))
    save('analysis-manifest.json',{'sources':{str(p.relative_to(D.parent)):sha(p) for p in [T/'exports/k3-full-raw-return.jsonl',N/'exports/k3-return-with-reuse.jsonl',N/'private/full-day-source.jsonl']},'script_sha256':sha(Path(__file__)),'summary_sha256':sha(D/'analysis-summary.json'),'all_answers_sha256':sha(D/'all-4402-original-answers.jsonl'),'unchanged_frozen_prompt_sha256':sha(N/'runner/prompt.txt')})
    print(json.dumps({k:v for k,v in summary.items() if k not in ['definitions','verification']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
