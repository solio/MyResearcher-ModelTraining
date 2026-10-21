"""Materialize manually read references; no keyword/rule-derived labels."""
from pathlib import Path
import json,hashlib,collections
D=Path(__file__).resolve().parent
C=json.load(open(D/'candidates-stage1.json')); by={r['candidate_id']:r for r in C}
contract=json.load(open(D.parent.parent/'ready-semantic-comparison-20261010/contract.json'))
keys=[x['id'] for x in contract['core_candidates']]
# Each tuple was written by a semantic reader after reading the whole post.
# (candidate_id, supported claims: (id, exact quote, reason, time), observations)
R=[
(2,[('action_clear','全卖了','作者说全部卖出','completed'),('action_sell_completed','昨天229.4全卖了','过去卖出陈述','completed'),('topic_price','229.4','报告卖出价位','past')],[]),
(9,[('emo_regret','后悔了，反弹没跑','对错失反弹卖出机会后悔','current'),('topic_price','反弹','讨论反弹走势','past')],[]),
(71,[('stance_bullish','坚定看好年底走势：88元收','明确看好年底走势','current'),('topic_price','88元收','给出年底收盘价预测','planned')],[]),
(18,[('stance_bullish','杰瑞业绩好，订单饱满，目标价180-200','正向经营评价和目标价格','current'),('topic_fundamentals','业绩好，订单饱满','讨论业绩和订单','current'),('topic_price','红三天，绿三天，很有规律','讨论涨跌规律','current')],[]),
(24,[('stance_bullish','非常不错的业绩','正向评价经营表现','current'),('topic_fundamentals','收入利润同比环比都增长','讨论收入利润','current')],[]),
(27,[('action_buy_completed','大胆买入了','明确买入完成','completed')],[('希望主力支持价格','主力给点力','main_force','unknown')]),
(35,[('stance_bearish','看空到25以下','明确看空和下方目标','current'),('topic_price','25以下','讨论价格目标','current')],[('劝他人不要加仓','不要加仓，不要加仓','other_retail','planned')]),
(38,[('topic_funds','今天资金流入了','描述资金流入','current')],[]),
(46,[('action_buy_completed','刚买了一手','作者刚完成买入','completed'),('topic_price','挂上山顶了','买入后处在价格高点的表述','current')],[]),
(49,[('action_sell_completed','我昨天卖了','作者明确昨天卖出','completed'),('topic_price','今天涨了吗','讨论今日上涨','current')],[('自述一个月亏损','一个月亏了本金一半','self','past')]),
(55,[('topic_funds','资金净流入流出统计数据','讨论资金统计','current')],[]),
(62,[('action_sell_completed','今天卖了','作者明确已卖','completed'),('stance_bearish','明天回调','明确预测明日回调','planned'),('topic_price','明天回调','讨论回调走势','planned')],[]),
(66,[('action_clear','清仓了清仓了','明确全部退出完成','completed'),('action_sell_completed','清仓了','清仓了陈述已卖出','completed')],[]),
(67,[('stance_bullish','坚定看好汇川，最优秀的企业','明确看好','current')],[]),
(79,[('topic_funds','每日最少资金超两亿','讨论资金规模及量能','current')],[]),
(85,[('topic_funds','半天流出资金2300多万','明确讨论流出资金规模','current')],[]),
(88,[('emo_regret','瞎了眼啊买了江丰','责备自己过去买入判断','current'),('action_buy_completed','买了江丰','明确已买江丰','completed')],[]),
(96,[('emo_regret','尾盘后悔了呀','明确对早盘卖出后悔','current'),('action_sell_completed','早盘一翻红就给你卖了','明确卖出完成','completed'),('topic_price','一翻红','讨论盘中价格转涨','past')],[]),
(104,[('action_sell_completed','反正卖了','明确卖出完成','completed')],[('明确否定后悔','不后悔','self','negated')]),
(107,[('wish_stop_decline','别跌了','直接要求停止下跌','current'),('topic_price','别跌了','讨论价格下跌','current')],[]),
(116,[('stance_bullish','周一直接涨停','直接预测未来涨停','planned'),('topic_price','涨停','讨论涨停','planned')],[]),
(123,[('topic_funds','资金高度控盘','讨论资金控盘和换手','current')],[]),
(125,[('wish_stop_decline','也别跌了','明确希望不要下跌','current'),('topic_price','别涨了，也别跌了','希望价格维持不变','current')],[('希望价格也不要上涨','别涨了','self','current')]),
(133,[('emo_regret','最后悔的事就是进了这个','后悔买入决定','current'),('action_buy_completed','进了这个','股票语境自述已进场','completed')],[]),
(141,[('action_t_unspecified','豆包说今天77卖3成仓做T，我试试准不准','作者明确尝试做T计划','planned'),('topic_price','今天77','给出计划价位','planned')],[('计划卖出三成而非已卖','卖3成仓做T，我试试准不准','self','planned')]),
(145,[('stance_bullish','中免将迎来涨停潮','明确未来大涨判断','planned'),('topic_price','涨停潮','讨论上涨走势','planned')],[]),
(148,[('action_clear','今日最后三分钟清仓了','作者已清仓','completed'),('action_sell_completed','清仓了','已全部卖出','completed')],[('计划调仓','准备调仓换股了','self','planned')]),
(159,[('emo_fear','真的害怕','作者直接表达害怕','current')],[]),
(162,[('action_sell_completed','242大仓位全卖了，留了几百户纪念下','卖出大部分且仍有余仓','completed'),('topic_price','242','卖出价位','past')],[('保留余仓，未全部清仓','留了几百户纪念下','self','current')]),
(168,[('stance_bullish','长期看好270','直接长期看好','current'),('topic_price','270','给出价格目标','planned')],[]),
(172,[('emo_disappointment','你真让人失望','直接表达失望','current')],[]),
(180,[('action_clear','今天全部清仓了','已全部退出','completed'),('action_sell_completed','今天全部清仓了','已卖出全部','completed')],[('持有期间持续亏损','拿了2个月今天全部清仓了，一直亏','self','past')]),
(185,[('action_clear','明天想清仓','自己的未来清仓意向','planned')],[('清仓尚未完成','明天想清仓','self','planned')]),
(187,[('stance_bullish','业绩好转，股价将迎来趋势上涨','正向经营预期与上涨预测','planned'),('topic_price','股价将迎来趋势上涨','讨论趋势上涨','planned'),('topic_fundamentals','接下来重点关注一季报，业绩好转','讨论季度业绩','planned')],[]),
(195,[('action_buy_completed','尾盘买入了','明确已买','completed'),('topic_price','连续两天假阴线','讨论K线形态','current')],[]),
(203,[('action_sell_completed','卖了','卖出已完成','completed')],[('自述亏损幅度','亏5个点','self','past')]),
(206,[('action_buy_completed','今天买了一点','今天已买','completed'),('topic_price','明天跌再买','讨论条件价格变化','conditional')],[('未来继续买入计划','明天跌再买，这周想买到5成仓','self','planned')]),
(216,[('stance_bullish','业绩这么好','肯定经营表现','current'),('topic_fundamentals','业绩这么好','讨论业绩','current'),('topic_price','为什么还跌','讨论下跌','current')],[]),
(220,[('action_clear','清仓了','明确清仓完成','completed'),('action_sell_completed','清仓了','已卖出全部','completed')],[('祝福其他持有人','留下得发财','other_retail','planned')]),
(225,[('topic_price','今天跌的','讨论今日下跌','current'),('topic_funds','资金流入','讨论资金流入','current')],[]),
(232,[('emo_anger','气死我了','明确生气','current'),('topic_price','自选就它绿','讨论下跌','current')],[]),
(233,[('emo_anger','气死我了','明确生气','current'),('topic_price','竟然还绿了','讨论下跌','current')],[]),
(239,[('emo_disappointment','失望透顶','明确失望','current'),('topic_price','两天跌十几个点','讨论下跌幅度','past')],[]),
(245,[('emo_disappointment','失望透顶','明确失望','current')],[]),
(253,[('emo_excitement','我激动得手抖','作者因表现激动','current')],[]),
(261,[('emo_fear','让我害怕踏空了','明确害怕错失','current'),('emo_fomo','害怕踏空了啊，下来把下来让我上车','害怕错过并希望参与','current'),('topic_price','最近两天跳空上涨','讨论跳空上涨','past')],[('希望回落以便买入','下来把下来让我上车','self','planned')]),
(268,[('emo_fear','好怕踏空啊','明确害怕错过','current'),('emo_fomo','AI让我18买入好怕踏空啊','买入建议引发踏空压力','current'),('topic_price','18买入','提及建议买入价位','planned')],[('AI的买入建议不是已买入','AI让我18买入','other','planned')]),
(289,[('emo_fear','亏怕了','亏损产生恐惧并退出','current'),('action_clear','割肉清仓了','自述已清仓','completed'),('action_sell_completed','割肉清仓了','自述已全部卖出','completed')],[('准备转向科技股','去科技抱团','self','planned')]),
(279,[('emo_regret','后悔死了','明确对早上调仓后悔','current'),('action_sell_completed','减了北京君正','已减持明确其他股票','completed'),('action_buy_completed','加了你','已增加帖子所指股票持仓','completed')],[]),
(281,[('emo_regret','早上180没走，真后悔','后悔未在早上卖出','current'),('topic_price','180','讨论错失的卖出价位','past')],[]),
]
assert len(R)==50
# Simple authored counterfactual controls, not real corpus examples.
S=[
('我今天已经买入了这只股票。',[('action_buy_completed','已经买入了','明确已完成买入','completed')],[]),
('我今天没有买入这只股票，一股也没买。',[],[('否定自己已买','没有买入这只股票，一股也没买','self','negated')]),
('我准备明天买入这只股票。',[],[('未来买入计划','准备明天买入','self','planned')]),
('他说他今天买入了这只股票，我还没买。',[('target_other_retail','他说他今天买入了这只股票','讨论另一投资者动作','completed')],[('他人已买，作者没买','他说他今天买入了这只股票，我还没买','other_retail','completed')]),
('我今天已经卖出一半持仓，另一半还留着。',[('action_sell_completed','已经卖出一半持仓','明确已部分卖出','completed')],[('仍保留一半仓位','另一半还留着','self','current')]),
('我今天没有卖出股票。',[],[('否定卖出','没有卖出股票','self','negated')]),
('我打算明天卖出一半持仓。',[],[('计划部分卖出','打算明天卖出一半持仓','self','planned')]),
('隔壁老王已经卖出股票，我还持有。',[('target_other_retail','隔壁老王已经卖出股票','讨论他人卖出行为','completed')],[('他人卖出','隔壁老王已经卖出股票','other_retail','completed')]),
('我刚把所有持仓都卖完了，已经清仓。',[('action_sell_completed','所有持仓都卖完了','已全部卖出','completed'),('action_clear','已经清仓','已清仓','completed')],[]),
('我准备明天清仓，现在还没卖。',[('action_clear','准备明天清仓','清仓命题包含明确意向','planned')],[('未完成卖出','现在还没卖','self','negated')]),
('如果明天跌到十元，我就全部清仓。',[('action_clear','如果明天跌到十元，我就全部清仓','明确条件清仓计划','conditional'),('topic_price','跌到十元','条件价位','conditional')],[]),
('我不准备清仓，也没有卖出。',[],[('否定清仓意向和卖出','不准备清仓，也没有卖出','self','negated')]),
('我今天先买后卖做了一次正T。',[('action_t_unspecified','做了一次正T','已完成正T','completed'),('action_buy_completed','今天先买后卖','作者已买','completed'),('action_sell_completed','今天先买后卖','作者已卖','completed')],[('正T顺序先买后卖','先买后卖','self','completed')]),
('我今天先卖后买做了一次逆T。',[('action_t_unspecified','做了一次逆T','已完成逆T','completed'),('action_buy_completed','今天先卖后买','作者已买','completed'),('action_sell_completed','今天先卖后买','作者已卖','completed')],[('逆T顺序先卖后买','先卖后买','self','completed')]),
('我打算明天做T，今天还没有交易。',[('action_t_unspecified','打算明天做T','明确做T计划','planned')],[('今天未交易','今天还没有交易','self','negated')]),
('这次下跌让我非常害怕，吓得不敢看账户。',[('emo_fear','非常害怕，吓得不敢看账户','明确本人恐惧','current'),('topic_price','这次下跌','讨论下跌','current')],[]),
('这次下跌我一点也不害怕。',[('topic_price','这次下跌','讨论下跌','current')],[('明确否定害怕','一点也不害怕','self','negated')]),
('群里的散户都很害怕，我自己并不害怕。',[('target_other_retail','群里的散户都很害怕','讨论他人情绪','current')],[('他人恐惧','群里的散户都很害怕','other_retail','current'),('否定本人恐惧','我自己并不害怕','self','negated')]),
('我真后悔昨天卖出了这只股票。',[('emo_regret','真后悔昨天卖出了','后悔过去卖出','current'),('action_sell_completed','昨天卖出了这只股票','已卖出','completed')],[]),
('我虽然卖早了，但我并不后悔。',[('action_sell_completed','卖早了','过去卖出','completed')],[('否定后悔','并不后悔','self','negated')]),
('这家公司又欺骗投资者，我气得手抖，太愤怒了！',[('emo_anger','我气得手抖，太愤怒了','明确本人愤怒','current')],[('指责公司欺骗投资者','这家公司又欺骗投资者','self','current')]),
('我以为这次能盈利，结果又亏了，太失望了。',[('emo_disappointment','以为这次能盈利，结果又亏了，太失望了','明确预期落差和失望','current')],[]),
('今天终于赚钱了，我兴奋得跳了起来！',[('emo_excitement','我兴奋得跳了起来','明确本人兴奋','current')],[]),
('我怕错过这次上涨机会，急得现在就想买进去。',[('emo_fear','我怕错过这次上涨机会','害怕错失机会','current'),('emo_fomo','怕错过这次上涨机会，急得现在就想买进去','错失恐惧推动参与压力','current'),('topic_price','上涨机会','讨论上涨','current')],[('买入意向不是成交','现在就想买进去','self','planned')]),
('我看好这家公司的前景，预计股价还会上涨。',[('stance_bullish','看好这家公司的前景，预计股价还会上涨','明确看多','current'),('topic_price','股价还会上涨','上涨预测','planned')],[]),
('我看空这只股票，预计下周还会下跌。',[('stance_bearish','看空这只股票，预计下周还会下跌','明确看空','current'),('topic_price','下周还会下跌','下跌预测','planned')],[]),
('这只股票今天收盘价是二十元，比昨天跌了百分之三。',[('topic_price','收盘价是二十元，比昨天跌了百分之三','只陈述价格与涨跌','current')],[]),
('今天主力资金净流出两亿元，筹码集中度下降。',[('topic_funds','主力资金净流出两亿元，筹码集中度下降','资金和筹码话题','current')],[]),
('公司公告称新工厂下个月投产，我还没判断这对利润的影响。',[('topic_fundamentals','新工厂下个月投产','公司产能话题','planned')],[]),
('求求你别再跌了，我只希望价格先稳住。',[('wish_stop_decline','别再跌了，我只希望价格先稳住','明确希望停止下跌','current'),('topic_price','别再跌了','讨论价格下跌','current')],[]),
]
assert len(S)==30
inputs=[];refs=[]
def build(d,positive,obs):
 t=d['text']; js={k:{'decision':'not_supported','quote':'','reason':'逐条通读后，原文未明确表达该完整命题；不代表现实中不存在。','subject':'unknown','target':'unknown','time':'unknown','confidence':None} for k in keys}
 for k,q,why,tm in positive:
  assert q in t,(d['id'],q,t)
  subject='text' if k.startswith('topic_') else 'other_retail' if k=='target_other_retail' else 'self'
  js[k]={'decision':'supported','quote':q,'reason':why,'subject':subject,'target':'unknown','time':tm,'confidence':None}
 ob=[]
 for meaning,q,subject,tm in obs:
  assert q in t,(d['id'],q,t)
  ob.append({'meaning':meaning,'quote':q,'subject':subject,'target':'unknown','time':tm})
 refs.append({'id':d['id'],'text':t,'judgments':js,'observations':ob,'reviewer':'q38_samples_semantic_reader','review_method':'This agent read full text, then authored claims and reasons. Scripts only materialize and check substring/schema.','reference_status':'single_assistant_read_not_human_gold'})
 inputs.append(d)
for i,(ci,pos,obs) in enumerate(R,1):
 r=by[ci]; d={k:r[k] for k in ['source','source_item_id','stock_code','stock_name','published_at','url','text','text_hash']};d.update(id=f'S1R{i:03}',source_kind='real_directed_easy',candidate_id=ci,sampling_probability=None);build(d,pos,obs)
for i,(t,pos,obs) in enumerate(S,1):
 d={'id':f'S1C{i:03}','text':t,'text_hash':hashlib.sha256(t.encode()).hexdigest(),'stock_name':'合成对照','stock_code':'synthetic','published_at':None,'source_kind':'synthetic_control','source':'authored_by_q38_samples','sampling_probability':None};build(d,pos,obs)
# Critical basic reversals are separately auditable; no overall-accuracy gate.
critical_specs={
'S1C002':[('action_buy_completed','N','明确否定本人买入')],
'S1C003':[('action_buy_completed','N','未来计划不能当成完成')],
'S1C004':[('action_buy_completed','N','他人买入且作者明确没买')],
'S1C005':[('action_clear','N','卖出一半且留有余仓不是清仓')],
'S1C006':[('action_sell_completed','N','明确否定卖出')],
'S1C007':[('action_sell_completed','N','未来计划不是卖出完成')],
'S1C008':[('action_sell_completed','N','他人卖出不能归给作者')],
'S1C010':[('action_sell_completed','N','清仓计划尚未成交'),('action_clear','Y','现契约清仓包含明确计划')],
'S1C011':[('action_sell_completed','N','条件清仓尚未成交')],
'S1C012':[('action_clear','N','否定清仓意向'),('action_sell_completed','N','否定卖出')],
'S1C015':[('action_buy_completed','N','计划做T不是已买'),('action_sell_completed','N','计划做T不是已卖')],
'S1C017':[('emo_fear','N','明确否定本人害怕')],
'S1C018':[('emo_fear','N','他人害怕且本人否定害怕')],
'S1C020':[('emo_regret','N','明确否定后悔')],
'S1C024':[('action_buy_completed','N','想买不是已买')],
'S1C030':[('emo_fear','N','希望止跌本身不自动证明恐惧')],
'S1R019':[('emo_regret','N','明确说不后悔')],
'S1R025':[('action_sell_completed','N','试试AI做T建议是计划，非成交')],
'S1R029':[('action_clear','N','卖掉大仓位仍保留几百股，不是全部退出')],
'S1R033':[('action_sell_completed','N','明天想清仓不是已卖')],
'S1R047':[('action_buy_completed','N','AI买入建议不等于本人买入')],
'S1R050':[('action_sell_completed','N','后悔没走不能当成已卖')],
}
uncertain_specs={
'S1R006':{'topic_funds':('主力给点力','主力行动诉求是否属于资金动向话题有边界，参考不硬判')},
'S1R016':{'stance_bearish':('还有想入非非的吗','可能间接否定乐观预期，也可能仅反问，保留歧义'),'target_other_retail':('还有想入非非的吗','可能指向看多散户，但主体省略，保留歧义')},
}
target_overrides={'S1R004':{'stance_bullish':'杰瑞','topic_fundamentals':'杰瑞','topic_price':'杰瑞'},'S1R014':{'stance_bullish':'汇川'},'S1R017':{'emo_regret':'江丰','action_buy_completed':'江丰'},'S1R026':{'stance_bullish':'中免','topic_price':'中免'},'S1R049':{'action_sell_completed':'北京君正','action_buy_completed':'你'}}
for r in refs:
 for k,(q,why) in uncertain_specs.get(r['id'],{}).items():
  assert q in r['text']
  r['judgments'][k].update(decision='abstain',quote=q,reason=why)
 for k,target in target_overrides.get(r['id'],{}).items():r['judgments'][k]['target']=target
 r['critical']=[{'id':k,'expected':expected,'reason':why} for k,expected,why in critical_specs.get(r['id'],[])]
 for c in r['critical']:
  assert r['judgments'][c['id']]['decision']==('supported' if c['expected']=='Y' else 'not_supported'),(r['id'],c)
for file,rows in [('stage1-inputs.json',inputs),('stage1-reference.json',refs)]:
 (D/file).write_text(json.dumps({'schema_version':'qwen38-staged-v1','contract_version':contract['version'],'rows':rows},ensure_ascii=False,indent=2))
print('WROTE',len(inputs),'stocks',len({r['stock_code'] for r in inputs if r['source_kind'].startswith('real')}),'months',dict(collections.Counter(r['published_at'][:7] for r in inputs if r['published_at'])))
print('positive counts',dict(collections.Counter(k for r in refs for k,v in r['judgments'].items() if v['decision']=='supported')))
