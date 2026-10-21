import json
import time
import datetime
import traceback
from pathlib import Path

ROOT = Path('/root/qwen38-a800-20261010')
OUT = ROOT / 'results/transformers-smoke.json'
CASES = [
    {'id': 'arithmetic_json', 'prompt': '计算2加3，只输出JSON对象，字段answer的值是计算结果。', 'expected': {'answer': 5}},
    {'id': 'buy_completed', 'prompt': '原文：“我昨天已经买入100股。”作者是否自述已经买入？只回答“是”或“否”。', 'expected': '是'},
    {'id': 'buy_negated', 'prompt': '原文：“我今天没有买入，一股也没买。”作者是否自述已经买入？只回答“是”或“否”。', 'expected': '否'},
    {'id': 'buy_planned', 'prompt': '原文：“我准备明天买入。”这句话是否表示作者已经完成买入？只回答“是”或“否”。', 'expected': '否'},
    {'id': 'other_subject', 'prompt': '原文：“老王昨天买入了，我还没有买。”这句话里，自述已经买入的人是谁？只回答“老王”或“作者”。', 'expected': '老王'},
    {'id': 'fear', 'prompt': '原文：“我真的害怕再跌了。”作者明确表达了哪种情绪？只回答“恐惧”、“兴奋”或“未表达”。', 'expected': '恐惧'},
    {'id': 'regret', 'prompt': '原文：“昨天不该卖，后悔死了。”作者明确表达了哪种情绪？只回答“后悔”、“兴奋”或“未表达”。', 'expected': '后悔'},
]
report = dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    model='Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17',
    scope='Seven handwritten deployment smoke cases, not a semantic-quality benchmark or evidence about 27B.', cases=[])
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    report['runtime'] = dict(torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda)
    path = str(ROOT/'models/Qwen3.5-0.8B')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True).to('cuda:0').eval()
    torch.cuda.synchronize()
    report['load_seconds'] = time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats()
    for case in CASES:
        prompt = tokenizer.apply_chat_template([{'role':'user','content':case['prompt']}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
        batch = tokenizer(prompt, return_tensors='pt').to('cuda:0')
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**batch, max_new_tokens=192, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        tokens = generated[0, batch['input_ids'].shape[1]:]
        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        passed = answer.strip('。.!！\n ') == case['expected']
        if isinstance(case['expected'], dict):
            try:
                passed = json.loads(answer) == case['expected']
            except Exception:
                passed = False
        receipt = dict(**case, answer=answer, passed=passed, elapsed_seconds=elapsed,
                       output_tokens=tokens.numel(), reached_output_limit=tokens.numel()>=192,
                       completion_tokens_per_second_including_prefill=tokens.numel()/elapsed)
        report['cases'].append(receipt)
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['generation_ran'] = True
except Exception:
    report['error'] = traceback.format_exc()
report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
raise SystemExit(0 if report.get('generation_ran') else 2)
