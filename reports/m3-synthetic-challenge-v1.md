# M3 Synthetic Challenge V1

这是一个固定的输入-only synthetic challenge fixture，服务于下一轮模型输入趋势诊断。
它不包含七头真实标签、模型预测、阈值、人工裁决或任何模型 artifact。

## 构成

- `IN_DOMAIN_CONTROL`：20 条正常中文证券讨论。
- `CLEAR_OOD`：20 条明显无关文本或非语言噪声；这里只是预期 scope，不能替代正式 OOD 集或 OOD gate。
- `SHIFT_CHALLENGE`：60 条分布变化挑战，每类 10 条：
  新股票/行业实体、新时间或事件、平台新黑话、繁体中文、错别字/谐音、缺失上下文或链接/emoji 重载。

每条记录只包含 `sample_id`、`stock_code`、`stock_name`、`model_text`、`expected_scope`、
`challenge_category` 和 `construction_note`。`expected_scope` 是构造时的诊断分层，不是模型输出，
也不是 Gold 或人工 adjudicated truth。

## 可重复检查

```bash
python tools/m3_synthetic_challenge_v1.py --check
python -m pytest -q tests/test_m3_synthetic_challenge_v1.py
```

检查固定 100 条、`sample_id` 唯一、字段完整、scope/category 数量和序列化确定性。该 fixture
只用于下一轮趋势诊断；不得据此选择模型、调阈值、宣称正式 OOD 性能或进入生产路径。
