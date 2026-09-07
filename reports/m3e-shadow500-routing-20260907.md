# M3-E Shadow500 输入质量路由收尾

状态：`M3_SHADOW500_ROUTING_COMPLETED_DIAGNOSTIC`

本轮只复用了 `runs/m3c-shadow500-20260907/` 已保存的 500 条输入与预测，
没有再次加载模型或执行推理。路由用于处理已有输入质量信号，不是语义 OOD
检测器，也不构成生产批准。

## 固定互斥规则

按以下顺序逐条应用，保证一条记录只有一个结果路由：

1. `model_eligibility=EXCLUDE_SEMANTIC` → `INPUT_EXCLUDED`；
2. 其余记录若 `six_head_mean_max_probability < 0.5243767648935318`
   → `LOW_CONFIDENCE_ABSTAIN`；
3. 其余 `CHALLENGE_REVIEW` → `REVIEW_RECOMMENDED`；
4. 其余 `ELIGIBLE` → `MODEL_OUTPUT`。

`INPUT_EXCLUDED` 的七头 `predictions` 在路由输出中被抑制，明确不得作为正常
七头结果交给下游，也不会映射为 `UNKNOWN`。`REVIEW_RECOMMENDED` 保留预测，
但带有独立路由和下游动作标记；它不等同于 `MODEL_OUTPUT`。

## 结果

| 路由 | 数量 | 预期 | 说明 |
| --- | ---: | ---: | --- |
| `INPUT_EXCLUDED` | 8 | 8 | 8/8 `EXCLUDE_SEMANTIC` 被拦截 |
| `LOW_CONFIDENCE_ABSTAIN` | 27 | 27 | 固定阈值严格小于规则 |
| `REVIEW_RECOMMENDED` | 158 | 158 | 非低分的 `CHALLENGE_REVIEW` |
| `MODEL_OUTPUT` | 307 | 307 | 非低分的 `ELIGIBLE` |
| 合计 | 500 | 500 | 互斥且穷尽 |

路由结果与摘要位于：
`runs/m3e-shadow500-routing-20260907/`。

- `shadow500-routed.jsonl`：500 条路由结果；8 条 `INPUT_EXCLUDED` 均无
  `predictions` 字段，其他 492 条保留原预测。
- `routing-summary.json`：记录输入/预测来源 SHA-256、固定阈值、优先级、计数
  和诊断边界。
- 上游 input SHA-256：
  `52e8aa09b3d036a225b04d34586dac1e07d362fa4350f096ba1e8489bf932286`。
- 上游 prediction SHA-256：
  `df772b6ee75ca1fc12645b9eaf3b3e7dbf3ceb847e796c6e461d2f0ffeb11a36`。

该路由只解决已知的 `model_eligibility` 与输入级置信度信号，不能声称解决
语义 OOD、分布外错误或生产安全问题。当前冻结候选仍为 diagnostic-only，
没有生产批准；本轮未访问 labels、Test、Anchor、Human100、Gold、OOD、
reference predictions、LLM、云服务或生产数据。
