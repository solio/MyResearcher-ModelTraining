# 七头语义模型版本（研究推理）

只用一个名字指向**完整七头组合**，不要再把 `M12`、`selected`、`Action 候选`混称为“现用模型”。`SEM7` 表示七头语义输出，日期是本次组合启用/实验日，`A0/A5` 是 Action 来源编号，不代表 head 数量。版本名不随 `ACTIVE`/`HISTORICAL` 状态变化；已有 artifact 目录不重命名。

| 版本 | 七头来源 | 状态（2026-09-29） |
| --- | --- | --- |
| `SEM7-20260915-A0` | `stance`、`emotion_target` 用 M12 Human-first；其余五头（含 Action）用 9 月 4 日 selected bundle | 历史默认版，保留作可回退对照；既有冻结预测属于此版 |
| `SEM7-20260929-A5` | 保持 A0 的其他六头，只把 Action 换为 9 月 29 日第五个对照训练 checkpoint | **后续研究推理默认版**；非生产模型，旧报告不改写 |

源路径：

- M12 两头：`/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m12-human-first-lert-oof-20260915/`
- selected 旧五头：`/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/m2-selected-candidate-frozen-for-m3-20260904/`
- A5 Action：`/Users/mac/Documents/trae_projects/MyResearcher/model-artifacts/semantic-action-contrast-role-negative-v1-20260929/action-checkpoint.pt`

9 月 29 日其余 Action 实验依次为 `A1` 一层语义成对训练、`A2` 两层、`A3` 加 65 条自然 silver、`A4` 再降低 30 条低证据旧标签权重；`A5` 再加 18 条真实主体/否定反例。A1～A4 是实验编号，不是可选的完整七头运行版本；完整目录和逐项结果见 [`../reports/semantic-action-contrast-20260929.md`](../reports/semantic-action-contrast-20260929.md)。

后续推理入口 [`../src/semantic_model/encoder_semantic_active_runtime.py`](../src/semantic_model/encoder_semantic_active_runtime.py) 默认 A5；传入 `version="SEM7-20260915-A0"` 可重跑旧组合。`market_extreme_refresh.py` 与 `market_data_bootstrap_v0.py` 的新运行默认 A5，并支持 `--model-version` 显式选择。每条新预测写顶层 `model_version`；旧文件没有该字段时视为 A0，**不能拿来当 A5 的缓存**。历史文件名 `m12-predictions.jsonl` 暂留接口兼容，其内部版本字段才决定是哪一版。

模型选择依据分两层：A5 在 Action 真实帖的盲读对照中语义方向进步，但仍有作者/他人、否定及部分/全部错读；旧版在事后市场极值中的个别较好结果不能证明其语义正确或预测能力更强。历史 A0 与 A5 影子报告均不提供真正时间外交易回测。A5 作为研究默认，不代表买卖建议、生产部署或市场有效性验收。
