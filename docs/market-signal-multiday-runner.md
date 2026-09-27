# 多日群体状态与中期收益评测脚本

Owner 于 2026-09-27 授权：复用尽量多的现有数据，评测过去 5/20 个交易日状态与未来 10/20/30 个交易日正收益的关系，并交付无需 Codex 的参数化脚本。

## 一条命令运行

已有 `.venv` 包含所需的 NumPy、SciPy 和 scikit-learn。脚本不需要 Torch、下载模型或网络，不重新运行语义模型。

```sh
cd /Users/mac/Documents/trae_projects/MyResearcher/MyResearcher-ModelTraining
./.venv/bin/python tools/market_signal_multiday.py --windows 5 20 --horizons 10 20 30
```

默认读取本项目已有历史预测、已完成的前向预测批次和最新本地行情；在 `pipeline-runs/market-signal-multiday-日期-时间/` 新建输出目录。

使用本轮保存的配置再跑一次，也可覆盖任何参数：

```sh
./.venv/bin/python tools/market_signal_multiday.py \
  --config /Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-signal-multiday-20260927-final/run-config.json \
  --as-of latest
```

从任何工作目录使用绝对脚本路径也能执行，不需要设置 `PYTHONPATH`。另一台机器需要 Python 3.11/3.12、项目代码与输入文件：创建虚拟环境后 `python -m pip install -e .`，再通过输入路径参数指定该机器的数据。

## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--windows` | `5 20` | 向过去观察的实际交易日数量 N，可提供多个整数 |
| `--horizons` | `10 20 30` | 从入场日算起的实际交易日期限 H，可提供多个整数 |
| `--as-of` | `latest` | 行情截止日期；`YYYY-MM-DD` 或最新本地行情 |
| `--min-train-dates` | `5` | 每折至少多少个已经到期的训练入场日；5 是短历史的探索性起点 |
| `--test-block-dates` | `1` | 每折测试连续多少个可评测入场日，测试块不重叠 |
| `--train-window-dates` | `0` | 0 使用全部已成熟历史；正数 K 只用最近 K 个符合条件的训练日期 |
| `--embargo-sessions` | `0` | 在训练结果到期与测试日之间额外留出的实际交易日数量 |
| `--min-signal-coverage` | `1.0` | 至少有多少比例的窗口状态；1.0 要求 N 天完整。降低会改变研究含义 |
| `--return-threshold` | `0.0` | 期限末收益须超过的比例，0.01 代表 +1%；不会自动推算费用 |
| `--logistic-c` | `0.1` | 固定逻辑回归正则化；脚本不搜索参数 |
| `--post-cap` | `16` | 每股每入场日概率抽样上限；可降低，不能超过输入历史概率已具备的样本上限 |
| `--bootstrap-repeats` | `500` | 日期区块重采样次数；0 关闭 |
| `--bootstrap-block-dates` | `0` | 0 使用 max(N,H)，处理窗口和收益的时间重叠；少于两个区块时不输出区间 |
| `--seed` | `20260927` | 模型与重采样随机种子 |
| `--state-contrast` | 关闭 | 增加同股去均值、行情线性残差两个固定对照；不搜索参数 |
| `--residual-alpha` | `1.0` | 残差解释器的固定 Ridge 正则化，只在对照开启时使用 |
| `--common-sample-control` | 开启 | 额外对照各 N 的共同训练/测试样本和固定行情特征；可用 `--no-common-sample-control` 关闭 |
| `--output-dir` | 自动新建目录 | 已存在且非空的目录不会被覆盖 |
| `--config` | 无 | JSON 配置；命令行同名参数优先 |

输入路径可通过 `--history-metadata`、`--history-predictions`、`--qfq-bars`、`--raw-bars`、`--benchmark-bars` 修改。

`--early-forward` 默认纳入 9/16 的既有预测，传入空字符串可以关闭。`--forward-glob` 默认扫描 `pipeline-runs/market-signal-v2-forward-*`，只纳入具有完整预测及 FORWARD 群体快照的目录。`--forward-dir` 可以重复指定额外目录。指定 `--forward-glob ''` 可以关闭目录扫描。一个股票×入场日只能有一个输入批次；重复时脚本明确报错。若自行提供已合并所有批次的历史输入，应同时指定 `--early-forward '' --forward-glob ''`。

## N 与 H 的精确定义

每个日状态已经归属于一个下一可入场交易日。预测入场日 t 时，使用归属于 t−N+1 到 t 的 N 个日状态：它们来自此前已经发表的帖子，最后一个是本次开盘前可用的状态。按交易日等权，避免热门日的帖子数量完全主导多日均值。

特征包括：多空、操作、情绪平衡、FOMO、分歧度、模型不确定度、集中度、讨论量的均值与变化斜率，最近 5 日相对整个 N 日窗口的变化，以及多空/情绪正负状态持续比例。完整每日类别分布仍保存在 `daily-states.jsonl`。

结果从入场日开盘价计算到第 H 个实际交易日收盘价，使用 qfq 收益，并复用已有 raw/qfq 异批次校验规则。周末/休市不算交易日。目标是期限结束时正收益，不是期间任一时点曾上涨；当前报告不代表扣费后的可执行交易策略。

## 交叉验证与数据不足

按入场日期向前推进，所有股票在同一个测试日期一起进入测试。训练只用 `outcome_date < test_start` 的结果；多日期测试块以第一天为界限。填补缺失行情特征、标准化与模型拟合只在该折训练数据中发生。没有随机打散股票或帖子。

五个输出概率：全局历史先验、同股 Beta(1,1) 历史胜率、纯行情、纯群体状态、行情与群体组合。同一 N/H 的所有模型使用完全相同的训练与测试样本。

为了直接比较 5 天与 20 天，默认还运行共同样本对照：同一期限下，两个窗口使用相同训练/测试股票日、相同日期和相同的固定行情特征，移除 N 专属的行情窗口字段。这样不会把训练样本数量差异误当成情绪窗口效果。最大可用样本与共同样本的输出分别保留。

若输入窗口完整、有连续日期且训练标签包含两类，产生第一个已到期 CV 结果至少需要约 `N + min_train_dates + 2H - 1 + embargo` 个行情交易日。实际状态缺失和单类训练会进一步推迟。本轮已有约 56 个交易日，不能凭股票数量制造足够的长期时间历史。

不能做 CV 的组合会记录 `INSUFFICIENT_TIME_HISTORY_FOR_CV`，给出成熟日期数、切分缺口及乐观数据下限。固定状态与收益的描述性排序相关仍单独输出，不冒充 CV 概率成绩。

历史 M12 概率是在历史市场日期之后生成的，本轮研究只能证明市场层时间切分下的诊断结果；不能当作整个系统在当时实际部署的前向成绩。

## 同股常态与行情残差的限定对照

```sh
./.venv/bin/python tools/market_signal_multiday.py \
  --config /Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-signal-multiday-20260927-final/run-config.json \
  --state-contrast --residual-alpha 1.0
```

此模式保留原五模型，并在每个 N/H 的相同训练/测试行上增加两个候选：

- `relative_combined`：行情 + 原状态减去该折训练样本的同股平均状态。
- `residual_combined`：行情 + 相对状态中不能被已知行情特征线性解释的剩余部分。

均值、缺失填补、行情缩放及 Ridge 解释器全部仅在该折已成熟训练行拟合，不在全历史上先做一次变换。行情解释器使用与该 N/H 行情基线相同的过去收益、指数、换手和量额字段，包括窗口 N 的已知历史收益；不读未来收益。Ridge 默认固定 alpha=1，不搜索；收益分类仍使用同一个 C。未出现在训练里的股票回退到训练总体均值。

原模型已经有股票 one-hot 截距。这两个版本没有增加信息，也不是独立拟合每只股票的新模型：在线性模型下，变换后与行情重新组合主要改变正则化几何。因此即使误差改善，也不能据此宣称找到了因果关系或完全排除了市场影响。对照是在看过前轮结果后提出的研究修订，不是独立确认试验。

报告额外区分每日跨股票排序与逐股时间相关。时间相关使用该股各日期的样本外概率/状态和后续收益；状态相关属于描述性分析，不是概率预测成绩。测试行情解释 R² 若为负，表示训练解释器在测试期不如直接用同股训练均值，不得把该残差宣传为已成功剥离行情的独立信号。

同时报告固定概率 0.5 的 Brier=0.25 参照，以及核心测试状态相对训练均值/标准差的偏离。前者避免把“比坏基线好”误判为有效预测；后者用于识别短历史下去均值/残差表达的外推不稳定，不据此事后调参。

共同样本 N 对照仍按原五模型运行。新增候选只在各 N/H 最大可用样本上作同样本比较，不能将不同 N 的绝对成绩直接相减来评价窗口优劣。

## 输出文件

| 文件 | 用途 |
| --- | --- |
| `REPORT.md` | 完整可读结果、方法与局限 |
| `REPORT.html` | 离线报告、Brier 增益图和概率校准表，可用浏览器直接打开 |
| `brier-gain.svg` | 六个组合的 Brier 增益及测试日期数量 |
| `report.json` | 全部指标、各股 Brier、排序、校准、区块区间、数据覆盖 |
| `summary.csv` | 一行一个 N/H，便于其他工具读取 |
| `run-config.json` | 实际算法与输入参数；可直接作为下一次输入 |
| `daily-states.jsonl` | 原始每日群体状态，保留完整类别概率统计 |
| `rolling-panel.jsonl` | 各 N/H 的窗口特征、收益及缺失/待到期状态 |
| `cv-predictions.jsonl` | 每条样本外五模型概率、真实收益与训练边界 |
| `fold-audit.jsonl` | 每折训练结果最晚日期、测试范围、训练数量、跳过原因 |
| `daily-cv-metrics.jsonl` | 每天的排序相关、前后四分位收益差，便于持续跟踪 |
| `common-sample-cv-predictions.jsonl` | 共同训练/测试样本下的窗口对照概率 |
| `common-sample-folds.jsonl` | 共同样本对照的时间切分 |
| `state-contrast-summary.csv` | 开启对照时，两个新增候选的 Brier、配对增益和逐股时序相关 |
| `state-contrast-temporal-metrics.jsonl` | 开启对照时，原/相对/残差状态的逐股相关和测试行情解释 R² |

开启对照后，`cv-predictions.jsonl` 保留七种市场层候选概率，以及三项核心状态的原值、相对值和残差值；这里的七种候选不等于语义模型的七个 head。原五模型概率、训练/测试身份不因开启对照而改变。

行情更新后可以直接重跑；新帖子需要先有对应的语义预测。脚本不会把 Collector 新文本自行转换成情绪，也不会重新训练语义模型。

报告中的正 Brier 增益表示该候选在当前测试日期上更好。短历史、重叠收益和多组合比较仍使结论只能用于下一轮方向判断，脚本不会自动选定或晋级模型。
