# V4 研究台：观察 / 训练 / 预测

这是 ModelTraining `deepseek-v4` 分支里的 v4 报表管线。它把“看数据”“学关系”“记预测”拆成三个版本，并在现有 Dashboard 的 8766 进程里以静态页面呈现。

**Dashboard 不新建分支、不新建工作树、不起新服务。** 页面只是被复制到现有 `worktrees/dashboard-k3-experiments/static/`，由已经运行的 8766 进程直接提供：

```
http://127.0.0.1:8766/static/v4.html
```

## 目录

| 路径 | 作用 |
|---|---|
| `build_v4.py` | 唯一数据构建入口；只读 collector/行情/K3 参考快照 |
| `lib/` | 时间基准、输入读取、统计与三段报表计算 |
| `config.json` | 口径、路径、48 个规格、bootstrap seed |
| `vocab/` | K3 的 `bearw.v2` + excite/declare + 3 个个股黑称，带 sha |
| `reference/` | 从 K3 panel 冻结的 H-1 事件快照（不含原文） |
| `stocks.json` | 20 股名称、研究分组、板块与涨跌停 |
| `hypotheses.json` | GPT/K3/v4 假设台账 |
| `lineage.json` | 三条工作线的版本谱系 |
| `external-evidence.json` | 外部文献/公开数据卡片 |
| `dashboard/` | 页面源码（v4.html / v4.css / v4.js） |
| `out/` | 冻结数据与 manifest |
| `tests/` | 时间基准、事件规则、JSON 不变量测试 |

## 构建与检查

```bash
# 从 ModelTraining deepseek-v4 工作树根目录运行
python3 v4/build_v4.py \
  --output v4/out/v4-data.json \
  --manifest v4/out/manifest.json

python3 v4/build_v4.py --check --output v4/out/v4-data.json
python3 -m unittest discover -s v4/tests -v
```

构建是确定性的：bootstrap 固定 seed `20261020`，`report.generated_at` 固定为 `2026-10-20T12:00:00+08:00`。同一输入重跑应得到相同 sha（除 manifest 的路径字段）。

## 发布到现有 Dashboard（不新起进程）

```bash
DASH=/Users/mac/Documents/trae_projects/MyResearcher/worktrees/dashboard-k3-experiments/static
cp v4/dashboard/v4.html v4/dashboard/v4.css v4/dashboard/v4.js "$DASH"/
cp v4/out/v4-data.json "$DASH"/v4-data.json
```

然后刷新 `http://127.0.0.1:8766/static/v4.html`。需要 8766 的 Dashboard 进程已在运行；**不要**为本页启动第二个服务。

也可以运行 `v4/publish_to_dashboard.sh`，它只做上述复制。

## 口径（冻结）

- 信息日：帖子在交易日 15:00（含）前发布，归该交易日；15:00 后、周末或节假日归下一个交易日。
- 入场：信息日后的下一个交易日**开盘**。
- 5 日持有：入场后第 5 个交易日**收盘**（信息日索引 + 6）。
- 密度：`bearw.v2` 正则 + 3 个个股黑称；标题先做与 K3 相同的归一化。
- 有效日：当日 ≥5 帖，单日 top1 作者占比 ≤60%，有行情。
- z：expanding 均值/标准差，只用此前 ≥60 个有效观测；不使用全样本未来信息。
- 极端：z ≥ 2。深回撤：收盘价距过去 21 个交易日最高收盘 ≤ -20%。
- F：同一簇（极端日间隔 ≤5 个交易日）内只取前两个极端日；第 3 个起停手。
- 成本：双边 20bp（单边 10bp）假设；不建模滑点、涨跌停、停牌、冲击成本。
- 多重比较：预声明 48 个规格（z × dd × horizon），全部展示；主规格固定 `z=2, dd=-20%, h=5`，不挑最高 t。
- 推断：逐事件、逐股、事件簇三种等权口径同时报告；n<3 不报 t/CI；bootstrap 按股票/事件簇重采样。

## 版本轴

| 轴 | 本页 | 不能混同 |
|---|---|---|
| 报告 | v4.0 观察 / v4.1 训练 / v4.2 预测 / v4.3 研究手记 | rule-triage v4.3 是测量规则版本 |
| 测量词表 | `bearw.v2` + 3 个黑称 | K3 panel 的 v3.x 是策略口径 |
| 参考事件 | K3 panel v3.4 的 26 笔 H-1 快照 | 只作“同事件换时点”的参照，不进入 v4 统计 |

## 数据边界

- 613,985 条帖子、20 只股票、2025-01-31 至 2026-10-16；正文非空仅 58,494 条（9.53%），大部分只有标题。
- K3 的 377,163 帖 / 1.2% 作者 / 29.5% 帖量来自更早快照；当前重算为 118,157 位作者、头部 1% 作者占 30.39% 帖量，不能混用。
- K3 健康表里的断档在当前 613,985 帖快照中可能已被回填；覆盖率是当前数据的描述，不是历史采集无缺口的证明。
- 20 只股票不是全 A 股，股吧发言者不是全体散户。
- 页面不包含帖子标题、正文、作者 id 或帖子 URL；只含聚合统计。

## 已知结论

- K3 的发布头条 +3.19pp（t=2.06，26 事件 / 12 簇）在改用因果 expanding z 并重选事件集后变为 +1.33pp（t=0.57，12 事件 / 11 簇，逐事件等权）；逐股等权 +2.44pp（t=1.02）。点估计仍为正，但两者都不显著；H-4 前瞻 -15.58%。
- 同一批事件换到次日开盘是 +3.81pp（t=2.56），所以**执行时点不是主要衰减源**；全样本阈值/事件选择才是。
- 2×2 对照：S+D+ +2.57%（16 个股票日 / 9 股）vs S-D+ +0.81%（349 / 19），差 +0.09pp（t=0.07）；S+D- vs S-D- 差 -0.27pp（t=-0.39）。情绪在深回撤内没有可辨识的正增量。
- 同股匹配只有 2 对，不报 t/CI；不能据此判断情绪增量。
- 48 个规格里最高 t 的角落是 z2.5/dd-25/h5：+7.52pp，t=2.25，仅 5 个事件，仍低于 t=3。
- 当前 0 个 S∩D 候选；市场 risk-off；H-4 江丰 2026-10-09 信号在 10-12 开盘 203.00 → 10-19 收盘 171.37，毛收益 -15.58%，第一笔预声明前瞻失败。H-4 不计入训练分布。
