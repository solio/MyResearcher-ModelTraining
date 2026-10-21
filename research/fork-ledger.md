# fork-ledger.md — 分叉账本（2026-10-21）

> 用途：RS-005（及任何新 RS）开工前，回答「哪些数据没被任何实验碰过收益」。
> 规则：登记 dashboard-k3-experiments（侦察机）每个脚本碰过的**日期段 × 变量 × 是否看收益**。
> 「看收益」= 计算或展示了未来收益/超额/t。「看帖」= 只做词表/纯度/一致性，不碰收益。

## 数据源的日期覆盖（先记底账）

| 数据源 | 覆盖 | 备注 |
|---|---|---|
| collector.db posts | 2025-01-31 ~ 2026-10-21 | 文本 |
| date-selection-20261004/…/daily-bars-qfq.csv | 2025-07-30 ~ 2026-09-30 | A+B 段行情 |
| market-data-v0-training-db-20261002/…/daily-bars-qfq.csv | **2024-12-02 ~ 2026-10-20** | 含 C 段！侦察机主用 |
| market-data-v0-20260916/…/daily-bars-qfq.csv | 2026-07-09 ~ 2026-10-15 | 补丁，含 C 段 |

## 逐脚本登记

| 脚本 | 读的行情 | 碰的日期段 | 变量 | 看收益? |
|---|---|---|---|---|
| gen_per_stock_panel.py | training-db + patch | **2025-02 ~ 2026-10-20（含 C 段）** | bearw 密度 × 收益；帖量 × 收益；回撤 × 收益 | **是（全版本全时段）** |
| gen_experiment_005.py | training-db + patch | 同上 | bearw 密度 × 收益（F 规则事件研究） | 是 |
| gen_experiment_007_exit_robustness.py | （经面板产物） | 同上 | 持有期 × 收益 | 是 |
| gen_feature_mining.py | training-db + patch | 同上 | 多特征 × 收益（含 posts_surge） | 是 |
| experiments/tmp-dd-volnorm.py | training-db + patch | 同上 | 归一化回撤闸门 × 收益 | 是 |
| experiments/tmp-prior-return-symmetry.py | training-db + patch | 同上 | 前置涨跌 × 收益 | 是 |
| experiments/tmp-channel-decompose.py | training-db + patch | 同上 | 注意力/约束/信念通道 × 收益 | 是 |
| experiments/tmp-condition-grid.py | 面板 HTML（间接含 C 段收益） | 同上 | 方向×帖量×密度×深度 16 格 × 收益 | 是 |
| experiments/tmp-z-regime.py | 面板 HTML | 同上 | 分段 z × 收益 | 是 |
| experiments/z_selection_report.py | （阈值审计, 无行情） | — | z 阈值机制审查 | 否 |
| verify_panel_integrity.py | bars + collector.db | 全时段 | 对账（密度/价格一致性） | **否**（只验一致性） |
| scan_hits.py | bars + collector.db | 全时段 | 词表命中计数 | **否** |
| forward_track_scan.py | 面板 HTML | — | 前瞻假设计数 | 否（只计数不判） |

## bearw 词表版本史（升级时是否看过收益）

| 版本 | 日期 | 升级依据 | 升级后是否看了收益 |
|---|---|---|---|
| v1（单字） | 2026-10-10 成文 | 从 gen_k3_experiment.py 提取 | — |
| v2（单字→词） | 2026-10-12 | qwen-27b 对账（只看帖，不看收益） | **是**（v3.3/v3.4 重算后对比了超额/t） |
| v3 | 2026-10-15 | 接口测试副产物 | 是（并入面板对比） |
| v4（用户去「跌」） | 2026-10-16 | 用户编辑 + qwen 基准（不看收益） | **是**（v3.7/v3.8 与 v3.4 并排对比收益） |
| v5 | 2026-10-16 后 | — | 是（同上机制） |

**结论**：bearw 每次升级后都看过收益。对 RS-004/005 的含义：词表选择已被收益信息污染过——RS-004 的贪婪词表必须新立，且纯度关（人判）先于任何收益计算。

## 对 RS-005 的硬结论

1. **A 段（≤2026-05-31）**：侦察机的面板版本从 2025-02 起就算过收益 → **A 段收益已被看过**。RS-005 若在 A 段发现关系，只能算「探索性」，不能算验证。
2. **B 段（2026-06-01~09-30）**：training-db 含 B 段且侦察机算过 → **B 段已被污染**。按补充约定第 2 条：任何侦察机来源的候选，B 段通过最高判「弱成立」。
3. **C 段（≥2026-10-01）**：training-db 含 C 段收益且 gen_per_stock_panel/tmp-* 在 10-16~10-20 的重算里看过 C 段收益 → **C 段的前段（10-01~10-20）已被看过**。RS-005 的干净前向验证只能从 **2026-10-21 之后**的数据开始（今日起冻结，之后的新帖/新行情没人碰过收益）。
4. 未触碰的干净集合：2026-10-21 起的前向数据。**这是唯一的未污染验证集。**
