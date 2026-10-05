# K3 语义审计工作线(k3-semantic-audit 分支)

**分支目的**:把 2026-10-05 会话中 K3(Claude)完成的"逆向投资信号瓶颈三层诊断 → 语义层人工仲裁 → 标签体系方法论评审 → 语态频率独立测量"整条工作线独立承载,供后续演进(标签体系 v2、K3 标注流程、真值集扩展)。

**基线**:`main` @ 79ced99(创建时主工作区有 235 个未提交改动,本分支不包含它们,是干净的隔离环境)。
**worktree 路径**:`/Users/mac/Documents/trae_projects/MyResearcher/worktrees/k3-semantic-audit`

---

## 内容地图

### 一、诊断主线(最终交付)
| 文件 | 内容 |
|---|---|
| `REPORT.md` | R-series 主报告 R1–R5:三层裁决(策略层/标的层/语义层)+ 仲裁结果 + 测量 bug 清单 |
| `session-notes-2026-10-05.md` | 会话工作记录(决策与教训的全过程) |

### 二、策略层(期限结构)
| 文件 | 内容 |
|---|---|
| `short_horizon_recalc.py` | 重算脚本:1/2/3/5/10 日持有期,入场 open→第 H 交易日 close(H=9 复现原口径 1096/1096) |
| `short-horizon-summary.json` / `short-horizon-per-stock.jsonl` | 结果:极值全期限无 alpha;冷却呈倒 U(5d +0.86% 峰值) |

### 三、标的层(散户定价权分层)
| 文件 | 内容 |
|---|---|
| `retail-proxy-stats.json` | 20 只票换手率/振幅代理指标(7 倍跨度) |
| (结果在 REPORT.md R2) | extreme 10d 在 HIGH 散户组 +2.22%(剔单股稳健),全池仅 +0.18% |

### 四、语义层(人工仲裁)
| 文件 | 内容 |
|---|---|
| `sample_k3_audit.py` | 三层抽样脚本(fixed-seed=42:极值日 60/高置信 30/随机 30) |
| `k3-audit-sample.jsonl` | 120 条待标样本 |
| `k3_labels.py` | K3 盲标结果与对比(含反讽维度) |
| `k3-audit-merged.jsonl` / `k3-arbitration-pack.jsonl` | 合并表 / 待仲裁分歧包 |
| `arbitration.html` + `build_arbitration_html.py` | 可交互仲裁页(键盘 1-4 判定,localStorage,导出 JSON) |
| `arbitration_page_test.js` | 页面行为端到端验证(jsdom,14/14 通过) |
| `arbitration-result.json` | **用户本人 84/87 条判定结果(人工真值来源)** |
| `ground-truth-v1.jsonl` | **79 条机器可用真值集**(K3确认53/学生17/双有效6/备注提取3) |
| `label-system-gaps.json` | 标签体系缺口清单(装不下的 5 条) |

**核心数字**:分歧集内 K3 63% vs 学生 20%;方向硬分歧 K3 81%(13/16);学生把 BULL 误判 BEAR 不对称 10:1。

### 五、方法论(标签体系评审)
| 文件 | 内容 |
|---|---|
| `label-system-review.md` | 评审 v2:方案逐条裁决 + 标签体系工程四层方法论 + P0–P5 优先级 |
| `freq-probe-report.md` | 语态真实频率独立测量(200 条盲判 + Wilson CI) |
| `freq-probe-sample.jsonl` | 频率测量抽样(seed=42) |

**关键裁决**:看跌买入真实 4.5%(值得一等特征,但用交叉特征实现);做T 真实 3.0% vs 输出 1.3%(teacher 漏标一半);FOMO 本窗口稀有 + 时间窗口偏差;NONE_EXPLICIT 必须保留。

---

## 下一步路线图(评审 v2 的 P0–P5)

- **P0**(零标签变更):补"逆向买入语态"训练样本 → 聚合层加 stance×action 交叉特征 → 重算极值日 stance_index
  - 种子样本:`ground-truth-v1.jsonl` + `freq-probe-sample.jsonl` 中 K3 判出的 9 条看跌买入
- **P1**:标注规程修复(决策树/类配额/难例必标)+ DO_T 漏标修复 + 死标签审计
- **P2**:判定输入加行情快照(消除"应查当日行情"类不可判定)
- **P3**:情绪对象扩展(其他散户/主力/场内/行业),逐个声明下游消费者
- **P4**:情绪词表修订(恐高/自我安慰,先作子标签观察,扩样 500 条)
- **P5**:DO_T 拆正T/逆T(前置:补漏标后复测可定向比例)

## 数据依赖(不在本仓库,用绝对路径引用)

- 信号与行情:`/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/date-selection-20261004/`(evaluation/all-method-opportunities.jsonl、market-source/daily-bars-qfq.csv)
- 帖子预测:`.../date-selection-20261004/refresh/refresh-v3/source/m12-predictions.jsonl`
- 固化副本:`/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/contrarian-bottleneck-20261005/`(本目录的原始出处)

## 复现

```bash
# 期限结构重算
python3 reports/k3-semantic-audit/short_horizon_recalc.py
# 三层抽样(seed=42,可复现)
python3 reports/k3-semantic-audit/sample_k3_audit.py
# 仲裁页(浏览器打开)与其行为验证
open reports/k3-semantic-audit/arbitration.html
node arbitration_page_test.js   # 需 jsdom
# 真值集重建
python3 reports/k3-semantic-audit/build_ground_truth.py
```
