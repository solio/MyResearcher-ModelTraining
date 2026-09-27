# 行情回填：2026-05-04 至最近已收盘交易日

## 本轮已经完成

2026-09-28 凌晨执行，仅采集日线，不训练、不生成语义预测、不改 Collector。
范围是 Collector 数据库实际已有帖子的16只个股，不是配置里尚未采集的新增股票。
另含沪深300，作为市场背景，不把它改成个股绝对收益目标的基准。

权威新目录：
`/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-data-backfill-20260928/`。

- 请求起点为2026-05-04，劳动节休市后首个实际交易日为2026-05-06。
- 最近已收盘交易日为2026-09-24；9月25～27日休市，执行时9月28日尚未开盘。
- 共101个实际交易日；16只股票各101条原始价、101条前复权价，分别1,616条。
- 沪深300为101条；33个下载请求全部成功，无缺失交易日、重复键或空缺个股行情字段。
- 相较旧56交易日快照，增加45个更早交易日。旧行情与旧研究报告不覆盖。

本轮直接使用 BaoStock。开、高、低、收分别保留原始价和前复权价；
同时含成交量（股）、成交额（元）、涨跌幅（百分数）、个股换手率（百分数）。
指数换手率为空，不以0冒充个股换手率。前复权全区间来自本轮取数，不拼接旧复权口径。
下载日期、provider和单位在逐条数据中记录。

休市说明：[上交所2026年休市安排](https://big5.sse.com.cn/site/cht/www.sse.com.cn/disclosure/dealinstruc/closed/)，
[中秋节休市公告](https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml)。
实际交易日列表保存在新目录的 `trading-calendar.json`。

## 文件和下游使用

- `daily-bars-raw.csv`：实际价格、成交量、成交额与换手率。
- `daily-bars-qfq.csv`：用于跨除权日的历史收益和价格路径比较。
- `benchmark-000300.csv`：沪深300日线。
- `instruments.json`、`fetch-report.json`：股票范围和下载结果。
- `coverage-summary.json`、`trading-calendar.json`：本轮覆盖统计、实际交易日。

后续新帖子回溯完成后，新的研究运行应显式把行情输入路径指向此目录。
旧运行的 `run-config.json` 仍指向旧行情，以保留旧结论；本轮未自动重跑任何实验。
行情回填不会自动补齐五月至七月的语义状态，须等新帖子进入预测链再做较长区间研究。

## 独立重复命令

在 ModelTraining 主目录执行。市场依赖隔离在仓库外的
`pipeline-runs/market-provider-packages`，未更改现有训练运行环境：

```sh
PYTHONPATH=/Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-provider-packages \
.venv/bin/python -m tools.market_data_bootstrap_v0 \
  --market-only --provider baostock \
  --stock-codes 002028 002463 002648 002891 300054 300487 300666 600312 \
                601012 601888 603039 603179 603806 603997 605020 688676 \
  --start-date 2026-05-04 --end-date 2026-09-24 \
  --output-dir /Users/mac/Documents/trae_projects/MyResearcher/pipeline-runs/market-data-backfill-20260928
```

以后将 `--end-date` 改为最近已收盘交易日，将 `--output-dir` 改为新的快照目录。
同目录重复下载只保留已有键；不同取数日期的前复权研究宜使用新的完整快照。
`--stock-codes` 可修改范围，不依赖旧预测文件中是否有该股票。
省略它则沿用原工具的预测文件范围；省略 `--provider` 则沿用 AKShare → BaoStock 回退。
不要省略 `--market-only`，以免进入工具原有的前向语义预测步骤。

本轮新增参数的回归测试与极端状态工具测试共15项通过；这证明下载入口可复用，
不是模型或投资效果验收。
