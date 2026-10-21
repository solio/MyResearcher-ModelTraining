**散户情绪项目的外部论文与公开资源检索**

检索与可用性核查日期：2026 年 10 月 13 日。

这份资料整理服务于方向判断：别人怎样定义和观察散户情绪、怎样发现问题、如何验证，以及哪些公开资源能够实际取得。当前收录 10 项核心材料和 3 项降级候选；本轮没有运行项目实验、下载模型或修改既有标注与预测口径。

已经找到直接研究中国东方财富股吧的论文，也找到细分表达含义、连续评分、作者行为和多重比较的材料。它们支持继续调查这些问题，但没有提供可以直接套用到本项目的盈利结论。本文将作者的做法、资源实际状态和对本项目的判断分别说明。

阅读状态有三种：**正文已通读**、**方法与相关章节已读**、**摘要或项目文档已核查**。未复现实验；“有代码”与“数据完整、环境可运行”分别记录。以下检索是有方向的资料筛选，不是覆盖全部文献的系统综述。

**先看这张索引**

| 材料 | 要回答的问题 | 本轮核查深度 |
| --- | --- | --- |
| [Measuring China’s Stock Market Sentiment](https://sites.duke.edu/jiali/files/2019/04/sentiment.pdf) | 中国股吧语料如何转成可研究的指标 | 数据、词典构造、标注附录、分类对照表、结论 |
| [Online investor disagreement, self-attribution bias, and future stock returns](https://link.springer.com/article/10.1186/s40854-025-00835-3) | 同一作者的观点怎样随涨跌变化 | 假设、数据筛选、指标定义、部分稳健性说明、结论与数据声明 |
| [Fin-SoMe](https://aclanthology.org/2020.lrec-1.749/) | 作者心情、行情判断、理由与盈亏是否混在一起 | 五页论文正文已通读；数据入口核查失败 |
| [FinLin](https://link.springer.com/article/10.1007/s10579-021-09555-3) | 怎样记录对象、连续分数、证据与标注分歧 | 语料设计、标注方法、工具、仓库说明 |
| [StockEmotions](https://github.com/adlnlp/StockEmotions) | 金融论坛能否单独记录情绪类别与多空方向 | 论文摘要、README、文件树与小段 CSV |
| [StockTwits 2008—2022](https://github.com/Jaxingjili/StockTwits-from-2008-to-2022) | 怎样组织作者、帖子、股票与时间的联合研究 | README、分析目录、数据访问文档、远端样本 |
| [中文金融情感词典](https://github.com/MengLingchao/Chinese_financial_sentiment_dictionary) | 怎样从已有词典与语料扩展领域用词 | 方法说明、文件树与词表文件存在性 |
| [arch 多重比较工具](https://bashtage.github.io/arch/multiple-comparison/multiple-comparisons.html) | 试过很多方法后，如何约束挑中偶然赢家的风险 | 官方说明、SPA 接口与示例 |
| [Alphalens Reloaded](https://github.com/stefan-jansen/alphalens-reloaded) | 怎样固定信号表现的比较维度 | 官方 README、功能与许可声明 |
| [The Twitter myth revisited](https://www.sciencedirect.com/science/article/pii/S0378426618302115) | 统计关联能否变成实际预测增量 | 出版社检索摘要与引言片段；未通读全文 |

**1 中国股吧研究中，值得读的是指标怎么来的**

Li、Chen、Shen、Wang、Huang 的《Measuring China’s Stock Market Sentiment》，本轮所读为作者站点的 2019 年 4 月 20 日工作论文版本。研究使用 2008—2018 年东方财富帖子，比较领域词典、SVM、CNN，并分别研究收益、波动和成交量。

方法上的价值在附录：标注者须提供判断关键词，经过培训、双人标注和抽查；作者发现正式财务用语词典不能充分覆盖论坛俚语，因而从人工阅读中构建领域词典。附表同时给出随机和按时间切分的分类比较。

**借鉴边界：**其正负标签合并了买卖意图、价格预期、盈亏抱怨等含义，且舍弃两人不一致的标注。本项目不能把这种综合语气直接命名为“恐惧强度”；其市场层面结果和较长时间尺度也不能直接支持单股短期信号。优先细读附录 I、表 A.1/A.2，并查清市场回归的时间设定。[作者 PDF](https://sites.duke.edu/jiali/files/2019/04/sentiment.pdf)

**2 同一作者的观点变化，已有中国市场研究路径**

Jin 与 Tian 的《Online investor disagreement, self-attribution bias, and future stock returns》，发表于 2026 年。作者报告使用 2019—2023 年、5,059 家 A 股公司的股吧帖子。

其观察单位包括“作者—股票—日”：先以正向概率减负向概率形成帖子分数，再汇总同一作者的当天表达，研究原本不同观点的人怎样回应后续涨跌。这是从总体情绪均值向个体变化追踪的具体方法。

**需要追问的地方：**论文排除间隔较长的发言，研究对象偏向持续发言者；删去不一致标注也会改变训练语料。概率差不自动等于心理强度。上述设计及控制变量提供机制证据，仍需区分机制解释与可执行预测。数据可用性栏写的是 “Not applicable”，本轮未找到配套公开数据或代码。[期刊全文](https://link.springer.com/article/10.1186/s40854-025-00835-3)

**3 Fin SoMe 把我们纠缠过的几种含义拆开了**

Chen 等人的 LREC 2020 论文研究一万条 StockTwits 帖子，由银行资金部门的两位专家分别标注行情方向、有无分析理由、作者心情和交易盈亏。论文用专家分歧与具体文本说明：作者自贴的多空标签未必由正文支持，获利卖出时的开心也不能直接解释成看涨。

可借鉴的是“先检查表达了什么、证据在哪里、不同判断为何分歧”的观察方法。两位专家对隐含信息的接受程度不同，所以文中的一致比例不能成为我们语料的准确率预期；也不能把无相关表达归为心理平静。

本轮已通读正文。作者数据入口请求返回 403，尚未取得数据；论文说明发布标签与帖子 ID，不能据此保证今天能取得全部原文。[论文与全文入口](https://aclanthology.org/2020.lrec-1.749/)

**4 FinLin 提供了连续评分的具体参考**

Daudert 的 FinLin 跨帖子、新闻、公司报告和投资者报告，对同一批公司、同一时段组织语料。标注包含目标对象、−1 到 1 的情绪分、0 到 1 的 relevance 分、证据片段与每位标注者的结果，工具允许“不知道”和分歧复核。

这里 relevance 还涉及文本对目标整体情绪的潜在影响或重要性，不能直接等同于模型置信度。它展示了连续标注如何组织，但没有证明这些分数是经过校准的真实心理量，也没有证明应如何拟合收益。

本轮已读语料与标注章节。GitHub 文件树只有 README 和两张图，README 要求联系作者取得语料，并说明非商业许可；不是已核实可直接下载全部数据的仓库。[论文](https://link.springer.com/article/10.1007/s10579-021-09555-3)、[仓库](https://github.com/TDaudert/FinLin)

**5 StockEmotions 有实际可读的数据，可用来研究标注体系的取舍**

StockEmotions 将用户自贴的 bullish/bearish 与另一套人机协作产生的 12 类情绪分开，提供一万条标注帖子、训练验证测试文件、行情 CSV 和分析 notebook。本轮已读取测试 CSV 的字段及开头记录，确认情绪类别和多空标签分栏存在。

值得借鉴的是两种含义并存的数据组织和采样说明。用户自贴方向、英语表达、2020 年样本及人机协作标签都有适用范围，不能把其标签视为人工金标准。未通读论文、未审计数据切分或运行 notebook；本轮在根目录未看到标准 LICENSE 文件。[作者仓库](https://github.com/adlnlp/StockEmotions)、[论文](https://arxiv.org/abs/2301.09279)

**6 StockTwits 长期数据项目，适合看研究数据怎样组织**

《StockTwits: Comprehensive records of a financial social media platform from 2008 to 2022》的配套仓库包含数据整理、热度变化、情绪与价格关系、股票网络及使用教程等 notebook。

数据将匿名作者、帖子时间、回复关系、股票标识及用户自贴方向联系起来。本轮按作者文档对公开 S3 文件做了 2 KB 范围读取，返回 HTTP 206，字段可读；只证明这个样本文件可访问，不代表已核验全部数据。

可借鉴字段组织和研究拆分；“某些作者预测更好”等 README 结论仍需读论文和验证代码，不能作为本项目选人或预测的现成依据。未执行 notebook，根目录未见标准 LICENSE 文件。[仓库及数据说明](https://github.com/Jaxingjili/StockTwits-from-2008-to-2022)

**7 中文金融词典有现成资源，词典的形成过程更值得借鉴**

姜富伟、孟令超、唐国豪的中文金融情感词典公开了 Excel 文件。README 描述了金融词典翻译、通用中文词筛选、word2vec 扩展和人工筛选，列出 9,228 个词，并说明可免费使用及引用要求。

这提供了可追溯的候选词发现方法。其研究场景涉及媒体文本，不能默认覆盖股吧黑称、反讽或交易宣言。对本项目的用途是对照词项来源、研究扩词方法；新词是否有效仍需看实际命中语境。本轮核验 README 与文件存在，未把词典并入现有规则。[作者仓库](https://github.com/MengLingchao/Chinese_financial_sentiment_dictionary)

**8 arch 对应我们反复试条件后如何判断有效的问题**

官方工具提供 SPA、StepM、MCS 等基于自助抽样的多重比较程序。SPA 使用一个基准与多个候选方法在同一组时间点上的损失序列，可选择保留时间依赖的分块抽样。

它能帮助研究“试过很多候选之后，发现的优胜是否仍有证据”，前提是候选、损失定义和时间序列都整理完整。它无法补回未记录的试验，也不能消除数据泄漏或替代未来验证。这里收录的是评估工具，本轮未安装或运行。[官方方法说明](https://bashtage.github.io/arch/multiple-comparison/multiple-comparisons.html)、[SPA 接口](https://bashtage.github.io/arch/multiple-comparison/generated/arch.bootstrap.SPA.html)

**9 Alphalens 可参考报告结构，不能代替业务评价定义**

Alphalens Reloaded 是公开的因子分析工具，提供收益、信息系数和换手等分析，输入因子与后续价格生成标准报告；仓库声明 Apache 2.0 许可。

可借鉴其固定比较维度和数据接口。它的横截面因子分析与本项目“每只股票多个时点”的事件研究存在区别；分组平均收益也不等于真实账户收益。本轮核查了功能、示例入口和许可，未运行，不建议直接搬用默认报告。[项目仓库](https://github.com/stefan-jansen/alphalens-reloaded)

**10 反面结果必须与正面结果一起保留**

Behrendt 与 Schmidt 的《The Twitter myth revisited》，Journal of Banking & Finance，2018，研究道指成分股的 5 分钟波动。出版社摘要报告：控制日内周期性后存在统计关联，但经济幅度很小，加入 Twitter 情绪和活动度没有改善样本外预测。

本轮只核查到出版社摘要与引言片段，全文尚未通读。其美国 Twitter、日内频率与我们有明显差别，不能据此判定股吧无效；同样也不能凭别处的正面结果认定“预测收益不行，预测风险就一定有用”。后续应把它与中国股吧研究并读，核查基准和控制项的差别。[出版社原始出处](https://www.sciencedirect.com/science/article/pii/S0378426618302115)

**检查后降低优先级的三个项目**

| 项目 | 实际发现 | 当前用途 |
| --- | --- | --- |
| [VincentWen0320/content-analysis](https://github.com/VincentWen0320/content-analysis) | 仓库有五份 Python 脚本；已读的两份含 Python 2 风格代码，并引用文件树里没有的分词词表、乐观悲观词表及语料 | 可读规则思路，暂不算完整可复现方案 |
| [Stock-Market-Sentiment-Analysis](https://github.com/ZijianWang1125/Stock-Market-Sentiment-Analysis) | 有采集、训练、预测脚本和报告；README 描述的训练数据、权重目录没有出现在本轮获取的仓库文件树中 | 教学流程参考；尚不能确认其数据与结果可复现 |
| [FinSight](https://github.com/ranafaraz/FinSight) | README 明确默认使用内置情绪与回报关系的合成市场，真实数据后端可选 | 可参考验证演示；展示成绩不能作为真实市场效果证据 |

**从这些材料里可以形成的研究问题**

下列是阅读后的综合判断，尚未转成项目实验计划。

1. **我们究竟要测什么？** 作者心情、行情预期、交易宣言、针对对象的评价可以同时存在。Fin-SoMe、FinLin 和 StockEmotions 提供不同拆分方式，应比较各自的定义、舍弃的信息和标注分歧，而非选择标签最多的一个。
2. **一个帖子的分数怎样成为群体指标？** 中国股吧论文有按帖汇总，也有先形成“作者—股票—日”再汇总的做法。它们代表的观察单位不同，值得研究其样本筛选和沉默者缺失如何影响解释。
3. **收益、波动、成交量中的哪种关系可重复？** 需要把样本、预测时间跨度、基准、控制变量和样本外结果放在一起比较。存在相关性与存在可用预测增量是两项不同证据。
4. **规则怎样扩展、结果怎样不被试验次数误导？** 中文词典给出了候选词形成过程；arch 给出了多候选比较工具。两者解决不同环节，使用前均需核对本项目的目标与输入条件。

**后续资料工作的交付形式**

优先深读中国股吧的两篇研究，再与 Fin-SoMe、FinLin 对照定义，并补足反面论文全文。每篇深读记录固定回答：提出什么问题、怎样观察到问题、如何构造样本和变量、用了哪些比较、结论依赖哪些条件、反例和不足在哪里、数据代码是否真实可取。只有摘要的材料保持摘要级结论；未运行的代码保持未复现状态。

公开资源的下一层核查应记录：实际输入输出、缺失文件、依赖、许可、复现实验范围与需要付出的成本。资料搜集本身不触发租卡、训练或替换现有研究口径。

可核验的来源与访问状态同时保存在 [sources.json](</Users/mac/Documents/trae_projects/MyResearcher/produce-docs/sentiment-external-research-20261013-sources.json>)，后续补充沿用这一份索引。
