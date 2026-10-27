# 本轮文献与证据范围

日期：2026-10-26。定向检索，不是穷尽查重。
仅使用下表明确读到的一手来源形成决定；检索未命中不证明没有相关研究。
没有安装额外科研skill：市场检索未找到匹配项。三轮由本会话主代理完成，
没有使用三名独立子代理，不以身份标签冒充外部评审。

| 来源 | 本会话实际读取范围 | 用途 |
|---|---|---|
| [Faulty Memories v2](https://arxiv.org/html/2605.12978v2) | 先前本会话定向提取§5、§7、L.3全文片段，本轮复用 | selective/delayed abstraction、快慢记忆分离已知 |
| [PACE v1](https://arxiv.org/html/2606.08106v1) | 先前本会话§4及定向§6，本轮复用 | 配对验收、单候选保证、功效与dev覆盖限制 |
| [SkillSmith v2](https://arxiv.org/html/2606.01314v2) | 本轮摘要、§1–3.2；定向§3.2/3.3/3.5/3.6/5、A.3/A.6 | 随机mask、二阶效应、Bayesian估计、版本作用域nogoods、验收 |
| [Low-Rank Bandits via Tight Two-to-Infinity Singular Subspace Recovery](https://arxiv.org/html/2402.15739v1) | 本轮摘要、§1–3及目录 | 低秩策略识别、两阶段方法、失配线性化与结构条件已有 |
| [Speed Up the Cold-Start Learning in Two-Sided Bandits with Many Arms](https://arxiv.org/html/2210.00340v3/) | 本轮检索返回摘要与引言，未完成全文 | 两侧因素组成矩阵、低秩筛选后搜索已有 |
| [HbBoPs](https://arxiv.org/html/2412.07820v2) | R4续审：摘要、§1–3.1及后文目录 | 指令/示例结构核、GP与Hyperband已知；E3只筛查实际提示表示的增量 |
| [LIMBO](https://arxiv.org/html/2609.14138v1) | R4续审：检索返回摘要与方法目录，未完成全文 | 逐任务记忆与预算的bandit控制已有近邻，不宣称资源控制首次提出 |
| [原2026-10-20新颖性评审](../r1_novelty.md) | 本轮全文 | 不重启已否决的整库回退＋检验、零差跳过路线 |
| [E2结果解读](../../../reports/e2_component/6b7faf403a0d91dc7d5b4a2497ad83a1a09d3eb28f9b55ea8f4c7a9b2bcb2434/interpretation_2026-10-25.md) | 本轮全文与此前核验的analysis | 新假设动机、实际预算、数据局限 |

SkillSmith有多篇同名论文；本次指arXiv **2606.01314v2**，
不是2605.15215的skill compiler或2608.08037的cloud/local方案。
未经核验的搜索片段不作为具体结论的独立依据。

本轮Skillsmith HTML本地下载缓存：
`.cache/review_20261026/skillsmith-v2.html`；
SHA256=`25485c9280d3668141eb72e038ac49ff935f5fdd89f3fb410724b4a1c896f81c`。
定向章节文本保存在同目录`skillsmith-v2-sections.txt`。缓存不提交Git。
其他来源保留固定版本URL和读取范围，不虚构本地全文档案。

与P2仍需进一步核验的近邻包括：cost-aware pure exploration、
misspecified/robust bandits、结构选择与独立验收的组合、经验型低秩LLM评估。
这项未完成的比较会影响方法新颖性，已在R1结论中保留HOLD。
不能把它解释成“相关论文没有做过，所以我们可以宣称首次”。
