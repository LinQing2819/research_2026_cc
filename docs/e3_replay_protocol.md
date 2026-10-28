# E3-R1：实际提示表示的等预算选择回放

2026-10-26，最终协议E3-R1。性质：已完成数据上的回顾性算法筛查。
对应科研草案CVS-P3；替代CVS-P2作为当前唯一实现目标。
准入结论与历轮问题见[收敛记录](reviews/e3_replay_design/convergence.md)。
旧E2/V5及其INCONCLUSIVE结论不修改。

## 1. 问题和准入范围

问题：在相同候选、相同预留token预算与独立阶段数据访问下，
使用检索/packing后实际组件文本，是否比原始记忆文本或不共享组件信息，
更有利于选择一个固定配置？

A=ExpeL的规则/经历版本；B=实际提示表示驱动的评估分配与候选复核。
方法暂名“rendered component selection”，简称RCS，只是工程标识。
目标增量是表示如何影响选择；GP、核方法、预算分配、样本分割都不是新发明。
HbBoPs已研究指令/示例结构核及多保真选择：
<https://arxiv.org/html/2412.07820v2>，本轮读取摘要、§1–3.1。
SkillSmith、PACE及低秩bandit约束继续适用，见原[来源](reviews/e3_replay_design/sources.md)。

**本阶段允许实现并运行只读导出和本地回放；新增LLM调用=0。**
它不验证新的题目泛化、不证明相对HbBoPs完整方法的优势，不发安全部署证书。
研究风险在于算法可能无优势；这是可以实测的失败结果，不是禁止筛查的理由。
既有score已被查看，所有回放数据都属于开发证据。

## 2. 唯一输入和完整性

固定E2运行：
`6b7faf403a0d91dc7d5b4a2497ad83a1a09d3eb28f9b55ea8f4c7a9b2bcb2434`，
软件0.2.1、dev64、test94、七臂、每臂每题两次。
只读SQLite一致事务必须包括已提交WAL，禁止实例化写Ledger、删除/重置meta。
导出核对manifest/所有列入manifest文件的封存和文件hash、selection、preflight、
analysis关联；核对每个accepted result及其request/raw hash、prompt IDs、seed、usage、
原官方答案提取器评分、首个完整成功响应。

导出包含manifest、snapshots、questions、聚合analysis和所需逐题投影，
不包含原始响应正文或未来E0后缀。历史trajectory已在questions/snapshots中，
用于原始表示和实际表示比较，不是当前test的预测特征。
导出失败则停止，不“尽量导出剩余行”。
JSON整体canonical封存并gzip；输入文件digest成为回放结果身份。

## 3. 固定候选和数据角色

候选永远为OO、ON、NO、NN、ZE、RZ、ZZ，与原E2一致。
不补缺失的旧规则-only/旧经历-only，不合成反事实结果。

- S：原manifest dev列表前40道，供搜索与表示构造；
- V：原manifest dev列表后24道，供搜索结束后的有限候选复核；
- T：原94道test，只在最终配置选定后作描述性测量。

S/V/T划分固定。既然旧结果已公开，本划分只是模拟部署的信息屏障，
不是恢复独立确认资格。原E2的selection不回改。
16次预定顺序seed=0…15，仅改变各阶段读取顺序：
按SHA256(`["E3-R1",stage,seed,qid]`)升序。
这些16次回放共用同一批回答，不是16条独立历史或16次新生成。
各种方法在同一seed使用相同S/V顺序，r=0/1回答全部平均，不能挑最好。

## 4. 表示和核，所有参数在本协议冻结

统一文本特征：Unicode字符三元组频数向量，L2归一化；空字符串用专用非空标记。
不训练encoder、不调用embedding或LLM，不用gold/score构造特征。
短于三字符时整串作为一个特征。字符、空格和数字保持原样。

实际表示：questions[q].rules[version].text和cases[version].text。
前packing表示：snapshots的原始规则，及questions[q].ranked[version]
通过原官方cases builder重建的完整top3文本；空组件用相同官方标记。
原存储表示：snapshots的原始规则，及该版本整个经历池的question/trajectory，
以canonical JSON编码；不把“整个经历池”假装成真正送入模型的文本。

在S的全部40题上，分别平均两个候选规则特征的内积和经历特征的内积，
得到R和E Gram矩阵。记忆组件不包含当前题干、选项或新回答。
定义：

`K = .25 × (.2R + .2E + .4(R⊙E) + .2I)`。

这是固定和核/积核的组合；正半定来自Gram、Schur乘积和非负加权，
不是本研究新定理。I是独立配置残差，避免完全由相似度替代直接测量。
K仅为working covariance，不能把后验标准差称为已校准的置信区间。
主均值先验.5，观测均值噪声方差`.25/n_answers`，对每臂累计均值拟合，
不把同一观测重复加入。数值求解使用float64及1e-10 diagonal jitter。

## 5. 六个方法，共享同一候选/数据/成本/复核

| 方法ID | 唯一区别 |
|---|---|
| rendered | 上述实际表示＋完整K，主候选RCS |
| unpacked | 同算法，改为前packing表示，隔离packing效应 |
| stored | 同算法，改为原存储表示，隔离检索后表示的价值 |
| additive | 实际表示，但K=.25×(.4R+.4E+.2I)，不共享乘积交互 |
| independent | K=.25I，不共享组件效用 |
| uniform | 相同初始化及预算，按累计已读题数从少到多轮转分配 |

它们是控制算法/消融，不冒称HbBoPs或SkillSmith官方复现。
对HbBoPs完整强基线的比较属于下一阶段门槛；本阶段不得据此宣称SOTA或论文创新成立。

## 6. 搜索规则，不能在T上调整

搜索预留预算 `B_S=524288` tokens。
每次购买一个臂接下来2道S题的两次回答，块成本为
`Σ_q 2×(question.prompts[arm].prompt_tokens+2048)`。
不依据真实completion长度预算分配。若下一块放不下，不能跳过块中昂贵题。
全部候选初始化各2题；若预算无法负担全部初始化则停止运行。

非uniform算法每步重算后验m_a、s_a：

1. 每4个初始化后的搜索动作，第4个做强制探索：
   从尚有可负担下一块的候选中，选已读题数最少者。
2. 其余动作：在可负担候选中保留
   `m_a+s_a >= max_b(m_b-s_b)` 的集合；
   为空则恢复全部可负担候选，不宣布其他臂已证明较差。
3. 选 `s_a² / next_block_cost(a)` 最大者。
4. 同值按候选顺序`ZZ,ZE,OO,ON,NO,NN,RZ`。

所有候选达到S上限或均无可负担下一块时结束。
这个采集函数是固定启发式，不主张无后悔或最佳臂识别保证。
任何臂都不因模型预测永久禁止，未测答案也不能被模型填入provider。
uniform按已读题数最少、同一候选顺序分配。

## 7. 有限候选复核与最终选择

搜索结束，用后验均值（uniform用直接观测均值）选前两名非ZZ配置，
加ZZ共3个不同配置，随后封存finalists。
复核预留预算 `B_V=393216`。
按共同V顺序每次购买下一道题的三个候选×两次回答，必须整块同时能放下。
取可负担最长前缀，至少8题，否则技术不合格，不能默默退回搜索分数。
最终按V平均正确率最高者选择；平局按S上的平均预留成本较低者，
再次同值用上述固定候选顺序。

这是候选复核，**不是3pp显著性验收**；不会声称ZZ或所选配置安全。
不另套PACE后将未经功效验证的门槛叫证书。最终所选配置固定用于全部T。
搜索和复核总预算上限917504 tokens；实际虚拟花费按已读取结果逐条计账。

## 8. 输出与判定

每方法/seed保存：完整搜索轨迹、各臂所读qid、预留/实际tokens、
finalists、V实际题数/均值、最终所选臂、T平均正确率。
T不向搜索/复核provider开放；反向更改T不能改变选择轨迹。
固定OO/ON/NO/NN/ZE/RZ/ZZ显示零选择成本与同一T准确率。

主筛查量：16次预定顺序下rendered相对stored、independent、uniform
最终T准确率差的平均值。另报告全部16次差值和所选配置频数，
不能只报rendered最好的seed。packed/unpacked/additive用于解释增量来源。
不对16次共享数据回放做iid置信区间或p值，不改变E2六个主区间。

固定实用筛查规则（不是显著性结论）：

- 任一数据/时序/预算不合格 → INVALID；
- rendered对stored、independent、uniform三个平均差都>=.02，
  且至少8/16次选出非ZZ配置 → WORTH_INDEPENDENT_TEST；
- 否则 → NO_INCREMENTAL_SIGNAL_ON_THIS_TABLE，不自动追加同题生成。

若所有方法选同一固定臂，说明本表不能展示复杂方法价值。
即便WORTH_INDEPENDENT_TEST，也不能据已看过的T确认2pp改善。
这个规则只约束是否值得付出下一阶段设计成本，非获批自动启动GPU。

## 9. 成本、效力和后续关口

实际新增生成成本=0；旧2212正式回答的生成成本不隐去，原运行约739万已接纳tokens。
每个方法/seed的虚拟评估成本<=917504 tokens；同一回答被多种算法读到时，
各自计入虚拟成本，实际不重新生成。16个seed的结果不当独立样本扩大功效。
全部Gram只需7×7、最多40题，CPU可完成。

筛查不能解决多版本结构或跨历史泛化；本阶段不承担这些主张。
若下一阶段值得进行，需另行冻结：
新语义隔离的B/S/V/T、至少两个任务与明确数量的新构建历史、
更多版本、HbBoPs完整/忠实适配强基线、固定效应目标与功效、
包括构建和全部方法的总成本。新的LLM实验未自动批准。

## 10. 实施与验收合同

代码放在`scripts/e3/`和`src/memgate/replay/`，不修改E2指纹覆盖的文件、
旧软件版本、官方子模块或旧账本。新脚本有独立E3-R1协议号。
export入口可在服务器直接运行；run入口在本机读取导出包。

必须验证：只读WAL、破坏hash拒绝、已接纳score与原scorer一致、
缺行/重复/外来行拒绝、S/V/T访问隔离、昂贵块不跳题、V共同分母、
GP核PSD及数值稳定、所有方法同预算、T结果变化不影响选择、
相同输入/seed结果完全可复现。合成验收不冒充真实E2筛查结果。

本机没有完整E2账本与questions；服务器SSH目前超时。
实现/合成验收可完成，真实回放必须等用户导出原件传回，
不能用聚合analysis伪造逐题数据。
