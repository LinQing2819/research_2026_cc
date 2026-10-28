# 第 3 轮评审：可行性、上下文与冻结保真

日期：2026-10-20。
唯一执行协议：`docs/component_replay_protocol.md` V4；统计前置：`docs/reviews/r2_recheck.md` PASS。
本轮只读代码、配置与评审材料，新增本文；未修改协议、代码或子模块，未运行测试、模型、embedding、服务器探测或实验，未检索论文。
“E0 Engineering rep0 有完整 969 条日志”采用用户提供的信息，本轮没有独立验真服务器源文件。

## 结论

**R3 = PASS：可以按 V4 编码独立的冻结机制 pilot，并具备单张 4090 上执行的工程路径。**

没有发现必须先改变统计范围、快照比例、七臂、样本量或研究上下文上限才能实现的阻断问题。
下面的必要实现项是 V4 实现验收的具体化，不是第二份实验协议，也不是宣称现有 E0/E1 已符合要求。
真实源数据有效性、tokenizer 一致性、显存和服务行为仍须在服务器按预定 prepare/preflight/smoke 验收；这些检查可以失败，不要求 pilot 先证明有效才允许编码。

放行边界：R1 的 V0/V1 FAIL 不撤销，双组件方向只作为待证伪研究通过；R2 原 FAIL 已由 V4 复审 PASS 替代。
三轮通过只允许实现冻结 pilot。**不等于远程长跑已成功、ON 有效、selector 有效或方法贡献成立。**
本轮子任务被委托为只评审，因此评审者不启动实现或服务器运行。
用户已授权三轮通过后编码；主代理可据此继续实现，不需要再次请求许可。

## 必要实现项

### 1. 冻结读取，不恢复在线状态

源码依据：[ExpelOnlineST](../../third_party/SeqMem-Eval/methods/expel_online_st.py#L612-L777)、
[Runner seed/resume](../../third_party/SeqMem-Eval/runner.py#L164-L233)。

- Engineering 没有交互 episode 接口，走 `generate()` 的非 action solver。在线成功由 `feedback.lower() == "success"` 决定，不是恢复函数使用的 `score >= 1`。导入时检查两者一致，不能只按 score 重建一个更大的池。
- 规则用边界记录的 `memory_meta.insights_after` 原文，并核对 `memory.insights`、`rules_count` 和前后规则链。不要重新 normalize、补句号、重排或给规则虚构权重。
- 经历按源顺序保留成功记录的 `question.strip()` 和 `memory.rollout.attempts[0].trajectory`；后者是官方从 `<trajectory>` 中提取的内容，无标签时才取原输出的 strip。缺字段就拒绝导入，不回退到 `model_output` 冒充保真。
- 原始记录完整封存；给 prompt 的案例只投影官方读取的字段。源里的 `options`、`gold_letter`、`predicted_letter`、正确性 score 不能因为“保留原记录”而额外插入案例。
- `restore_state_from_memory()` 会把规则权重设为 1，按 score 重建池，也不恢复完整的待更新状态；通用 Runner 更会 `pop("memory")`。两条路径均不用。冻结答题不需要规则权重或 pending batch，直接使用不可变规则文本和成功池即可，不存在必须先修好在线恢复的前置依赖。
- E2 不调用 `generate()`、`update_memory()`、curator 或通用 Runner；只复用纯 prompt builder/评分逻辑，检索使用明确冻结的池。方法对象只可作为隔离的离线参照，不能成为正式运行的可变状态容器。

这能保留已有历史的实际规则和经历，包括它们可能存在的缺点；不能把重放解释为“修正所有 E0 bug 后本来会得到的历史”。

### 2. 忠实检索与真实 prompt

源码依据：[检索及池追加](../../third_party/SeqMem-Eval/methods/expel_online_st.py#L488-L532)、
[few-shot 与 solver builder](../../third_party/SeqMem-Eval/methods/expel_online_st.py#L244-L303)、
[MMLU 数据接口](../../third_party/SeqMem-Eval/tasks/task_mmlu_pro.py#L123-L162)、
[任务模板](../../third_party/SeqMem-Eval/prompts/prompt_templates.json#L7)。

| 对象 | 必须保留的官方语义 |
| --- | --- |
| 当前检索 query | 题干加按 A-J 编号的选项；不含 gold、解释或历史该题回答。 |
| 历史案例的检索文本 | 存储的题干，不追加历史选项或 trajectory。与当前 query 不对称是源码现状，不在本轮“修正”。 |
| embedding | E0 配置的 `Qwen/Qwen3-Embedding-0.6B`，normalize embeddings；固定模型文件、tokenizer、长度设置、库版本、dtype 和 CPU 路径，不改成 BM25/规则相关检索或新增 query instruction。 |
| 排序 | 官方 cosine、降序稳定排序；分数相同按源池顺序。每题 E旧/E新各取一次 top-3，固定 ID、原始浮点相似度和顺序，跨规则臂及回答重复复用。 |
| 案例渲染 | `_build_fewshot_examples_block()` 的 `### Example`、两位小数 `Similarity`、`Task`、`Trajectory`；空经历为 `(none)`。不是名称相似但未用于本路径的 `_build_retrieved_cases_block()`。 |
| 规则与任务 | 空规则为 `(empty)`；同一个 ROLE/OBJECTIVE/RULEBOOK/FEWSHOT/TRACE/CURRENT PROBLEM scaffold；当前题要求最后输出 `\\boxed{字母}`，不改成 Baseline prompt。 |

MMLU 的 `build_inputs()` 不提供 manual fewshots，因此这里没有需要另外补入的 seed demo。
导入时仍应核对池大小；意外存在额外 seed 或池数量不明时拒绝，不能静默猜测。
embedding 输出需要检查维度、有限性和有效范数，持久化向量或可核验的缓存，以及最终排名/渲染结果，恢复时不重新漂移检索。

离线参照必须调用真实官方纯 builder，并给真实官方检索函数注入固定向量，验证 OO/NN 未打包 prompt 的逐字节一致性。
不能只比较两份新写的实现，也不能像旧 E1 测试那样替换整个子进程后便宣称官方保真。
源日志没有保存原始向量和所有历史 token IDs，因此只能承诺源字段保真及锁定环境下的重建一致性，不能声称逐 bit 复现旧服务器所有请求。

### 3. 源完整性、gold 与近重复

[官方 MMLU qid/gold](../../third_party/SeqMem-Eval/tasks/task_mmlu_pro.py#L72-L105) 与
[Runner eval qid](../../third_party/SeqMem-Eval/runner.py#L367-L394) 不是同一种可靠标识：
后者找 `entry["qid"]`，而任务实际存 `_qid`，可能退化成顺序号。源使用 memory JSONL 的真实 qid，不能拿 readable eval 的行号作公共 ID。

必要导入检查：

1. 校验完整公共 Engineering 数据与完整源日志的一一对应、唯一 qid、顺序、题干和选项；不靠 `.done` 或“恰有 969 行”判断完整。截断、重复、错序、拼接、缺专有字段均明确失败。
2. 若 N=969，旧边界为已完成 193 题后的状态，新边界为 581 题后的状态，后缀 388 题。选择边界行分别是零基 192/580；不能把边界题的更新前规则误当更新后规则。
3. 全源验证器与组件构造器分开：前者可流式校验全源 ID/公开内容及计算文件 hash，后者只接收截至新边界的副本。未来 E0 回答、规则、score 不作为构造、选择或分析输入。源后缀完整性检查不是读取未来记忆用于构造的许可。
4. 前缀逐条检查 ST 单次 rollout、feedback/score、累计成功数与 `experience_pool_size`、规则前后链。对未知 schema 或不能解释的池跳变拒绝导入，不补数据。`recent_success_size` 等字段作为审计证据保留，但不据此继续在线更新。
5. 公共输入投影只含 qid、题干、选项等必要信息；gold/answer/answer_index/cot_content 等答案字段只进入独立评分数据。禁止把官方 `build_inputs()` 的整个 dict 或源记录直接序列化进 prompt。历史成功 trajectory 中的历史答案是合法记忆，不应被统一抹除。
6. 在首次生成前检查构建前缀与 dev/test、以及 dev 与 test 的精确和近重复。至少覆盖规范化题干/选项相同、不同 qid 同题、选项重排和高文本相似候选；算法、阈值、候选列表及不看答案的确认记录须冻结。保留数字/公式差异，不能只凭 embedding 相似就断言同题。

V4 未授权按近重复临时删题补位或重抽 split。发现已确认的跨边界泄漏，或候选未决而无法保证隔离，按数据不合格停止 prepare，先修订重审；不能偷偷修改固定题集。
这不是要求证明训练语料绝无污染，也不是重开 discovery/confirmation 口径。
本轮没有真实源文件，不能给出“969 条已通过完整性/去重检查”的结论。

### 4. 独立 prefix packing 与 8192 上限

V4 的 `8192 - 2048 - 256 = 5888` 算术闭合；服务原生窗口再大也不扩展研究上限。
实现不能把 8192 当成“输入上限”，也不能使用 token 数约等于字符数的估计。

- 先固定每题两套检索结果，再分别打包 R旧/R新 和 E旧/E新；保留最长的合格源前缀。只从尾部删整条规则/整例 trajectory，首项太长则该块可为空，不跳过它改取后面短例。
- 规则额度 1024、经历额度 3072，各自标题/编号/Similarity/空标记的归属明确且计入对应额度；不借额度，不按另一个组件是否为空重新塞内容。
- 逐条规则的识别须符合源编号格式，不能按句号随意拆分；不再提取或截断已保存的 trajectory。最终问题、所有选项、答案格式和固定 scaffold 完整保留。
- 分块 token 计数不具可加性。最终完整 messages 必须通过同一个 non-thinking chat template 一次展开，再检查实际发送的全部 token IDs；不能用独立编码片段的简单拼接冒充最终模板编码。
- 对齐组件边界后比较文本、案例 ID/顺序和实际 token span，不要求它们在整串中的绝对位置相同。固定 E 换 R 时 E 不变，固定 R 换 E 时 R 不变；组件长度相同不等于内容不变。
- 任意一臂超过 5888，生成前共同排除该题七臂，不补位。资格表对 dev/test 的公开输入预先生成，不能在看到答案或失败后重算资格。
- 同时报告源组件变化率、打包后变化率、共同变化率、丢弃比例和相同 prompt 比例。组件没有实际改变时标记干预退化，不能把它包装成版本机制被充分检验；这不是要求先出现正效应。

### 5. 单 4090 与精确服务计数

版本证据：[client lock](../../requirements/client.lock.txt#L218) 为 transformers 4.57.6，
[server lock](../../requirements/vllm.lock.txt#L461) 为 transformers 4.51.3、vLLM 0.8.5。
两环境应保持分离。包版本不同本身不否决，但同模型名字或相同 token 总数都不能证明 tokenizer 一致。

现有 [E0 启动脚本](../../scripts/e0/serve_vllm.sh#L20-L43) 默认 `MAX_LEN=32768`、`GPU_UTIL=0.95`，不能直接把默认值当成 E2 的单卡验收配置。
E2 启动说明应固定单卡、TP=1、`max_model_len=8192`、单请求/低并发、明确 dtype 和 `generation-config=vllm`。
保守起点是 `max_num_seqs=1`、eager 执行、与 8192 匹配的 batching 配置，CPU 执行 embedding；不要默认让 SentenceTransformer 自动占用生成卡。
这属于独立 `scripts/e2` 的职责，不要求改 E0 脚本。

8B 的 BF16 权重约 15-16 GiB；若实际配置为 32 层、8 KV heads、head_dim=128，8192 的单序列 BF16 KV 约 1 GiB。
再加运行时 workspace 后，24 GiB 4090 有合理执行空间，但具体可用显存仍以启动日志和 preflight 为准。
不应为了 OOM 静默换量化、换模型、缩输出或提高/降低研究窗口；失败时先诊断服务配置。
2240 个逻辑请求最多预留 4,587,520 个输出 tokens，不承诺远程总耗时；检索和 packing 可在 CPU 预计算，不存在逐题 curator 开销。

preflight 必须可操作地完成：

1. 保存服务启动实参、实际 engine `max_model_len`、served model 映射、模型配置/权重与 tokenizer 文件标识、两端库版本。不能只读模型原生 `max_position_embeddings` 或本地环境变量推断服务实际上限。
2. 客户端 `apply_chat_template(..., tokenize=True, add_generation_prompt=True, enable_thinking=False)` 得到最终 IDs；固定所有 sampling/stop 默认值，禁止再次添加 BOS、套模板或 `truncate_prompt_tokens`。以整数列表发 `/v1/completions`，`n=1`，不用原 OpenRouter chat/retry 封装。
3. 用含空白、Unicode、选项和模板特殊 token 的无答案 sentinel，比较两环境完整模板 IDs；再通过服务 tokenization 接口核对实际 tokenization。vLLM 常用入口是根路径 `/tokenize`，不是当然存在的 `/v1/tokenize`。具体安装返回能力必须实查，本轮没有联网核验运行中的 0.8.5。
4. 只有返回长度而没有可核验 IDs/identity，或拿不到实际服务上限时，不降级为“看起来一样”。版本差异导致 IDs 不同就 preflight 失败；同一个 tokenizer 名称不豁免。
5. 至多两个无评估答案的生成探测：短 sentinel 和接近 5888 输入的合法边界请求；边界请求仍预留正式 2048 输出额度。保存原始请求/响应，核对 `usage.prompt_tokens == len(sent_ids)`、输出额度及成功状态。tokenization/元数据读取本身不是额外生成，但不得暗藏未记账的生成健康检查。
6. 后续每条正式响应继续核对 usage；400/上下文错误不重试，不能临时截断后再发。合法 `finish_reason=length` 是预定输出预算内的结果，记录并按固定评分器评分；网络失败、缺响应或 schema 错误不是答错。

这条路径可实现精确计数，不需要为了本地无模型先安装或下载真实模型；服务器验收失败仍会阻止正式生成。

### 6. 指纹、原子结果与全局尝试账本

只新增 `scripts/e2` 与 `src/memgate/experiments` 的独立实现及针对性测试，不复用 E0 `.done`、旧 E1 bootstrap 或旧 gate/CS 语义。
数据流应是不可变输入、独立生成 transport、独立评分和受限选择，而不是在线 Agent 的恢复运行。

必须形成可校验的指纹链：

| 产物 | 最少绑定内容 |
| --- | --- |
| 源 | 完整源文件 hash、公共数据 hash/有序 qid、两份前缀 hash、边界与 schema、源审计结果；不只记录路径。 |
| prepare manifest | V4 hash、实际相关代码内容及子模块版本/脏状态、模板、embedding/tokenizer 身份、全部固定参数、split/排除表、检索 ID/分数、packing 结果、每请求 token IDs/hash、种子和调度顺序。 |
| preflight 记录 | 上述 manifest、实际服务配置/身份及 sentinel 证据；在首条 dev 生成前绑定到本次运行。 |
| 结果行 | 稳定的 qid/arm/rep、输入指纹、seed、attempt ID、原始响应、usage、finish reason、评分器版本及评分；原始响应与派生评分可分别校验。 |
| selection.json | manifest、完整已验收 dev 结果摘要、各 selector 的授权投影摘要/qid/成本/余额、平局规则和所选 S/F；封存 hash。 |
| test/报告 | 同一输入链与已封存 selection hash、完整预期行集合；不可跨运行目录择优拼接。 |

不要造成自引用 hash：先冻结输入/授权清单，再绑定实际服务，再追加结果及 selection；结果不能反过来修改输入 manifest。
只存 git commit 不足以覆盖未提交文件或换过的模板。改源、参数、prompt、packing、种子、tokenizer 或评分器后，旧结果不能续用。

账本在“实验身份”层持久化，不能通过换 run 子目录、删除 `.done` 或重启进程重置预算。
在 dispatch 前事务性预留 attempt、落盘并锁定计数；崩溃后无法确认是否已提交的预留也保守计费，不退回额度。
可用单写者锁配合事务数据库，或具有同等持久性保证的原子文件/日志；只 `rename` 而不处理账本与结果的崩溃窗口不够。

分别约束首次逻辑尝试不超过 2240、后端生成探测不超过 2、传输重试不超过 44，总尝试不超过 2286；每逻辑请求最多 3 次。
禁用 SDK 隐式重试，所有超时、429、5xx 和可能已经执行的尝试都计账。探测重启也不能获得第三次额度。
retry 保持同 payload/seed，但 HTTP 没有天然 exactly-once：保存各 attempt，按预先固定的响应接纳规则只提交一个有效逻辑结果，不按正确性或长度择优。
已落盘的有效 raw response 应恢复评分/提交，不能因为 score 文件未写完又生成；已完成逻辑行绝不重发。
失败或未决行不能靠缩小 n、换题、补预算或记成 0 分处理；analyze 必须拒绝。

### 7. smoke 继续运行与 test 封存

阶段必须可由程序强制：prepare/资格与预算清单冻结 -> preflight -> dev smoke -> dev 剩余 -> selection 封存 -> test -> analyze。
prepare 可访问 test 的无答案问题进行计数和资格判定，不能生成 test 回答或向 selector 提供其标签。

- smoke 是合格 dev 顺序前 4 题，最多 56 个正式逻辑行；qid/arm/rep、seed 和输出位置与后续 dev 完全相同。不要引入不同的 smoke seed 命名空间或 smoke 专属目录后再拷贝拼接。
- 若合格 dev 不足 4，不能从 test/备用题凑 smoke，按实际合格前缀处理并记录覆盖率；零合格 dev 或组件 selector 不能消费一个完整题块时执行 V4 的拒绝规则。
- 首次生成前冻结两个 selector 的授权 qid/候选/成本清单。完整 selector 看 OO/NN 的全部合格 dev；组件 selector 只看其预算可负担的完整题块前缀，首个放不下就停止。不能把物理上已生成的其他标签免费提供给 selector。
- seed 使用稳定 hash 和明确阶段命名空间，不用进程随机化的 Python `hash()`；检查不同逻辑键的种子碰撞。即便两臂 prompt 相同，也不共享回答。hash、打散和两次请求不是独立性证明，仍保留 R2 的后端稳定/条件独立假设。
- dev 完整通过后原子封存 `selection.json`，test 命令必须验证其内容 hash 与全部依赖，不能只检查文件存在。封存后修改 dev、授权清单或所选臂，test/run/analyze 均拒绝。
- 同一 manifest 重复 prepare/preflight/smoke/run 是验证并继续缺失工作，不覆盖旧行、重做选择或无限重复探测。服务重启后必须验证身份未变；已用完两次探测时不得以恢复为由额外生成。
- smoke 后如需改变影响实验的设计或实现，按 V4 作废本次运行并先修订重审，不按 smoke 准确率调快照、检索或 packing。只查看技术正确性不要求任何效应方向。

## 离线验收清单

以下均为后续必须编写并运行的测试，**本轮没有执行，不能标成已通过**。
使用合成题目/日志、固定向量、fake tokenizer 和可计数的 fake HTTP 服务；所有真实 LLM、embedding 加载及外网请求默认禁止。
其中真实 builder/任务函数可以直接调用，不能被一并 stub 掉；fake tokenizer 通过不替代服务器两端真 tokenizer 验收。

| 编号 | 输入/故障注入 | 可验证的通过标准 |
| --- | --- | --- |
| T01 源完整性 | 完整源、缺首/尾、重复/错序 qid、同 qid 异题、残留 `.done`、缺 rollout | 只接受完整匹配源；边界 193/581 正确；构造器观察不到第 582 条及以后 payload。 |
| T02 冻结字段 | 带/不带 trajectory 标签、成功/失败、score/feedback 矛盾、池大小和规则链断裂 | 保留存储 trajectory 与 insights 原文；不合法源拒绝；把 Runner/restore/update/curator 替换成必抛异常后，冻结路径仍可完成。 |
| T03 官方检索 | 固定向量、平分、空池、少于 3 项、题干相同但选项不同 | 新实现与真实 `retrieve_memory()` 的 ID/顺序/分数一致；历史仅编码题干，query 含选项；两套 E 各检索一次。 |
| T04 官方 prompt | 七臂、空块、特殊字符、多行 trajectory、相似度格式 | 未打包 OO/NN 与真实 builder 字节相等；共同 scaffold 相同；不用另一套 retrieved-cases 模板。 |
| T05 泄漏与近重复 | 给 gold/cot/历史后缀注入唯一哨兵；不同 qid 同题、选项重排、近似改写 | 查询/案例/prompt/selector 中无禁用哨兵；历史合法答案不被删；跨边界重复按冻结审计规则拒绝，不重抽 split。 |
| T06 独立打包 | 长首项、长尾项、空块、刚好 1024/3072、BPE 跨边界的非加性例子 | 只保留整项 prefix；不借额/跳项；七臂因子文本/ID/token span 不串扰；完整题不变。 |
| T07 总上下文 | 最终输入 5888/5889、服务上限 8191、模板重复/BOS 重复、tokenizer 同长度异 IDs | 正好上限可用，超限题七臂共同预排除；后端/身份不合格拒绝；实际发送整数 IDs 且 usage 严格匹配。 |
| T08 请求预算 | 两次探测、44 次 retry、第三次重试、400/429/5xx/超时、SDK 重试 | fake 服务实际接收次数与持久账本一致或被保守上界覆盖；各类限额不能互相挪用；400 不重试。 |
| T09 崩溃恢复 | 预留后/发送后/响应落盘后/行提交前分别杀断；并发启动两个客户端 | 无免费重发、覆盖或双写；已存 raw response 无需再生成；不明 attempt 保守计账；有效行只接纳一次。 |
| T10 指纹拒绝 | 分别改源、模板、规则、embedding 缓存、参数、IDs、seed、结果、selection | 每阶段验证依赖并拒绝混用；仅移动路径且内容相同不应冒充新实验绕预算。 |
| T11 smoke 与选择 | smoke 完成/中断后 run；预算刚好容纳一块/差一 token；后续便宜题；平局 | 正式前 4 题不重生成；选择只读授权投影；首个超预算即停、分母一致、平局顺序固定。 |
| T12 test 与分析 | 无 selection、伪造封存、dev 改动、test 缺行/重复/失败、跨指纹拼接 | 封存前零 test 请求；非法数据全部拒绝；原始完整分母可追溯，不删除失败题后分析。 |
| T13 固定评分/统计接口 | 官方字母抽取、无答案、合法 length 截止、S=F/S=ON、n=0/n=1 | 固定官方评分；技术失败不算错；六对比与 R2 公式一致，零对比 [0,0]、n=0 拒绝、n=1 题间方差不适用。 |

旧测试只能作为已有行为的参考，不能代替上述验收。
服务器另需人工按文档运行源审计、真实 tokenizer/API preflight 与正式 dev 前四题 smoke，保存产物后才决定是否继续同一协议 run。

## 最终放行范围

**PASS，放行独立冻结 pilot 的实现；没有必要修改 V4 的统计设计。**
必须落实上述导入、保真、packing、计数、账本和隔离测试；发现源不完整、泄漏、token 不一致、干预退化或资源不足时如实停止，不通过改题/改预算/换后端消化失败。
不要求修复所有旧 E0/E1 问题，不实施在线 gate，不自动启动远程长跑，不把本评审当成运行结果。
