# V3 九题诊断：独立执行与恢复

本轮已获用户授权，无需再次批准。固定9题、4配置、2次解码，共72个正式请求；另有2个不可重试探测，和最多6次正式请求重试，共享8次预留，总计最多80次。无模型训练、curator或embedding调用，不借用旧E2/E4预算。

2026-11-04已完成执行包及本机模拟验收，真实模型调用仍为0。服务器`zlq@10.154.63.41:22`两次连接均在认证前超时；当前无法上传、读取GPU状态或启动服务。恢复该地址的SSH连接即可接续。

## 封存身份

| 项目 | 值 |
|---|---|
| 准备身份 | `285636a7bb4c884c4581403fe9dd286f7f360ee65864367e4cbf20d700b70ce0` |
| 执行身份 | `0526ef7b0e4530a8bf9cac6f39a00342f07c0ae42b401278ae35a177efa27924` |
| execution.json.gz SHA-256 | `a8e44233e88a6b6feca75552bbc89de73739d653ae47a4db5f4af6ee0a9afe75` |
| 传输包 | `.cache/stage-a-execution-v1.tar.gz`，860,756 bytes |
| 传输包 SHA-256 | `b4a68a3ac1d55636d91acf54c304242f9ea99ec8362e87a1baca437db1707c9c` |
| 独立预算槽 | `v3-curated-9q-20261104` |

执行包绑定原始准备包、全部72个原始请求hash与顺序、9题评分专用gold、官方评分源码、完整模型文件hash、2个探测和运行器源码。原准备包的0授权及空执行清单保持不变；本包单独记录本次授权。只有已封存候选的9个gold进入评分，不回流到内容、选题、种子或顺序。

默认无重试时为74次尝试：正式输入49,392 tokens，两个探测分别40和5888 tokens，合计55,320输入；输出上限151,552。正式请求每个最多重试一次，六个最大可重试输入为1449、1449、1356、1356、1159、1159，因此本执行清单的全额上界为**63,248输入、163,840输出tokens**。这比准备阶段按8×5888计算的保守输入上界更紧，不改变80次授权。实际用量只从有效响应统计，失败或未知用量另列，不当作零。

## 执行行为

1. 校验传输hash、封存包、运行代码和官方评分源码；创建唯一预算槽。
2. 读取服务身份，逐个核验36个正式prompt和2个探测的完整chat文本、token IDs、8192上下文及模型配置；这些请求不生成回答。
3. 两个探测各最多发送一次，所有可能发送的尝试先落盘计费。探测失败即停，恢复也不重新探测；持久化的成功探测直接复用。
4. 按封存顺序完成前8个正式请求，检查结构和账本后自动继续其余64个。成绩、空答案、输出达到长度上限均不触发重试或效果筛选。
5. 对网络失败、429、5xx和完整响应前的连接中断最多重试一次，且保留原payload和seed。进程在发送状态不明时退出，已预留尝试仍计费；收到并持久化的首个有效响应直接恢复。
6. 自动导出原始响应、尝试、结果和元数据，并生成逐题四配置、交互差及三个组合对比。失败时也尽力导出部分结果；未完成不生成总效应结论。

账本固定为服务器`runs/.stage-a-ledgers/v3-curated-9q-20261104.sqlite`；独立标记在仓库`.stage-a-authorizations/`。换运行目录、重解压包或重跑命令不产生新预算。任一已知标记/账本丢失、身份变更、账本被空库替换，均停止，不能删除标记来重开额度。两个文件都人为删除的情况无法仅凭本地软件检测，因此也禁止手工删除整个槽位。

服务固定使用`.venv-vllm/bin/python`，客户端使用`.venv-client/bin/python`；独立pyz只需Python标准库。包不会更新服务器checkout，也不需要git pull或pip安装。服务默认端口8001；若该端口已有服务，客户端仍做完整身份验收。若端口空闲但GPU已有进程，启动脚本停止，保留现有任务；可在确认现有服务符合身份后让客户端screen使用其实际端口。

## Mac本机：校验并上传

以下命令在服务器SSH恢复后执行：

```bash
set -euo pipefail
stage_a_local=/Users/bytedance/research_26/research_2026_cc
"${stage_a_local}/.venv/bin/python" \
  "${stage_a_local}/.cache/stage-a-execution-v1/stage-a.pyz" verify
scp "${stage_a_local}/.cache/stage-a-execution-v1.tar.gz" \
  zlq@10.154.63.41:/data/zlq/stage-a-execution-v1.tar.gz
```

## 服务器：首次解压并自动串联

在服务器执行整块命令，遇错停止。若已解压，只使用下一节的恢复命令。

```bash
set -euo pipefail
cd /data/zlq/research_2026_cc
source scripts/e2/env.sh
printf '%s\n' \
  'b4a68a3ac1d55636d91acf54c304242f9ea99ec8362e87a1baca437db1707c9c  /data/zlq/stage-a-execution-v1.tar.gz' \
  | sha256sum -c -
test ! -e /data/zlq/stage-a-execution-v1
tar --keep-old-files -xzf /data/zlq/stage-a-execution-v1.tar.gz -C /data/zlq
bash /data/zlq/stage-a-execution-v1/launch.sh
```

启动脚本从第一步将模型服务与客户端分别放入`stage-a-service`和`stage-a-client`两个screen，日志在`runs/v3-curated-9q-20261104/{service,client}.screen.log`。客户端最多等待服务就绪15分钟，然后自动执行前置验收→前8个正式请求→其余请求→导出分析；就绪检查不发送模型探测。

## 服务器：查看与恢复

```bash
set -euo pipefail
cd /data/zlq/research_2026_cc
source scripts/e2/env.sh
export SCREENDIR="$PWD/.cache/screen"
screen -ls
tail -n 40 runs/v3-curated-9q-20261104/client.screen.log
```

需要进入客户端时执行`SCREENDIR=/data/zlq/research_2026_cc/.cache/screen screen -r stage-a-client`；`Ctrl+A`松开，再按`D`退出到后台。

客户端不在运行且需要恢复时：

```bash
set -euo pipefail
cd /data/zlq/research_2026_cc
source scripts/e2/env.sh
bash /data/zlq/stage-a-execution-v1/launch.sh
```

如果提示探测不可重试、预算耗尽、身份不符、账本丢失，先保留所有原文件并分析，不能重开额度。若只需导出已保存数据，执行以下只读导出：SQLite以`mode=ro`读取，并正常包含有效WAL；无需模型服务。

```bash
set -euo pipefail
cd /data/zlq/research_2026_cc
source scripts/e2/env.sh
.venv-client/bin/python /data/zlq/stage-a-execution-v1/stage-a.pyz export
```

## Mac本机：取回并独立回放

```bash
set -euo pipefail
stage_a_local=/Users/bytedance/research_26/research_2026_cc
mkdir -p "${stage_a_local}/.cache/stage-a-results-v1"
scp -r \
  zlq@10.154.63.41:/data/zlq/research_2026_cc/runs/v3-curated-9q-20261104/exports \
  "${stage_a_local}/.cache/stage-a-results-v1/"
for stage_a_export in "${stage_a_local}"/.cache/stage-a-results-v1/exports/export-*.json.gz; do
  "${stage_a_local}/.venv/bin/python" \
    "${stage_a_local}/.cache/stage-a-execution-v1/stage-a.pyz" analyze \
    --export-file "${stage_a_export}" \
    --output "${stage_a_local}/.cache/stage-a-results-v1/replayed"
done
```

本地回放重新计算每次payload、原始响应和结果hash，校验首个成功响应、模型用量、预算、官方评分、preflight和输入绑定，并要求整个分析与服务器导出一致。输出包括内容寻址的gzip、JSON和Markdown。不要只复制SQLite主文件而遗漏WAL。

## 本机验收与研究边界

[17项新验收](../tests/test_stage_a_execution.py)已全部通过，覆盖真实HTTP传输接口的模拟服务、74次正常路径、前8项冒烟和自动继续、重启零新增发送、gzip逐字节回放、模型/token错误零派发、共享与总预算、种子/请求篡改、不可重试探测、非重试响应、进程硬退出、有效WAL、双写锁、账本/标记丢失、空库替换、gold隔离和导出篡改。这里的74次是本机模拟HTTP请求，**不是真实模型实验结果**。

随后原准备/内容验收及复用E2运行时回归共39项全部通过；本轮合计**56项通过**。9个传输文件的SHA-256、pyz中运行源码与本地源码一致性、原始封存包以及启动脚本语法均校验通过。真实服务器screen、vLLM和GPU验收仍待SSH连通后执行。

从Mac仓库根目录复现：

```bash
set -euo pipefail
PYTHONPATH=.cache/stage-a-deps:src .venv/bin/python -m pytest -q \
  tests/test_stage_a_execution.py tests/test_stage_a_preparation.py \
  tests/test_stage_a_curated.py tests/test_e2_runtime.py
```

九题均是人工提名、已暴露的开发材料。两个解码先在题内平均；交互差为`p11−p10−p01+p00`，并报告`p11−p00`、`p11−p10`、`p11−p01`。只作描述，不设显著性准入门槛，也不凭正点估计自动扩至2592次。本轮不能验证自由选择器、失败更新、缓存淘汰或V3全系统贡献。
