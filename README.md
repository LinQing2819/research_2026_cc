# memgate：自进化记忆的逐样本门控与在线回退监控

研究问题：自进化 Agent 记忆（ExpRAG / DC / AWM / ExpeL 等）在**平均**上提升准确率，但对单个样本可能有害。
本仓库检验并利用这一点：

1. **E0 预实验（当前）**：在公开基准 [SeqMem-Eval](https://github.com/ShenGroup/SeqMem-Eval) 上，用官方实现跑 Baseline 与各记忆方法，按 qid 配对，检验
   - **H1** 记忆导致的伤害率超过采样噪声底（excess hurt rate ≥ 5%）；
   - **H2** 伤害可由生成前特征预测（AUC ≥ 0.65）；
   - **H3** 影子配对探测 + anytime-valid 置信序列（PrPl-EB, Waudby-Smith & Ramdas 2024）在合理探测率下有可用分辨率。
2. **E1（后续）**：逐查询门控 + 记忆版本选择（基于 `--memory-seed-file` / `--no-update-memory` 冻结版本快照）。

## 目录结构

```
configs/                 实验配置（模型、任务、方法、阈值）
src/memgate/             可复用的分析库（pip install -e .）
  io/                    读取 SeqMem-Eval 的逐样本 eval log
  features/              生成前特征（仅用当前与过去信息，无未来泄漏）
  analysis/              配对翻转统计、可预测性 AUC、门控上界
  monitor/               置信序列与影子探测模拟
scripts/
  setup/                 一次性准备（下载公开数据）
  e0/                    E0：起 vLLM、跑实验矩阵、分析出报告
tests/                   单元测试（合成数据只在临时目录生成，不入库）
third_party/SeqMem-Eval  官方基准（git submodule，固定 commit，不做修改）

# 以下由脚本生成，已 gitignore：
data/                    下载的公开数据集
runs/<exp>/<model>/<method>/rep<k>/   原始运行输出与日志
reports/<exp>/<model>/   分析表格、图、verdict.md
```

## 学校服务器（能访问镜像、家目录磁盘已满）

家目录所在分区只剩几 GB 时，把仓库放到数据盘（例如 `/data`），安装脚本会把 Miniconda、两个虚拟环境、模型、数据集和缓存都放在仓库目录里，并强制要求该目录所在磁盘至少有 40 GB 空闲。pip 走清华镜像，模型走 `hf-mirror.com`。

```bash
bash scripts/setup/install_on_server.sh
screen -S vllm
CUDA_VISIBLE_DEVICES=0 bash scripts/e0/serve_vllm.sh    # 只用 0 号卡，端口 8000
# 另开 screen，等日志出现 Application startup complete：
.venv-client/bin/python scripts/e0/run_matrix.py --smoke 5
.venv-client/bin/python scripts/e0/analyze.py --smoke
.venv-client/bin/python scripts/e0/run_matrix.py
.venv-client/bin/python scripts/e0/analyze.py
```

实验进程的 embedding 固定在 CPU 上，GPU 只给这一个 vLLM。多卡时再改用 `scripts/e0/serve_vllm_multi.sh`。

## 环境

```bash
git clone --recurse-submodules git@github.com:LinQing2819/research_2026_cc.git
cd research_2026_cc
# 若已 clone：git submodule update --init

# GPU 机器上：SeqMem-Eval 环境 + 本包装进同一个环境
conda env create -f third_party/SeqMem-Eval/environment.yml
conda activate seqmem-eval
pip install -r third_party/SeqMem-Eval/requirements.txt
pip install vllm
pip install -e ".[dev]"

pytest            # 本地（无 GPU）也可以跑
```

## 运行 E0

```bash
# 1. 下载公开数据到 data/seqmem/
python scripts/setup/download_seqmem_data.py

# 2. 起本地 OpenAI 兼容服务（Qwen3-8B，单卡 24GB 以上）
bash scripts/e0/serve_vllm.sh            # 另开一个终端保持运行

# 3. 冒烟测试：每个任务只跑前 5 条，输出到 runs/e0_smoke/
python scripts/e0/run_matrix.py --smoke 5
python scripts/e0/analyze.py --smoke

# 4. 完整矩阵（5 任务 × (Baseline×2 + 4 方法) = 30 个作业，断点续跑）
python scripts/e0/run_matrix.py --dry-run     # 先看命令
python scripts/e0/run_matrix.py
python scripts/e0/analyze.py
```

`run_matrix.py` 支持 `--only-task` / `--only-method` 拆分作业；每个完成的作业会写 `.done` 标记，重复执行会跳过。
`analyze.py` 会对照 SeqMem-Eval 论文 Table 3 检查复现准确率（容差见配置 `reproduction_tolerance`）。

## 判定标准（见 `reports/e0/<model>/verdict.md`）

| 结果 | 含义 | 下一步 |
| --- | --- | --- |
| GO | H1 与 H2 均成立 | E1：逐查询门控 + 版本选择 + 监控 |
| PARTIAL | 有伤害但不可预测 | 侧重版本回退 + anytime-valid 监控 |
| NO-GO | 伤害不超过噪声底 | 换方向 |
