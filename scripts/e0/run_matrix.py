"""Launch the E0 run matrix (baseline repeats + memory methods) through SeqMem-Eval.

Each (method, repetition, task) is one SeqMem-Eval process writing to
``runs/e0/<model_tag>/<log_name>/rep<k>/``. A ``<task>.done`` marker is written
on success, so re-running the script skips finished jobs.

Examples:
    python scripts/e0/run_matrix.py --dry-run
    python scripts/e0/run_matrix.py --smoke 5            # 5 samples per job, separate dir
    python scripts/e0/run_matrix.py --only-task MATH500 --only-method Baseline
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from memgate.config import load_config, model_tag, resolve  # noqa: E402
from memgate.io.seqmem_logs import run_dir  # noqa: E402

DATA_ROOT = resolve("data/seqmem")
DATA_OVERRIDES = {
    "MATH500": DATA_ROOT / "MATH500" / "test-2.jsonl",
    "MMLU-Pro-Engineering": DATA_ROOT / "MMLU-Pro" / "test.parquet",
    "MMLU-Pro-Physics": DATA_ROOT / "MMLU-Pro" / "test.parquet",
    "MMLU-Pro-Math": DATA_ROOT / "MMLU-Pro" / "test.parquet",
    "HumanEval": DATA_ROOT / "HumanEval" / "test.parquet",
    "APIBench-HF": DATA_ROOT / "APIBench",
    "APIBench-TF": DATA_ROOT / "APIBench",
    "APIBench-TH": DATA_ROOT / "APIBench",
}


def build_jobs(cfg: dict):
    base = cfg["baseline"]
    for rep in range(base["repeats"]):
        yield base["cli_name"], base["log_name"], rep, True
    for m in cfg["methods"]:
        for rep in range(cfg.get("method_repeats", 1)):
            yield m["cli_name"], m["log_name"], rep, False


def build_command(cfg: dict, cli_name: str, task: str, out_dir: Path, is_baseline: bool, smoke):
    llm = cfg["llm"]
    cmd = [
        cfg["harness"]["python"], "main.py",
        "--method", cli_name,
        "--tasks", task,
        "--generation-model", f"openrouter/{llm['served_model']}",
        "--temperature", str(llm["temperature"]),
        "--max-new-tokens", str(llm["max_new_tokens"]),
        "--top-k", str(llm["top_k"]),
        "--embedding-model", llm["embedding_model"],
        "--output-dir", str(out_dir),
        "--timing",
    ]
    if task in DATA_OVERRIDES:
        cmd += ["--task-data-override", f"{task}={DATA_OVERRIDES[task]}"]
    if is_baseline:
        cmd += ["--max-parallel-samples", str(cfg["baseline"].get("max_parallel_samples", 1))]
    if smoke:
        cmd += ["--run-first-k", str(smoke)]
    return cmd


def build_env(cfg: dict) -> dict:
    llm = cfg["llm"]
    env = dict(os.environ)
    env.update(
        OPENROUTER_API_KEY=str(llm["api_key"]),
        OPENROUTER_BASE_URL=llm["base_url"],
        OPENROUTER_ENABLE_THINKING="on" if llm.get("enable_thinking") else "off",
        PYTHONUNBUFFERED="1",
    )
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/e0_pilot.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", type=int, default=None, help="Run only the first K samples per job.")
    parser.add_argument("--only-task", action="append", default=None)
    parser.add_argument("--only-method", action="append", default=None, help="Match cli_name or log_name.")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    seqmem_root = resolve(cfg["harness"]["seqmem_root"])
    if not (seqmem_root / "main.py").exists():
        print(f"SeqMem-Eval not found at {seqmem_root}; run `git submodule update --init`.", file=sys.stderr)
        return 1
    runs_root = resolve(cfg["runs_dir"] + ("_smoke" if args.smoke else ""))
    tag = model_tag(cfg["llm"]["served_model"])
    env = build_env(cfg)
    tasks = [t for t in cfg["tasks"] if not args.only_task or t in args.only_task]

    failures = 0
    for cli_name, log_name, rep, is_baseline in build_jobs(cfg):
        if args.only_method and cli_name not in args.only_method and log_name not in args.only_method:
            continue
        out_dir = run_dir(runs_root, tag, log_name, rep)
        for task in tasks:
            done = out_dir / f"{task}.done"
            if done.exists():
                print(f"[done] {log_name} rep{rep} {task}")
                continue
            cmd = build_command(cfg, cli_name, task, out_dir, is_baseline, args.smoke)
            print(f"[run ] {log_name} rep{rep} {task}\n       {' '.join(cmd)}")
            if args.dry_run:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / f"{task}.stdout.log").open("w") as log:
                ret = subprocess.run(cmd, cwd=seqmem_root, env=env, stdout=log, stderr=subprocess.STDOUT)
            if ret.returncode == 0:
                done.touch()
            else:
                failures += 1
                print(f"[fail] {log_name} rep{rep} {task} (exit {ret.returncode}); see {log.name}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
