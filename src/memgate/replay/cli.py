"""Model-free E3 command line. Export may run from a standalone zipapp."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from memgate.experiments.common import Invalid, atomic_json, digest, seal
from memgate.experiments.official import Official
from . import PROTOCOL
from .export import export_table, read_export
from .provenance import implementation_identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"{PROTOCOL}: verified E2 export and offline replay")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="protocol, implementation hashes and runtime versions")
    export = sub.add_parser("export", help="read-only original E2 ledger export, no model calls")
    export.add_argument("--repo", type=Path, default=Path.cwd())
    export.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="offline six-method replay, no model calls")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--repo", type=Path, default=Path.cwd())
    run.add_argument("--output-dir", type=Path, default=Path("reports/e3_replay"))
    args = parser.parse_args(argv)
    try:
        if args.command == "version":
            print(json.dumps(implementation_identity(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "export":
            export_table(args.repo, args.output)
            return 0
        from .selection import run_replay
        table = read_export(args.input)
        from memgate.experiments.common import file_hash, require
        official_root = args.repo / "third_party/SeqMem-Eval"
        for name, expected in table["manifest"]["fingerprint"]["official"]["files"].items():
            require(file_hash(official_root / name) == expected, f"Official source changed: {name}")
        result = run_replay(table, Official(official_root))
        implementation = implementation_identity()
        result["implementation"] = implementation
        result["implementation_sha256"] = implementation["source_sha256"]
        directory = args.output_dir / digest([PROTOCOL, result["input_sha256"], implementation])
        atomic_json(directory / "replay.json", seal(result))
        lines = [
            "# E3-R1 retrospective selection replay", "",
            f"Status: **{result['status']}**", "",
            "No new model calls. These 16 orders reuse one table and are not independent histories.",
            "This does not change the E2 INCONCLUSIVE conclusion or certify deployment.", "",
            "| Method | Mean test accuracy | Mean reserved tokens | Selected counts |",
            "|---|---:|---:|---|",
        ]
        for name, s in result["summary"].items():
            lines.append(f"| {name} | {100*s['mean_test_accuracy']:.2f}% | "
                         f"{s['mean_reserved_tokens']:.0f} | {json.dumps(s['selected_counts'])} |")
        lines += ["", "Fixed policies (zero selection cost):", ""]
        lines += [f"- {a}: {100*v:.2f}%" for a, v in result["fixed_policy_accuracy"].items()]
        lines += ["", "All paired order differences, complete search/validation traces and costs: replay.json.",
                  "No p-value is computed from the 16 dependent orders.",
                  "An independent new-data experiment requires a separately frozen protocol."]
        (directory / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        print(f"{result['status']}; report: {directory / 'report.md'}; new model calls=0")
        return 0
    except (Invalid, OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 2
