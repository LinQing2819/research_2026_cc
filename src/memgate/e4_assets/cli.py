"""No generation commands: export original evidence, inspect it, or identify code."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sqlite3
import sys
import zlib

from memgate.experiments.common import Invalid
from . import VERSION
from .assets import export_assets, inspect_assets
from .provenance import implementation_identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"{VERSION}: read-only assets, NOT E4 qualification")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="asset version and implementation hashes")
    export = commands.add_parser("export", help="verify original E2 ledger and export sealed assets")
    export.add_argument("--repo", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="offline bundle validation and qualification gaps")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--output", type=Path, help="optional new sealed JSON summary")
    args = parser.parse_args(argv)
    try:
        if args.command == "version":
            result = implementation_identity()
        elif args.command == "export":
            with redirect_stdout(sys.stderr):
                result = export_assets(args.repo, args.output)
        else:
            result = inspect_assets(args.input, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (Invalid, OSError, ValueError, KeyError, TypeError, IndexError, AttributeError,
            EOFError, sqlite3.Error, zlib.error) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


def entrypoint():
    raise SystemExit(main())
