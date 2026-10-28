"""Package validation code without touching the server's frozen E2 checkout."""
import argparse
from io import BytesIO
from pathlib import Path
import shutil
import sys
import tempfile
import zipapp

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from memgate.e4_assets.assets import publish_bytes
from memgate.e4_assets import VERSION
from memgate.e4_assets.provenance import MODULE_FILES
from memgate.experiments.common import Invalid


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a nonoverwriting E4 assets zipapp")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/releases" / f"{VERSION}.pyz")
    args = parser.parse_args(argv)
    try:
        with tempfile.TemporaryDirectory(prefix="e4-assets-build-") as temp:
            temp = Path(temp)
            for name in MODULE_FILES:
                target = temp / "memgate" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / "src/memgate" / name, target)
            archive = BytesIO()
            zipapp.create_archive(temp, archive, compressed=True,
                                  main="memgate.e4_assets.cli:entrypoint")
            publish_bytes(args.output, archive.getvalue())
        print(args.output)
        return 0
    except (Invalid, OSError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
