"""Build a single-file exporter so the server's completed E2 checkout need not change."""
import zipapp
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
output = ROOT / ".cache/releases/e3-r1-tools.pyz"
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=ROOT / ".cache", prefix="e3-zip-") as temp:
    temp = Path(temp)
    for folder in ("replay", "experiments"):
        (temp / "memgate" / folder).mkdir(parents=True)
    names = ["__init__.py"] + [f"replay/{p.name}" for p in (ROOT / "src/memgate/replay").glob("*.py")]
    names += [f"experiments/{n}" for n in ("__init__.py", "common.py", "ledger.py", "official.py", "inference.py")]
    for name in names:
        shutil.copyfile(ROOT / "src/memgate" / name, temp / "memgate" / name)
    zipapp.create_archive(temp, output, main="memgate.replay.cli:main", compressed=True)
print(output)
