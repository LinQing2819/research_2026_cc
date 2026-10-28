"""Standalone source entry for read-only E4 asset tooling."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from memgate.e4_assets.cli import entrypoint

if __name__ == "__main__":
    entrypoint()
