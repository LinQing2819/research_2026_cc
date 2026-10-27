"""E3 standalone source entry; never invokes a model service."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from memgate.replay.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
