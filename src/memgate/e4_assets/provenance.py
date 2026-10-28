"""Hash the code actually loaded, including the inherited E2/E3 validation code."""
import hashlib
import pkgutil
import sys

from memgate.experiments.common import digest, require, versions
from memgate.replay.provenance import MODULE_FILES as REPLAY_FILES
from . import VERSION

MODULE_FILES = tuple(dict.fromkeys((
    *REPLAY_FILES, "experiments/__init__.py", "experiments/source.py",
    "e4_assets/__init__.py", "e4_assets/assets.py", "e4_assets/cli.py",
    "e4_assets/provenance.py",
)))


def implementation_identity():
    files = {}
    for name in MODULE_FILES:
        content = pkgutil.get_data("memgate", name)
        require(content is not None, f"Missing implementation resource: {name}")
        files[name] = hashlib.sha256(content).hexdigest()
    return dict(version=VERSION, files=files, source_sha256=digest(files),
                python=sys.version, packages=versions(["numpy"]))
