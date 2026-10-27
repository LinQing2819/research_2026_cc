"""Source identity works both in the checkout and in the single-file zipapp."""
import hashlib
import pkgutil
import sys

from memgate.experiments.common import digest, require, versions
from . import PROTOCOL

MODULE_FILES = (
    "__init__.py",
    "experiments/common.py", "experiments/official.py",
    "experiments/ledger.py", "experiments/inference.py",
    "replay/__init__.py", "replay/export.py", "replay/selection.py",
    "replay/cli.py", "replay/provenance.py",
)


def implementation_identity():
    files = {}
    for name in MODULE_FILES:
        content = pkgutil.get_data("memgate", name)
        require(content is not None, f"Missing implementation resource: {name}")
        files[name] = hashlib.sha256(content).hexdigest()
    return dict(protocol=PROTOCOL, files=files, source_sha256=digest(files),
                python=sys.version, packages=versions(["numpy"]))
