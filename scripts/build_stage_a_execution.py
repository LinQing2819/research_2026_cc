#!/usr/bin/env python3
"""Build a portable, separately authorized Stage A tool without model calls."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_stage_a import OfflineTokenizer
from memgate.experiments.common import canonical, digest, file_hash, require, seal, unseal
from memgate.experiments.prompts import sentinel_prompts
from memgate.stage_a import (
    CAPS, PREPARATION_FILE, PREPARATION_DIGEST, SLOT, requests, validate_package,
)

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    output = Path(output).resolve()
    require(output.parent.is_dir() and not output.exists(), "Use a new output directory")
    prep_path = ROOT / ".cache/stage-a-curated-v1/preparation-v2.json.gz"
    require(file_hash(prep_path) == PREPARATION_FILE, "Preparation file changed")
    prep = unseal(json.loads(gzip.decompress(prep_path.read_bytes())))
    require(digest(prep) == PREPARATION_DIGEST, "Preparation value changed")
    assets_path = ROOT / ".cache/e4-assets-v1/assets.json.gz"
    assets = unseal(json.loads(gzip.decompress(assets_path.read_bytes())))
    require(digest(assets) == prep["library_config"]["bundle_sha256"], "Scoring source changed")
    fingerprint = assets["e2_table"]["manifest"]["fingerprint"]
    # Verify frozen local code before building an executable wrapper.
    for name, expected in prep["provenance"]["code"].items():
        require(file_hash(ROOT / name) == expected, f"Frozen preparation source changed: {name}")
    tokenizer = OfflineTokenizer(ROOT / ".cache/stage-a-tokenizer", fingerprint["model"])
    sources = [
        "src/memgate/__init__.py", "src/memgate/stage_a.py",
        "src/memgate/experiments/__init__.py", "src/memgate/experiments/common.py",
        "src/memgate/experiments/ledger.py", "src/memgate/experiments/official.py",
    ]
    runtime = {name.removeprefix("src/"): (ROOT / name).read_bytes() for name in sources}
    runtime["__main__.py"] = b"from memgate.stage_a import main\nmain()\n"
    # Gold is read only after the candidate and request identities are immutable.
    all_gold = unseal(json.loads(assets["artifacts_utf8"]["scoring_gold.json"]))
    gold = {p["qid"]: all_gold[p["qid"]] for p in prep["pairs"]}
    data = dict(
        schema="V3_STAGE_A_EXECUTION_1", slot=SLOT, preparation=prep,
        authorization=dict(
            date="2026-11-04",
            user_instruction="可以可以，这种就不必找我批准了，你自动继续执行，尽可能一次会话多做任务，到训练阶段或者做不下去了再喊我",
            scope="9 frozen pairs x 4 arms x 2 repeats; 8 shared probes/retries; no training or expansion"),
        limits=CAPS, shared_probe_retry_cap=8,
        scoring_gold=gold, scoring_source_sha256=hashlib.sha256(
            assets["artifacts_utf8"]["scoring_gold.json"].encode()).hexdigest(),
        sentinels=[dict(s, sha256=digest(s["ids"])) for s in sentinel_prompts(tokenizer)],
        model_files=fingerprint["model"],
        official_files=fingerprint["official"]["files"],
        launcher_sha256=fingerprint["code"]["scripts/e2/server.py"],
        runtime_code={k: hashlib.sha256(v).hexdigest() for k, v in runtime.items()},
        builder_sha256=file_hash(__file__),
    )
    require(file_hash(ROOT / "scripts/e2/server.py") == data["launcher_sha256"],
            "Server launcher differs from historical model runtime")
    data["allowed_requests"] = {k: digest(v["request"]) for k, v in requests(data).items()}
    validate_package(data)
    encoded = gzip.compress(canonical(seal(data)), mtime=0)
    encoded_hash = hashlib.sha256(encoded).hexdigest()
    runtime["stage_a_authorization.py"] = f'INPUT_SHA256 = "{encoded_hash}"\n'.encode()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, content in sorted(runtime.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zipped.writestr(info, content)
    output.mkdir()
    (output / "stage-a.pyz").write_bytes(archive.getvalue())
    (output / "execution.json.gz").write_bytes(encoded)
    shutil.copyfile(prep_path, output / "preparation-v2.json.gz")
    for name, expected in data["official_files"].items():
        source = ROOT / "third_party/SeqMem-Eval" / name
        require(file_hash(source) == expected, f"Official file changed: {name}")
        target = output / "official" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(ROOT / "scripts/stage_a_launch.sh", output / "launch.sh")
    hashes = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "SHA256SUMS").write_text(
        "".join(f"{h}  {name}\n" for name, h in hashes.items()), encoding="utf-8")
    tar_path = output.with_suffix(".tar.gz")
    require(not tar_path.exists(), "Transfer archive already exists")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    print(json.dumps(dict(execution=digest(data), input_sha256=encoded_hash,
                          archive=str(tar_path), archive_sha256=file_hash(tar_path),
                          bytes=tar_path.stat().st_size, formal=72, attempts_cap=80,
                          model_calls=0), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
