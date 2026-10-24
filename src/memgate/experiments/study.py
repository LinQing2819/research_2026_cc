"""Orchestration for the single pre-registered V4 frozen discovery."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from .common import (ARMS, LIMITS, PARAMETERS, ROOT, Invalid, atomic_json, digest,
                     file_hash, logical_key, read_json, require, seal, stable_seed,
                     tree_hashes, unseal, versions)
from .inference import (analyze_scores, authorized_projection, select,
                        selector_authorizations, sensitivity)
from .ledger import Ledger, Transport, payload
from .official import Official
from .prompts import checked_vectors, prepare_question, sentinel_prompts
from .source import (build_snapshots, duplicate_audit, load_public,
                     split_public, validate_source)


def configuration(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {"source", "data", "seqmem_root", "model_path", "embedding_path", "base_url"}
    require(set(cfg) == expected, "Config only accepts frozen resource paths and base_url")
    return {k: (ROOT / v).resolve() if k != "base_url" else v for k, v in cfg.items()}


def input_fingerprint(cfg):
    files = list((ROOT / "src/memgate/experiments").glob("*.py"))
    files += list((ROOT / "scripts/e2").glob("*.py")) + list((ROOT / "scripts/e2").glob("*.sh"))
    files += [ROOT / "docs/component_replay_protocol.md"]
    official_names = ["methods/expel_online_st.py", "tasks/task_mmlu_pro.py",
                      "tasks/math_answer_utils.py", "prompts/prompt_templates.json",
                      "utils/embedder.py"]
    code = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}
    official = {s: file_hash(cfg["seqmem_root"] / s) for s in official_names}
    commit = subprocess.check_output(["git", "-C", str(cfg["seqmem_root"]), "rev-parse", "HEAD"],
                                     text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(cfg["seqmem_root"]), "status", "--porcelain"],
                                    text=True).strip()
    print("Verifying source, code and local model content hashes (no model calls)...", flush=True)
    return dict(parameters=PARAMETERS, source=file_hash(cfg["source"]), data=file_hash(cfg["data"]),
                code=code, official=dict(files=official, commit=commit, dirty=dirty),
                model=tree_hashes(cfg["model_path"]), embedding=tree_hashes(cfg["embedding_path"]),
                client_versions=versions(["transformers", "tokenizers", "sentence-transformers",
                                          "torch", "numpy", "pandas", "pyarrow"]))


def write_artifact(directory, name, value):
    atomic_json(directory / name, seal(value))
    return file_hash(directory / name)


def load_artifact(directory, name):
    return unseal(read_json(directory / name))


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    require(tokenizer.is_fast, "Exact token spans require the fast tokenizer")
    return tokenizer


def make_embeddings(path, snapshots, split, official):
    from sentence_transformers import SentenceTransformer
    import torch
    # Explicit CPU, fp32; no query prompt/instruction is added.
    embedder = SentenceTransformer(str(path), device="cpu", local_files_only=True,
                                   model_kwargs={"torch_dtype": torch.float32})
    texts = [r["question"] for r in snapshots["new"]["pool"]]
    public = split["dev"] + split["test"]
    queries = [official.query(r) for r in public]
    require(texts, "No successful experiences in the new prefix")
    started = time.perf_counter()
    def encode_all(items):
        # Match historical one-text _encode calls to avoid batching-dependent drift.
        return checked_vectors(
            [embedder.encode([text], normalize_embeddings=True, show_progress_bar=False,
                             prompt="")[0] for text in items], len(items))
    vectors, qvectors = encode_all(texts), encode_all(queries)
    return dict(pool=vectors.tolist(), queries=dict(zip((r["qid"] for r in public), qvectors.tolist())),
                texts_sha256=digest(texts), queries_sha256=digest(queries),
                dtype="float32", device="cpu", normalize=True, prompt="",
                max_seq_length=embedder.max_seq_length,
                dimension=vectors.shape[1], seconds=time.perf_counter()-started)


def prepare(cfg, directory, fingerprint, ledger):
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        return verify_manifest(directory, fingerprint, ledger)
    require(ledger.counts()["attempts"] == 0, "Cannot prepare after generation without a manifest")
    official = Official(cfg["seqmem_root"])
    public, gold = load_public(cfg["data"])
    old, new, splits = split_public(public)
    prefix, raw_prefix, source_hash = validate_source(cfg["source"], public, new)
    require(source_hash == fingerprint["source"], "Source changed during prepare")
    snapshots = build_snapshots(prefix, old, new)
    hashes = {}
    artifacts = dict(public=public, scoring_gold=gold, prefix_raw=raw_prefix,
                     old_prefix_raw="".join(raw_prefix.splitlines(keepends=True)[:old]),
                     snapshots=snapshots, splits=splits)
    for name, value in artifacts.items():
        hashes[name + ".json"] = write_artifact(directory, name + ".json", value)
    audit = duplicate_audit(public[:new], splits)
    hashes["duplicate_audit.json"] = write_artifact(directory, "duplicate_audit.json", audit)
    require(audit["qualified"], f"Cross-boundary duplicate candidates require review: {directory / 'duplicate_audit.json'}")
    tokenizer = load_tokenizer(cfg["model_path"])
    cache = directory / "embeddings.json"
    if cache.exists():
        embeddings = load_artifact(directory, "embeddings.json")
    else:
        print("Encoding frozen historical questions and public queries on CPU...", flush=True)
        embeddings = make_embeddings(cfg["embedding_path"], snapshots, splits, official)
        write_artifact(directory, "embeddings.json", embeddings)
    require(embeddings["texts_sha256"] == digest([r["question"] for r in snapshots["new"]["pool"]]),
            "Embedding cache source mismatch")
    require(embeddings["queries_sha256"] == digest([official.query(r) for r in splits["dev"]+splits["test"]]),
            "Embedding cache query mismatch")
    hashes["embeddings.json"] = file_hash(cache)
    vectors = {"new": embeddings["pool"],
               "old": embeddings["pool"][:len(snapshots["old"]["pool"])]}
    questions, eligible = {}, {}
    for phase, items in splits.items():
        eligible[phase] = []
        for row in items:
            name = f"questions/{row['qid']}.json"
            # Public numeric MMLU IDs; never allow path-like IDs.
            require(row["qid"].isdigit(), "Unexpected non-numeric public qid")
            if (directory / name).exists():
                question = load_artifact(directory, name)
            else:
                question = prepare_question(official, tokenizer, row, snapshots, vectors,
                                            embeddings["queries"][row["qid"]])
                write_artifact(directory, name, question)
            hashes[name] = file_hash(directory / name)
            questions[row["qid"]] = question
            if question["eligible"]:
                eligible[phase].append(row["qid"])
        print(f"{phase}: eligible {len(eligible[phase])}/{len(items)}", flush=True)
    authorization = selector_authorizations(eligible["dev"], questions)
    changes = [questions[q]["change"] for q in eligible["test"]]
    intervention = dict(eligible_test=len(changes),
                        rules_changed=sum(c["packed_rules"] for c in changes),
                        experiences_changed=sum(c["packed_experiences"] for c in changes),
                        jointly_changed=sum(c["packed_rules"] and c["packed_experiences"] for c in changes))
    hashes["intervention.json"] = write_artifact(directory, "intervention.json", intervention)
    require(intervention["jointly_changed"] > 0, "No jointly changed test intervention; inspect intervention.json")
    schedule = {}
    for phase, qids in eligible.items():
        items = [dict(key=logical_key(phase, qid, arm, rep), phase=phase, qid=qid, arm=arm, rep=rep,
                      seed=stable_seed(phase, qid, arm, rep))
                 for qid in qids for arm in ARMS for rep in range(2)]
        schedule[phase] = sorted(items, key=lambda r: digest(["e2-v4-schedule", r["key"]]))
    seeds = [r["seed"] for rows in schedule.values() for r in rows]
    require(len(set(seeds)) == len(seeds), "Decode seed collision")
    sentinels = sentinel_prompts(tokenizer)
    hashes["sentinels.json"] = write_artifact(directory, "sentinels.json", sentinels)
    manifest = dict(identity=digest(fingerprint), fingerprint=fingerprint, files=hashes,
                    boundaries=dict(total=len(public), old=old, new=new),
                    eligible=eligible, original_counts={"dev": 64, "test": 96},
                    authorization=authorization, schedule=schedule)
    write_artifact(directory, "manifest.json", manifest)
    ledger.bind("manifest", digest(manifest))
    return manifest


def verify_manifest(directory, fingerprint, ledger):
    require((directory / "manifest.json").exists(), "Missing manifest; run prepare first")
    manifest = load_artifact(directory, "manifest.json")
    require(manifest["fingerprint"] == fingerprint, "Source/code/model/runtime changed; cannot reuse this run")
    ledger.bind("manifest", digest(manifest))
    for name, expected in manifest["files"].items():
        require(file_hash(directory / name) == expected, f"Prepared artifact changed: {name}")
        load_artifact(directory, name)
    return manifest


def check_service(identity, manifest):
    require(identity["actual_max_model_len"] >= LIMITS["context"], "Server max_model_len is below 8192")
    require(identity["served_models"] == [PARAMETERS["model"]], "Unexpected served model mapping")
    require(identity["model_files"] == manifest["fingerprint"]["model"], "Server model/tokenizer files differ")
    require(identity["dtype"] == "torch.bfloat16" and identity["tensor_parallel_size"] == 1 and
            identity["max_num_seqs"] == 1 and identity["generation_config"] == "vllm",
            "Server launch settings differ from the single-GPU protocol")
    require(identity["enforce_eager"] is True and identity["max_num_batched_tokens"] == 8192
            and len(identity["gpu"]) == 1, "Server must use eager, one visible GPU and batch cap 8192")
    require(identity["launcher_sha256"] == manifest["fingerprint"]["code"]["scripts/e2/server.py"],
            "Service launcher code differs from frozen code")


def service_evidence(transport, manifest, directory):
    identity = transport.json("/e2/identity")
    check_service(identity, manifest)
    models = transport.json("/v1/models")
    require([m["id"] for m in models["data"]] == [PARAMETERS["model"]], "API model listing differs")
    evidence = []
    for sent in load_artifact(directory, "sentinels.json"):
        server = transport.json("/e2/chat_ids", {"messages": sent["messages"]})
        require(server["ids"] == sent["ids"] and server["text"] == sent["text"],
                "Client/server complete chat template IDs differ")
        tokenized = transport.json("/tokenize", dict(model=PARAMETERS["model"], prompt=sent["text"],
                                                     add_special_tokens=False))
        require(tokenized.get("tokens") == sent["ids"], "Backend /tokenize IDs differ or are unavailable")
        require(tokenized.get("max_model_len") == identity["actual_max_model_len"],
                "Backend /tokenize actual context differs")
        evidence.append(dict(server=server, tokenized=tokenized))
    return dict(identity=identity, tokenization=evidence)


def preflight(transport, manifest, directory, ledger):
    evidence = service_evidence(transport, manifest, directory)
    ledger.bind("service", evidence)
    for index, sent in enumerate(load_artifact(directory, "sentinels.json")):
        print(f"Preflight probe {index+1}/2 (persisted results are reused)...", flush=True)
        ledger.complete(f"probe:{index}", payload(sent["ids"], 2000000000+index), transport,
                        lambda text: {}, binding=dict(manifest=digest(manifest), service=digest(evidence)),
                        probe=True)
    record = dict(manifest=digest(manifest), service=evidence,
                  probes=[digest(ledger.result(f"probe:{i}")) for i in range(2)])
    write_artifact(directory, "preflight.json", record)
    ledger.bind("preflight", digest(record))
    return record


def accepted_preflight(transport, manifest, directory, ledger, check_live=True):
    require((directory / "preflight.json").exists(), "Run preflight before dev generation")
    record = load_artifact(directory, "preflight.json")
    require(ledger.meta("preflight") == digest(record) and record["manifest"] == digest(manifest),
            "Preflight binding changed")
    if check_live:
        evidence = service_evidence(transport, manifest, directory)
        require(evidence == record["service"], "Service identity/tokenization changed after preflight")
    for index, sent in enumerate(load_artifact(directory, "sentinels.json")):
        row = ledger.verify_result(f"probe:{index}", payload(sent["ids"], 2000000000+index),
                                   lambda text: {}, dict(manifest=digest(manifest),
                                                        service=digest(record["service"])))
        require(digest(row) == record["probes"][index], "Probe evidence changed")
    return record


def score_function(official, gold_letter):
    def score(text):
        predicted = official.extract_letter(text)
        return {"predicted": predicted, "score": int(bool(predicted) and predicted == gold_letter)}
    return score


def request_context(item, manifest, directory, ledger, official, gold, selection_hash=None):
    question = load_artifact(directory, f"questions/{item['qid']}.json")
    request = payload(question["prompts"][item["arm"]]["ids"], item["seed"])
    binding = dict(manifest=digest(manifest), preflight=ledger.meta("preflight"),
                   phase=item["phase"], qid=item["qid"], arm=item["arm"], rep=item["rep"],
                   seed=item["seed"], selection=selection_hash)
    return request, binding, score_function(official, gold[item["qid"]])


def collect(phase, manifest, directory, ledger, official, gold, selection_hash=None):
    result = []
    for item in manifest["schedule"][phase]:
        request, binding, scorer = request_context(item, manifest, directory, ledger, official, gold,
                                                   selection_hash)
        value = ledger.verify_result(item["key"], request, scorer, binding)
        result.append(dict(qid=item["qid"], arm=item["arm"], rep=item["rep"], score=value["score"],
                           result_sha256=digest(value), usage=value["usage"],
                           finish_reason=value["finish_reason"]))
    return result


def seal_selection(manifest, directory, ledger, official, gold):
    dev = collect("dev", manifest, directory, ledger, official, gold)
    selections = {}
    for name, authorization in manifest["authorization"].items():
        projection = authorized_projection(authorization, dev)
        selections[name] = select(authorization, projection)
    selection = dict(manifest=digest(manifest), preflight=ledger.meta("preflight"),
                     dev_sha256=digest(dev), selectors=selections, tie_order=["NN", "OO", "ON", "NO"])
    write_artifact(directory, "selection.json", selection)
    ledger.bind("selection", digest(selection))
    return selection


def verified_selection(manifest, directory, ledger, official, gold):
    require((directory / "selection.json").exists(), "Test forbidden until complete dev and sealed selection")
    stored = load_artifact(directory, "selection.json")
    require(ledger.meta("selection") == digest(stored), "Selection missing from ledger or modified")
    expected = seal_selection(manifest, directory, ledger, official, gold)
    require(stored == expected, "Selection/dev dependencies changed")
    return stored


def run_phase(phase, manifest, directory, ledger, transport, official, gold, smoke=False):
    accepted_preflight(transport, manifest, directory, ledger)
    if phase == "dev" and not smoke:
        require((directory / "smoke.json").exists(),
                "Run and accept the frozen dev smoke before full dev generation")
        smoke_record = load_artifact(directory, "smoke.json")
        smoke_qids = manifest["eligible"]["dev"][:4]
        smoke_items = [item for item in manifest["schedule"]["dev"] if item["qid"] in set(smoke_qids)]
        require(smoke_record["manifest"] == digest(manifest)
                and smoke_record["preflight"] == ledger.meta("preflight")
                and smoke_record["qids"] == smoke_qids
                and smoke_record["rows"] == len(smoke_items),
                "Smoke evidence is incomplete or bound to a different protocol state")
        expected_results = [digest(ledger.result(item["key"])) for item in smoke_items]
        require(all(ledger.result(item["key"]) is not None for item in smoke_items)
                and smoke_record["results"] == expected_results,
                "Smoke results are missing or changed")
    selection_hash = None
    if phase == "test":
        selection_hash = digest(verified_selection(manifest, directory, ledger, official, gold))
    items = manifest["schedule"][phase]
    if smoke:
        allowed = set(manifest["eligible"]["dev"][:4])
        items = [r for r in items if r["qid"] in allowed]
    for index, item in enumerate(items, 1):
        request, binding, scorer = request_context(item, manifest, directory, ledger, official, gold,
                                                   selection_hash)
        if ledger.result(item["key"]) is None:
            print(f"{phase}{' smoke' if smoke else ''} {index}/{len(items)}: "
                  f"{item['qid']} {item['arm']} rep{item['rep']} "
                  f"input={len(request['prompt'])} reserved_output=2048", flush=True)
        ledger.complete(item["key"], request, transport, scorer, binding)
    if phase == "dev" and not smoke:
        seal_selection(manifest, directory, ledger, official, gold)
    if smoke:
        values = [ledger.result(item["key"]) for item in items]
        technical = dict(rows=len(values), qids=manifest["eligible"]["dev"][:4],
                         largest_input=max(v["usage"]["prompt_tokens"] for v in values),
                         length_finishes=sum(v["finish_reason"] == "length" for v in values),
                         attempts=ledger.counts(), manifest=digest(manifest),
                         preflight=ledger.meta("preflight"),
                         results=[digest(v) for v in values])
        # Counts can grow on later runs; the first smoke evidence remains immutable.
        if not (directory / "smoke.json").exists():
            write_artifact(directory, "smoke.json", technical)
        print(json.dumps(technical, ensure_ascii=False), flush=True)


def report(manifest, directory, ledger, transport, official, gold):
    accepted_preflight(transport, manifest, directory, ledger, check_live=False)
    selection = verified_selection(manifest, directory, ledger, official, gold)
    shash = digest(selection)
    dev = collect("dev", manifest, directory, ledger, official, gold)
    test = collect("test", manifest, directory, ledger, official, gold, shash)
    expected = {r["key"] for items in manifest["schedule"].values() for r in items} | {"probe:0", "probe:1"}
    require(set(ledger.results()) == expected, "Unexpected/foreign accepted results in ledger")
    analysis = analyze_scores(manifest["eligible"]["test"], test,
                              selection["selectors"]["S"]["arm"], selection["selectors"]["F"]["arm"])
    questions = {q: load_artifact(directory, f"questions/{q}.json")
                 for q in manifest["eligible"]["dev"] + manifest["eligible"]["test"]}
    packing_summary = {}
    for phase, qids in manifest["eligible"].items():
        subset = [questions[q] for q in qids]
        packing_summary[phase] = {
            "change_rates": {k: sum(q["change"][k] for q in subset) / len(subset)
                             for k in ("source_rules", "packed_rules", "retrieved_experiences",
                                       "packed_experiences")},
            "joint_packed_change_rate": sum(q["change"]["packed_rules"] and
                                           q["change"]["packed_experiences"] for q in subset) / len(subset),
            "identical_arm_pair_fraction": sum(
                q["prompts"][a]["ids"] == q["prompts"][b]["ids"]
                for q in subset for i, a in enumerate(ARMS) for b in ARMS[i+1:]) / (21*len(subset)),
            "mean_discard_fraction": {component: sum(
                (q[component][v]["total"]-q[component][v]["kept"]) / max(1, q[component][v]["total"])
                for q in subset for v in ("old", "new")) / (2*len(subset))
                for component in ("rules", "cases")},
            "max_input_tokens": max(q["prompts"][a]["prompt_tokens"] for q in subset for a in ARMS),
            "endpoint_unchanged_from_official_fraction": sum(
                q["prompts"][a]["unchanged_from_official_unpacked"]
                for q in subset for a in ("OO", "NN")) / (2*len(subset)),
        }
    mean_reserved = sum(2*(questions[q]["prompts"][a]["prompt_tokens"]+2048)
                        for q in manifest["eligible"]["test"] for a in ARMS) / analysis["n"]
    analysis.update(manifest=digest(manifest), selection=selection,
                    selection_sha256=shash, sensitivity=sensitivity(analysis, mean_reserved),
                    sensitivity_note="Marginal 80% power scenarios, not joint GO power or an information-theoretic lower bound; observed normal scenarios are exploratory",
                    coverage={p: dict(eligible=len(manifest["eligible"][p]), original=n)
                              for p, n in manifest["original_counts"].items()},
                    attempt_counts=ledger.counts(), result_sha256=digest(dev+test),
                    fixed_policy_selection_cost=0,
                    actual_accepted_usage={k: sum(r["usage"][k] for r in dev+test)
                                           for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                    length_finishes=sum(r["finish_reason"] == "length" for r in dev+test),
                    intervention=load_artifact(directory, "intervention.json"),
                    packing_summary=packing_summary,
                    exclusions=[load_artifact(directory, name)["exclusion"]
                                | {"qid": load_artifact(directory, name)["public"]["qid"]}
                                for name in manifest["files"] if name.startswith("questions/")
                                and not load_artifact(directory, name)["eligible"]],
                    component_audit={qid: dict(change=q["change"], rules={v: {
                        k: q["rules"][v][k] for k in ("kept", "total", "tokens")} for v in ("old", "new")},
                        experiences={v: {k: q["cases"][v][k] for k in ("ids", "kept", "total", "tokens")}
                                     for v in ("old", "new")}, timings=q["timings"])
                                     for qid, q in questions.items()})
    # Account for every persisted response, including probes/retries. Unknown usage stays unknown.
    raw_usage, unknown = [], 0
    for row in ledger.db.execute("SELECT raw FROM attempts"):
        raw = json.loads(row["raw"]) if row["raw"] else {}
        try:
            usage = json.loads(raw["body"])["usage"]
            raw_usage.append(usage)
        except (KeyError, TypeError, json.JSONDecodeError):
            unknown += 1
    analysis["all_attempt_usage"] = dict(
        known={k: sum(u[k] for u in raw_usage) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        attempts_without_known_usage=unknown,
        reserved_output_token_upper_bound=ledger.counts()["attempts"] * 2048)
    write_artifact(directory, "analysis.json", analysis)
    lines = [
        "# E2 V4 frozen discovery", "",
        f"Mechanism: **{analysis['mechanism_status']}**, eligible test n={analysis['n']}.",
        f"S={selection['selectors']['S']['arm']}; F={selection['selectors']['F']['arm']}.",
        "Conditional on this history, fixed question set, packing and frozen dev selection.",
        "This does not establish new-query, multi-history, online recovery or general selector performance.", "",
        "| Contrast | Estimate | Simultaneous 95% CI | Status |",
        "|---|---:|---:|---|",
    ]
    for name, c in analysis["contrasts"].items():
        lines.append(f"| {name} | {c['estimate']:.4f} | [{c['ci'][0]:.4f}, {c['ci'][1]:.4f}] | "
                     f"{c['status'] if name != 'I' else 'additive interaction only'} |")
    lines += ["", "Full variance, coverage, packing, cost and confirmation scenarios: analysis.json.",
              "An INCONCLUSIVE result does not authorize adding repetitions or changing this run."]
    # Derived human-readable report; canonical evidence is the sealed analysis JSON.
    (directory / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Mechanism: {analysis['mechanism_status']}; report: {directory / 'report.md'}", flush=True)
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description="E2 V4 frozen component replay (no curator or online update)")
    parser.add_argument("command", choices=["prepare", "preflight", "smoke", "dev", "test", "run", "analyze"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e2_component.yaml")
    args = parser.parse_args(argv)
    # Prevent accidental embedding GPU use even when CUDA is available in the client environment.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        cfg = configuration(args.config)
        fingerprint = input_fingerprint(cfg)
        identity = digest(fingerprint)
        directory = ROOT / "runs/e2_component" / identity
        # One study slot. Changing code after any submitted call cannot silently start a new study.
        with Ledger(ROOT / "runs/.e2-ledgers/component-v4.sqlite", identity) as ledger:
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(ROOT / "runs/e2_component/active.json", seal({"identity": identity}), immutable=False)
            print(f"E2 artifacts: {directory}", flush=True)
            if args.command == "prepare":
                manifest = prepare(cfg, directory, fingerprint, ledger)
                print(f"Prepared and sealed {len(manifest['schedule']['dev']) + len(manifest['schedule']['test'])} logical calls.", flush=True)
                return 0
            manifest = verify_manifest(directory, fingerprint, ledger)
            transport = Transport(cfg["base_url"])
            if args.command == "preflight":
                preflight(transport, manifest, directory, ledger)
                return 0
            official = Official(cfg["seqmem_root"])
            gold = load_artifact(directory, "scoring_gold.json")
            if args.command in ("smoke", "dev", "run"):
                run_phase("dev", manifest, directory, ledger, transport, official, gold,
                          smoke=args.command == "smoke")
            if args.command in ("test", "run"):
                run_phase("test", manifest, directory, ledger, transport, official, gold)
            if args.command in ("analyze", "run"):
                report(manifest, directory, ledger, transport, official, gold)
            return 0
    except (Invalid, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
