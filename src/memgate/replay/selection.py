"""Frozen E3-R1 algorithm; all LLM outcomes come from authorized stored rows."""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache

import numpy as np

from memgate.experiments.common import ARMS, FACTORS, digest, require
from . import PROTOCOL

ORDER = ("ZZ", "ZE", "OO", "ON", "NO", "NN", "RZ")
METHODS = ("rendered", "unpacked", "stored", "additive", "independent", "uniform")
SEARCH_BUDGET, VALIDATION_BUDGET = 524288, 393216


def validate_table(table):
    splits, questions, rows = table["splits"], table["questions"], table["rows"]
    require(set(splits) == {"dev", "test"} and len(splits["dev"]) == 64
            and len(splits["test"]) == 94, "Expected fixed dev64/test94")
    all_ids = splits["dev"] + splits["test"]
    require(len(set(all_ids)) == 158 and set(questions) == set(all_ids), "Duplicate/missing public IDs")
    expected = {(p, q, a, r) for p, qs in splits.items() for q in qs for a in ARMS for r in range(2)}
    require(len(rows) == len(expected) and
            {(r["phase"], r["qid"], r["arm"], r["rep"]) for r in rows} == expected,
            "Missing/duplicate/foreign scores")
    require(all(type(r["score"]) is int and r["score"] in (0, 1) for r in rows), "Invalid scores")
    for question in questions.values():
        require(question["eligible"] and set(question["prompts"]) == set(ARMS), "Ineligible question")
        for prompt in question["prompts"].values():
            require(prompt["prompt_tokens"] == len(prompt["ids"]) <= 5888,
                    "Prompt length differs or exceeds protocol")


@lru_cache(maxsize=4096)
def feature(text):
    text = text or "<E3_EMPTY>"
    pieces = [text[i:i+3] for i in range(max(1, len(text)-2))]
    values = Counter(pieces)
    norm = math.sqrt(sum(x*x for x in values.values()))
    return {k: v/norm for k, v in values.items()}


def similarity(a, b):
    a, b = feature(a), feature(b)
    if len(a) > len(b):
        a, b = b, a
    return sum(v*b.get(k, 0) for k, v in a.items())


def texts(table, qid, arm, representation, official):
    r, e = FACTORS[arm]
    question, snapshots = table["questions"][qid], table["snapshots"]
    if representation in ("rendered", "additive"):
        return question["rules"][r]["text"], question["cases"][e]["text"]
    rules = "(empty)" if r == "empty" else snapshots[r]["rules"]
    if representation == "unpacked":
        cases = official.cases([] if e == "empty" else question["ranked"][e])
    else:
        from memgate.experiments.common import canonical
        cases = (official.cases([]) if e == "empty" else
                 canonical([{k: x[k] for k in ("question", "trajectory")}
                            for x in snapshots[e]["pool"]]).decode())
    return rules, cases


def kernel(table, qids, method, official):
    require(method in METHODS, "Unknown E3 method")
    if method in ("uniform", "independent"):
        return .25*np.eye(7)
    R, E = np.zeros((7, 7)), np.zeros((7, 7))
    for qid in qids:
        pairs = [texts(table, qid, arm, method, official) for arm in ORDER]
        for i in range(7):
            for j in range(i, 7):
                R[i, j] += similarity(pairs[i][0], pairs[j][0])
                E[i, j] += similarity(pairs[i][1], pairs[j][1])
    R = (R+R.T-np.diag(R.diagonal()))/len(qids)
    E = (E+E.T-np.diag(E.diagonal()))/len(qids)
    K = .25*(.4*R+.4*E+.2*np.eye(7)) if method == "additive" else (
        .25*(.2*R+.2*E+.4*R*E+.2*np.eye(7)))
    require(np.isfinite(K).all() and np.linalg.eigvalsh(K).min() >= -1e-9, "Invalid kernel")
    return K


class Provider:
    """Only authorized phase/qids are retained; reads are prefix-only and charged."""
    def __init__(self, rows, questions, phase, qids, budget):
        allowed = set(qids)
        self.scores = {(r["qid"], r["arm"], r["rep"]): r for r in rows
                       if r["phase"] == phase and r["qid"] in allowed}
        require(len(self.scores) == len(qids)*14, "Incomplete provider projection")
        self.questions, self.qids, self.budget = questions, list(qids), budget
        self.spent = 0
        self.count = dict.fromkeys(ORDER, 0)
        self.correct = dict.fromkeys(ORDER, 0)
        self.actual = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        self.trace = []

    def next_qids(self, arm, size):
        start = self.count[arm]
        return self.qids[start:start+size]

    def cost(self, arm, size):
        qids = self.next_qids(arm, size)
        if len(qids) != size:
            return None
        return sum(2*(self.questions[q]["prompts"][arm]["prompt_tokens"]+2048) for q in qids)

    def available(self, size=2):
        return [a for a in ORDER if self.cost(a, size) is not None
                and self.spent+self.cost(a, size) <= self.budget]

    def buy(self, arm, size, reason):
        cost = self.cost(arm, size)
        require(cost is not None and self.spent+cost <= self.budget, "Unauthorized/overbudget read")
        qids = self.next_qids(arm, size)
        wins = 0
        for q in qids:
            for rep in range(2):
                row = self.scores[(q, arm, rep)]
                wins += row["score"]
                for key in self.actual:
                    self.actual[key] += row["usage"][key]
        self.spent += cost
        self.count[arm] += size
        self.correct[arm] += wins
        self.trace.append(dict(arm=arm, qids=qids, reason=reason, correct=wins,
                               reserved_tokens=cost, cumulative_reserved=self.spent))

    def mean(self, arm):
        require(self.count[arm] > 0, "No authorized observations")
        return self.correct[arm]/(2*self.count[arm])


def posterior(K, provider):
    counts = np.array([provider.count[a]*2 for a in ORDER], dtype=float)
    require((counts > 0).all(), "All candidates must be initialized")
    y = np.array([provider.mean(a) for a in ORDER])
    system = K + np.diag(.25/counts + 1e-10)
    mean = .5 + K @ np.linalg.solve(system, y-.5)
    cov = K-K @ np.linalg.solve(system, K)
    std = np.sqrt(np.maximum(0, np.diag(cov)))
    return mean, std


def search(provider, K, method):
    for arm in ORDER:
        provider.buy(arm, 2, "initialize")
    iteration = 0
    while provider.available():
        arms = provider.available()
        iteration += 1
        if method == "uniform" or iteration % 4 == 0:
            arm = min(arms, key=lambda a: (provider.count[a], ORDER.index(a)))
            reason = "uniform" if method == "uniform" else "forced_exploration"
        else:
            mean, std = posterior(K, provider)
            floor = max(mean-std)
            plausible = [a for a in arms if mean[ORDER.index(a)]+std[ORDER.index(a)] >= floor]
            plausible = plausible or arms
            arm = max(plausible, key=lambda a: (
                std[ORDER.index(a)]**2/provider.cost(a, 2), -ORDER.index(a)))
            reason = "variance_per_reserved_token"
        provider.buy(arm, 2, reason)
    means = (np.array([provider.mean(a) for a in ORDER]) if method == "uniform"
             else posterior(K, provider)[0])
    ranked = sorted((a for a in ORDER if a != "ZZ"),
                    key=lambda a: (-means[ORDER.index(a)], ORDER.index(a)))
    return ranked[:2]+["ZZ"], {a: float(means[i]) for i, a in enumerate(ORDER)}


def select_one(table, method, seed, K):
    S, V = table["splits"]["dev"][:40], table["splits"]["dev"][40:]
    order = lambda stage, ids: sorted(ids, key=lambda q: digest([PROTOCOL, stage, seed, q]))
    provider = Provider(table["rows"], table["questions"], "dev", order("S", S), SEARCH_BUDGET)
    finalists, predicted = search(provider, K, method)
    verification = Provider(table["rows"], table["questions"], "dev", order("V", V), VALIDATION_BUDGET)
    while verification.count[finalists[0]] < len(V):
        costs = [verification.cost(a, 1) for a in finalists]
        if any(c is None for c in costs) or verification.spent+sum(costs) > verification.budget:
            break
        for arm in finalists:
            verification.buy(arm, 1, "common_validation_block")
    n = verification.count[finalists[0]]
    require(n >= 8 and all(verification.count[a] == n for a in finalists), "Invalid V coverage")
    mean_cost = {a: sum(2*(table["questions"][q]["prompts"][a]["prompt_tokens"]+2048)
                        for q in S)/len(S) for a in finalists}
    chosen = min(finalists, key=lambda a: (-verification.mean(a), mean_cost[a], ORDER.index(a)))
    return dict(method=method, seed=seed, finalists=finalists, selected=chosen,
                search=dict(qids=provider.qids, trace=provider.trace, counts=provider.count,
                            reserved_tokens=provider.spent, actual_tokens=provider.actual,
                            model_means=predicted),
                validation=dict(qids=verification.qids[:n], n=n, trace=verification.trace,
                                means={a: verification.mean(a) for a in finalists},
                                reserved_tokens=verification.spent, actual_tokens=verification.actual),
                reserved_tokens=provider.spent+verification.spent)


def run_replay(table, official):
    validate_table(table)
    S = table["splits"]["dev"][:40]
    kernels = {m: kernel(table, S, m, official) for m in METHODS}
    # Freeze all choices before accessing the test score projection.
    choices = [select_one(table, m, seed, kernels[m]) for m in METHODS for seed in range(16)]
    test = [r for r in table["rows"] if r["phase"] == "test"]
    accuracy = {a: sum(r["score"] for r in test if r["arm"] == a)/(94*2) for a in ORDER}
    for choice in choices:
        choice["test_accuracy"] = accuracy[choice["selected"]]
    summary = {}
    for method in METHODS:
        runs = [r for r in choices if r["method"] == method]
        summary[method] = dict(mean_test_accuracy=float(np.mean([r["test_accuracy"] for r in runs])),
                               selected_counts=dict(Counter(r["selected"] for r in runs)),
                               mean_reserved_tokens=float(np.mean([r["reserved_tokens"] for r in runs])))
    pairs = {}
    for control in METHODS[1:]:
        pairs[control] = [
            next(r["test_accuracy"] for r in choices if r["method"] == "rendered" and r["seed"] == seed) -
            next(r["test_accuracy"] for r in choices if r["method"] == control and r["seed"] == seed)
            for seed in range(16)]
    advance = all(np.mean(pairs[c]) >= .02 for c in ("stored", "independent", "uniform"))
    advance = advance and sum(r["selected"] != "ZZ" for r in choices if r["method"] == "rendered") >= 8
    return dict(protocol=PROTOCOL, input_sha256=digest(table), summary=summary, choices=choices,
                fixed_policy_accuracy=accuracy, paired_differences=pairs,
                status="WORTH_INDEPENDENT_TEST" if advance else "NO_INCREMENTAL_SIGNAL_ON_THIS_TABLE",
                new_generation_calls=0,
                interpretation="Retrospective shared-table screening; 16 orders are not independent histories; no p-values or deployment certificate")
