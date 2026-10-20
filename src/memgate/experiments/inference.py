"""Budget-limited dev selection and the six preregistered fixed-sample contrasts."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .common import ARMS, TIE_ORDER, digest, require


def selector_authorizations(dev_order, questions):
    require(dev_order, "No eligible dev questions")

    def block(qid, arms):
        return sum(2 * (questions[qid]["prompts"][a]["prompt_tokens"] + 2048) for a in arms)

    endpoints, components = ("OO", "NN"), ("OO", "ON", "NO", "NN")
    budget = sum(block(qid, endpoints) for qid in dev_order)
    chosen, spent = [], 0
    for qid in dev_order:
        cost = block(qid, components)
        if spent + cost > budget:
            break  # Never skip an expensive earlier question for a later cheap one.
        chosen.append(qid)
        spent += cost
    require(chosen, "Component selector cannot afford one complete dev block")
    return {
        "F": dict(qids=list(dev_order), arms=list(endpoints), budget=budget, spent=budget, remaining=0),
        "S": dict(qids=chosen, arms=list(components), budget=budget, spent=spent, remaining=budget-spent),
    }


def authorized_projection(authorization, rows):
    expected = {(qid, arm, rep) for qid in authorization["qids"]
                for arm in authorization["arms"] for rep in range(2)}
    by_key = {(r["qid"], r["arm"], r["rep"]): r for r in rows}
    require(len(by_key) == len(rows) and expected <= set(by_key), "Missing/duplicate dev rows")
    # This is the only data returned to the selector. No prompts, test or unbudgeted labels.
    return [dict(qid=q, arm=a, rep=r, score=by_key[q, a, r]["score"])
            for q, a, r in sorted(expected)]


def select(authorization, projection):
    expected = {(q, a, r) for q in authorization["qids"]
                for a in authorization["arms"] for r in range(2)}
    actual = {(r["qid"], r["arm"], r["rep"]) for r in projection}
    require(actual == expected and len(projection) == len(expected), "Unauthorized selector projection")
    require(all(set(r) == {"qid", "arm", "rep", "score"} and r["score"] in (0, 1)
                for r in projection), "Unexpected information supplied to selector")
    means = {a: sum(r["score"] for r in projection if r["arm"] == a) /
             (2 * len(authorization["qids"])) for a in authorization["arms"]}
    best = max(means.values())
    arm = next(a for a in TIE_ORDER if a in means and means[a] == best)
    return dict(arm=arm, means=means, authorization=authorization,
                projection_sha256=digest(projection))


def coefficients(selected, full):
    require(selected in ("OO", "ON", "NO", "NN") and full in ("OO", "NN"), "Invalid selected arm")

    def difference(a, b):
        values = dict.fromkeys(ARMS, 0)
        values[a] += 1
        values[b] -= 1
        return [values[x] for x in ARMS]

    return dict(C1=difference("ON", "OO"), C2=difference("ON", "NN"),
                C3=difference("ON", "ZE"), I=[1, -1, -1, 1, 0, 0, 0],
                D_F=difference(selected, full), D_ON=difference(selected, "ON"))


def status(low, high, delta=0.03):
    return "GO" if low > delta else ("NO_GO" if high < delta else "INCONCLUSIVE")


def analyze_scores(qids, rows, selected, full):
    n = len(qids)
    require(n > 0 and len(set(qids)) == n, "No eligible test set/duplicate test IDs")
    expected = {(q, a, r) for q in qids for a in ARMS for r in range(2)}
    by_key = {(r["qid"], r["arm"], r["rep"]): r for r in rows}
    require(set(by_key) == expected and len(rows) == len(expected), "Incomplete/duplicate/foreign test rows")
    require(all(r["score"] in (0, 1) for r in rows), "Invalid score, technical failures cannot be zero")
    y = np.array([[[by_key[q, a, r]["score"] for r in range(2)] for a in ARMS] for q in qids])
    contrasts = {}
    for name, values in coefficients(selected, full).items():
        c = np.asarray(values)
        sq = int(c @ c)
        per_query = y.mean(axis=2) @ c
        estimate = float(per_query.mean())
        half = math.sqrt(sq * math.log(240) / (4 * n))
        lo, hi = max(float(c[c < 0].sum()), estimate-half), min(float(c[c > 0].sum()), estimate+half)
        contrasts[name] = dict(coefficients=values, estimate=estimate, half_width=half,
                               ci=[lo, hi], status=status(lo, hi),
                               decode_variance=float((((y[:, :, 0]-y[:, :, 1])**2) *
                                                      c**2).sum() / (4*n*n)),
                               query_variance=float(np.var(per_query, ddof=1)) if n > 1 else None)
    necessary = {k: contrasts[k] for k in ("C1", "C2", "C3")}
    go = all(v["ci"][0] > 0.03 for v in necessary.values())
    rejected = [k for k, v in necessary.items() if v["ci"][1] < 0.03]
    main = "GO" if go else ("NO_GO" if rejected else "INCONCLUSIVE")
    c2, inter = contrasts["C2"], contrasts["I"]
    return dict(n=n, arm_accuracy=dict(zip(ARMS, map(float, y.mean(axis=(0, 2))))),
                contrasts=contrasts, mechanism_status=main, excluded_conditions=rejected,
                simple_effect_a=dict(estimate=-c2["estimate"]-inter["estimate"],
                                     ci=[max(-1, -c2["ci"][1]-inter["ci"][1]),
                                         min(1, -c2["ci"][0]-inter["ci"][0])]),
                simple_effect_b=contrasts["C1"],
                target="New independent decoding means on this fixed eligible test set, conditional on history/dev/selection",
                assumptions=["Conditionally independent requests", "Stable backend distribution", "Fixed scorer"],
                interaction_note="An additive interaction is not semantic compatibility; its significance is not a GO requirement")


def sensitivity(analysis, mean_reserved_per_question):
    result = []
    normal = NormalDist()
    # Optimistic observed-variance normal scenarios are NOT the registered CI.
    z = normal.inv_cdf(1 - .05/12) + normal.inv_cdf(.8)
    for name in ("C1", "C2", "C3", "I"):
        contrast = analysis["contrasts"][name]
        sq = sum(c*c for c in contrast["coefficients"])
        observed = contrast["decode_variance"] * analysis["n"]
        for effect in (.03, .06, .10):
            for threshold in (0.0, .03):
                gap = effect-threshold
                sufficient = (math.ceil(sq * (math.sqrt(math.log(240)) + math.sqrt(math.log(5)))**2 /
                                        (4 * gap**2)) if gap > 0 else None)
                scenario = math.ceil(observed*z*z/gap**2) if gap > 0 else None
                result.append(dict(contrast=name, true_effect=effect, threshold=threshold,
                                   marginal_power=.8, conservative_sufficient_n=sufficient,
                                   observed_variance_normal_scenario_n=scenario,
                                   conservative_reserved_tokens=(math.ceil(sufficient*mean_reserved_per_question)
                                                                 if sufficient is not None else None)))
    return result
