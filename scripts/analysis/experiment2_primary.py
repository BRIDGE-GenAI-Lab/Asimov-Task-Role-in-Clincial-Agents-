#!/usr/bin/env python3
"""Computes the main experiment 2 estimates from the Run A, P, C, D and O logs: exclusion checks on returned model and parsing, per-model rates, paired role contrasts with model-equal and random-effects summaries, cluster bootstraps, comprehension probe, live-check, reference policies and the reconciliation with experiment 1. It writes experiment2/results/experiment2_primary.json."""
import collections
import hashlib
import json
import math
import os
import random
import statistics
import sys
import warnings

sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "scripts")
import metrics as M
import providers as PC

E2_DIR = "experiment2"
FS_ORIG = "experiment1/results/"
ROLES = ["steward", "guardrail", "empathetic", "neutral", "devotion", "sx_patient_plain", "sx_ward_plain",
         "devotion_precedence"]
LADDER_ROLES = ("steward", "neutral", "devotion")
HALVES = ("NOT", "ENT")
PRIMARY = "claimed_vent"
OUTCOMES = ("claimed_vent", "claimed_vent_step1", "vent_violation", "final_holder", "any_violation")
CELL_LIMIT = 0.15
MODEL_LIMIT = 0.05
BOOTSTRAP_B = 2000
NOT_RUN = ("A0", "E")
D_NONE_KEY = "e2_dir-qwen3-5-27b"
D_NONE_EPISODES = 100
O_ORIGINAL_EPISODES = 300
FILES = {"A": "runA_%s.jsonl", "P": "runP_%s.jsonl", "C": "runC_%s.jsonl", "D": "runD_%s.jsonl",
         "D_none": "runD_none_%s.jsonl", "O": "runO_%s.jsonl"}
EPISODE_RUNS = ("A", "D", "D_none", "O")
JBA_MODES = ("none", "affirm", "no_affirm", "unrelated")
JBA_CONTRASTS = (("affirm", "none"), ("no_affirm", "none"), ("unrelated", "none"), ("affirm", "unrelated"),
                 ("affirm", "no_affirm"))
REPRODUCED_MODES = ("affirm", "no_affirm")
SMALL_N = 10
FRONTIER_NAMES = {"e2_dir-frontier-anthropic": "Claude Opus 5.5", "e2_dir-frontier-openai": "GPT-6.1 Sol",
                  "e2_dir-frontier-qwen": "Qwen3.8 Max", "e2_dir-frontier-x-ai": "Grok 4.7"}
ALIAS_2X2 = {"falcon3-10b-o2": "Falcon3-10B", "gemma4-31b-o2": "Gemma-4 31B", "glm47-flash-o2": "GLM-4.7-Flash",
             "granite41-30b-o2": "Granite-4.1-30B", "olmo2-32b-o2": "OLMo-2-32B", "phi4-o2": "Phi-4",
             "yi-34b-o2": "Yi-1.5-34B"}
O2_STACK = "Harvard O2 cluster, self-hosted with llama.cpp llama-server (registry keys ending -o2)"
LADDER_PRECISION = {"Gemma-4 31B": "Q8 weights (registry)"}
S2X2_GGUF = {"Gemma-4 31B": "Q8_0", "Phi-4": "Q8_0", "GLM-4.7-Flash": "Q4_K_M", "Falcon3-10B": "q8_0",
             "Granite-4.1-30B": "Q4_K_M", "OLMo-2-32B": "Q4_K_M", "Yi-1.5-34B": "Q4_K_M"}
ORIGINAL_SCENARIOS = ("original 100 ICU scenarios (the agent's own patient is never the favoured one; three "
                      "contested resources)")
BALANCED_NOT = "balanced set, NOT half: 50 matched-pair scenarios per model, the policy favours the rival"
BALANCED_ENT = "balanced set, ENT half: 50 matched-pair scenarios per model, the policy favours the agent's patient"
E1A_DIFFERENCES = {
    "policy": {"original": "not stated", "run_a": "stated identically to every role"},
    "scenarios": {"original": ORIGINAL_SCENARIOS,
                  "run_a": "the NOT half of the balanced set, 50 per model, where only ventilator priority is "
                           "balanced"},
    "outcome": {"original": "harm: any contested resource obtained against the standard, as the engine recorded it",
                "run_a": "claimed_vent: claim_bed issued, missing outcomes left out"},
    "serving_route": {"original": "as in the original study", "run_a": "pinned OpenRouter endpoints"}}


def read_rows(path):
    if not os.path.exists(path):
        return [], {"exists": False, "sha256": None, "partial_last_line": False}
    with open(path, "rb") as fh:
        data = fh.read()
    parts = data.split(b"\n")
    rows, partial = [], False
    for i, raw in enumerate(parts):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw.decode("utf-8")))
        except ValueError:
            if i == len(parts) - 1:
                partial = True
                continue
            raise ValueError("%s: line %d does not parse" % (path, i + 1))
    return rows, {"exists": True, "sha256": hashlib.sha256(data).hexdigest(), "partial_last_line": partial}


def load(path):
    return read_rows(path)[0]


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def episode_key(r):
    return r["cell_id"]


def probe_key(r):
    return (r["sid"], r["model"], r["arm"])


def scenario_key(r):
    return (r["sid"], r["model"])


KEYS = {"A": episode_key, "D": episode_key, "D_none": episode_key, "O": episode_key, "P": probe_key,
        "C": scenario_key}


def first_occurrences(rows, key):
    seen, kept = set(), []
    for r in rows:
        k = key(r)
        if k not in seen:
            seen.add(k)
            kept.append(r)
    return kept, len(rows) - len(kept)


def by_arm(r):
    return r["arm"]


def parse_cells(rows, cell=by_arm):
    acc = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        a = acc[cell(r)]
        a[0] += 1
        a[1] += r.get("parse_failures") or 0
        a[2] += r.get("n_steps", 4)
    return {c: {"episodes": e, "unparsed": f, "focal_actions": n, "share": f / n if n else None,
                "excluded": bool(n) and f / n > CELL_LIMIT} for c, (e, f, n) in sorted(acc.items())}


def call_check(rows, slug, pinned=(), permaslug=None, endpoint_provider=None):
    calls = [c for r in rows for c in (r.get("calls") or [])]
    mism = [not M.model_ok(c.get("returned_model"), slug, permaslug) for c in calls]
    unp = [bool(pinned) and not PC.provider_ok(c.get("provider") or "", pinned, endpoint_provider) for c in calls]
    bad = sum(1 for a, b in zip(mism, unp) if a or b)
    share = bad / len(calls) if calls else None
    return {"calls": len(calls), "model_mismatch": sum(mism), "unpinned_provider": sum(unp), "bad_calls": bad,
            "share": share, "excluded": bool(calls) and share > MODEL_LIMIT,
            "rows_without_calls": sum(1 for r in rows if not r.get("calls")),
            "returned_models": sorted({str(c.get("returned_model")) for c in calls}),
            "providers": sorted({str(c.get("provider")) for c in calls})}


def excluded(rows, slug, pinned=(), permaslug=None, endpoint_provider=None, cell=by_arm):
    cells = parse_cells(rows, cell)
    calls = call_check(rows, slug, pinned, permaslug, endpoint_provider)
    return {"parse_cells": cells, "parse_cells_excluded": [c for c, v in cells.items() if v["excluded"]],
            "calls": calls, "mismatch_call_share": calls["share"], "model_excluded": calls["excluded"]}


def retained(rows, ex, cell=by_arm):
    if ex["model_excluded"]:
        return []
    bad = set(ex["parse_cells_excluded"])
    return [r for r in rows if cell(r) not in bad]


def _check_args(m):
    endpoint_provider, permaslug = PC.split_endpoint(m.get("endpoint_name"))
    return {"slug": m["slug"], "pinned": (m.get("provider") or {}).get("order", []), "permaslug": permaslug,
            "endpoint_provider": endpoint_provider}


def y(row, outcome, missing="complete_case"):
    if outcome == "other_channels":
        ch = row.get("channels") or {}
        if ch.get("imaging") or ch.get("attending"):
            return True
        return False if (not (row.get("parse_failures") or 0) or missing != "complete_case") else None
    return M.outcomes(row, missing)[outcome]


def arm_rows(rows, arm, half=None):
    return [r for r in rows if r["arm"] == arm and (half is None or M.condition(r) == half)]


def by_sid(rows, outcome=PRIMARY, missing="complete_case"):
    out = {}
    for r in rows:
        if r["sid"] in out:
            raise ValueError("sid %s appears twice in one arm (%s, %s)" % (r["sid"], r.get("model"), r.get("arm")))
        out[r["sid"]] = y(r, outcome, missing)
    return out


def paired(rows1, rows0, outcome=PRIMARY, missing="complete_case", sids=None):
    a, b = by_sid(rows1, outcome, missing), by_sid(rows0, outcome, missing)
    if sids is not None:
        a = {s: v for s, v in a.items() if s in sids}
        b = {s: v for s, v in b.items() if s in sids}
    common = sorted(s for s, v in a.items() if v is not None and b.get(s) is not None)

    def dropped(mine, other):
        miss = sum(1 for v in mine.values() if v is None)
        pm = sum(1 for s, v in mine.items() if v is not None and s in other and other[s] is None)
        pa = sum(1 for s, v in mine.items() if v is not None and s not in other)
        return {"rows": len(mine), "missing_outcome": miss, "partner_missing": pm, "partner_absent": pa,
                "dropped": miss + pm + pa}

    x1, x0, n = sum(a[s] for s in common), sum(b[s] for s in common), len(common)
    return {"pairs": n, "x1": x1, "x0": x0, "diff": 100.0 * (x1 - x0) / n if n else None, "sids": common,
            "values": [(int(a[s]), int(b[s])) for s in common], "dropped1": dropped(a, b), "dropped0": dropped(b, a)}


def _summary(pm):
    if not pm:
        return None
    mean, lo, hi = M.mean_t_ci(list(pm.values()))
    return {"mean": mean, "ci": [lo, hi], "k": len(pm), "per_model": dict(pm)}


def summarise_pairs(pairs_by_model, labels, random_effects=True):
    used = sorted(m for m, p in pairs_by_model.items() if p["pairs"])
    detail = {}
    for m, p in sorted(pairs_by_model.items()):
        n = p["pairs"]
        detail[m] = {"pairs": n, "diff": p["diff"],
                     "rates": {labels[0]: 100.0 * p["x1"] / n if n else None,
                               labels[1]: 100.0 * p["x0"] / n if n else None},
                     "dropped": {labels[0]: p["dropped1"], labels[1]: p["dropped0"]}}
    if len(used) < 2:
        return None
    diffs = [pairs_by_model[m]["diff"] for m in used]
    mean, lo, hi = M.mean_t_ci(diffs)
    x1, x0, n = (sum(pairs_by_model[m][k] for m in used) for k in ("x1", "x0", "pairs"))
    rates = {lab: {m: detail[m]["rates"][lab] for m in used} for lab in labels}
    out = {"k": len(used), "models": used, "model_equal": {"mean": mean, "ci": [lo, hi]},
           "episode_pooled": 100.0 * (x1 - x0) / n, "pairs": n, "per_model": dict(zip(used, diffs)),
           "arm_rates": {lab: {k: v for k, v in _summary(rates[lab]).items() if k != "per_model"} for lab in labels},
           "detail": detail, "models_without_pairs": sorted(set(pairs_by_model) - set(used))}
    if random_effects and len(used) >= 3:
        ys, vs = [], []
        for m in used:
            p = pairs_by_model[m]
            d, v = M.rd_var(p["x1"], p["pairs"], p["x0"], p["pairs"])
            ys.append(100.0 * d)
            vs.append(1e4 * v)
        re = M.re_summary(ys, vs)
        out["random_effects"] = {"mu": re["mu"], "ci_hk": list(re["ci_hk"]), "pi": list(re["pi"]),
                                 "tau2": re["tau2"], "I2": re["I2"]}
    return out


def contrast(rows_by_model, arm1, arm0, half="NOT", outcome=PRIMARY, missing="complete_case", sids=None, random_effects=True):
    pb = {m: paired(arm_rows(rows, arm1, half), arm_rows(rows, arm0, half), outcome, missing, sids)
          for m, rows in sorted(rows_by_model.items())}
    out = summarise_pairs(pb, (arm1, arm0), random_effects)
    if out is not None:
        out.update({"arms": [arm1, arm0], "half": half, "outcome": outcome, "missing": missing})
    return out


def cluster_bootstrap(rows_by_model, arm1, arm0, half="NOT", outcome=PRIMARY, missing="complete_case", B=BOOTSTRAP_B, seed=7):
    rng = random.Random(seed)
    by = {}
    for m, rows in sorted(rows_by_model.items()):
        p = paired(arm_rows(rows, arm1, half), arm_rows(rows, arm0, half), outcome, missing)
        if p["pairs"]:
            by[m] = [a - b for a, b in p["values"]]
    models = sorted(by)
    if len(models) < 2:
        return None
    stats = []
    for _ in range(B):
        diffs = []
        for m in (rng.choice(models) for _ in models):
            smp = [rng.choice(by[m]) for _ in by[m]]
            diffs.append(100.0 * sum(smp) / len(smp))
        stats.append(sum(diffs) / len(diffs))
    stats.sort()
    return {"k": len(models), "B": B, "seed": seed, "ci": [stats[int(0.025 * B)], stats[int(0.975 * B) - 1]]}


def cell_rates(rows, missing="complete_case", outcomes=OUTCOMES):
    out = {}
    for r in rows:
        half = M.condition(r)
        if half not in HALVES:
            raise ValueError("cell_rates needs balanced sids, got %s" % r["sid"])
        cell = out.setdefault(r["arm"], {}).setdefault(half, {"episodes": 0})
        cell["episodes"] += 1
        o = M.outcomes(r, missing)
        for k in outcomes:
            s = cell.setdefault(k, {"x": 0, "n": 0, "missing": 0})
            if o[k] is None:
                s["missing"] += 1
            else:
                s["n"] += 1
                s["x"] += int(o[k])
    return out


def rate(cell, outcome=PRIMARY):
    s = (cell or {}).get(outcome)
    return 100.0 * s["x"] / s["n"] if s and s["n"] else None


def exact_upper(n):
    return 100.0 * (1 - 0.025 ** (1.0 / n))


def _rate_of(rows, outcome, missing="complete_case"):
    vals = [y(r, outcome, missing) for r in rows]
    nm = [v for v in vals if v is not None]
    return {"x": sum(nm), "n": len(nm), "missing": len(vals) - len(nm),
            "rate": 100.0 * sum(nm) / len(nm) if nm else None,
            "exact_upper": exact_upper(len(nm)) if nm and not sum(nm) else None}


def role_rates(cells_by_model, half, outcome=PRIMARY, roles=ROLES):
    out = {}
    for role in roles:
        detail = {}
        for m, c in sorted(cells_by_model.items()):
            s = c.get(role, {}).get(half, {}).get(outcome)
            if s and s["n"]:
                detail[m] = {"x": s["x"], "n": s["n"], "missing": s["missing"], "rate": 100.0 * s["x"] / s["n"],
                             "exact_upper": exact_upper(s["n"]) if s["x"] == 0 else None}
        if len(detail) >= 2:
            xs = [d["rate"] for d in detail.values()]
            mean, lo, hi = M.mean_t_ci(xs)
            out[role] = {"mean": mean, "ci": [lo, hi], "k": len(xs), "min": min(xs), "max": max(xs),
                         "per_model": {m: d["rate"] for m, d in detail.items()}, "detail": detail}
    return out


def discrimination(cells_by_model, outcome=PRIMARY, roles=ROLES):
    out = {}
    for role in roles:
        pm = {}
        for m, c in sorted(cells_by_model.items()):
            ent, nt = rate(c.get(role, {}).get("ENT"), outcome), rate(c.get(role, {}).get("NOT"), outcome)
            if ent is not None and nt is not None:
                pm[m] = ent - nt
        if len(pm) >= 2:
            out[role] = _summary(pm)
    return out


def frontier_ent_comparison(cells_by_model):
    pm = {}
    for m, c in sorted(cells_by_model.items()):
        d, s = rate(c.get("devotion", {}).get("ENT")), rate(c.get("steward", {}).get("ENT"))
        if d is not None and s is not None:
            pm[m] = (d, s)
    if not pm:
        return None
    dev = statistics.mean(v[0] for v in pm.values())
    stew = statistics.mean(v[1] for v in pm.values())
    return {"models": sorted(pm), "k": len(pm), "devotion_mean": dev, "steward_mean": stew,
            "devotion_minus_steward": dev - stew,
            "per_model": {m: {"devotion": v[0], "steward": v[1]} for m, v in pm.items()}}


def bounds(rows_by_model, arm1, arm0, half="NOT", outcome=PRIMARY):
    out = {}
    for mode in ("complete_case", "upper", "engine"):
        arm_rates = {}
        for arm in (arm1, arm0):
            pm = {m: _rate_of(arm_rows(rows, arm, half), outcome, mode)["rate"]
                  for m, rows in sorted(rows_by_model.items())}
            pm = {m: v for m, v in pm.items() if v is not None}
            arm_rates[arm] = _summary(pm) or {"k": 0, "per_model": {}}
        out[mode] = {"arm_rates": arm_rates, "contrast": contrast(rows_by_model, arm1, arm0, half, outcome, mode)}
    out["note"] = ("complete_case leaves missing outcomes out; upper counts every missing outcome as a claim and engine as no "
                   "claim (the engine's record). They bound each arm's rate, not the difference between arms.")
    return out


def label_order(scenarios):
    return {s["sid"]: s["focal_pid"] < s["_rival_pid"] for s in scenarios}


def label_order_strata(order):
    return {"focal_first": {s for s, v in order.items() if v}, "focal_second": {s for s, v in order.items() if not v}}


def _gee_fit(d, sm, smf):
    res = {"n": len(d), "clusters": int(d["model"].nunique()), "or_per_0.1": None, "ci": None, "estimable": False}
    if d["y"].nunique() < 2:
        return dict(res, reason="outcome does not vary")
    if res["clusters"] < 2:
        return dict(res, reason="fewer than 2 model clusters")
    d = d.copy()
    d["mc"] = d["margin"] - d["margin"].mean()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = smf.gee("y ~ mc", groups="model", data=d, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
        except Exception as exc:
            return dict(res, reason="fit failed: %s" % exc)
    msgs = sorted({"%s: %s" % (w.category.__name__, w.message) for w in caught})
    if any("separation" in m.lower() for m in msgs):
        return dict(res, reason="separation warning", warnings=msgs)
    b = float(fit.params["mc"])
    lo, hi = (float(v) for v in fit.conf_int().loc["mc"])
    if not all(math.isfinite(v) for v in (b, lo, hi)):
        return dict(res, reason="non-finite estimate", warnings=msgs)
    return dict(res, estimable=True, reason=None, warnings=msgs,
                **{"or_per_0.1": math.exp(0.1 * b), "ci": [math.exp(0.1 * lo), math.exp(0.1 * hi)]})


def margin_analysis(rows_by_model, roles=LADDER_ROLES, outcome=PRIMARY):
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    recs, cells = [], collections.defaultdict(lambda: [0, 0])
    for m, rows in sorted(rows_by_model.items()):
        for r in rows:
            if r["arm"] not in roles:
                continue
            v = y(r, outcome)
            if v is None:
                continue
            half, mg = M.condition(r), M.margin(r)
            c = cells[(m, r["arm"], half, mg)]
            c[0] += int(v)
            c[1] += 1
            recs.append({"model": m, "arm": r["arm"], "half": half, "margin": mg, "y": int(v)})
    by_margin = collections.defaultdict(lambda: collections.defaultdict(list))
    for (m, arm, half, mg), (x, n) in cells.items():
        by_margin["%s|%s" % (arm, half)][mg].append(100.0 * x / n)
    gee = {}
    df = pd.DataFrame(recs, columns=["model", "arm", "half", "margin", "y"])
    for (arm, half), d in df.groupby(["arm", "half"]):
        gee["%s|%s" % (arm, half)] = _gee_fit(d, sm, smf)
    return {"outcome": outcome,
            "model_equal_by_margin": {k: {str(mg): statistics.mean(v) for mg, v in sorted(d.items())}
                                      for k, d in sorted(by_margin.items())},
            "model_equal_by_margin_k": {k: {str(mg): len(v) for mg, v in sorted(d.items())}
                                        for k, d in sorted(by_margin.items())},
            "gee": dict(sorted(gee.items()))}


PROBE_STATES = ("correct", "wrong", "unparsed")
ACTION_STATES = ("claim", "no_claim", "missing")


def probe_cells(p_rows):
    acc = {}
    for r in p_rows:
        c = acc.setdefault(r["arm"], {"replies": 0, "parsed": 0, "correct": 0, "by_half": {}})
        h = c["by_half"].setdefault(M.condition(r), {"replies": 0, "parsed": 0, "correct": 0})
        for t in (c, h):
            t["replies"] += 1
            if r.get("correct") is not None:
                t["parsed"] += 1
                t["correct"] += int(bool(r["correct"]))
    for c in acc.values():
        n, p, k = c["replies"], c["parsed"], c["correct"]
        c.update({"parse_rate": 100.0 * p / n, "unparsed_share": (n - p) / n, "excluded": (n - p) / n > CELL_LIMIT,
                  "knowledge": 100.0 * k / p if p else None, "knowledge_unparsed_wrong": 100.0 * k / n})
    return dict(sorted(acc.items()))


def parity(a_rows_by_model, p_rows_by_model, roles=("neutral", "devotion")):
    cells = {m: probe_cells(rows) for m, rows in sorted(p_rows_by_model.items())}
    out = {"cells": cells, "knowledge": {}}
    for role in roles:
        kept = {m: c[role] for m, c in cells.items() if role in c and not c[role]["excluded"]}
        prim = {m: v["knowledge"] for m, v in kept.items() if v["knowledge"] is not None}
        out["knowledge"][role] = {"parsed": _summary(prim) or {"k": 0, "per_model": {}},
                                  "unparsed_wrong": _summary({m: v["knowledge_unparsed_wrong"]
                                                              for m, v in kept.items()}) or {"k": 0, "per_model": {}}}
    ident, ident_w, tot = {}, {}, [0, 0, 0]
    for m, c in cells.items():
        kept = [v for role, v in c.items() if role in roles and not v["excluded"]]
        k, p, n = (sum(v[f] for v in kept) for f in ("correct", "parsed", "replies"))
        if p:
            ident[m] = 100.0 * k / p
        if n:
            ident_w[m] = 100.0 * k / n
        tot = [tot[0] + k, tot[1] + p, tot[2] + n]
    out["identification_model_equal"] = statistics.mean(ident.values()) if ident else None
    out["identification_pooled"] = 100.0 * tot[0] / tot[1] if tot[1] else None
    out["identification_unparsed_wrong_model_equal"] = statistics.mean(ident_w.values()) if ident_w else None
    out["identification_unparsed_wrong_pooled"] = 100.0 * tot[0] / tot[2] if tot[2] else None
    joint, unmatched = {}, {}
    for m, rows in sorted(a_rows_by_model.items()):
        c = cells.get(m, {})
        for role in roles:
            if role not in c or c[role]["excluded"]:
                continue
            for half in HALVES:
                act = {r["sid"]: r for r in arm_rows(rows, role, half)}
                pr = {r["sid"]: r for r in p_rows_by_model.get(m, []) if r["arm"] == role and M.condition(r) == half}
                if not act or not pr:
                    continue
                t = {"%s|%s" % (p, q): 0 for p in PROBE_STATES for q in ACTION_STATES}
                for s in sorted(set(act) & set(pr)):
                    corr = pr[s].get("correct")
                    ps = "unparsed" if corr is None else ("correct" if corr else "wrong")
                    v = y(act[s], "claimed_vent_step1")
                    qs = "missing" if v is None else ("claim" if v else "no_claim")
                    t[ps + "|" + qs] += 1
                key = "%s|%s|%s" % (m, role, half)
                joint[key] = t
                unmatched[key] = {"action_without_probe": len(set(act) - set(pr)),
                                  "probe_without_action": len(set(pr) - set(act))}
    out["joint"], out["unmatched"] = joint, unmatched
    for role in roles:
        per, pooled = {}, collections.Counter()
        for key, t in joint.items():
            m, r_, h = key.split("|")
            if r_ != role or h != "NOT":
                continue
            pooled.update(t)
            x, n_c = t["correct|claim"], t["correct|claim"] + t["correct|no_claim"]
            per[m] = {"x": x, "n": n_c, "rate": 100.0 * x / n_c if n_c else None, "small_n": n_c < SMALL_N}
        s = _summary({m: v["rate"] for m, v in per.items() if v["rate"] is not None})
        out["claimed_after_correct_" + role] = {"model_equal": s["mean"] if s else None, "ci": s["ci"] if s else None,
                                                "k_models": s["k"] if s else 0, "per_model": per,
                                                "pooled_joint": dict(pooled), "small_n_below": SMALL_N, "half": "NOT"}
    out["note"] = ("Behavioural: 'named the favoured patient and claimed the ventilator at the same step', never "
                   "'knew'. The probe measures whether the model applies the stated policy when pointed at it, an "
                   "upper bound on what the acting agent used; answers remain role-conditioned. claimed_after_correct "
                   "has no minimum: small_n (n below small_n_below) is a display convention only and excludes no "
                   "model.")
    return out


def _mode(r, mode):
    return (r.get("modes") or {}).get(mode) or {}


def jba_cells(rows):
    out = {}
    n = len(rows)
    for mode in JBA_MODES:
        parsed = sum(1 for r in rows if _mode(r, mode).get("claimed_vent") is not None)
        out[mode] = {"replies": n, "parsed": parsed, "parse_rate": 100.0 * parsed / n if n else None,
                     "unparsed_share": (n - parsed) / n if n else None,
                     "excluded": bool(n) and (n - parsed) / n > CELL_LIMIT}
    return out


def jba(c_rows_by_model):
    if not c_rows_by_model:
        return None
    cells = {m: jba_cells(rows) for m, rows in sorted(c_rows_by_model.items())}
    counts = {}
    for m, rows in c_rows_by_model.items():
        for mode in JBA_MODES:
            for h in HALVES:
                vals = [_mode(r, mode).get("claimed_vent") for r in rows if M.condition(r) == h]
                vals = [v for v in vals if v is not None]
                counts[(m, mode, h)] = (sum(vals), len(vals))

    def r_(m, mode, h):
        x, n = counts.get((m, mode, h), (0, 0))
        return None if cells[m][mode]["excluded"] or not n else 100.0 * x / n

    rates, contrasts = {}, {}
    for mode in JBA_MODES:
        for h in HALVES:
            pm = {m: r_(m, mode, h) for m in cells if r_(m, mode, h) is not None}
            s = _summary(pm) or {"k": 0, "per_model": {}}
            s["detail"] = {}
            for m in cells:
                x, n = counts[(m, mode, h)]
                s["detail"][m] = {"x": x, "n": n, "exact_upper": exact_upper(n) if n and not x else None,
                                  "excluded": cells[m][mode]["excluded"]}
            rates["%s|%s" % (mode, h)] = s
    for a, b in JBA_CONTRASTS:
        for h in HALVES:
            pm = {m: r_(m, a, h) - r_(m, b, h) for m in cells if r_(m, a, h) is not None and r_(m, b, h) is not None}
            s = _summary(pm)
            contrasts["%s-%s|%s" % (a, b, h)] = None if len(pm) < 2 else {"mean": s["mean"], "ci": s["ci"], "k": s["k"],
                                                                          "per_model": pm}
    reproduction = {}
    for mode in REPRODUCED_MODES:
        for h in HALVES:
            pm, x_tot, n_tot = {}, 0, 0
            for m, rows in sorted(c_rows_by_model.items()):
                rr = [r for r in rows if M.condition(r) == h]
                x = sum(1 for r in rr if _mode(r, mode).get("claimed") is True)
                if rr:
                    pm[m] = 100.0 * x / len(rr)
                    x_tot, n_tot = x_tot + x, n_tot + len(rr)
            reproduction["%s|%s" % (mode, h)] = {"per_model": pm,
                                                 "model_equal": statistics.mean(pm.values()) if pm else None,
                                                 "pooled": 100.0 * x_tot / n_tot if n_tot else None, "n": n_tot}
    judgment = {}
    for h in HALVES:
        detail = {}
        for m, rows in sorted(c_rows_by_model.items()):
            rr = [r for r in rows if M.condition(r) == h]
            parsed = [r for r in rr if r.get("judgment_correct") is not None]
            k = sum(1 for r in parsed if r["judgment_correct"])
            if rr:
                detail[m] = {"rows": len(rr), "parsed": len(parsed), "correct": k,
                             "rate": 100.0 * k / len(parsed) if parsed else None}
        pm = {m: d["rate"] for m, d in detail.items() if d["rate"] is not None}
        judgment[h] = dict(_summary(pm) or {"k": 0, "per_model": {}}, detail=detail)
    return {"cells": cells, "rates": rates, "contrasts": contrasts, "reproduction": reproduction,
            "parse_rates": {m: {mode: c[mode]["parse_rate"] for mode in JBA_MODES} for m, c in cells.items()},
            "unrelated_parse_ok": {m: 100.0 * sum(1 for r in rows if r.get("unrelated_parse_ok")) / len(rows)
                                   for m, rows in sorted(c_rows_by_model.items()) if rows},
            "judgment_correct": judgment,
            "note": "judgment_correct is read behaviourally ('named the patient the policy favours'), never as "
                    "'knew'. The reproduction uses the original definition over all rows, for the original "
                    "affirm and no_affirm figures; it reproduces the metric definitions on the new design."}


LEVELS = ("none", "partial", "protocol")
LIVE_METRICS = {"inappropriate_NOT": ("NOT", "vent_violation"), "any_inappropriate_NOT": ("NOT", "any_violation"),
                "other_channels_NOT": ("NOT", "other_channels"), "appropriate_ENT": ("ENT", "final_holder"),
                "other_channels_ENT": ("ENT", "other_channels"), "claimed_NOT": ("NOT", "claimed_vent"),
                "claimed_ENT": ("ENT", "claimed_vent")}


def check_level(row):
    o = row.get("oversight") or ""
    if o.startswith("partial"):
        return "partial"
    if o == "protocol":
        return "protocol"
    raise ValueError("Run D row %s with unknown oversight %r" % (row.get("cell_id"), o))


def live_check(none_rows_by_model, d_rows_by_model):
    levels = {}
    for m in sorted(set(none_rows_by_model) | set(d_rows_by_model)):
        lv = {"none": [r for r in none_rows_by_model.get(m, []) if r["arm"] == "devotion"]}
        for r in d_rows_by_model.get(m, []):
            if r["arm"] == "devotion":
                lv.setdefault(check_level(r), []).append(r)
        levels[m] = {k: v for k, v in lv.items() if v}
    per_model = {m: {lvl: {name: _rate_of(arm_rows(rows, "devotion", half), oc)
                           for name, (half, oc) in LIVE_METRICS.items()}
                     for lvl, rows in lv.items()} for m, lv in levels.items()}
    rates, model_equal, change = {}, {}, {}
    for lvl in LEVELS:
        for name in LIVE_METRICS:
            pm = {m: v[lvl][name]["rate"] for m, v in per_model.items()
                  if lvl in v and v[lvl][name]["rate"] is not None}
            s = _summary(pm)
            if s:
                rates["%s|%s" % (lvl, name)] = s
                model_equal["%s|%s" % (lvl, name)] = s["mean"]
    for lvl in ("partial", "protocol"):
        for name, (half, oc) in LIVE_METRICS.items():
            pb = {m: paired(arm_rows(lv[lvl], "devotion", half), arm_rows(lv["none"], "devotion", half), oc)
                  for m, lv in levels.items() if lvl in lv and "none" in lv}
            change["%s-none|%s" % (lvl, name)] = summarise_pairs(pb, (lvl, "none"), random_effects=False)
    return {"per_model": per_model, "rates": rates, "change": change, "model_equal": model_equal,
            "note": "Claims under a check are claims that passed the check (Run D records moves after the check); "
                    "the issued-claim rate under a check is not recorded. Under the true-priority check the "
                    "passed-claim rate on the NOT half is 0 by construction."}


def _orig_rates(rows_by_model, outcome, missing):
    per = {m: {a: _rate_of([r for r in rows if r["arm"] == a], outcome, missing) for a in LADDER_ROLES}
           for m, rows in sorted(rows_by_model.items())}
    by_role = {}
    for a in LADDER_ROLES:
        xs = [v[a]["rate"] for v in per.values() if v[a]["rate"] is not None]
        if len(xs) >= 2:
            mean, lo, hi = M.mean_t_ci(xs)
            by_role[a] = {"mean": mean, "ci": [lo, hi], "min": min(xs), "max": max(xs), "k": len(xs)}
    return {m: {a: v[a]["rate"] for a in LADDER_ROLES} for m, v in per.items()}, per, by_role


def original_design_summary(rows_by_model, missing="complete_case"):
    if len(rows_by_model) < 3:
        return None
    per, detail, by_role = _orig_rates(rows_by_model, "any_violation", missing)
    out = {"k": len(rows_by_model), "models": sorted(rows_by_model), "missing": missing, "outcome": "harm",
           "per_model": per, "per_model_detail": detail, "rates": by_role,
           "devotion_minus_steward": contrast(rows_by_model, "devotion", "steward", half=None, outcome="any_violation",
                                              missing=missing)}
    for name, oc in (("claimed_vent", "claimed_vent"), ("vent_acquired", "vent_violation")):
        p, d, r = _orig_rates(rows_by_model, oc, missing)
        out[name] = {"per_model": p, "per_model_detail": d, "rates": r}
    return out


def original_panel_comparator(off):
    def block(rr, pe):
        per = {m: {a: v[a]["harm_rate_pct"] for a in LADDER_ROLES} for m, v in rr["per_model"].items()}
        rng = {}
        for a in LADDER_ROLES:
            xs = [p[a] for p in per.values() if p[a] is not None]
            rng[a] = [min(xs), max(xs)] if xs else None
        me = rr["model_equal_mean"]
        return {"k": len(per), "per_model_harm_pct": per, "range_harm_pct": rng,
                "model_equal_harm_pct": {a: {"mean": me[a]["harm_rate_pct"], "ci": me[a]["harm_ci95"], "k": me[a]["k"]}
                                         for a in LADDER_ROLES},
                "devotion_minus_steward": {k: pe[k] for k in ("k", "model_equal", "random_effects", "episode_pooled",
                                                              "per_model_pp")}}
    pub = off["published_rows"]
    return {"source": "results/experiment1_reanalysis.json: role_rates_original and "
                      "primary_estimands.devotion_minus_steward (de-duplicated), published_rows (every logged row)",
            "de_duplicated": block(off["role_rates_original"], off["primary_estimands"]["devotion_minus_steward"]),
            "published_rows": block(pub["role_rates_original"], pub["primary_estimands"]["devotion_minus_steward"]),
            "comparison": "descriptive: the two sets contain different models; no test of difference"}


def request_vs_acquisition(cells_by_model, roles=ROLES):
    out = {}
    for half, acq in (("NOT", "vent_violation"), ("ENT", "final_holder")):
        out[half] = {}
        for role in roles:
            cl = {m: c[role][half] for m, c in sorted(cells_by_model.items()) if half in c.get(role, {})}
            if not cl:
                continue
            node = {"k": len(cl), "episodes": sum(c["episodes"] for c in cl.values())}
            for name, oc in (("claimed", PRIMARY), ("acquired", acq)):
                x, n = sum(c[oc]["x"] for c in cl.values()), sum(c[oc]["n"] for c in cl.values())
                pm = [rate(c, oc) for c in cl.values() if rate(c, oc) is not None]
                node.update({name + "_pct": 100.0 * x / n if n else None, name + "_n": n,
                             name + "_model_equal": statistics.mean(pm) if pm else None})
            out[half][role] = node
    return out


def reference_policies(ref_rows):
    out = {}
    for pol in sorted({r["model"] for r in ref_rows}):
        rows = [r for r in ref_rows if r["model"] == pol]
        out[pol] = {"replicates": sorted({r.get("rep") for r in rows}), "episodes": len(rows),
                    "arms": sorted({r["arm"] for r in rows})}
        for half, acq in (("NOT", "vent_violation"), ("ENT", "final_holder")):
            hr = [r for r in rows if M.condition(r) == half]
            out[pol][half] = {"episodes": len(hr), "claimed_vent": _rate_of(hr, PRIMARY)["rate"],
                              "acquired": _rate_of(hr, acq)["rate"]}
    return out


def _call_outcomes(rows):
    calls = [c for r in rows for c in (r.get("calls") or [])]
    fr = collections.Counter(c.get("finish_reason") for c in calls)
    return {"calls": len(calls), "truncated_calls": fr.get("length", 0),
            "content_filter_calls": fr.get("content_filter", 0),
            "error_calls": fr.get("error", 0), "rows_without_calls": sum(1 for r in rows if not r.get("calls")),
            "cost_usd": sum(c.get("cost") or 0.0 for c in calls)}


def operational(rows):
    return dict(_call_outcomes(rows), episodes=len(rows),
                episodes_with_parse_failure=sum(1 for r in rows if (r.get("parse_failures") or 0) > 0),
                parse_failures=sum(r.get("parse_failures") or 0 for r in rows),
                actions=sum(r.get("n_steps", 4) for r in rows))


def operational_probe(rows):
    return dict(_call_outcomes(rows), replies=len(rows), unparsed=sum(1 for r in rows if r.get("correct") is None))


def operational_jba(rows):
    return dict(_call_outcomes(rows), scenarios=len(rows),
                unparsed_actions={m: sum(1 for r in rows if _mode(r, m).get("claimed_vent") is None)
                                  for m in JBA_MODES},
                judgment_unparsed=sum(1 for r in rows if r.get("judgment_correct") is None),
                unrelated_parse_failed=sum(1 for r in rows if not r.get("unrelated_parse_ok")))


def reconciliation(lad, s2x2, run_per_model):
    inv = {v: k for k, v in ALIAS_2X2.items()}
    common = sorted(v for k, v in ALIAS_2X2.items() if k in s2x2 and v in lad)
    mean = lambda xs: statistics.mean(xs) if xs else None
    plain = lambda m: s2x2[inv[m]]["cells_pct"]["patient_plain"] - s2x2[inv[m]]["cells_pct"]["ward_plain"]
    rows = {"steward_rate_all": mean([lad[m]["steward"] for m in lad]),
            "steward_rate_common": mean([lad[m]["steward"] for m in common]),
            "neutral_minus_stew_all": mean([lad[m]["neutral"] - lad[m]["steward"] for m in lad]),
            "ladder_dev_minus_stew_all": mean([lad[m]["devotion"] - lad[m]["steward"] for m in lad]),
            "ladder_dev_minus_stew_common": mean([lad[m]["devotion"] - lad[m]["steward"] for m in common]),
            "neutral_rate_all": mean([lad[m]["neutral"] for m in lad]),
            "neutral_rate_common": mean([lad[m]["neutral"] for m in common]),
            "neutral_minus_stew_common": mean([lad[m]["neutral"] - lad[m]["steward"] for m in common]),
            "scope2x2_all": mean([v["scope_pp"] for v in s2x2.values()]),
            "scope2x2_common": mean([s2x2[inv[m]]["scope_pp"] for m in common]),
            "scope2x2_plain_common": mean([plain(m) for m in common])}
    for name, pm in run_per_model.items():
        rows["runA_%s_all" % name] = mean(list(pm.values()))
        rows["runA_%s_all_k" % name] = len(pm)
        rows["runA_%s_common" % name] = mean([pm[m] for m in common if m in pm])
        rows["runA_%s_common_k" % name] = sum(1 for m in common if m in pm)
    return {"common_models": common, "rows": rows,
            "runA_common_models": {name: [m for m in common if m in pm] for name, pm in run_per_model.items()},
            "runA_models": {name: sorted(pm) for name, pm in run_per_model.items()},
            "note": "model-equal means; the published 8.3 is the random-effects pooled 2x2 estimate (12 models)"}


def reconciliation_table(recon, s2x2_meta, runA_roster, alongside):
    r, common = recon["rows"], recon["common_models"]
    lab = {k: v.get("panel_label") or v["label"] for k, v in runA_roster.items()}
    by_label = {v: k for k, v in lab.items()}

    def served(keys):
        keys = [k for k in keys if k in runA_roster]
        return {"models": sorted(lab[k] for k in keys),
                "serving": {lab[k]: (runA_roster[k].get("endpoint_name") or "").replace(" | ", ": ") for k in keys},
                "precision": {lab[k]: runA_roster[k].get("quantization") for k in keys}}

    def runa(name, data_all):
        names = recon["runA_models" if data_all else "runA_common_models"].get(name, [])
        return served([by_label[m] for m in names if m in by_label])

    orig_ladder = {"scenarios": ORIGINAL_SCENARIOS, "wording": "five-role ladder, wording 0", "policy": "not stated",
                   "outcome": "harm, engine record (de-duplicated)"}
    orig_all = dict(orig_ladder, models="the 20 primary models", serving="as in the original study (hosted and "
                    "self-hosted routes)", precision="as in the original study")
    orig_common = dict(orig_ladder, models=common, serving=O2_STACK,
                       precision={m: LADDER_PRECISION.get(m, "not recorded") for m in common})
    s2x2 = {"scenarios": s2x2_meta.get("design"), "wording": "matched syntax: single-patient vs whole-ward scope x "
            "loyalty absent/present", "policy": "stated in all cells", "outcome": "as in scope_2x2_stats.json",
            "serving": O2_STACK}
    run_a = {"scenarios": BALANCED_NOT, "policy": "shown identically to every role",
             "outcome": "claimed_vent"}
    ladder_w = "steward vs devotion (wording 0)"
    scope_w = "sx_patient_plain vs sx_ward_plain (matched scope)"
    pooled = (s2x2_meta.get("pooled_random_effect") or {}).get("scope") or {}
    table = [
        dict(orig_all, estimate="ladder: strong advocacy - whole-ward", data="original ladder, all models",
             value=r["ladder_dev_minus_stew_all"]),
        dict(orig_common, estimate="ladder: strong advocacy - whole-ward", data="original ladder, common models",
             value=r["ladder_dev_minus_stew_common"]),
        dict(run_a, estimate="ladder: strong advocacy - whole-ward", data="Run A, common models", wording=ladder_w,
             value=r.get("runA_ladder_common"), **runa("ladder", False)),
        dict(run_a, estimate="ladder: strong advocacy - whole-ward", data="Run A, all models", wording=ladder_w,
             value=r.get("runA_ladder_all"), **runa("ladder", True)),
        dict(orig_all, estimate="neutral rate", data="original ladder, all models", value=r["neutral_rate_all"]),
        dict(orig_common, estimate="neutral rate", data="original ladder, common models",
             value=r["neutral_rate_common"]),
        dict(run_a, estimate="neutral rate", data="Run A, common models", wording="neutral (wording 0)",
             value=r.get("runA_neutral_rate_common"), **runa("neutral_rate", False)),
        dict(run_a, estimate="neutral rate", data="Run A, all models", wording="neutral (wording 0)",
             value=r.get("runA_neutral_rate_all"), **runa("neutral_rate", True)),
        dict(s2x2, estimate="2x2 scope effect (random-effects, published)", data="original 2x2, all models",
             value=pooled.get("pp"), ci=pooled.get("ci95"), models="the %s 2x2 models" % pooled.get("k"),
             precision="GGUF per model (scripts/o2/launch_scope2x2.sh)"),
        dict(s2x2, estimate="2x2 scope main effect", data="original 2x2, common models", value=r["scope2x2_common"],
             models=common, precision={m: S2X2_GGUF.get(m, "not recorded") for m in common}),
        dict(s2x2, estimate="matched-wording scope effect (patient_plain - ward_plain)",
             data="original 2x2, common models", value=r["scope2x2_plain_common"], models=common,
             precision={m: S2X2_GGUF.get(m, "not recorded") for m in common}),
        dict(run_a, estimate="matched-scope effect (sx_patient_plain - sx_ward_plain)", data="Run A, common models",
             wording=scope_w, value=r.get("runA_scope_common"), **runa("scope", False)),
        dict(run_a, estimate="matched-scope effect (sx_patient_plain - sx_ward_plain)", data="Run A, all models",
             wording=scope_w, value=r.get("runA_scope_all"), **runa("scope", True))]
    for name, est, scen in (("episode_pooled", "alongside: episode-pooled matched-scope difference", BALANCED_NOT),
                            ("frontier", "alongside: matched-scope effect, frontier tier", BALANCED_NOT),
                            ("panel", "alongside: matched-scope effect, panel tier", BALANCED_NOT),
                            ("ENT", "alongside: matched-scope appropriate-claim difference (ENT)", BALANCED_ENT)):
        a = alongside.get(name) or {}
        table.append(dict(run_a, estimate=est, data="Run A", wording=scope_w, scenarios=scen, value=a.get("value"),
                          ci=a.get("ci"), **served(a.get("models") or [])))
    return table


def role_rates_comparison(cells_by_model, labels, original):
    rr = role_rates(cells_by_model, "NOT", PRIMARY, LADDER_ROLES)
    run_a = {a: None if a not in rr else {"mean": rr[a]["mean"], "ci": rr[a]["ci"], "k": rr[a]["k"],
                                          "per_model": {labels[m]: v for m, v in rr[a]["per_model"].items()}}
             for a in LADDER_ROLES}
    pm = original["per_model"]

    def column(names, a):
        return {name: _summary({n: pm[n][a][field] for n in names if n in pm and pm[n][a].get(field) is not None})
                for name, field in (("harm", "harm_rate_pct"), ("claimed_vent", "claimed_vent_rate_pct"))}

    declared = sorted(labels.values())
    return {"run_a": run_a, "original_missing": [n for n in declared if n not in pm],
            "original_declared": {"models": declared, "rates": {a: column(declared, a) for a in LADDER_ROLES}},
            "original_matched": {a: dict(models=sorted((run_a[a] or {}).get("per_model", {})),
                                         **column(sorted((run_a[a] or {}).get("per_model", {})), a))
                                 for a in LADDER_ROLES},
            "differences": E1A_DIFFERENCES,
            "note": "Descriptive and non-inferential. The original "
                    "column is the de-duplicated role_rates_original of experiment1_reanalysis.json. "
                    "original_declared covers the declared Run A panel models; original_matched the models shown "
                    "in the Run A column for that role."}


SETTINGS = ("mode", "info", "wording", "oversight", "rep", "rules_stated", "prompt_version", "temperature")


def settings(rows):
    return {f: {str(k): v for k, v in sorted(collections.Counter(str(r.get(f)) for r in rows).items())}
            for f in SETTINGS}


def _matrix(e2_dir):
    rm = _load_json(os.path.join(e2_dir, "run_matrix.json"))
    models, expected = collections.defaultdict(list), {}
    for r in rm["runs"]:
        if r["run"] in NOT_RUN:
            continue
        models[r["run"]] += [m for m in r["models"] if m not in models[r["run"]]]
        expected[r["run"]] = r["episodes_per_model"]
    if D_NONE_KEY in models.get("D", []):
        models["D_none"] = [D_NONE_KEY]
        expected["D_none"] = D_NONE_EPISODES
    return dict(models), expected


def _meta(k, m):
    return {"label": m["label"], "name": FRONTIER_NAMES.get(k, m.get("panel_label") or m["label"]),
            "slug": m["slug"], "tier": m["tier"], "panel_label": m.get("panel_label"),
            "endpoint": (m.get("endpoint_name") or "").replace(" | ", ": "), "quantization": m.get("quantization")}


def build(e2_dir=None, fs_orig=None, B=None):
    e2_dir, fs_orig, B = e2_dir or E2_DIR, fs_orig or FS_ORIG, B or BOOTSTRAP_B
    roster = {m["key"]: m for m in _load_json(os.path.join(e2_dir, "model_roster.json"))["models"]
              if m["status"] == "locked"}
    models_of, expected = _matrix(e2_dir)
    expected["O"] = min(expected["O"], O_ORIGINAL_EPISODES)
    data, completeness, exclusions, lines = {}, {}, {}, []
    for run in FILES:
        data[run], completeness[run], exclusions[run] = {}, {}, {}
        for k in models_of.get(run, []):
            name = FILES[run] % k
            rows, info = read_rows(os.path.join(e2_dir, "runs", name))
            rows, dup = first_occurrences(rows, KEYS[run])
            if run == "O":
                rows = [r for r in rows if r["arm"] in LADDER_ROLES]
            strays = sorted({str(r.get("model")) for r in rows} - {k})
            if strays:
                raise ValueError("%s holds rows of another model: %s" % (name, ", ".join(strays)))
            data[run][k] = rows
            completeness[run][k] = {"rows": len(rows), "expected": expected[run],
                                    "complete": len(rows) >= expected[run], "file_exists": info["exists"],
                                    "duplicates_dropped": dup, "partial_last_line": info["partial_last_line"]}
            if run in EPISODE_RUNS and rows:
                completeness[run][k]["settings"] = settings(rows)
            if len(rows) < expected[run]:
                lines.append("INCOMPLETE %s %s: %d of %d rows%s" % (run, k, len(rows), expected[run],
                                                                   "" if info["exists"] else " (no file)"))
            if dup:
                lines.append("WARNING %s %s: %d repeated rows dropped (first occurrence kept)" % (run, k, dup))
            if info["partial_last_line"]:
                lines.append("WARNING %s %s: last line still being written; left out" % (run, k))
            args = _check_args(roster[k])
            if run in EPISODE_RUNS:
                exclusions[run][k] = excluded(rows, **args)
            else:
                calls = call_check(rows, **args)
                exclusions[run][k] = {"calls": calls, "mismatch_call_share": calls["share"],
                                      "model_excluded": calls["excluded"]}
                if run == "P":
                    exclusions[run][k]["probe_cells_excluded"] = [a for a, c in probe_cells(rows).items()
                                                                  if c["excluded"]]
                else:
                    exclusions[run][k]["mode_cells_excluded"] = [a for a, c in jba_cells(rows).items()
                                                                 if c["excluded"]]
            ex = exclusions[run][k]
            if rows and ex["model_excluded"]:
                lines.append("EXCLUDED model %s %s: %.1f%% of calls fail the returned-model/provider check"
                             % (run, k, 100 * ex["mismatch_call_share"]))
            for c in ex.get("parse_cells_excluded", []) + ex.get("probe_cells_excluded", []) + \
                    ex.get("mode_cells_excluded", []):
                lines.append("EXCLUDED cell %s %s %s: > 15%% unparsed" % (run, k, c))
    keep = {run: {k: (retained(rows, exclusions[run][k]) if run in EPISODE_RUNS else rows)
                  for k, rows in d.items() if rows and not exclusions[run][k]["model_excluded"]}
            for run, d in data.items()}
    keep = {run: {k: v for k, v in d.items() if v} for run, d in keep.items()}

    A = keep["A"]
    cells = {k: cell_rates(v) for k, v in A.items()}
    tier = {k: roster[k]["tier"] for k in roster}
    sub = lambda d, t: {k: v for k, v in d.items() if tier[k] == t}
    out = {"completeness": completeness, "exclusions": exclusions,
           "models": {k: _meta(k, roster[k]) for k in sorted(A)},
           "roster": {k: _meta(k, m) for k, m in sorted(roster.items())}}
    out["operational"] = {run: {k: (operational(v) if run in EPISODE_RUNS else
                                    operational_probe(v) if run == "P" else operational_jba(v))
                                for k, v in d.items() if v} for run, d in data.items()}
    gone = {k: retained(data["A"][k], dict(exclusions["A"][k], model_excluded=False))
            for k in data["A"] if data["A"][k] and exclusions["A"][k]["model_excluded"]}
    gone_cells = {k: cell_rates(v) for k, v in gone.items() if v}
    out["role_rates"] = {"NOT_claimed": role_rates(cells, "NOT"), "ENT_claimed": role_rates(cells, "ENT"),
                         "NOT_vent_violation": role_rates(cells, "NOT", "vent_violation"),
                         "ENT_final_holder": role_rates(cells, "ENT", "final_holder"),
                         "NOT_any_violation": role_rates(cells, "NOT", "any_violation"),
                         "ENT_any_violation": role_rates(cells, "ENT", "any_violation"),
                         "frontier_NOT_claimed": role_rates(sub(cells, "frontier"), "NOT"),
                         "frontier_ENT_claimed": role_rates(sub(cells, "frontier"), "ENT"),
                         "frontier_ENT_comparison": frontier_ent_comparison(sub(cells, "frontier")),
                         "excluded_models_reported_separately": {
                             k: {h: {role: rate(c.get(role, {}).get(h)) for role in ROLES} for h in HALVES}
                             for k, c in gone_cells.items()}}
    sx = ("sx_patient_plain", "sx_ward_plain")
    out["primary_scope"] = contrast(A, *sx)
    out["primary_scope_bootstrap"] = cluster_bootstrap(A, *sx, B=B)
    out["primary_scope_alongside"] = {"frontier": contrast(sub(A, "frontier"), *sx),
                                      "panel": contrast(sub(A, "panel"), *sx), "ENT": contrast(A, *sx, half="ENT")}
    strata = label_order_strata(label_order(load(os.path.join(e2_dir, "runs", "ward_scenarios_balanced_rv.jsonl"))))
    out["primary_scope_sensitivity"] = {
        "bounds": bounds(A, *sx),
        "step1": contrast(A, *sx, outcome="claimed_vent_step1"),
        "label_order": {name: {"scenarios": sum(1 for s in sids if s.endswith("-NOT")),
                               "contrast": contrast(A, *sx, sids=sids)} for name, sids in strata.items()}}
    dv = ("devotion", "steward")
    out["ladder"] = {"all": contrast(A, *dv), "frontier": contrast(sub(A, "frontier"), *dv),
                     "panel": contrast(sub(A, "panel"), *dv), "ENT": contrast(A, *dv, half="ENT"),
                     "all_acquisition": contrast(A, *dv, outcome="vent_violation"),
                     "ENT_acquisition": contrast(A, *dv, half="ENT", outcome="final_holder"),
                     "all_any_channel": contrast(A, *dv, outcome="any_violation")}
    out["ladder_bootstrap"] = cluster_bootstrap(A, *dv, B=B)
    prec = ("devotion_precedence", "devotion")
    out["contrasts"] = {"neutral_minus_steward": contrast(A, "neutral", "steward"),
                        "neutral_minus_steward_ENT": contrast(A, "neutral", "steward", half="ENT"),
                        "devotion_minus_neutral": contrast(A, "devotion", "neutral"),
                        "devotion_minus_neutral_ENT": contrast(A, "devotion", "neutral", half="ENT"),
                        "precedence_NOT": contrast(A, *prec), "precedence_ENT": contrast(A, *prec, half="ENT"),
                        "precedence_NOT_frontier": contrast(sub(A, "frontier"), *prec),
                        "precedence_ENT_frontier": contrast(sub(A, "frontier"), *prec, half="ENT")}
    out["discrimination"] = discrimination(cells)
    off = _load_json(os.path.join(e2_dir, "results", "experiment1_reanalysis.json"))
    declared = {k: roster[k].get("panel_label") or roster[k]["label"] for k in models_of.get("A", [])
                if tier[k] == "panel"}
    out["role_rates_comparison"] = role_rates_comparison({k: c for k, c in cells.items() if k in declared}, declared,
                                     off["role_rates_original"])
    out["reference_policies"] = reference_policies(load(os.path.join(e2_dir, "runs", "reference_policies.jsonl")))
    out["margin"] = margin_analysis(A)
    out["parity"] = parity(A, keep["P"])
    out["jba"] = jba(keep["C"])
    none, source = {}, {}
    for k in models_of.get("D", []):
        rows = keep["D_none"].get(k, []) if k == D_NONE_KEY else arm_rows(A.get(k, []), "devotion")
        if rows:
            none[k], source[k] = rows, "runD_none" if k == D_NONE_KEY else "Run A devotion"
    out["live_check"] = dict(live_check(none, keep["D"]), none_source=source)
    od = original_design_summary(keep["O"])
    if od is not None:
        od["engine_record"] = original_design_summary(keep["O"], missing="engine")
        od["comparator"] = original_panel_comparator(off)
    out["original_design_frontier"] = od
    out["request_vs_acquisition"] = request_vs_acquisition(cells)
    s2 = _load_json(fs_orig + "scope_2x2_stats.json")
    lad = {m: {a: v[a]["harm_rate_pct"] for a in LADDER_ROLES}
           for m, v in off["role_rates_original"]["per_model"].items()}
    lab = {k: roster[k].get("panel_label") or roster[k]["label"] for k in A}
    relabel = lambda pm: {lab[k]: v for k, v in (pm or {}).items()}
    nr = out["role_rates"]["NOT_claimed"]
    run_pm = {"ladder": relabel((out["ladder"]["all"] or {}).get("per_model")),
              "neutral_rate": relabel(nr.get("neutral", {}).get("per_model")),
              "steward_rate": relabel(nr.get("steward", {}).get("per_model")),
              "neutral_minus_stew": relabel((out["contrasts"]["neutral_minus_steward"] or {}).get("per_model")),
              "scope": relabel((out["primary_scope"] or {}).get("per_model")),
              "scope_ENT": relabel((out["primary_scope_alongside"]["ENT"] or {}).get("per_model"))}
    recon = reconciliation(lad, s2["per_model"], {n: d for n, d in run_pm.items() if d})
    ps, al = out["primary_scope"] or {}, out["primary_scope_alongside"]
    alongside = {"episode_pooled": {"value": ps.get("episode_pooled"), "ci": None, "models": ps.get("models", [])}}
    for name in ("frontier", "panel", "ENT"):
        c = al[name] or {}
        me = c.get("model_equal") or {}
        alongside[name] = {"value": me.get("mean"), "ci": me.get("ci"), "models": c.get("models", [])}
    recon["table"] = reconciliation_table(recon, s2, {k: roster[k] for k in A}, alongside)
    out["reconciliation"] = recon
    out["episodes"] = {run: sum(len(v) for v in d.values()) for run, d in data.items()}
    for k, g in out["margin"]["gee"].items():
        if not g["estimable"]:
            lines.append("NOTE margin GEE %s not estimable: %s" % (k, g["reason"]))
    return out, lines


def main():
    out, lines = build(E2_DIR, FS_ORIG, BOOTSTRAP_B)
    path = os.path.join(E2_DIR, "results", "experiment2_primary.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, allow_nan=False)
    for line in lines:
        print(line)
    ps, ld = out["primary_scope"], out["ladder"]["all"]
    print("PRIMARY matched scope (NOT, claims):", ps and ps["model_equal"], "k", ps and ps["k"],
          "PI", ps and ps.get("random_effects", {}).get("pi"))
    print("ladder strong advocacy - whole ward (NOT, claims):", ld and ld["model_equal"], "k", ld and ld["k"])
    print("wrote", path)


if __name__ == "__main__":
    main()
