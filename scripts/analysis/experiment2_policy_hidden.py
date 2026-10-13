#!/usr/bin/env python3
"""Compares Run A0 (policy hidden) with Run A (policy shown) for five models, and summarises Run O, which repeats experiment 1's design on four high-capability models. It reads the run logs in experiment2/runs and writes experiment2/results/experiment2_policy_hidden.json; with --check it only reports whether the stored file is current."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment2_primary as RR

E2_DIR = RR.E2_DIR
OUT = os.path.join(E2_DIR, "results", "experiment2_policy_hidden.json")
A0_MODELS = ("e2_dir-frontier-qwen", "e2_dir-deepseek-v4-flash", "e2_dir-claude-haiku-4-5", "e2_dir-gemma-4-31b", "e2_dir-phi-4")
O_MODELS = ("e2_dir-frontier-anthropic", "e2_dir-frontier-openai", "e2_dir-frontier-qwen", "e2_dir-frontier-x-ai")
ROLES3 = ("steward", "neutral", "devotion")
ROLES5 = ("steward", "guardrail", "empathetic", "neutral", "devotion")
HALF_NAME = {"NOT": "inappropriate", "ENT": "appropriate"}
A0_EPISODES, O_EPISODES = 300, 500


def load_run(e2_dir, name, roster_row, expected, arms=None):
    path = os.path.join(e2_dir, "runs", name)
    rows, info = RR.read_rows(path)
    rows, dup = RR.first_occurrences(rows, RR.episode_key)
    if arms is not None:
        rows = [r for r in rows if r["arm"] in arms]
    ex = RR.excluded(rows, **RR._check_args(roster_row))
    kept = RR.retained(rows, ex)
    rec = {"rows": len(rows), "expected": expected, "complete": len(rows) >= expected, "sha256": info["sha256"],
           "duplicates_dropped": dup, "partial_last_line": info["partial_last_line"],
           "model_excluded": ex["model_excluded"], "cells_excluded": ex["parse_cells_excluded"],
           "mismatch_call_share": ex["mismatch_call_share"]}
    return kept, rec


def _pct(x, n):
    return 100.0 * x / n if n else None


def visibility(a0, a):
    out = {"roles": {}}
    for role in ROLES3:
        for half, name in HALF_NAME.items():
            p = RR.paired(RR.arm_rows(a0, role, half), RR.arm_rows(a, role, half))
            out["roles"].setdefault(role, {})[name] = {
                "pairs": p["pairs"], "hidden": _pct(p["x1"], p["pairs"]), "shown": _pct(p["x0"], p["pairs"]),
                "hidden_minus_shown": p["diff"]}
    gh = RR.paired(RR.arm_rows(a0, "devotion", "NOT"), RR.arm_rows(a0, "steward", "NOT"))
    gs = RR.paired(RR.arm_rows(a, "devotion", "NOT"), RR.arm_rows(a, "steward", "NOT"))
    out["devotion_minus_steward"] = {
        "hidden": gh["diff"], "hidden_pairs": gh["pairs"], "shown": gs["diff"], "shown_pairs": gs["pairs"],
        "change": None if gh["diff"] is None or gs["diff"] is None else gh["diff"] - gs["diff"]}
    return out


def across(values):
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return {"mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs), "range": [min(xs), max(xs)], "k": len(xs)}


def summarise_visibility(per_model):
    s = {"roles": {}}
    for role in ROLES3:
        for name in HALF_NAME.values():
            s["roles"].setdefault(role, {})[name] = {
                k: across([m["roles"][role][name][k] for m in per_model.values()])
                for k in ("hidden", "shown", "hidden_minus_shown")}
    s["devotion_minus_steward"] = {k: across([m["devotion_minus_steward"][k] for m in per_model.values()])
                                   for k in ("hidden", "shown", "change")}
    s["role_span"] = {}
    for name in HALF_NAME.values():
        means = [s["roles"][r][name]["hidden_minus_shown"]["mean"] for r in ROLES3
                 if s["roles"][r][name]["hidden_minus_shown"]]
        s["role_span"][name] = {"hidden_minus_shown": [min(means), max(means)] if means else None}
    return s


def o_rates(rows):
    return {a: RR._rate_of([r for r in rows if r["arm"] == a], "any_violation", "engine") for a in ROLES5}


def panel_rates(e2_dir):
    off = RR._load_json(os.path.join(e2_dir, "results", "experiment1_reanalysis.json"))
    me = off["role_rates_original"]["model_equal_mean"]
    return {a: {"mean": me[a]["harm_rate_pct"], "ci": me[a]["harm_ci95"], "k": me[a]["k"]} for a in ROLES5}


def build(e2_dir=None):
    e2_dir = e2_dir or E2_DIR
    roster = {m["key"]: m for m in RR._load_json(os.path.join(e2_dir, "model_roster.json"))["models"]}
    a0_rec, o_rec, lines = {}, {}, []
    vis = {}
    for k in A0_MODELS:
        a0, rec = load_run(e2_dir, "runA0_%s.jsonl" % k, roster[k], A0_EPISODES)
        a, _ = load_run(e2_dir, "runA_%s.jsonl" % k, roster[k], 800, arms=ROLES3)
        a0_rec[k] = rec
        if not rec["complete"] or rec["model_excluded"]:
            lines.append("A0 %s left out: %s" % (k, "excluded" if rec["model_excluded"] else
                                                 "%d of %d rows" % (rec["rows"], A0_EPISODES)))
            continue
        vis[k] = visibility(a0, a)
    o = {}
    for k in O_MODELS:
        rows, rec = load_run(e2_dir, "runO_%s.jsonl" % k, roster[k], O_EPISODES)
        o_rec[k] = rec
        if not rec["complete"] or rec["model_excluded"]:
            lines.append("O %s incomplete or excluded: %d of %d rows" % (k, rec["rows"], O_EPISODES))
        if rows:
            o[k] = {"complete": rec["complete"] and not rec["model_excluded"], "rates": o_rates(rows)}
    out = {"a0": {"models": list(vis), "records": a0_rec, "per_model": vis,
                  "across_models": summarise_visibility(vis) if vis else None},
           "o": {"models": list(o), "records": o_rec, "per_model": o, "original_panel": panel_rates(e2_dir)},
           "notes": lines}
    return out, lines


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out, lines = build()
    data = json.dumps(out, indent=2, sort_keys=True) + "\n"
    for l in lines:
        print(l)
    if "--check" in argv:
        same = os.path.exists(OUT) and open(OUT).read() == data
        print("experiment2_policy_hidden.json is %s" % ("current" if same else "STALE"))
        return 0 if same else 1
    with open(OUT, "w") as fh:
        fh.write(data)
    print("wrote %s (sha256 %s)" % (OUT, hashlib.sha256(data.encode()).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.exit(main())
