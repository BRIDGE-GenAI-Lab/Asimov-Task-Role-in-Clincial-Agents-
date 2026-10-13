#!/usr/bin/env python3
"""Summarises the repeated-sampling runs of Run E (temperature 0.7 replicates, temperature 0 and a deterministic check) for the steward and devotion roles: replicate-to-replicate SD, scenario agreement, ICC(1), the temperature 0 versus 0.7 difference and the contrast by replicate. It writes experiment2/results/experiment2_replicates.json."""
import hashlib
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import experiment2_primary as RR

E2_DIR = RR.E2_DIR
OUT = os.path.join(E2_DIR, "results", "experiment2_replicates.json")
ROLES = ("steward", "devotion")
EXPECTED = {"T07": 1000, "T0": 200, "T0det": 80}


def once(rows):
    return RR.first_occurrences(rows, RR.episode_key)[0]


def _matrix(rows, arm, half="NOT"):
    out = {}
    for r in rows:
        if r["arm"] != arm or (half is not None and M.condition(r) != half):
            continue
        out.setdefault(r["sid"], {})[r["rep"]] = RR.y(r, RR.PRIMARY)
    return out


def _complete(rows, arm, half="NOT", reps=None):
    mat = _matrix(rows, arm, half)
    reps = sorted({k for v in mat.values() for k in v}) if reps is None else list(reps)
    return {s: [int(v[k]) for k in reps] for s, v in sorted(mat.items())
            if all(v.get(k) is not None for k in reps)}, reps


def rates_by_rep(rows, arm, half="NOT"):
    mat = _matrix(rows, arm, half)
    out = {}
    for k in sorted({k for v in mat.values() for k in v}):
        vals = [v[k] for v in mat.values() if v.get(k) is not None]
        out[k] = 100.0 * sum(vals) / len(vals) if vals else None
    return out


def sd_across_reps(rows, arm, half="NOT"):
    rates = [r for r in rates_by_rep(rows, arm, half).values() if r is not None]
    return statistics.stdev(rates) if len(rates) > 1 else None


def scenario_agreement(rows, arm, half="NOT", with_n=False):
    comp, _ = _complete(rows, arm, half)
    if not comp:
        return (0, None) if with_n else None
    a = M.agreement(list(comp.values()))
    return (len(comp), a) if with_n else a


def icc(rows, arm, half="NOT"):
    comp, _ = _complete(rows, arm, half)
    return M.icc1(list(comp.values())) if len(comp) > 1 and len(next(iter(comp.values()))) > 1 else None


def t0_minus_t07(t07, t0, arm, half="NOT"):
    z = _matrix(t0, arm, half)
    s = _matrix(t07, arm, half)
    sids = sorted(x for x in z if z[x].get(0) is not None and x in s
                  and all(v is not None for v in s[x].values()) and s[x])
    if not sids:
        return None
    r0 = 100.0 * sum(z[x][0] for x in sids) / len(sids)
    r7 = 100.0 * statistics.mean(statistics.mean(s[x].values()) for x in sids)
    return {"t0": r0, "t07": r7, "diff": r0 - r7, "n": len(sids)}


def determinism(rows, arm, half=None):
    comp, reps = _complete(rows, arm, half, reps=(0, 1))
    return M.agreement(list(comp.values())) if comp else None


def e_models():
    runs = RR._load_json(os.path.join(E2_DIR, "run_matrix.json"))["runs"]
    return [r["models"] for r in runs if r["run"] == "E"][0]


def operational_new(rows, seeded):
    return RR.operational([r for r in rows if r.get("rep") != 0] if seeded else rows)


def _read(path):
    rows, info = RR.read_rows(path)
    return once(rows), info


def _per_model(key, roster):
    T = {}
    sha = {}
    for tag, name in (("T07", "runE_T07_%s.jsonl"), ("T0", "runE_T0_%s.jsonl"), ("T0det", "runE_T0det_%s.jsonl")):
        path = os.path.join(E2_DIR, "runs", name % key)
        rows, info = _read(path)
        if rows:
            ex = RR.excluded(rows, **RR._check_args(roster[key]))
            rows = [] if ex["model_excluded"] else RR.retained(rows, ex)
        T[tag] = rows
        sha["runs/" + os.path.basename(path)] = info["sha256"]
    return T, sha


def build():
    roster = {m["key"]: m for m in RR._load_json(os.path.join(E2_DIR, "model_roster.json"))["models"]
              if m["status"] == "locked"}
    keys = e_models()
    out = {"models": {}}
    for k in keys:
        T, _ = _per_model(k, roster)
        blk = {"operational": {t: operational_new(v, seeded=(t == "T07")) for t, v in T.items()},
               "episodes": {t: len(v) for t, v in T.items()}, "expected": EXPECTED,
               "complete": all(len(T[t]) >= EXPECTED[t] for t in EXPECTED)}
        for arm in ROLES:
            blk[arm] = {"rate_by_rep": rates_by_rep(T["T07"], arm),
                        "sd_across_reps": sd_across_reps(T["T07"], arm),
                        "agreement": dict(zip(("n", "share"), scenario_agreement(T["T07"], arm, with_n=True))),
                        "icc1": icc(T["T07"], arm), "t0_minus_t07": t0_minus_t07(T["T07"], T["T0"], arm),
                        "determinism": determinism(T["T0det"], arm)}
        a, b = rates_by_rep(T["T07"], "devotion"), rates_by_rep(T["T07"], "steward")
        blk["contrast_by_rep"] = {k2: a[k2] - b[k2] for k2 in a if a.get(k2) is not None and b.get(k2) is not None}
        out["models"][k] = blk
    done = {k: b for k, b in out["models"].items() if b["complete"]}
    out["complete_models"] = sorted(done)
    out["incomplete_models"] = sorted(set(out["models"]) - set(done))
    if len(done) >= 2:
        P = {}
        for arm in ROLES:
            def col(f):
                return [f(b[arm]) for b in done.values() if f(b[arm]) is not None]
            P[arm] = {"k": len(done),
                      "sd_across_reps": {"mean": statistics.mean(col(lambda x: x["sd_across_reps"])),
                                         "range": [min(col(lambda x: x["sd_across_reps"])),
                                                   max(col(lambda x: x["sd_across_reps"]))]},
                      "agreement": {"mean": 100 * statistics.mean(col(lambda x: x["agreement"]["share"])),
                                    "range": [100 * min(col(lambda x: x["agreement"]["share"])),
                                              100 * max(col(lambda x: x["agreement"]["share"]))]},
                      "icc1": {"mean": statistics.mean(col(lambda x: x["icc1"])),
                               "range": [min(col(lambda x: x["icc1"])), max(col(lambda x: x["icc1"]))]},
                      "determinism": {"mean": 100 * statistics.mean(col(lambda x: x["determinism"])),
                                      "range": [100 * min(col(lambda x: x["determinism"])),
                                                100 * max(col(lambda x: x["determinism"]))]}}
            d = col(lambda x: x["t0_minus_t07"]["diff"] if x["t0_minus_t07"] else None)
            m, lo, hi = M.mean_t_ci(d)
            P[arm]["t0_minus_t07"] = {"mean": m, "ci": [lo, hi], "k": len(d)}
        reps = sorted({r for b in done.values() for r in b["contrast_by_rep"]})
        per_rep = [statistics.mean(b["contrast_by_rep"][r] for b in done.values()) for r in reps]
        P["contrast"] = {"per_rep_model_equal": per_rep, "mean": statistics.mean(per_rep),
                         "sd_across_reps": statistics.stdev(per_rep), "min": min(per_rep), "max": max(per_rep),
                         "per_model_sd": {k: statistics.stdev(list(b["contrast_by_rep"].values()))
                                          for k, b in done.items()}}
        out["panel"] = P
    return out


def main():
    out = build()
    text = json.dumps(out, indent=1, allow_nan=False, ensure_ascii=False)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("wrote", OUT, hashlib.sha256(text.encode()).hexdigest()[:12])
    print("complete:", out["complete_models"], "| incomplete:", out["incomplete_models"])
    for k, b in out["models"].items():
        print(k, b["episodes"])


if __name__ == "__main__":
    main()
