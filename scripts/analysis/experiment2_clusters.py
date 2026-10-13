#!/usr/bin/env python3
"""Computes paired variances, paired random-effects estimates and a developer-cluster bootstrap (10,000 resamples) for the Run A contrasts: scope, ladder, neutral versus steward, devotion versus neutral and precedence. It reads the Run A logs and experiment2_primary.json and writes experiment2/results/experiment2_clusters.json."""
import hashlib
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import experiment2_primary as RR

E2_DIR = RR.E2_DIR
OUT = os.path.join(E2_DIR, "results", "experiment2_clusters.json")
B_DEV, SEED = 10000, 20261007
CONTRASTS = (("scope", "sx_patient_plain", "sx_ward_plain", "primary_scope", "primary_scope_bootstrap"),
             ("ladder", "devotion", "steward", None, "ladder_bootstrap"),
             ("neutral_steward", "neutral", "steward", None, None),
             ("devotion_neutral", "devotion", "neutral", None, None),
             ("precedence", "devotion_precedence", "devotion", None, None))


def indep_var(pairs):
    n = len(pairs)
    p1, p0 = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    return p1 * (1 - p1) / n + p0 * (1 - p0) / n


def paired_var(pairs):
    d = [a - b for a, b in pairs]
    return statistics.variance(d) / len(d) if len(d) > 1 else 0.0


def model_row(pairs):
    n = len(pairs)
    d = 100.0 * sum(a - b for a, b in pairs) / n
    v = paired_var(pairs)
    return {"pairs": n, "diff": d, "var": v, "var_indep": indep_var(pairs), "no_information": v == 0.0}


def paired_random_effects(rows, floor=False):
    pos = [r["var"] for r in rows.values() if r["var"] > 0]
    if len(pos) < 2:
        return None
    lo = min(pos)
    use = {m: r for m, r in rows.items() if r["var"] > 0 or floor}
    ys = [r["diff"] for r in use.values()]
    vs = [1e4 * (r["var"] if r["var"] > 0 else lo) for r in use.values()]
    re = M.re_summary(ys, vs)
    return {"k": len(use), "k_no_information": sum(1 for r in rows.values() if r["var"] == 0),
            "models": sorted(use), "mu": re["mu"], "ci_hk": list(re["ci_hk"]),
            "pi": list(re["pi"]) if re["pi"] else None, "tau2": re["tau2"], "I2": re["I2"], "floor": floor}


def variance_ratio(rows):
    r = sorted(x["var"] / x["var_indep"] for x in rows.values() if x["var"] > 0 and x["var_indep"] > 0)
    return {"k": len(r), "min": r[0], "median": statistics.median(r), "max": r[-1]}


def developer_of(roster):
    return {k: m["slug"].split("/")[0] for k, m in roster.items()}


def developer_bootstrap(pairs_by_model, dev, B=B_DEV, seed=SEED):
    rng = random.Random(seed)
    by_dev = {}
    for m in sorted(pairs_by_model):
        by_dev.setdefault(dev[m], []).append(m)
    devs = sorted(by_dev)
    stats = []
    for _ in range(B):
        diffs = []
        for d in (rng.choice(devs) for _ in devs):
            for m in by_dev[d]:
                p = pairs_by_model[m]
                smp = [rng.choice(p) for _ in p]
                diffs.append(100.0 * sum(a - b for a, b in smp) / len(smp))
        stats.append(sum(diffs) / len(diffs))
    stats.sort()
    return stats[int(0.025 * B)], stats[int(0.975 * B) - 1]


def load_retained(e2_dir, run="A"):
    roster = {m["key"]: m for m in RR._load_json(os.path.join(e2_dir, "model_roster.json"))["models"]
              if m["status"] == "locked"}
    models_of, _ = RR._matrix(e2_dir)
    out, sha = {}, {}
    for k in models_of[run]:
        path = os.path.join(e2_dir, "runs", RR.FILES[run] % k)
        rows, info = RR.read_rows(path)
        rows, _ = RR.first_occurrences(rows, RR.KEYS[run])
        if not rows:
            continue
        ex = RR.excluded(rows, **RR._check_args(roster[k]))
        if ex["model_excluded"]:
            continue
        out[k] = RR.retained(rows, ex)
        sha["runs/" + os.path.basename(path)] = info["sha256"]
    return out, roster, sha


def build(e2_dir=E2_DIR, B=B_DEV):
    A, roster, sha = load_retained(e2_dir)
    RJ_path = os.path.join(e2_dir, "results", "experiment2_primary.json")
    RJ = json.load(open(RJ_path))
    dev = developer_of({k: roster[k] for k in A})
    out = {"seed": SEED, "B": B,
           "developers": {d: sorted(k for k in dev if dev[k] == d) for d in sorted(set(dev.values()))},
           "n_developers": len(set(dev.values())), "contrasts": {}}
    for name, a1, a0, key, bkey in CONTRASTS:
        pairs = {}
        for m, rows in sorted(A.items()):
            p = RR.paired(RR.arm_rows(rows, a1, "NOT"), RR.arm_rows(rows, a0, "NOT"))
            if p["pairs"]:
                pairs[m] = list(p["values"])
        rows = {m: model_row(p) for m, p in pairs.items()}
        mean, lo, hi = M.mean_t_ci([r["diff"] for r in rows.values()])
        block = {"arms": [a1, a0], "k": len(rows), "model_equal": {"mean": mean, "ci": [lo, hi]},
                 "per_model": {m: r["diff"] for m, r in rows.items()},
                 "developer_bootstrap": {"ci": list(developer_bootstrap(pairs, dev, B)), "B": B, "seed": SEED},
                 "re_paired": paired_random_effects(rows), "re_paired_floor": paired_random_effects(rows, True),
                 "paired_variance_ratio": variance_ratio(rows)}
        ref = {"primary_scope": "primary_scope", "ladder": ("ladder", "all"), "neutral_steward":
               ("contrasts", "neutral_minus_steward"), "devotion_neutral": ("contrasts", "devotion_minus_neutral"),
               "precedence": ("contrasts", "precedence_NOT")}.get(name, name)
        node = RJ
        for step in ([ref] if isinstance(ref, str) else ref):
            node = node.get(step) if node else None
        if node is None and name == "scope":
            node = RJ["primary_scope"]
        if node:
            assert abs(node["model_equal"]["mean"] - mean) < 1e-9, (name, node["model_equal"]["mean"], mean)
            block["reference_experiment2_primary"] = {"model_equal": node["model_equal"],
                                                "random_effects_independent": node.get("random_effects")}
        if bkey and RJ.get(bkey):
            block["two_stage_bootstrap"] = {"ci": RJ[bkey]["ci"], "B": RJ[bkey]["B"], "seed": RJ[bkey]["seed"]}
        out["contrasts"][name] = block
    return out


def main():
    out = build()
    text = json.dumps(out, indent=1, allow_nan=False, ensure_ascii=False)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("wrote", OUT, hashlib.sha256(text.encode()).hexdigest()[:12])
    for n, b in out["contrasts"].items():
        rp = b["re_paired"]
        print("%-17s k=%d ME %.1f (%.1f to %.1f) | dev-boot %.1f to %.1f | RE-paired %s" % (
            n, b["k"], b["model_equal"]["mean"], *b["model_equal"]["ci"], *b["developer_bootstrap"]["ci"], "k=%d mu %.1f (%.1f to %.1f) PI %s" % (
                rp["k"], rp["mu"], *rp["ci_hk"], "%.1f to %.1f" % tuple(rp["pi"]) if rp["pi"] else "n/a") if rp else "n/a"))


if __name__ == "__main__":
    main()
