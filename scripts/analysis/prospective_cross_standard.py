import json, pathlib, glob, math
from collections import defaultdict

STANDARDS = ["sickest_first", "lottery"]
REF = "gemma3-12b-qat"
EXCLUDE = {"owl-alpha", "nemotron-ultra-550b"}
OUT = pathlib.Path("experiment1/results/prospective_cross_standard.json")


def wil(k, n):
    if not n:
        return (0.0, 0.0, 0.0)
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * p, 1), round(100 * (c - h), 1), round(100 * (c + h), 1))


def arms_block(harms_by_arm):
    d = {}
    for arm in ("steward", "devotion"):
        h = harms_by_arm.get(arm, [])
        k, n = sum(h), len(h)
        pt, lo, hi = wil(k, n)
        d[arm] = {"chr": pt, "ci": [lo, hi], "n": n}
    d["gap_pp"] = round(d["devotion"]["chr"] - d["steward"]["chr"], 1)
    return d


out = {
    "note": ("agents PROSPECTIVELY instructed under each standard (rule stated), scored "
             "against that standard. Gemma-3 12B is the primary reference (n=100); the "
             "frontier panel is descriptive, at smaller n."),
    "reference_model": REF,
    "per_standard": {},
}

for std in STANDARDS:
    files = sorted(glob.glob(f"output/intermediate/prospective_{std}.jsonl") +
                   glob.glob(f"output/intermediate/prospective_{std}_*.jsonl"))
    by_model = defaultdict(lambda: defaultdict(list))
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("arm") in ("steward", "devotion") and r.get("model") not in EXCLUDE:
                by_model[r.get("model")][r["arm"]].append(bool(r["harm"]))
    per_model = {m: arms_block(arms) for m, arms in by_model.items()}
    pooled = defaultdict(list)
    for m, arms in by_model.items():
        if m == REF:
            continue
        for arm in ("steward", "devotion"):
            pooled[arm] += arms.get(arm, [])
    out["per_standard"][std] = {
        "reference": per_model.get(REF),
        "per_model": per_model,
        "frontier_pooled": arms_block(pooled) if pooled else None,
        "n_models": len(per_model),
    }

OUT.write_text(json.dumps(out, indent=2))
print("wrote", OUT)
print()
for std in STANDARDS:
    blk = out["per_standard"][std]
    print(f"=== {std}  ({blk['n_models']} models) ===")
    print(f"  {'model':28s}  {'steward':>16s}  {'devotion':>16s}  {'gap':>7s}")
    for m, d in sorted(blk["per_model"].items()):
        s, dv = d["steward"], d["devotion"]
        tag = " (REF)" if m == REF else ""
        print(f"  {m+tag:28s}  {s['chr']:>6.1f}% (n={s['n']:>3d})  "
              f"{dv['chr']:>6.1f}% (n={dv['n']:>3d})  {d['gap_pp']:>+6.1f}")
    fp = blk["frontier_pooled"]
    if fp:
        print(f"  {'FRONTIER POOLED':28s}  {fp['steward']['chr']:>6.1f}% (n={fp['steward']['n']:>3d})  "
              f"{fp['devotion']['chr']:>6.1f}% (n={fp['devotion']['n']:>3d})  {fp['gap_pp']:>+6.1f}")
    print()
