import json, collections, pathlib

SRC = pathlib.Path("output/intermediate/rationale_codes.jsonl")
OUT = pathlib.Path("experiment1/results/rationale_by_arm.json")

rows = [json.loads(l) for l in SRC.open() if l.strip()]
by_arm = collections.defaultdict(collections.Counter)
harm_by_arm = collections.Counter()
n_by_arm = collections.Counter()
for r in rows:
    arm = r.get("arm")
    code = r.get("primary") or r.get("code") or "OTHER_EMERGENT"
    if arm is None:
        continue
    by_arm[arm][code] += 1
    n_by_arm[arm] += 1
    if r.get("harm"):
        harm_by_arm[arm] += 1

out = {"n_total": len(rows), "arms": {}}
for arm in by_arm:
    n = n_by_arm[arm]
    out["arms"][arm] = {
        "n": n,
        "harm_rate": round(harm_by_arm[arm] / n, 4) if n else None,
        "code_pct": {c: round(100 * k / n, 1) for c, k in by_arm[arm].most_common()},
        "code_n": dict(by_arm[arm].most_common()),
    }
OUT.write_text(json.dumps(out, indent=2))
print(f"wrote {OUT}  arms={list(out['arms'])}  n_total={out['n_total']}")
