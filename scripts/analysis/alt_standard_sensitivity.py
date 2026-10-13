"""Replays the recorded moves of the DeepSeek-V4 Flash ladder log under five allocation standards through the ward engine and scores harm under each. It reports steward and devotion violation rates with Wilson intervals and their gap per standard, and writes experiment1/results/alt_standard_sensitivity.json."""
import json, pathlib, math
import sys; sys.path.insert(0, "code")
from collections import defaultdict
from dataclasses import replace
from scarcity_ward.hospital.ward_actions import WardAction, WardMove
from scarcity_ward.hospital.ward_env import WardState, apply_step
from scarcity_ward.hospital.ward_harm import collateral_harm
from scarcity_ward.hospital.ward_scenarios import load_ward_scenarios

STANDARDS = ["sofa_survival","sickest_first","first_come","lottery","strict_exclusion"]
SRC = "output/intermediate/ward_ds_ladder_v2.jsonl"
OUT = pathlib.Path("experiment1/results/alt_standard_sensitivity.json")

def wil(k, n):
    if not n:
        return (0.0, 0.0, 0.0)
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * p, 1), round(100 * (c - h), 1), round(100 * (c + h), 1))

scn = {ws.scenario.sid: ws for ws in load_ward_scenarios("data/scenarios/ward_scenarios.jsonl")}
rows = [json.loads(l) for l in open(SRC) if l.strip()]

res = defaultdict(lambda: defaultdict(list))
skipped = 0

for r in rows:
    if r["arm"] not in ("steward", "devotion") or r.get("oversight") != "blind":
        continue
    if r["sid"] not in scn:
        skipped += 1
        continue
    base = scn[r["sid"]]
    by_step = defaultdict(list)
    for mv in r["transcript"]:
        by_step[mv["step"]].append(
            WardMove(mv["pid"], mv["step"], mv.get("rationale", ""),
                     WardAction(mv["verb"], mv.get("target")))
        )
    for std in STANDARDS:
        ws2 = replace(base,
                      seed=f"{r['sid']}|{r.get('rep', 0)}",
                      scenario=replace(base.scenario, protocol=std))
        st = WardState()
        for s in sorted(by_step):
            apply_step(ws2, st, by_step[s])
        res[std][r["arm"]].append(bool(collateral_harm(ws2, st)))

if skipped:
    print(f"WARNING: {skipped} rows skipped (sid not in scenarios file)")

out = {"reference_model": "deepseek-v4-flash", "per_standard": {}}
for std in STANDARDS:
    d = {}
    for arm in ("steward", "devotion"):
        k = sum(res[std][arm])
        n = len(res[std][arm])
        pt, lo, hi = wil(k, n)
        d[arm] = {"chr": pt, "ci": [lo, hi], "n": n}
    d["gap_pp"] = round(d["devotion"]["chr"] - d["steward"]["chr"], 1)
    out["per_standard"][std] = d

OUT.write_text(json.dumps(out, indent=2))
print("wrote", OUT)
print()
print(f"  {'standard':18s}  {'steward':>10s}  {'devotion':>10s}  {'gap':>8s}  n")
for std, d in out["per_standard"].items():
    print(f"  {std:18s}  {d['steward']['chr']:>10.1f}  {d['devotion']['chr']:>10.1f}  {d['gap_pp']:>+8.1f}  {d['devotion']['n']}")
