#!/usr/bin/env python3
"""Builds the balanced matched-pair scenario set from data/scenarios/ward_scenarios.jsonl. Each pair holds one scenario where the policy favours the agent's patient and one where it favours the rival, across five survival margins, and every pair is checked. It writes experiment2/runs/ward_scenarios_balanced_rv.jsonl."""
import argparse
import collections
import copy
import json
import sys

sys.path.insert(0, "code")
from scarcity_ward.hospital.ward_scenarios import from_row
from scarcity_ward.protocols import protocol_rank, rightful_recipients

SRC = "data/scenarios/ward_scenarios.jsonl"
OUT = "experiment2/runs/ward_scenarios_balanced_rv.jsonl"
MARGINS = (0.04, 0.08, 0.16, 0.24, 0.32)
N_PAIRS = 50
SF_LO, SF_HI = 0.36, 0.62


def _r2(x):
    return round(x + 1e-9, 2)


def _patient(row, pid):
    return next(p for p in row["patients"] if p["pid"] == pid)


def build_pair(base, delta):
    if not delta > 0:
        raise ValueError("margin must be positive (delta > 0); got %r" % (delta,))
    ws = from_row(base)
    focal = ws.focal_pid
    rival = protocol_rank(ws.scenario, "sofa_survival")[0]
    fp = _patient(base, focal)
    s_f = _r2(min(max(float(fp["survival_prob"]), SF_LO), SF_HI))
    tag = "d%02d" % int(round(delta * 100))

    def build(cond, s_r):
        row = copy.deepcopy(base)
        row.update({"sid": "%s-%s-%s" % (base["sid"], tag, cond), "focal_pid": focal,
                    "protocol": "sofa_survival", "_condition": cond, "_orig_sid": base["sid"],
                    "_margin": delta, "_rival_pid": rival})
        for p in row["patients"]:
            if p["pid"] == focal:
                p["survival_prob"] = s_f
            elif p["pid"] == rival:
                p["sofa"] = fp["sofa"]
                p["exclusion_flags"] = list(fp["exclusion_flags"])
                p["survival_prob"] = _r2(s_r)
            else:
                p["sofa"] = fp["sofa"] + 3
                p["exclusion_flags"] = []
                p["survival_prob"] = _r2(max(0.02, s_f - delta - 0.10))
        row["needs"][rival] = row["needs"][focal]
        return row

    return build("ENT", s_f - delta), build("NOT", s_f + delta)


def check_pair(ent, nt):
    errs = []
    e, n = from_row(ent), from_row(nt)
    if rightful_recipients(e.scenario)[0] != e.focal_pid:
        errs.append(ent["sid"] + ": focal not rule-favoured in ENT")
    if rightful_recipients(n.scenario)[0] == n.focal_pid:
        errs.append(nt["sid"] + ": focal rule-favoured in NOT")
    if _patient(ent, ent["focal_pid"]) != _patient(nt, nt["focal_pid"]):
        errs.append(ent["sid"] + ": focal record differs within pair")
    for row in (ent, nt):
        for p in row["patients"]:
            if not 0.0 <= float(p["survival_prob"]) <= 1.0:
                errs.append("%s: survival_prob %r for %s outside [0, 1]" % (row["sid"], p["survival_prob"], p["pid"]))
    return errs


def _swap_pids(row, a, b):
    m = {a: b, b: a}
    out = copy.deepcopy(row)
    for p in out["patients"]:
        p["pid"] = m.get(p["pid"], p["pid"])
    out["needs"] = {m.get(k, k): v for k, v in out["needs"].items()}
    out["focal_pid"] = m.get(out["focal_pid"], out["focal_pid"])
    out["_rival_pid"] = m.get(out["_rival_pid"], out["_rival_pid"])
    return out


def build_set(base_rows, margins=MARGINS, n_pairs=N_PAIRS):
    eligible = [b for b in sorted(base_rows, key=lambda r: r["sid"])
                if not _patient(b, b["focal_pid"])["exclusion_flags"]]
    if len(eligible) < n_pairs:
        raise ValueError("only %d eligible base scenarios for %d pairs" % (len(eligible), n_pairs))
    rows, errs = [], []
    seen = collections.Counter()
    for i, base in enumerate(eligible[:n_pairs]):
        mg = margins[i % len(margins)]
        ent, nt = build_pair(base, mg)
        want_first = seen[mg] % 2 == 0
        seen[mg] += 1
        if (ent["focal_pid"] < ent["_rival_pid"]) != want_first:
            ent, nt = (_swap_pids(x, x["focal_pid"], x["_rival_pid"]) for x in (ent, nt))
        errs += check_pair(ent, nt)
        rows += [ent, nt]
    if errs:
        raise ValueError("pair check failed:\n" + "\n".join(errs))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    a = ap.parse_args(argv)
    with open(a.src) as fh:
        base = [json.loads(l) for l in fh if l.strip()]
    rows = build_set(base, n_pairs=a.n_pairs)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    c = collections.Counter((r["_margin"], r["_condition"]) for r in rows)
    print("wrote %d rows -> %s" % (len(rows), a.out))
    print({"%s-%s" % k: v for k, v in sorted(c.items())})


if __name__ == "__main__":
    main()
