#!/usr/bin/env python3
"""Summarises the ladder and comprehension-probe logs of four additional models (MedGemma-27B, MedGemma-1.5-4B, Gemini 3.1 Pro, Claude Opus 4.8): violation rate per role with Wilson intervals, steward-to-devotion gap and probe outcomes. It writes experiment1/results/replication_panel.json."""
import json, math, os, collections

Z = 1.959964
ARMS = ["steward", "guardrail", "empathetic", "neutral", "devotion"]
FROZEN = "experiment1/results/replication_panel.json"
LADDER = "output/intermediate/ward_%s_icu.jsonl"
PROBE = "output/intermediate/comprehension_probe_%s.jsonl"

PANEL = [
    ("medgemma27b_o2",     "MedGemma-27B (text)", "self-hosted O2, pinned Q8", True),
    ("medgemma_1.5_4b_o2", "MedGemma-1.5-4B",     "self-hosted O2, pinned Q8", True),
    ("gemini-3.1-pro",     "Gemini 3.1 Pro",      "agy subscription CLI", False),
    ("claude-opus-4.8-cc", "Claude Opus 4.8",     "Claude Code CLI subscription", False),
]


def wilson(k, n):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return [round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1)]


def per_model(tag):
    lf = LADDER % tag
    if not os.path.exists(lf):
        return None
    rows = [json.loads(l) for l in open(lf) if l.strip()]
    if not rows:
        return None
    byarm = collections.defaultdict(lambda: [0, 0])
    pf = 0
    for r in rows:
        byarm[r.get("arm")][1] += 1
        if r.get("harm"):
            byarm[r.get("arm")][0] += 1
        pf += r.get("parse_failures", 0)
    if min((byarm[a][1] for a in ARMS), default=0) < 20:
        return None
    per_arm = {}
    for a in ARMS:
        h, n = byarm[a]
        per_arm[a] = ({"apvr_pct": round(100 * h / n, 1), "harm": h, "n": n, "ci95": wilson(h, n)}
                      if n else {"apvr_pct": None, "n": 0})
    out = {"n_episodes": len(rows), "n_scenarios": len(set(r.get("sid") for r in rows)),
           "parse_failures": pf, "per_arm": per_arm}
    sh, sn = byarm["steward"]
    dh, dn = byarm["devotion"]
    nh, nn = byarm["neutral"]
    if sn and dn:
        s, d = 100 * sh / sn, 100 * dh / dn
        out["steward_pct"] = round(s, 1)
        out["devotion_pct"] = round(d, 1)
        out["gap_pp"] = round(d - s, 1)
        if nn:
            out["neutral_pct"] = round(100 * nh / nn, 1)
            if (d - s):
                out["neutral_frac_of_gap_pct"] = round(100 * (100 * nh / nn - s) / (d - s))
    prf = PROBE % tag
    if os.path.exists(prf):
        pr = [json.loads(l) for l in open(prf) if l.strip()]
        correct = [p for p in pr if p.get("correct")]
        ggc = [p for p in correct if p.get("harm")]
        out["comprehension"] = {
            "n": len(pr),
            "named_rightful_n": len(correct),
            "named_rightful_pct": round(100 * len(correct) / len(pr), 1) if pr else None,
            "grab_given_correct_n": len(ggc),
            "grab_given_correct_denom": len(correct),
            "grab_given_correct_pct": round(100 * len(ggc) / len(correct), 1) if correct else None,
            "grab_given_correct_ci95": wilson(len(ggc), len(correct)),
        }
    return out


def main():
    panel = {}
    for tag, label, route, repro in PANEL:
        r = per_model(tag)
        if r is None:
            continue
        r.update(label=label, route=route, reproducible=repro)
        panel[tag] = r
    out = {
        "design": ("Prospective contemporary-frontier + domain-specialized replication panel. "
                   "50 ICU scenarios x 5-arm instruction ladder + pre-action comprehension probe per model. "
                   "Reported separately from and does not alter the frozen primary 20-model ladder "
                   "(ladder_pooled.json) or 12-model matched-syntax 2x2 (scope_2x2_stats.json)."),
        "models": panel,
        "n_models": len(panel),
    }
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    with open(FROZEN, "w") as fh:
        json.dump(out, fh, indent=1)
    for tag, m in panel.items():
        c = m.get("comprehension", {})
        print(f"{m['label']:22s} steward {m.get('steward_pct','-')}% -> devotion {m.get('devotion_pct','-')}% "
              f"(gap {m.get('gap_pp','-')}pp; neutral {m.get('neutral_pct','-')}%) | "
              f"grab|correct {c.get('grab_given_correct_pct','-')}% ({c.get('grab_given_correct_n','-')}/{c.get('grab_given_correct_denom','-')}) | "
              f"named-rightful {c.get('named_rightful_pct','-')}% | parse_fail {m['parse_failures']}")
    print(f"wrote {FROZEN} ({len(panel)} models)")


if __name__ == "__main__":
    main()
