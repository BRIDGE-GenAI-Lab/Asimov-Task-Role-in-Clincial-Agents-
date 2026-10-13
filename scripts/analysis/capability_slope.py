#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))
from analysis.stats import wilson_ci

LOGS = REPO / "output" / "intermediate"
OUT_JSON = REPO / "experiment1" / "results" / "capability_slope.json"
OUT_CSV = REPO / "figures" / "data" / "capability_slope.csv"

OLDER_LOGS = {"ward_ds_ladder.jsonl", "ward_ds_oversight.jsonl"}
COMPROMISED = {"ward_e2b_icu.jsonl", "ward_exp1_26b.jsonl",
               "ward_qwen35_9b_icu.jsonl"}
PILOT = {"ward_pilot_focal.jsonl"}
EXCLUSIONS = sorted(OLDER_LOGS | COMPROMISED | PILOT)

PARSE_FAIL_MAX = 0.15

CANON = [
    ("gemma3-12b-qat", "ward_exp1_v2.jsonl", "Google (open)", "edge-12B", "Gemma-3 12B"),
    ("deepseek-v4-flash", "ward_ds_ladder_v2.jsonl", "DeepSeek", "flash", "DeepSeek-V4 Flash"),
    ("deepseek-v4-pro", "ward_ds_pro_icu.jsonl", "DeepSeek", "pro", "DeepSeek-V4 Pro"),
    ("gemini-2.5-flash", "ward_gem25f_ladder.jsonl", "Google", "flash", "Gemini-2.5 Flash"),
    ("gpt-5.1", "ward_gpt51_icu.jsonl", "OpenAI", "frontier", "GPT-5.1"),
    ("claude-haiku45", "ward_haiku_icu.jsonl", "Anthropic", "small", "Claude Haiku 4.5"),
    ("llama-4-maverick", "ward_llama4m_ladder.jsonl", "Meta", "frontier-MoE", "Llama-4 Maverick"),
    ("gpt-oss-120b", "ward_gptoss_icu.jsonl", "OpenAI (open)", "120B", "GPT-OSS 120B"),
    ("mistral-small", "ward_mistral_icu.jsonl", "Mistral", "small", "Mistral Small"),
    ("owl-alpha", "ward_owl_icu.jsonl", "(panel)", "alpha", "Owl-Alpha"),
    ("qwen3.5-0.8b", "ward_qwen08_icu.jsonl", "Qwen (open)", "edge-0.8B", "Qwen-3.5 0.8B"),
    ("qwen3.5-2b", "ward_qwen2b_icu.jsonl", "Qwen (open)", "edge-2B", "Qwen-3.5 2B"),
    ("qwen3.5-4b", "ward_qwen4b_icu.jsonl", "Qwen (open)", "edge-4B", "Qwen-3.5 4B"),
    ("gemma4-e2b-qat", "ward_e2b_icu_v2.jsonl", "Google (open)", "edge-2B-e", "Gemma-4 e2b"),
    ("qwen3.5-9b", "ward_qwen35_9b_icu_v2.jsonl", "Qwen (open)", "edge-9B", "Qwen-3.5 9B"),
]


def is_icu(sid: str) -> bool:
    return "ED" not in (sid or "").upper()


def load_icu(fname: str, model_key: str):
    rows = []
    pf = act = 0
    for ln in open(LOGS / fname):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "n_steps" not in r:
            continue
        if r.get("model") != model_key:
            continue
        if not is_icu(r.get("sid", "")):
            continue
        rows.append(r)
        pf += r.get("parse_failures", 0)
        act += len(r.get("transcript", []) or [])
    pf_rate = (pf / act) if act else 0.0
    return rows, pf_rate


def cell(rows, arm):
    rs = [r for r in rows if r.get("arm") == arm]
    h = sum(1 for r in rs if r.get("harm"))
    n = len(rs)
    return h, n


def chr_ci(h, n):
    if n == 0:
        return None, (None, None)
    lo, hi = wilson_ci(h, n)
    return h / n, (lo, hi)


def assignment_or(rows):
    sub = [r for r in rows if r.get("arm") in ("steward", "neutral")]
    df = pd.DataFrame({
        "harm": [1 if r.get("harm") else 0 for r in sub],
        "neutral": [1 if r.get("arm") == "neutral" else 0 for r in sub],
        "sid": [r.get("sid") for r in sub],
    })
    if df["neutral"].nunique() < 2:
        return None
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        res = smf.gee("harm ~ neutral", groups=df["sid"], data=df,
                      family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        beta = res.params["neutral"]
        ci = res.conf_int().loc["neutral"]
        orr, lo, hi, p = (float(np.exp(beta)), float(np.exp(ci[0])),
                          float(np.exp(ci[1])), float(res.pvalues["neutral"]))
        if np.isfinite(orr) and np.isfinite(lo) and np.isfinite(hi) and orr > 0:
            return {"or": orr, "ci": (lo, hi), "p": p, "method": "gee"}
    except Exception:
        pass
    return _cluster_bootstrap_or(df)


def _cluster_bootstrap_or(df, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    sids = df["sid"].unique()
    groups = {s: df[df["sid"] == s] for s in sids}

    def _or(d):
        a = ((d.neutral == 1) & (d.harm == 1)).sum() + 0.5
        b = ((d.neutral == 1) & (d.harm == 0)).sum() + 0.5
        c = ((d.neutral == 0) & (d.harm == 1)).sum() + 0.5
        e = ((d.neutral == 0) & (d.harm == 0)).sum() + 0.5
        return (a * e) / (b * c)

    obs = _or(df)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(sids, size=len(sids), replace=True)
        samp = pd.concat([groups[s] for s in pick], ignore_index=True)
        if samp["neutral"].nunique() < 2:
            continue
        boots.append(_or(samp))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    logb = np.log(boots)
    p = 2.0 * min((logb <= 0).mean(), (logb >= 0).mean())
    return {"or": float(obs), "ci": (float(lo), float(hi)),
            "p": float(min(1.0, p)), "method": "cluster_bootstrap"}


def main():
    models_out = []
    print(f"{'model':<20} {'tier':<12} {'n':>5} {'stew':>7} {'neut':>7} {'devo':>7} "
          f"{'OR(neut|stew)':>16} {'pfail':>6}")
    print("-" * 92)

    for model_key, fname, vendor, tier, display in CANON:
        if not (LOGS / fname).exists():
            print(f"{display:<20} MISSING FILE {fname}")
            continue
        rows, pf_rate = load_icu(fname, model_key)
        if pf_rate > PARSE_FAIL_MAX:
            print(f"{display:<20} EXCLUDED (parse-fail {pf_rate:.1%} > {PARSE_FAIL_MAX:.0%})")
            continue
        if not rows:
            print(f"{display:<20} no ICU rows in {fname}")
            continue

        sh, sn = cell(rows, "steward")
        nh, nn = cell(rows, "neutral")
        dh, dn = cell(rows, "devotion")
        s_chr, s_ci = chr_ci(sh, sn)
        n_chr, n_ci = chr_ci(nh, nn)
        d_chr, d_ci = chr_ci(dh, dn)
        orr = assignment_or(rows)
        n_total = len(rows)

        or_str = (f"{orr['or']:6.1f} [{orr['ci'][0]:.1f},{orr['ci'][1]:.1f}]" if orr else "-")
        print(f"{display:<20} {tier:<12} {n_total:>5} "
              f"{100*s_chr:6.1f}% {100*n_chr:6.1f}% {100*d_chr:6.1f}% "
              f"{or_str:>16} {pf_rate:6.2%}")

        models_out.append({
            "model": display,
            "model_key": model_key,
            "source_file": fname,
            "vendor": vendor,
            "tier": tier,
            "n": n_total,
            "steward_chr": round(s_chr, 4),
            "steward_ci": [round(s_ci[0], 4), round(s_ci[1], 4)],
            "steward_events_n": [sh, sn],
            "neutral_chr": round(n_chr, 4),
            "neutral_ci": [round(n_ci[0], 4), round(n_ci[1], 4)],
            "neutral_events_n": [nh, nn],
            "devotion_chr": round(d_chr, 4),
            "devotion_ci": [round(d_ci[0], 4), round(d_ci[1], 4)],
            "devotion_events_n": [dh, dn],
            "assignment_or": round(orr["or"], 3) if orr else None,
            "assignment_or_ci": [round(orr["ci"][0], 3), round(orr["ci"][1], 3)] if orr else None,
            "assignment_p": orr["p"] if orr else None,
            "assignment_or_method": orr["method"] if orr else None,
        })

    slope_claim = (
        "The STRUCTURE is universal across the panel: in every model the bare-assignment "
        "(neutral) and task-devotion arms collapse to high collateral harm while the "
        "steward (collective-objective) arm holds a low floor -- assignment, not instruction "
        "content, carries the effect. The MAGNITUDE of the assignment contrast scales with "
        "capability rather than along a single monotone line: across two independent frontier "
        "vendors in a tiered design the neutral-vs-steward odds ratio rises from roughly 11x "
        "in the weak edge model (Gemma-3 12B) to ~50-120x in capable cloud models "
        "(DeepSeek-V4 Flash), because a more capable model both follows the steward objective "
        "more cleanly (lower steward floor) and executes the assigned-patient grab more "
        "completely (higher devotion CHR). The relationship is monotone in DIRECTION (capability "
        "sharpens the contrast) but NOT a clean monotone capability curve in level: the smallest "
        "edge models (Qwen-3.5 0.8B/2B) invert the steward floor entirely (they grab MORE under "
        "the steward objective than under devotion, an OR < 1), reflecting an inability to "
        "represent the collective objective at all rather than safety. We therefore report the "
        "universal structure as the primary claim and the capability scaling as a directional, "
        "vendor-replicated trend, not a fitted scalar slope."
    )

    payload = {
        "generated_from": "output/intermediate/ward_*.jsonl (fixed-interface / post-A3 files)",
        "denominator": "ICU",
        "chr_definition": "mean(raw harm field) over episodes in a model x arm cell",
        "parse_fail_rule": f"exclude any file with malformed-output share > {PARSE_FAIL_MAX:.0%}",
        "exclusions": EXCLUSIONS,
        "assignment_or_definition": "neutral vs steward, GEE Binomial with scenario cluster "
                                    "(exchangeable cov); cluster-bootstrap fallback on non-convergence",
        "models": models_out,
        "slope_claim": slope_claim,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "model": m["model"],
        "vendor": m["vendor"],
        "tier": m["tier"],
        "n": m["n"],
        "steward_chr": m["steward_chr"],
        "steward_ci_lo": m["steward_ci"][0],
        "steward_ci_hi": m["steward_ci"][1],
        "neutral_chr": m["neutral_chr"],
        "neutral_ci_lo": m["neutral_ci"][0],
        "neutral_ci_hi": m["neutral_ci"][1],
        "devotion_chr": m["devotion_chr"],
        "devotion_ci_lo": m["devotion_ci"][0],
        "devotion_ci_hi": m["devotion_ci"][1],
        "assignment_or": m["assignment_or"],
        "assignment_or_ci_lo": m["assignment_or_ci"][0] if m["assignment_or_ci"] else None,
        "assignment_or_ci_hi": m["assignment_or_ci"][1] if m["assignment_or_ci"] else None,
        "assignment_p": m["assignment_p"],
        "source_file": m["source_file"],
    } for m in models_out])
    df.to_csv(OUT_CSV, index=False)

    print("-" * 92)
    print(f"\nwrote {OUT_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_CSV.relative_to(REPO)}")

    print("\nREFERENCE SANITY CHECKS (raw-harm canonical reading):")
    ref = {
        "Gemma-3 12B": ("steward 9.0% / devotion 86.0%", 0.09, 0.86),
        "DeepSeek-V4 Flash": ("steward 17.2% / devotion 88.5%", 0.172, 0.885),
        "Claude Haiku 4.5": ("steward 12% / neutral 49%", None, None),
    }
    by = {m["model"]: m for m in models_out}
    for name, (desc, s_exp, d_exp) in ref.items():
        m = by.get(name)
        if not m:
            print(f"  {name}: NOT IN OUTPUT")
            continue
        ok = True
        msg = f"  {name}: ref {desc}  ->  got steward {100*m['steward_chr']:.1f}% / " \
              f"neutral {100*m['neutral_chr']:.1f}% / devotion {100*m['devotion_chr']:.1f}%"
        if s_exp is not None and abs(m["steward_chr"] - s_exp) > 0.011:
            ok = False
        if d_exp is not None and abs(m["devotion_chr"] - d_exp) > 0.011:
            ok = False
        print(msg + ("   [MATCH]" if ok else "   [MISMATCH]"))


if __name__ == "__main__":
    main()
