#!/usr/bin/env python3
"""Aggregates the matched-syntax 2x2 logs (scope by loyalty wording) matched by --glob, one log per model. It computes per-model scope and loyalty effects, DerSimonian-Laird and Hartung-Knapp pooled estimates, a fixed-effect estimate and leave-one-out contrasts, skipping models with a cell below --min-cell. It writes experiment1/results/scope_2x2_stats.json (or --out)."""
import argparse, glob, json, math, os
from collections import defaultdict
from scipy.stats import t as tdist

CELL = {"sx_patient_plain": "patient_plain", "sx_patient_loyal": "patient_loyal",
        "sx_ward_plain": "ward_plain", "sx_ward_loyal": "ward_loyal"}
NEED = ["patient_plain", "patient_loyal", "ward_plain", "ward_loyal"]
Z = 1.959964


def load_cells(path):
    c = defaultdict(lambda: [0, 0])
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            cell = CELL.get(d.get("arm"))
            if cell is None:
                continue
            c[cell][1] += 1
            if d.get("harm"):
                c[cell][0] += 1
    return c


def two_sided_p(est, se):
    if not se or math.isnan(se):
        return float("nan")
    z = abs(est) / se
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def per_model(path, min_cell):
    c = load_cells(path)
    if any(c[k][1] < min_cell for k in NEED):
        return None
    p = {k: c[k][0] / c[k][1] for k in NEED}
    v = {k: p[k] * (1 - p[k]) / c[k][1] for k in NEED}
    pp, pl, wp, wl = (p[k] for k in NEED)
    vpp, vpl, vwp, vwl = (v[k] for k in NEED)
    scope = (pp + pl) / 2 - (wp + wl) / 2
    se_scope = math.sqrt((vpp + vpl + vwp + vwl) / 4)
    loy = (pl + wl) / 2 - (pp + wp) / 2
    se_loy = se_scope
    inter = (pp - wp) - (pl - wl)
    se_inter = math.sqrt(vpp + vwp + vpl + vwl)
    contrast = pp - wl
    se_contrast = math.sqrt(vpp + vwl)
    return {
        "n_total": sum(c[k][1] for k in NEED),
        "n_per_cell": min(c[k][1] for k in NEED),
        "cells_pct": {k: round(100 * p[k], 1) for k in NEED},
        "scope_pp": round(100 * scope, 1),
        "scope_ci95": [round(100 * (scope - Z * se_scope), 1), round(100 * (scope + Z * se_scope), 1)],
        "loyalty_pp": round(100 * loy, 1),
        "loyalty_ci95": [round(100 * (loy - Z * se_loy), 1), round(100 * (loy + Z * se_loy), 1)],
        "interaction_pp": round(100 * inter, 1),
        "interaction_ci95": [round(100 * (inter - Z * se_inter), 1), round(100 * (inter + Z * se_inter), 1)],
        "scope_gt_loyalty": scope > loy,
        "contrast_scope_minus_wording_pp": round(100 * contrast, 1),
        "contrast_ci95": [round(100 * (contrast - Z * se_contrast), 1), round(100 * (contrast + Z * se_contrast), 1)],
        "contrast_p": round(two_sided_p(contrast, se_contrast), 4),
        "_scope": scope, "_se_scope": se_scope, "_loy": loy, "_se_loy": se_loy,
        "_contrast": contrast, "_se_contrast": se_contrast,
    }


def pooled_iv(models, est_key, se_key):
    num = den = 0.0
    for m in models:
        se = m[se_key]
        if se and not math.isnan(se) and se > 0:
            w = 1 / se ** 2
            num += w * m[est_key]
            den += w
    if den == 0:
        return None, None, None
    est = num / den
    se = math.sqrt(1 / den)
    return est, se, [round(100 * (est - Z * se), 1), round(100 * (est + Z * se), 1)]


def hartung_knapp(y, v, tau2, k):
    w = [1.0 / (vi + tau2) for vi in v]
    sw = sum(w)
    mu = sum(wi * yi for wi, yi in zip(w, y)) / sw
    q = sum(wi * (yi - mu) ** 2 for wi, yi in zip(w, y)) / (k - 1)
    se = math.sqrt(q / sw)
    t = tdist.ppf(0.975, k - 1)
    return {"pp": round(mu, 1),
            "ci95": [round(mu - t * se, 1), round(mu + t * se, 1)],
            "se": round(se, 3), "df": k - 1, "t_crit": round(float(t), 3),
            "method": "Hartung-Knapp-Sidik-Jonkman, t interval on k-1 df"}


def v_from_ci95(ci):
    return ((ci[1] - ci[0]) / (2 * 1.96)) ** 2


def dl_random_effect(models, est_key, se_key):
    ys, ws, vs = [], [], []
    for m in models:
        se = m[se_key]
        if se and se > 0 and not math.isnan(se):
            ys.append(m[est_key]); vs.append(se ** 2); ws.append(1 / se ** 2)
    if len(ys) < 2:
        return None
    sw = sum(ws)
    y_fix = sum(w * y for w, y in zip(ws, ys)) / sw
    Q = sum(w * (y - y_fix) ** 2 for w, y in zip(ws, ys))
    df = len(ys) - 1
    C = sw - sum(w ** 2 for w in ws) / sw
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    wr = [1 / (v + tau2) for v in vs]
    swr = sum(wr)
    y_ran = sum(w * y for w, y in zip(wr, ys)) / swr
    se_ran = math.sqrt(1 / swr)
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    return {
        "pp": round(100 * y_ran, 1),
        "ci95": [round(100 * (y_ran - Z * se_ran), 1), round(100 * (y_ran + Z * se_ran), 1)],
        "p": round(two_sided_p(y_ran, se_ran), 4),
        "tau2_pp2": round(1e4 * tau2, 2), "I2_pct": round(I2, 1),
        "Q": round(Q, 2), "df": df, "k": len(ys),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="output/intermediate/ward_scope2x2_*_2ward.jsonl")
    ap.add_argument("--min-cell", type=int, default=95)
    ap.add_argument("--out", default="experiment1/results/scope_2x2_stats.json")
    a = ap.parse_args()
    pm, skipped = {}, {}
    for path in sorted(glob.glob(a.glob)):
        name = os.path.basename(path).replace("ward_scope2x2_", "").replace("_2ward.jsonl", "")
        r = per_model(path, a.min_cell)
        if r:
            pm[name] = r
        else:
            c = load_cells(path)
            skipped[name] = min((c[k][1] for k in NEED), default=0)
    models = list(pm.values())
    scope_e, scope_se, scope_ci = pooled_iv(models, "_scope", "_se_scope")
    loy_e, loy_se, loy_ci = pooled_iv(models, "_loy", "_se_loy")
    con_e, con_se, con_ci = pooled_iv(models, "_contrast", "_se_contrast")
    scope_re = dl_random_effect(models, "_scope", "_se_scope")
    loy_re = dl_random_effect(models, "_loy", "_se_loy")
    con_re = dl_random_effect(models, "_contrast", "_se_contrast")

    n_models_hk = len(models)
    scope_hk = hartung_knapp([m["scope_pp"] for m in models],
                             [v_from_ci95(m["scope_ci95"]) for m in models],
                             scope_re["tau2_pp2"], n_models_hk)
    loy_hk = hartung_knapp([m["loyalty_pp"] for m in models],
                           [v_from_ci95(m["loyalty_ci95"]) for m in models],
                           loy_re["tau2_pp2"], n_models_hk)
    con_hk = hartung_knapp([m["contrast_scope_minus_wording_pp"] for m in models],
                           [v_from_ci95(m["contrast_ci95"]) for m in models],
                           con_re["tau2_pp2"], n_models_hk)

    def loo_contrast(drop_name):
        num = den = 0.0
        for nm, m in pm.items():
            if nm == drop_name:
                continue
            se = m["_se_contrast"]
            if se > 0:
                w = 1 / se ** 2
                num += w * m["_contrast"]
                den += w
        return round(100 * num / den, 1) if den else None

    loo = {nm: loo_contrast(nm) for nm in pm}
    most_infl = min(loo, key=lambda k: loo[k]) if loo else None
    for m in models:
        for k in ["_scope", "_se_scope", "_loy", "_se_loy", "_contrast", "_se_contrast"]:
            m.pop(k, None)
    out = {
        "design": "orthogonal 2x2, matched syntax: scope {single-patient, whole-ward} x loyalty {absent, present}; rules stated in all cells; 2-ward bed instrument",
        "note": f"Expanded to {len(pm)} self-hosted pinned-weight models (from 4). PRIMARY pooled estimate is DerSimonian-Laird random-effects given between-model heterogeneity (down-weights outliers); fixed-effect inverse-variance, per-model spread, and leave-one-out also reported. Models with any 2x2 cell < {a.min_cell} obs excluded.",
        "excluded_incomplete": skipped,
        "per_model": pm,
        "pooled_random_effect": {"scope": scope_re, "loyalty": loy_re,
                                  "note": "DerSimonian-Laird; pp = risk difference in percentage points; I2 and tau2 quantify between-model heterogeneity"},
        "pooled_hartung_knapp": {"scope": scope_hk, "loyalty": loy_hk,
                                  "contrast_scope_minus_wording": con_hk,
                                  "note": ("Hartung-Knapp-Sidik-Jonkman t interval on k-1 df, reusing the DL tau2; "
                                           "reported alongside pooled_random_effect (DL), not in place of it, because "
                                           "the DL normal interval is anticonservative at k=12 with I2=94.8%.")},
        "pooled_fixed_effect": {
            "scope_pp": round(100 * scope_e, 1), "scope_ci95": scope_ci,
            "loyalty_pp": round(100 * loy_e, 1), "loyalty_ci95": loy_ci,
        },
        "contrast_scope_minus_wording": {
            "random_effect": con_re,
            "fixed_pooled_pp": round(100 * con_e, 1), "fixed_ci95": con_ci,
            "fixed_p": round(two_sided_p(con_e, con_se), 4),
            "note": "direct contrast = (patient-plain cell) - (ward-loyal cell). Random-effect (DL) is primary; fixed-effect shown for comparison.",
        },
        "leave_one_out_fixed_contrast": {"by_model_pp": loo, "most_influential": most_infl,
                                          "contrast_without_most_influential_pp": loo.get(most_infl) if most_infl else None},
        "scope_gt_loyalty_in": sum(1 for m in models if m["scope_gt_loyalty"]),
        "n_models": len(pm),
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"aggregated {len(pm)} models (excluded incomplete: {skipped})")
    print(f"RANDOM-EFFECTS (primary): scope +{scope_re['pp']} {scope_re['ci95']} (I2={scope_re['I2_pct']}%) | "
          f"loyalty +{loy_re['pp']} {loy_re['ci95']} | contrast +{con_re['pp']} {con_re['ci95']} P={con_re['p']} (I2={con_re['I2_pct']}%)")
    print(f"fixed-effect: scope +{out['pooled_fixed_effect']['scope_pp']} | contrast +{out['contrast_scope_minus_wording']['fixed_pooled_pp']} P={out['contrast_scope_minus_wording']['fixed_p']}")
    print(f"scope>loyalty in {out['scope_gt_loyalty_in']}/{len(pm)} | LOO most-influential={most_infl} -> contrast without it +{loo.get(most_infl)}")


if __name__ == "__main__":
    main()
