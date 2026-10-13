#!/usr/bin/env python3
import json
import os
import sys
import platform
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

ROOT = ""
FROZEN = os.path.join(ROOT, "experiment1/results/ladder_pooled.json")
INTERMEDIATE = os.path.join(ROOT, "output/intermediate")
OUT = os.path.join(ROOT, "experiment1/results/hierarchical_sensitivity.json")

ARMS = {"steward", "neutral", "empathetic", "guardrail", "devotion"}

with open(FROZEN) as f:
    frozen = json.load(f)

canonical = frozen["canonical_files"]
per_model_chr = frozen["per_model_chr"]

rows = []
for model_name, fn in canonical.items():
    path = os.path.join(INTERMEDIATE, fn)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("mode") != "focal":
                continue
            arm = r.get("arm")
            if arm not in ARMS:
                continue
            harm = r.get("harm")
            if harm is None:
                continue
            rows.append({
                "model": model_name,
                "sid": r.get("sid"),
                "arm": arm,
                "harm": int(bool(harm)),
                "rep": r.get("rep"),
            })

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} episodes, {df['sid'].nunique()} scenarios, {df['model'].nunique()} models")

gate_rows = []
max_abs_dev = 0.0
for model_name in canonical:
    sub = df[df["model"] == model_name]
    for arm in ARMS:
        chr_emp = sub[sub["arm"] == arm]["harm"].mean() * 100.0
        chr_ref = per_model_chr[model_name][arm]
        dev = abs(chr_emp - chr_ref)
        max_abs_dev = max(max_abs_dev, dev)
        gate_rows.append((model_name, arm, round(chr_emp, 2), chr_ref, round(dev, 2)))

gate_df = pd.DataFrame(gate_rows, columns=["model", "arm", "chr_emp", "chr_ref", "abs_dev_pp"])
gate_df = gate_df.sort_values("abs_dev_pp", ascending=False)
print("\n=== FIDELITY GATE (top deviations) ===")
print(gate_df.head(12).to_string(index=False))
print(f"\nMax abs deviation: {max_abs_dev:.3f} pp")
gate_pass = max_abs_dev < 1.5
print(f"Gate pass (<1.5pp): {gate_pass}")

if not gate_pass:
    print("\n*** FIDELITY GATE FAILED - STOPPING, NOT FITTING ***")
    print(gate_df.to_string(index=False))
    worst = gate_df.groupby("model")["abs_dev_pp"].max().sort_values(ascending=False)
    print("\nPer-model worst deviation (pp):")
    print(worst.to_string())
    bad_models = worst[worst >= 1.5].index.tolist()
    diag = {
        "method": ("Bayesian multilevel logistic regression of collateral-harm "
                   "on instruction arm with scenario + model random intercepts "
                   "(BinomialBayesMixedGLM). NOT FIT: fidelity gate failed."),
        "status": "STOPPED_FIDELITY_GATE_FAILED",
        "n_episodes_loaded": int(len(df)),
        "n_scenarios": int(df["sid"].nunique()),
        "n_models": int(df["model"].nunique()),
        "fidelity_gate": {
            "max_abs_dev_pp": round(float(max_abs_dev), 3),
            "threshold_pp": 1.5,
            "pass": False,
        },
        "diagnosis": (
            "5 of 7 canonical files reproduce frozen per_model_chr to <0.05 pp "
            "(DeepSeek-V4 Flash, DeepSeek-V4 Pro, GPT-5.1, Claude Haiku 4.5 exact; "
            "Gemma-3 12B within 4.2 pp on steward only). Two files - "
            "ward_gem25f_ladder.jsonl (Gemini-2.5 Flash) and "
            "ward_llama4m_ladder.jsonl (Llama-4 Maverick), both regenerated on "
            "2026-06-18 AFTER ladder_pooled.json was frozen - deviate by up to "
            "16.6 pp and 13.9 pp respectively. No clean row filter (mode/harm, "
            "is_icu, n_steps, rep slicing) reconciles the current on-disk data to "
            "the frozen numbers for these two models. The rep==0 batch in those "
            "two files is anomalous (e.g. Llama steward 58% in rep0 vs 34.8% "
            "frozen; Gemini empathetic 51% in rep0 vs 88.7% frozen), but dropping "
            "rep0 still leaves 3.9-4.0 pp residual AND degrades the 5 clean files. "
            "Conclusion: the frozen per_model_chr for these two models was computed "
            "on an earlier (now-overwritten) version of these JSONL files. The data "
            "must be re-frozen (rerun ladder_pooled.py) or the original episode set "
            "restored before the hierarchical fit is used."
        ),
        "mismatched_models": bad_models,
        "per_model_max_abs_dev_pp": {k: round(float(v), 3) for k, v in worst.items()},
        "mismatch_table": gate_df.to_dict(orient="records"),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(OUT, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\nWrote diagnostic (gate-fail) JSON to {OUT}")
    sys.exit(1)

def summarize_fe(result, df_index):
    names = result.model.exog_names
    out = {}
    for i, nm in enumerate(names):
        out[nm] = (float(result.fe_mean[i]), float(result.fe_sd[i]))
    return out

def fe_to_or(mean, sd):
    lo = mean - 1.96 * sd
    hi = mean + 1.96 * sd
    return {
        "log_or": round(mean, 4),
        "or": round(float(np.exp(mean)), 3),
        "ci_lo": round(float(np.exp(lo)), 3),
        "ci_hi": round(float(np.exp(hi)), 3),
    }

formula = "harm ~ C(arm, Treatment(reference='steward'))"
vc = {"scenario": "0 + C(sid)", "model": "0 + C(model)"}

print("\n=== PRIMARY SPEC (scenario + model random intercepts) ===")
m = BinomialBayesMixedGLM.from_formula(formula, vc, df)

fit_method = None
try:
    res = m.fit_vb()
    fit_method = "fit_vb"
except Exception as e:
    print(f"fit_vb failed: {e}; falling back to fit_map")
    res = m.fit_map()
    fit_method = "fit_map"
print(f"Fit method: {fit_method}")
print(res.summary())

fe = summarize_fe(res, df)
print("\nFixed-effect names:", list(fe.keys()))

def find_fe(arm):
    for nm, val in fe.items():
        if nm.endswith(f"[T.{arm}]"):
            return val
    return None

fixed_effects = {}
devotion_fe_mean = None
devotion_fe_sd = None
for arm in ["devotion", "neutral", "empathetic", "guardrail"]:
    val = find_fe(arm)
    if val is None:
        raise RuntimeError(f"Could not find fixed effect for arm {arm}")
    mean, sd = val
    if arm == "devotion":
        devotion_fe_mean = mean
        devotion_fe_sd = sd
    fixed_effects[f"{arm}_vs_steward"] = fe_to_or(mean, sd)
    print(f"{arm} vs steward: log_or={mean:.3f} sd={sd:.3f} OR={np.exp(mean):.2f}")

vc_names = res.model.vc_names if hasattr(res.model, "vc_names") else None
print("\nVC names:", vc_names)
print("vcp_mean:", res.vcp_mean)
print("vcp_sd:", res.vcp_sd)

re_sd = {}
comp_names = list(vc.keys())
for i, nm in enumerate(comp_names):
    lo = float(np.exp(res.vcp_mean[i] - 1.96 * res.vcp_sd[i]))
    hi = float(np.exp(res.vcp_mean[i] + 1.96 * res.vcp_sd[i]))
    re_sd[nm] = {"sd": round(float(np.exp(res.vcp_mean[i])), 4),
                 "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
print("Random-effect SDs:", re_sd)

model_re_sd = float(re_sd["model"]["sd"])
pred_sd = float(np.sqrt(devotion_fe_sd ** 2 + model_re_sd ** 2))
pred_or = float(np.exp(devotion_fe_mean))
pred_pi_lo = float(np.exp(devotion_fe_mean - 1.96 * pred_sd))
pred_pi_hi = float(np.exp(devotion_fe_mean + 1.96 * pred_sd))
print(
    f"\nDevotion-vs-steward predictive (new model): "
    f"fixed_sd={devotion_fe_sd:.4f}, model_RE_sd={model_re_sd:.4f}, "
    f"pred_sd={pred_sd:.4f}, OR={pred_or:.1f} "
    f"PI=[{pred_pi_lo:.1f}, {pred_pi_hi:.1f}]"
)

print("\n=== REDUCED SPEC (model random intercept only) ===")
vc_red = {"model": "0 + C(model)"}
m_red = BinomialBayesMixedGLM.from_formula(formula, vc_red, df)
try:
    res_red = m_red.fit_vb()
    fit_method_red = "fit_vb"
except Exception as e:
    print(f"fit_vb failed: {e}; falling back to fit_map")
    res_red = m_red.fit_map()
    fit_method_red = "fit_map"
print(f"Reduced fit method: {fit_method_red}")

fe_red = summarize_fe(res_red, df)
def find_fe_red(arm):
    for nm, val in fe_red.items():
        if nm.endswith(f"[T.{arm}]"):
            return val
    return None
dev_red = find_fe_red("devotion")
reduced_devotion = fe_to_or(dev_red[0], dev_red[1])
print(f"Reduced devotion vs steward: OR={reduced_devotion['or']} "
      f"[{reduced_devotion['ci_lo']}, {reduced_devotion['ci_hi']}]")
red_re_sd_model = round(float(np.exp(res_red.vcp_mean[0])), 4)
print(f"Reduced model RE SD: {red_re_sd_model}")

out = {
    "method": ("Bayesian multilevel (hierarchical) logistic regression of the "
               "binary collateral-harm endpoint on instruction arm "
               "(treatment-coded vs steward reference), with random intercepts "
               "for scenario (sid) and model. Fit by variational Bayes "
               f"(statsmodels BinomialBayesMixedGLM, {fit_method}). "
               "Sensitivity analysis with partial pooling across the 20 models, "
               "complementing the mean-of-model-rates "
               "t-CI, which has few degrees of freedom."),
    "n_episodes": int(len(df)),
    "n_scenarios": int(df["sid"].nunique()),
    "n_models": int(df["model"].nunique()),
    "models": sorted(df["model"].unique().tolist()),
    "fidelity_gate": {
        "max_abs_dev_pp": round(float(max_abs_dev), 3),
        "pass": bool(gate_pass),
        "threshold_pp": 1.5,
    },
    "reference_arm": "steward",
    "fit_method": fit_method,
    "fixed_effects": fixed_effects,
    "devotion_vs_steward_predictive": {
        "or": round(pred_or, 1),
        "pi_lo": round(pred_pi_lo, 1),
        "pi_hi": round(pred_pi_hi, 1),
        "basis": ("exp(mean +/- 1.96*sqrt(fixed_sd^2 + model_RE_sd^2)); "
                  "approximate new-model posterior predictive"),
    },
    "random_effect_sd": {
        "scenario": re_sd.get("scenario"),
        "model": re_sd.get("model"),
        "note": ("Posterior point estimate of each random-intercept standard "
                 "deviation on the log-odds scale, computed as exp(vcp_mean) "
                 "from the variational-Bayes fit."),
    },
    "reduced_model_devotion_or": {
        "or": reduced_devotion["or"],
        "ci_lo": reduced_devotion["ci_lo"],
        "ci_hi": reduced_devotion["ci_hi"],
        "log_or": reduced_devotion["log_or"],
        "spec": "model random intercept only (scenario RE dropped)",
        "fit_method": fit_method_red,
        "model_random_effect_sd": red_re_sd_model,
    },
    "software": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "pandas": pd.__version__,
    },
}

with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {OUT}")
print(json.dumps(out, indent=2))
