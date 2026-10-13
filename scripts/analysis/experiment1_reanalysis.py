#!/usr/bin/env python3
"""Recomputes the experiment 1 estimates from the 20 ladder logs after keeping the first row of each cell_id: scenario margins, dose response, devotion-minus-steward contrasts with model-equal and random-effects summaries, resource channels, replicate variability, the temperature sweep, role rates, alternative-standard summaries and the episode flow. It writes experiment2/results/experiment1_reanalysis.json."""
import collections
import json
import math
import statistics
import sys

sys.path.insert(0, "code")
sys.path.insert(0, "scripts/analysis")
sys.path.insert(0, "scripts")
import metrics as M
from logs_io import load_jsonl, primary_files
from scarcity_ward.hospital.ward_scenarios import load_ward_scenarios
from scarcity_ward.protocols import rightful_recipients

E2_DIR = "experiment2"
FS = "experiment1/results/"
INTER = "output/intermediate/"
ARMS = ["steward", "guardrail", "empathetic", "neutral", "devotion"]
FLOW = {"executed": 24939, "flagged_excluded": 2023, "primary_corpus": 22916, "ward_episodes": 20129,
        "probe_episodes": 2787, "primary_ladder": 12961}


def _cell_id(r):
    return r["cell_id"]


def _sweep_key(r):
    return (r["temperature"], r["arm"], r["sid"], r["rep"])


def _first_occurrences(rows, key=_cell_id):
    seen, kept = set(), []
    for r in rows:
        k = key(r)
        if k not in seen:
            seen.add(k)
            kept.append(r)
    return kept


def _icu_rows(fn, dedup=True):
    rows = [r for r in load_jsonl(INTER + fn) if "n_steps" in r and "ED" not in r.get("sid", "").upper()]
    return _first_occurrences(rows) if dedup else rows


def ladder(dedup=True):
    return {label: _icu_rows(fn, dedup) for label, fn in primary_files()}


def scenario_margins():
    out = {}
    for ws in load_ward_scenarios(INTER + "ward_scenarios.jsonl"):
        pts = {p.pid: p for p in ws.scenario.patients}
        fav = pts[rightful_recipients(ws.scenario)[0]]
        foc = pts[ws.focal_pid]
        out[ws.scenario.sid] = {"sofa_gap": foc.sofa - fav.sofa,
                                "survival_gap": round(fav.survival_prob - foc.survival_prob, 2)}
    return out


def margins_summary(m):
    gaps = [v["sofa_gap"] for v in m.values()]
    sgaps = [v["survival_gap"] for v in m.values()]
    return {"n": len(m), "sofa_gap_median": statistics.median(gaps), "sofa_gap_range": [min(gaps), max(gaps)],
            "sofa_gap_iqr": statistics.quantiles(gaps, n=4)[::2],
            "survival_gap_median": statistics.median(sgaps), "survival_gap_range": [min(sgaps), max(sgaps)],
            "focal_sicker_in": sum(g > 0 for g in gaps)}


def _rate(rows):
    return 100.0 * sum(bool(r.get("harm")) for r in rows) / len(rows) if rows else None


def per_model_arm(lad):
    return {m: {a: _rate([r for r in rows if r["arm"] == a]) for a in ARMS} for m, rows in lad.items()}


def _contrast(lad, a1, a0):
    diffs, ys, vs, k = [], [], [], [0, 0, 0, 0]
    per_model = {}
    for label, rows in lad.items():
        r1 = [r for r in rows if r["arm"] == a1]
        r0 = [r for r in rows if r["arm"] == a0]
        x1, x0 = sum(bool(r.get("harm")) for r in r1), sum(bool(r.get("harm")) for r in r0)
        diffs.append(100.0 * (x1 / len(r1) - x0 / len(r0)))
        per_model[label] = diffs[-1]
        d, v = M.rd_var(x1, len(r1), x0, len(r0))
        ys.append(100 * d)
        vs.append(1e4 * v)
        k = [k[0] + x1, k[1] + len(r1), k[2] + x0, k[3] + len(r0)]
    m, lo, hi = M.mean_t_ci(diffs)
    re = M.re_summary(ys, vs)
    return {"k": len(diffs), "model_equal": {"mean": m, "ci": [lo, hi]},
            "random_effects": {"mu": re["mu"], "ci_hk": list(re["ci_hk"]), "pi": list(re["pi"]),
                               "tau2": re["tau2"], "I2": re["I2"]},
            "episode_pooled": 100.0 * (k[0] / k[1] - k[2] / k[3]),
            "per_model_median": statistics.median(diffs),
            "per_model_iqr": [statistics.quantiles(diffs, n=4)[0], statistics.quantiles(diffs, n=4)[2]],
            "per_model_pp": per_model}


def primary_estimands(lad=None):
    lad = lad or ladder()
    return {"devotion_minus_steward": _contrast(lad, "devotion", "steward"),
            "neutral_minus_steward": _contrast(lad, "neutral", "steward"),
            "devotion_minus_neutral": _contrast(lad, "devotion", "neutral"),
            "note": "Odds ratio 11.26 (multilevel, conditional on model and scenario) is a subject-specific "
                    "scale and is not convertible to these marginal percentage-point differences; this "
                    "analysis reports risk differences only."}


def dose_response_original(lad=None, margins=None):
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    lad = lad or ladder()
    margins = margins or scenario_margins()
    cuts = statistics.quantiles([v["sofa_gap"] for v in margins.values()], n=3)
    band = lambda g: "low" if g <= cuts[0] else ("mid" if g <= cuts[1] else "high")
    table = collections.defaultdict(dict)
    recs = []
    for m, rows in lad.items():
        for a in ARMS:
            for b in ("low", "mid", "high"):
                rr = [r for r in rows if r["arm"] == a and band(margins[r["sid"]]["sofa_gap"]) == b]
                table[a].setdefault(b, []).append(_rate(rr))
        for r in rows:
            if r["arm"] in ("steward", "devotion"):
                recs.append({"model": m, "arm": r["arm"], "gap": margins[r["sid"]]["sofa_gap"],
                             "y": int(bool(r.get("harm")))})
    df = pd.DataFrame(recs)
    df["gc"] = df["gap"] - df["gap"].mean()
    fit = smf.gee("y ~ C(arm, Treatment('steward')) * gc", groups="model", data=df,
                  family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    p, cov = fit.params, fit.cov_params()
    gi = "C(arm, Treatment('steward'))[T.devotion]:gc"
    slope_dev = p["gc"] + p[gi]
    se_dev = (cov.loc["gc", "gc"] + cov.loc[gi, gi] + 2 * cov.loc["gc", gi]) ** 0.5
    orr = lambda b, s: [math.exp(b), math.exp(b - 1.96 * s), math.exp(b + 1.96 * s)]
    return {"band_cutpoints_sofa_gap": cuts,
            "model_equal_rate_by_band": {a: {b: statistics.mean([x for x in v if x is not None])
                                             for b, v in d.items()} for a, d in table.items()},
            "gee_or_per_sofa_point": {"steward": orr(p["gc"], cov.loc["gc", "gc"] ** 0.5),
                                      "devotion": orr(slope_dev, se_dev)}}


def channels(lad=None):
    lad = lad or ladder()
    out = {}
    for a in ARMS:
        shares = collections.defaultdict(list)
        for rows in lad.values():
            harm = [r for r in rows if r["arm"] == a and r.get("harm")]
            if not harm:
                continue
            for ch in ("bed", "imaging", "attending"):
                shares[ch].append(100.0 * sum(bool(r["channels"].get(ch)) for r in harm) / len(harm))
            shares["withhold_any"].append(100.0 * sum(bool(r.get("withhold")) for r in rows if r["arm"] == a)
                                          / len([r for r in rows if r["arm"] == a]))
        out[a] = {k: statistics.mean(v) for k, v in shares.items()}
    return out


def _replicate_stats(rows, arm):
    by = collections.defaultdict(dict)
    for r in rows:
        if r["arm"] == arm:
            if r["rep"] in by[r["sid"]]:
                raise ValueError(f"{r['sid']} {arm}: replicate {r['rep']} appears more than once")
            by[r["sid"]][r["rep"]] = int(bool(r.get("harm")))
    reps = sorted({k for d in by.values() for k in d})
    full = [[d[k] for k in reps] for d in by.values() if all(k in d for k in reps)]
    rep_rates = [100.0 * sum(row[i] for row in full) / len(full) for i in range(len(reps))] if full else []
    return {"replicates": len(reps), "scenarios_complete": len(full),
            "scenarios_incomplete_dropped": len(by) - len(full),
            "rep_rates": rep_rates, "sd_pp": statistics.stdev(rep_rates) if len(rep_rates) > 1 else None,
            "agreement": M.agreement(full) if full else None,
            "icc1": M.icc1(full) if len(full) > 1 and len(reps) > 1 else None}


def replicate_variability(dedup=True):
    out = {}
    for label, fn in (("DeepSeek-V4 Flash", "ward_ds_ladder_v2.jsonl"), ("Gemma-3 12B", "ward_exp1_v2.jsonl")):
        rows = _icu_rows(fn, dedup)
        out[label] = {a: _replicate_stats(rows, a) for a in ARMS}
    return out


def temperature_sweep(dedup=True):
    rows = load_jsonl(INTER + "temp_sweep.jsonl")
    cells = collections.defaultdict(list)
    for r in (_first_occurrences(rows, _sweep_key) if dedup else rows):
        cells[(r["temperature"], r["arm"])].append(r)
    by_temperature = {}
    for t in sorted({k[0] for k in cells}):
        by_temperature[str(t)] = {}
        for a in (x for x in ARMS if (t, x) in cells):
            c = cells[(t, a)]
            harm = sum(bool(r["harm"]) for r in c)
            by_temperature[str(t)][a] = {"n": len(c), "harm": harm, "rate_pct": 100.0 * harm / len(c),
                                         "scenarios": len({r["sid"] for r in c}),
                                         "replicates": len({r["rep"] for r in c})}
    return {"source": INTER + "temp_sweep.jsonl",
            "note": "The model is not recorded in the rows (fields: temperature, sid, arm, rep, harm); it is "
                    "Gemma-3 12B, the local open-weight reference model, so that the temperature could be set "
                    "directly. The sweep belongs to experiment 1, in which the allocation rule was not stated. "
                    "Descriptive only: counts and rates, no interval or test.",
            "by_temperature": by_temperature}


def role_rates_original(lad=None):
    lad = lad or ladder()
    harm = per_model_arm(lad)
    per_model, repeated = {}, {}
    for m, rows in lad.items():
        per_model[m] = {}
        for a in ARMS:
            cell = [r for r in rows if r["arm"] == a]
            n_claim = sum(M.outcomes(r, missing="engine")["claimed_vent"] for r in cell)
            per_model[m][a] = {"n": len(cell), "harm_episodes": sum(bool(r.get("harm")) for r in cell),
                               "harm_rate_pct": harm[m][a], "claimed_vent_episodes": n_claim,
                               "claimed_vent_rate_pct": 100.0 * n_claim / len(cell) if cell else None}
        ids = {r.get("cell_id") for r in rows}
        if len(ids) < len(rows):
            repeated[m] = {"rows": len(rows), "distinct_cell_ids": len(ids)}
    pooled = lambda a, key: M.mean_t_ci([per_model[m][a][key] for m in per_model if per_model[m][a][key] is not None])
    return {"roles": ARMS,
            "scoring": "Experiment 1, scored on the engine's record. "
                       "harm: the episode endpoint (a contested resource obtained against the study's "
                       "allocation standard), as in per_model_arm and ladder_pooled.json per_model_chr. "
                       "claimed_vent: the focal agent issued claim_bed at one or more of its 4 steps, read by "
                       "metrics.outcomes(missing='engine'), where an unparsed step is the engine's wait. "
                       "Rates are over the de-duplicated episodes of each cell (the first occurrence of each "
                       "cell_id per log).",
            "note": "Per-model values are for all 20 primary models. The model-equal mean over the 16 "
                    "Run A panel models is computed from per_model over those models, not from "
                    "model_equal_mean (all 20). No difference, ratio or test is computed against Run A.",
            "per_model": per_model,
            "model_equal_mean": {a: {"k": len(per_model), "harm_rate_pct": pooled(a, "harm_rate_pct")[0],
                                     "harm_ci95": list(pooled(a, "harm_rate_pct")[1:]),
                                     "claimed_vent_rate_pct": pooled(a, "claimed_vent_rate_pct")[0],
                                     "claimed_vent_ci95": list(pooled(a, "claimed_vent_rate_pct")[1:])}
                                 for a in ARMS},
            "models_with_repeated_cell_ids": repeated}


def _load_json(fn):
    with open(FS + fn) as fh:
        return json.load(fh)


def alt_standards():
    rescored = _load_json("alt_standard_sensitivity.json")
    prospective = _load_json("prospective_cross_standard.json")
    cell = lambda src: {"steward_pct": src["steward"]["chr"], "devotion_pct": src["devotion"]["chr"],
                        "gap_pp": src["gap_pp"], "n": src["steward"]["n"]}
    return {"rescored": rescored, "prospective": prospective,
            "summary": {"rescored_reference_model": rescored["reference_model"],
                        "rescored_by_standard": {s: cell(v) for s, v in rescored["per_standard"].items()},
                        "prospective_reference_model": prospective["reference_model"],
                        "prospective_by_standard": {s: cell(v["reference"])
                                                    for s, v in prospective["per_standard"].items()},
                        "note": "Rescoring holds the recorded actions fixed and changes only the scoring rule, so "
                                "the steward rate and the gap under a standard other than the study's are "
                                "standard-specific by construction and are not a behavioural contrast; "
                                "the prospective block, where the standard was stated in the prompt, is the "
                                "behavioural contrast. strict_exclusion has the same ranking key as "
                                "sofa_survival in code/scarcity_ward/protocols.py, so its rescored values equal the "
                                "study standard's: the five rescored rows are four distinct standards."}}


def _models_in(fn, key):
    v = _load_json(fn)
    for part in key.split("."):
        v = v[part]
    models = sorted(v.keys() if isinstance(v, dict) else v)
    if not models:
        raise ValueError(f"{fn}: {key} lists no models")
    return models


def inclusion_matrix():
    return {"primary_ladder": [m for m, _ in primary_files()],
            "matched_wording_2x2": _models_in("scope_2x2_stats.json", "per_model"),
            "comprehension_probe": _models_in("comprehension_probe.json", "per_model"),
            "judgment_before_action": _models_in("action_time_panel.json", "summary.models_self_binding"),
            "live_check_balanced": _models_in("live_oversight_balanced.json", "per_model")}


def flow(lad):
    n = sum(len(rows) for rows in lad.values())
    basis = (f"De-duplicated (first occurrence of each cell_id per log): {FLOW['primary_ladder'] - n} repeated "
             f"rows are dropped from the {FLOW['primary_ladder']:,} logged.")
    return dict(FLOW, primary_ladder=n, primary_ladder_basis=basis)


def build():
    lad = ladder()
    m = scenario_margins()
    return {"margins": margins_summary(m),
            "dose_response_original": dose_response_original(lad, m),
            "primary_estimands": primary_estimands(lad),
            "channels": channels(lad),
            "replicate_variability": replicate_variability(),
            "temperature_sweep": temperature_sweep(),
            "role_rates_original": role_rates_original(lad),
            "alt_standards": alt_standards(),
            "inclusion_matrix": inclusion_matrix(),
            "flow": flow(lad)}


def main():
    out = build()
    text = json.dumps(out, indent=2, allow_nan=False)
    with open(E2_DIR + "/results/experiment1_reanalysis.json", "w") as fh:
        fh.write(text + "\n")
    print(json.dumps({k: out[k] for k in ("margins",)}, indent=1))
    d = out["primary_estimands"]["devotion_minus_steward"]
    print("primary devotion-steward: mean %.1f (95%% CI %.1f to %.1f), 95%% prediction interval %.1f to %.1f, k = %d"
          % (d["model_equal"]["mean"], *d["model_equal"]["ci"], *d["random_effects"]["pi"], d["k"]))
    print("primary ladder: %d episodes after removing repeated rows" % out["flow"]["primary_ladder"])


if __name__ == "__main__":
    main()
