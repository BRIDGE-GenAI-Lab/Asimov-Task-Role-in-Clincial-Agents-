#!/usr/bin/env python3
import json
import math
import pathlib
import random
import statistics

from scipy.stats import t as tdist

ROOT = pathlib.Path("")
FS = ROOT / "experiment1/results"
OUT = FS / "developer_cluster_bootstrap.json"

N_BOOT = 10000
SEED = 20260726

LADDER_DEVELOPER = {
    "Gemma-3 12B": "Google", "Gemini-2.5 Flash": "Google", "Gemma-4 31B": "Google",
    "DeepSeek-V4 Flash": "DeepSeek", "DeepSeek-V4 Pro": "DeepSeek",
    "GPT-5.1": "OpenAI", "GPT-OSS 120B": "OpenAI", "GPT-OSS 20B": "OpenAI",
    "Claude Haiku 4.5": "Anthropic", "Nemotron-3-Ultra 550B": "NVIDIA",
    "Llama-4-Scout": "Meta", "Phi-4": "Microsoft", "Granite-4.1-30B": "IBM",
    "Mistral-Medium-3.5": "Mistral", "GLM-4.7-Flash": "Zhipu",
    "Qwen3.5-122B": "Alibaba", "Cohere Command A+": "Cohere",
    "OLMo-2-32B": "AI2", "Yi-1.5-34B": "01.AI", "Falcon3-10B": "TII",
}

SCOPE2X2_DEVELOPER = {
    "gemma4-12b-o2": "Google", "gemma4-31b-o2": "Google",
    "phi4-o2": "Microsoft", "phi4-mini-o2": "Microsoft",
    "falcon3-10b-o2": "TII", "glm47-flash-o2": "Zhipu", "granite41-30b-o2": "IBM",
    "internlm3-8b-o2": "Shanghai AI Lab", "mistral-small32-o2": "Mistral",
    "olmo2-32b-o2": "AI2", "qwen35-27b-o2": "Alibaba", "yi-34b-o2": "01.AI",
}

DRIFT_EXCLUDED_MODELS = ["Gemini-2.5 Flash"]


def cluster_bootstrap(values_by_cluster, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    keys = list(values_by_cluster)
    observed = statistics.mean(v for k in keys for v in values_by_cluster[k])
    means = []
    for _ in range(n_boot):
        drawn = [values_by_cluster[rng.choice(keys)] for _ in keys]
        flat = [v for grp in drawn for v in grp]
        means.append(statistics.mean(flat))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(int(0.975 * n_boot), n_boot - 1)]
    return {"mean": round(observed, 1), "ci_lo": round(lo, 1), "ci_hi": round(hi, 1),
            "n_clusters": len(keys), "n_boot": n_boot, "seed": seed}


def group(per_model_value, developer_map):
    by_model = {m: [v] for m, v in per_model_value.items()}
    by_dev = {}
    for m, v in per_model_value.items():
        by_dev.setdefault(developer_map[m], []).append(v)
    return by_model, by_dev


def t_based_pooled_gap(gaps_dict):
    vals = list(gaps_dict.values())
    n = len(vals)
    m = statistics.mean(vals)
    sd = statistics.stdev(vals)
    se = sd / math.sqrt(n)
    tcrit = tdist.ppf(0.975, n - 1)
    return {"mean": round(m, 1), "ci_lo": round(m - tcrit * se, 1),
            "ci_hi": round(m + tcrit * se, 1), "n_models": n, "df": n - 1}


def main():
    ladder = json.load(open(FS / "ladder_pooled.json"))
    chr_ = ladder["per_model_chr"]
    gaps = {m: chr_[m]["devotion"] - chr_[m]["steward"] for m in chr_}
    assert set(gaps) == set(LADDER_DEVELOPER), sorted(set(gaps) ^ set(LADDER_DEVELOPER))
    bm, bd = group(gaps, LADDER_DEVELOPER)

    s2 = json.load(open(FS / "scope_2x2_stats.json"))["per_model"]
    scope = {m: s2[m]["scope_pp"] for m in s2}
    assert set(scope) == set(SCOPE2X2_DEVELOPER), sorted(set(scope) ^ set(SCOPE2X2_DEVELOPER))
    sm, sd = group(scope, SCOPE2X2_DEVELOPER)

    gaps_19 = {m: v for m, v in gaps.items() if m not in DRIFT_EXCLUDED_MODELS}
    bm_19, bd_19 = group(gaps_19, LADDER_DEVELOPER)
    drift_excluded = {
        "dropped_models": DRIFT_EXCLUDED_MODELS,
        "note": ("Llama-4 Maverick also drifted on re-query but was "
                 "already moved to the descriptive panel for that reason and so is not a member "
                 "of the 20-model primary panel to exclude here; Gemini-2.5 Flash is the only "
                 "primary-panel hosted endpoint flagged provisional."),
        "full_panel_t_based": t_based_pooled_gap(gaps),
        "drift_excluded_t_based": t_based_pooled_gap(gaps_19),
        "full_panel_model_bootstrap": cluster_bootstrap(bm),
        "drift_excluded_model_bootstrap": cluster_bootstrap(bm_19),
        "drift_excluded_developer_bootstrap": cluster_bootstrap(bd_19),
    }

    payload = {
        "note": ("Cluster bootstrap over models and over developers. The panel is a designed, "
                 "non-random set of models, so these intervals quantify uncertainty over the "
                 "models studied and do not license extrapolation beyond the panel."),
        "ladder_devotion_minus_steward": {
            "model_bootstrap": cluster_bootstrap(bm),
            "developer_bootstrap": cluster_bootstrap(bd),
            "n_models": len(bm), "n_developers": len(bd)},
        "scope_2x2_scope_main_effect": {
            "model_bootstrap": cluster_bootstrap(sm),
            "developer_bootstrap": cluster_bootstrap(sd),
            "n_models": len(sm), "n_developers": len(sd)},
        "drift_excluded": drift_excluded,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
