#!/usr/bin/env python3
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FROZEN_DIR = os.path.join(REPO, "experiment1", "results")
LADDER_POOLED = os.path.join(FROZEN_DIR, "ladder_pooled.json")
OUT = os.path.join(FROZEN_DIR, "model_set_sensitivity.json")

RAW_DIR = os.path.join(REPO, "output", "intermediate")
LLAMA_NAME = "Llama-4 Maverick"
LLAMA_FILE = os.path.join(RAW_DIR, "ward_llama4m_ladder.jsonl")

ARMS = ["steward", "empathetic", "guardrail", "neutral", "devotion"]


def chr_per_arm(path):
    num = {a: 0 for a in ARMS}
    den = {a: 0 for a in ARMS}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("mode") != "focal":
                continue
            if "n_steps" not in r:
                continue
            if "ED" in str(r.get("sid", "")):
                continue
            a = r.get("arm")
            if a not in num:
                continue
            den[a] += 1
            if r.get("harm"):
                num[a] += 1
    out = {}
    for a in ARMS:
        if den[a] == 0:
            raise ValueError(f"{path}: no rows for arm {a}")
        out[a] = round(100.0 * num[a] / den[a], 1)
    return out


def pooled(set_models, per_model):
    arm_chr = {}
    for a in ARMS:
        vals = [per_model[m][a] for m in set_models]
        arm_chr[a] = round(sum(vals) / len(vals), 1)
    gap = round(arm_chr["devotion"] - arm_chr["steward"], 1)
    return {
        "n_models": len(set_models),
        "models": list(set_models),
        "pooled": arm_chr,
        "gap_pp": gap,
    }


def main():
    lp = json.load(open(LADDER_POOLED))
    canonical_files = lp["canonical_files"]

    per_model = {}
    for name, fname in canonical_files.items():
        per_model[name] = chr_per_arm(os.path.join(RAW_DIR, fname))
    per_model[LLAMA_NAME] = chr_per_arm(LLAMA_FILE)

    six = list(canonical_files.keys())
    local_open = "Gemma-3 12B"

    sets = {
        "six_frozen_primary": six,
        "seven_incl_llama": six + [LLAMA_NAME],
        "hosted_only": [m for m in six if m != local_open],
        "local_open_only": [local_open],
        "all_full_ladder_attempts": six + [LLAMA_NAME],
    }

    result = {
        "note": (
            "Model-set sensitivity of the ICU prompt-ladder headline contrast. "
            "Per-(model, arm) compliant-harm rate (CHR) = 100 * mean(harm), "
            "recomputed from raw canonical ICU ladder files "
            "(mode==focal, has n_steps, ICU sid) declared in "
            "ladder_pooled.json::canonical_files, plus a fresh Llama-4 Maverick "
            "file (output/intermediate/ward_llama4m_ladder.jsonl). For each model "
            "SET, 'pooled' is the model-equal mean of the per-model CHR and "
            "'gap_pp' is pooled devotion minus pooled steward. Arms ordered "
            "steward, empathetic, guardrail, neutral, devotion."
        ),
        "sets": {k: pooled(v, per_model) for k, v in sets.items()},
    }

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"wrote {OUT}")
    print(f"{'set':<26}{'steward':>9}{'devotion':>10}{'gap_pp':>9}{'n':>4}")
    for k in sets:
        s = result["sets"][k]
        print(f"{k:<26}{s['pooled']['steward']:>9.1f}"
              f"{s['pooled']['devotion']:>10.1f}{s['gap_pp']:>9.1f}{s['n_models']:>4}")


if __name__ == "__main__":
    main()
