#!/usr/bin/env python3
"""Aggregates the task-scope ladder logs (ward_boundary_*_2ward.jsonl) found in an input directory (default output/intermediate): per model and role it computes the violation rate with a Wilson interval and checks whether the rate falls as the assigned scope widens. It writes experiment1/results/boundary_ladder.json (or the path given as the second argument)."""
import json
import glob
import os
import math
import sys

ARM_ORDER = ["neutral", "empathetic", "guardrail", "ward_steward", "global_steward"]

DISPLAY = {
    "gemma4-31b-o2": "Gemma-4 31B",
    "gemma4-12b-o2": "Gemma-4 12B",
    "qwen35-27b-o2": "Qwen3.5-27B",
    "olmo2-32b-o2": "OLMo-2-32B",
    "qwen35-122b-o2": "Qwen3.5-122B",
    "mistral-medium35-o2": "Mistral-Medium-3.5",
    "glm47-flash-o2": "GLM-4.7-Flash",
    "yi-34b-o2": "Yi-1.5-34B",
    "phi4-o2": "Phi-4",
    "falcon3-10b-o2": "Falcon3-10B",
    "granite41-30b-o2": "Granite-4.1-30B",
    "nemotron-super-120b": "Nemotron-Super-120B",
    "llama3.3-70b": "Llama-3.3-70B",
    "qwen3-next-80b": "Qwen3-Next-80B",
    "hermes3-405b": "Hermes-3-405B",
    "gpt-oss-120b": "GPT-OSS-120B",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (round(100 * p, 1), round(100 * max(0.0, centre - half), 1),
            round(100 * min(1.0, centre + half), 1))


def summarize_file(fp):
    by = {}
    rules_flags = set()
    n_parsefail = 0
    model_tag = None
    with open(fp) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            m, a = o.get("model"), o.get("arm")
            if m is None or a is None:
                continue
            model_tag = m
            if "rules_stated" in o:
                rules_flags.add(bool(o["rules_stated"]))
            h = o.get("harm")
            if h is None:
                n_parsefail += 1
                continue
            by.setdefault(a, []).append(1 if h else 0)
    arms = {}
    for a in ARM_ORDER + [x for x in by if x not in ARM_ORDER]:
        if a in by:
            k, n = sum(by[a]), len(by[a])
            pct, lo, hi = wilson(k, n)
            arms[a] = {"pct": pct, "ci": [lo, hi], "k": k, "n": n}
    return model_tag, arms, rules_flags, n_parsefail


def main(indir, outpath):
    files = sorted(glob.glob(os.path.join(indir, "ward_boundary_*_2ward.jsonl")))
    if not files:
        print("no ward_boundary_*_2ward.jsonl files found in", indir)
        return 1
    models = {}
    for fp in files:
        model_tag, arms, rules_flags, n_pf = summarize_file(fp)
        if model_tag is None or not arms:
            print("  SKIP (no usable episodes):", os.path.basename(fp))
            continue
        seq = [arms[a]["pct"] for a in ARM_ORDER if a in arms]
        gap = None
        if "neutral" in arms and "global_steward" in arms:
            gap = round(arms["neutral"]["pct"] - arms["global_steward"]["pct"], 1)
        monotone = None
        if len(seq) >= 2:
            monotone = all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1))
        models[DISPLAY.get(model_tag, model_tag)] = {
            "tag": model_tag,
            "source_file": os.path.basename(fp),
            "rules_stated": (rules_flags == {True}),
            "rules_stated_values": sorted(str(x) for x in rules_flags),
            "parse_failures": n_pf,
            "arms": arms,
            "narrow_to_wide_pcts": seq,
            "neutral_minus_global_pp": gap,
            "monotone_containment": monotone,
        }
    out = {
        "note": ("Boundary ladder: scope-order "
                 "ladder neutral->empathetic->guardrail->ward_steward->global_steward "
                 "(narrow->wide) over the two-ward bed instrument "
                 "(ward_scenarios_2ward_bed.jsonl, 100 scenarios), RULES STATED in all "
                 "arms (WARD_RULES=1), heavy self-hosted O2 model panel. CHR = "
                 "mean(harm), Wilson 95% CI. Expected: CHR falls as the assigned task "
                 "boundary widens (protection relocates outward but does not dissolve)."),
        "arm_order_narrow_to_wide": ARM_ORDER,
        "instrument": "output/intermediate/ward_scenarios_2ward_bed.jsonl",
        "n_models": len(models),
        "models": models,
    }
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {outpath}: {len(models)} model(s)")
    for name, d in models.items():
        seq = " -> ".join(str(p) for p in d["narrow_to_wide_pcts"])
        print(f"  {name}: [{seq}]  rules_stated={d['rules_stated']} "
              f"mono={d['monotone_containment']} gap(neutral-global)={d['neutral_minus_global_pp']}pp "
              f"pf={d['parse_failures']}")
    return 0


if __name__ == "__main__":
    indir = sys.argv[1] if len(sys.argv) > 1 else "output/intermediate"
    outpath = (sys.argv[2] if len(sys.argv) > 2 else
               "experiment1/results/boundary_ladder.json")
    sys.exit(main(indir, outpath))
