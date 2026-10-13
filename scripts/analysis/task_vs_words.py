#!/usr/bin/env python3
import json, math, collections, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "output" / "intermediate"
OUT = REPO / "experiment1" / "results" / "task_vs_words.json"

FILES = [("Gemma-3 12B", "ward_exp1_v2.jsonl"), ("DeepSeek-V4 Flash", "ward_ds_ladder_v2.jsonl"),
         ("DeepSeek-V4 Pro", "ward_ds_pro_icu.jsonl"), ("Gemini-2.5 Flash", "ward_gem25f_ladder.jsonl"),
         ("GPT-5.1", "ward_gpt51_icu.jsonl"), ("Claude Haiku 4.5", "ward_haiku_icu.jsonl"),
         ("GPT-OSS 120B", "ward_gptoss_icu.jsonl"), ("Gemma-4 31B", "ward_gemma4_31b_o2_icu.jsonl"),
         ("GPT-OSS 20B", "ward_oss20b_icu.jsonl"), ("Nemotron-3-Ultra 550B", "ward_nemotron550b_icu.jsonl"),
         ("Llama-4-Scout", "ward_llama4_scout_o2_icu.jsonl"), ("Phi-4", "ward_phi4_o2_icu.jsonl"),
         ("Granite-4.1-30B", "ward_granite41_30b_o2_icu.jsonl"), ("Mistral-Medium-3.5", "ward_mistral35_off_o2_icu.jsonl"),
         ("GLM-4.7-Flash", "ward_glm47_flash_o2_icu.jsonl"), ("Qwen3.5-122B", "ward_qwen35_122b_o2_icu.jsonl"),
         ("Cohere Command A+", "ward_cohere_command_a_o2_icu.jsonl"), ("OLMo-2-32B", "ward_olmo2_32b_o2_icu.jsonl"),
         ("Yi-1.5-34B", "ward_yi_34b_o2_icu.jsonl"), ("Falcon3-10B", "ward_falcon3_10b_o2_icu.jsonl")]
ARMS = ["steward", "guardrail", "empathetic", "neutral", "devotion"]


def is_icu(s):
    return "ED" not in (s or "").upper()


def load(fn):
    return [r for r in (json.loads(l) for l in open(LOGS / fn) if l.strip())
            if "n_steps" in r and is_icu(r.get("sid", ""))]


def wilson(k, n):
    if not n:
        return (0.0, 0.0, 0.0)
    p = k / n; z = 1.96; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * p, 1), round(100 * (c - h), 1), round(100 * (c + h), 1))


def odds_ratio(k1, n1, k0, n0):
    a, b, c, d = k1, n1 - k1, k0, n0 - k0
    if 0 in (a, b, c, d):
        a += .5; b += .5; c += .5; d += .5
    o = (a * d) / (b * c); se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return (round(o, 1), round(o * math.exp(-1.96 * se), 1), round(o * math.exp(1.96 * se), 1),
            math.log(o), se)


def main():
    counts = {}
    for disp, fn in FILES:
        c = collections.defaultdict(lambda: [0, 0])
        for r in load(fn):
            a = c[r["arm"]]; a[1] += 1; a[0] += 1 if r.get("harm") else 0
        counts[disp] = {a: tuple(c[a]) for a in ARMS}

    per_model = {}
    assign_lor, word_lor, shares, residuals = [], [], [], []
    for disp, _ in FILES:
        C = counts[disp]
        chr_ = {a: wilson(*C[a]) for a in ARMS}
        oA = odds_ratio(*C["neutral"], *C["steward"])
        oW = odds_ratio(*C["devotion"], *C["neutral"])
        oE = odds_ratio(*C["empathetic"], *C["neutral"])
        oG = odds_ratio(*C["guardrail"], *C["neutral"])
        best_pro = min(chr_["empathetic"][0], chr_["guardrail"][0])
        residual = round(best_pro - chr_["steward"][0], 1)
        gap = chr_["devotion"][0] - chr_["steward"][0]
        share = round(100 * (chr_["neutral"][0] - chr_["steward"][0]) / gap, 0) if gap else None
        assign_lor.append(oA[3]); word_lor.append(oW[3]); residuals.append(residual)
        if share is not None:
            shares.append(share)
        per_model[disp] = {
            "chr": {a: chr_[a] for a in ARMS},
            "assignment_or_neutral_vs_steward": oA[:3],
            "wording_or_devotion_vs_neutral": oW[:3],
            "prosocial_or_empathetic_vs_neutral": oE[:3],
            "prosocial_or_guardrail_vs_neutral": oG[:3],
            "best_prosocial_chr": best_pro,
            "residual_above_floor_pp": residual,
            "assignment_share_of_gap_pct": share,
            "assignment_effect_exceeds_wording": oA[3] > oW[3],
        }

    pooled_chr = {a: round(statistics.mean([wilson(*counts[d][a])[0] for d, _ in FILES]), 1) for a in ARMS}
    summary = {
        "pooled_chr_per_arm": pooled_chr,
        "n_assignment_exceeds_wording": sum(1 for a, w in zip(assign_lor, word_lor) if a > w),
        "n_models": len(FILES),
        "mean_assignment_or": round(math.exp(statistics.mean(assign_lor)), 1),
        "mean_wording_or": round(math.exp(statistics.mean(word_lor)), 1),
        "assignment_share_median_pct": statistics.median(shares),
        "assignment_share_range_pct": [min(shares), max(shares)],
        "residual_above_floor_median_pp": statistics.median(residuals),
        "residual_above_floor_range_pp": [min(residuals), max(residuals)],
    }
    try:
        from scipy.stats import wilcoxon
        st, p = wilcoxon([abs(x) for x in assign_lor], [abs(x) for x in word_lor])
        summary["paired_wilcoxon_W"] = float(st)
        summary["paired_wilcoxon_p"] = round(float(p), 4)
    except Exception as e:
        summary["paired_test"] = f"unavailable: {e}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "note": "Task-not-words dissociation on the deterministic CHR (judge-free), 20 primary models. "
                "ASSIGNMENT effect = neutral vs steward (bare one-patient assignment vs whole-ward role, "
                "no persuasive words); WORDING effect = devotion vs neutral; pro-social wording = "
                "empathetic/guardrail vs neutral. Odds ratios with logit 95% CI; CHR with Wilson 95% CI.",
        "summary": summary,
        "per_model": per_model,
    }, indent=2))

    print("POOLED CHR:", pooled_chr)
    print(f"assignment effect > wording effect: {summary['n_assignment_exceeds_wording']}/{summary['n_models']} models")
    print(f"mean OR  assignment={summary['mean_assignment_or']}  wording(devotion)={summary['mean_wording_or']}")
    print(f"assignment share of gap: median {summary['assignment_share_median_pct']}% "
          f"(range {summary['assignment_share_range_pct']})")
    print(f"best pro-social wording residual above floor: median {summary['residual_above_floor_median_pp']}pp "
          f"(range {summary['residual_above_floor_range_pp']})")
    if "paired_wilcoxon_p" in summary:
        print(f"paired Wilcoxon |logOR| assignment vs wording: W={summary['paired_wilcoxon_W']}, "
              f"p={summary['paired_wilcoxon_p']}")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
