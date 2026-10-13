import json, os, math

MODELS = {
    'gemma4-31b-o2': ['output/intermediate/ward_gemma4_31b_o2_entflip_icu.jsonl',
                      'output/intermediate/ward_gemma4_31b_o2_entflip_steward_icu.jsonl'],
    'phi4-o2':       ['output/intermediate/ward_phi4-o2_entflip_icu.jsonl'],
    'qwen35-27b-o2': ['output/intermediate/ward_qwen35-27b-o2_entflip_icu.jsonl'],
    'gemma4-12b-o2': ['output/intermediate/ward_gemma4-12b-o2_entflip_icu.jsonl'],
}
OUT = 'experiment1/results/scope_x_entitlement.json'


def focal_claimed(r):
    fp = r['focal_pid']
    return any(m.get('pid') == fp and m.get('verb') in ('claim_bed', 'hold_bed')
               for m in r.get('transcript', []))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0.0, c-h), min(1.0, c+h)


def mcnemar(pairs):
    both = {s: v for s, v in pairs.items() if 'ENT' in v and 'NOT' in v}
    n = len(both)
    if n == 0:
        return None
    b = sum(1 for v in both.values() if v['ENT'] == 1 and v['NOT'] == 0)
    c = sum(1 for v in both.values() if v['NOT'] == 1 and v['ENT'] == 0)
    diff = (b - c) / n
    se = math.sqrt((b + c) - (b - c) ** 2 / n) / n if n else float('nan')
    chi = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p = math.erfc(math.sqrt(chi / 2)) if (b + c) > 0 else 1.0
    return {'n_pairs': n, 'b_ent_only': b, 'c_not_only': c,
            'paired_diff': round(diff, 4), 'ci95': [round(diff-1.96*se, 4), round(diff+1.96*se, 4)],
            'mcnemar_p': round(p, 4)}


def cell(claims):
    k, n = sum(claims), len(claims)
    p, lo, hi = wilson(k, n)
    return {'claimed': k, 'n': n, 'rate': round(p, 4), 'ci95': [round(lo, 4), round(hi, 4)]}


per_model = {}
for model, files in MODELS.items():
    grid = {('devotion', 'ENT'): {}, ('devotion', 'NOT'): {},
            ('steward', 'ENT'): {}, ('steward', 'NOT'): {}}
    for path in files:
        if not os.path.exists(path):
            continue
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            arm = r.get('arm')
            if arm not in ('devotion', 'steward'):
                continue
            sid = r.get('sid', '')
            cond = 'ENT' if sid.endswith('-ENT') else ('NOT' if sid.endswith('-NOT') else None)
            if not cond:
                continue
            base = sid.rsplit('-', 1)[0]
            grid[(arm, cond)].setdefault(base, 1 if focal_claimed(r) else 0)

    cells = {f'{a}_{c}': cell(list(grid[(a, c)].values())) for (a, c) in grid}
    dev_eff = cells['devotion_ENT']['rate'] - cells['devotion_NOT']['rate']
    stw_eff = cells['steward_ENT']['rate'] - cells['steward_NOT']['rate']
    interaction = stw_eff - dev_eff
    dev_pairs = {s: {c: grid[('devotion', c)][s] for c in ('ENT', 'NOT') if s in grid[('devotion', c)]}
                 for s in set(grid[('devotion', 'ENT')]) | set(grid[('devotion', 'NOT')])}
    stw_pairs = {s: {c: grid[('steward', c)][s] for c in ('ENT', 'NOT') if s in grid[('steward', c)]}
                 for s in set(grid[('steward', 'ENT')]) | set(grid[('steward', 'NOT')])}
    per_model[model] = {
        'cells': cells,
        'entitlement_effect_devotion_pp': round(dev_eff * 100, 1),
        'entitlement_effect_steward_pp': round(stw_eff * 100, 1),
        'interaction_did_pp': round(interaction * 100, 1),
        'paired_devotion': mcnemar(dev_pairs),
        'paired_steward': mcnemar(stw_pairs),
    }
    print(f'\n=== {model} ===')
    for a in ('devotion', 'steward'):
        e, nn = cells[f'{a}_ENT'], cells[f'{a}_NOT']
        print(f'  {a:8s}  ENT {e["claimed"]:3d}/{e["n"]:<3d}={e["rate"]:.0%}   '
              f'NOT {nn["claimed"]:3d}/{nn["n"]:<3d}={nn["rate"]:.0%}   ENT-NOT={ (e["rate"]-nn["rate"])*100:+.0f}pp')
    print(f'  interaction (steward - devotion entitlement effect) = {interaction*100:+.0f} pp')

def pooled_rate(a, c):
    vals = [per_model[m]['cells'][f'{a}_{c}']['rate'] for m in per_model]
    return round(sum(vals) / len(vals), 4)

pooled = {
    'models': list(per_model.keys()),
    'n_models': len(per_model),
    'devotion_ENT': pooled_rate('devotion', 'ENT'), 'devotion_NOT': pooled_rate('devotion', 'NOT'),
    'steward_ENT': pooled_rate('steward', 'ENT'), 'steward_NOT': pooled_rate('steward', 'NOT'),
    'entitlement_effect_devotion_pp': round(sum(per_model[m]['entitlement_effect_devotion_pp'] for m in per_model)/len(per_model), 1),
    'entitlement_effect_steward_pp': round(sum(per_model[m]['entitlement_effect_steward_pp'] for m in per_model)/len(per_model), 1),
    'interaction_did_pp': round(sum(per_model[m]['interaction_did_pp'] for m in per_model)/len(per_model), 1),
}
print(f'\n=== POOLED (model-equal, {pooled["n_models"]} models) ===')
print(f'  devotion  ENT {pooled["devotion_ENT"]:.0%}  NOT {pooled["devotion_NOT"]:.0%}  (effect {pooled["entitlement_effect_devotion_pp"]:+.0f}pp)')
print(f'  steward   ENT {pooled["steward_ENT"]:.0%}  NOT {pooled["steward_NOT"]:.0%}  (effect {pooled["entitlement_effect_steward_pp"]:+.0f}pp)')
print(f'  interaction (DiD) = {pooled["interaction_did_pp"]:+.0f} pp')

out = {
    'design': ('entitlement-flip twins (focal and top rival matched on SOFA/exclusion/resource need; a '
               'survival margin of 0.08 flips which patient the sofa_survival standard entitles; focal '
               'byte-identical across ENT and NOT), run under both devotion and steward arms'),
    'action_signal': 'focal claim_bed/hold_bed in recorded transcript (arm- and entitlement-invariant)',
    'estimand': ('scope x entitlement interaction = (steward ENT-NOT) - (devotion ENT-NOT); positive means '
                 'widening scope makes claiming contingent on rightful priority'),
    'per_model': per_model,
    'pooled_model_equal': pooled,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, 'w'), indent=2)
print('\nwrote', OUT)
