import json, math

d = json.load(open('experiment1/results/scope_2x2.json'))
Z = 1.96

def cell_kn(pct, n_cell):
    k = round(pct / 100.0 * n_cell)
    return k, n_cell

out = {}
pooled_scope_num = pooled_scope_den = 0.0
pooled_loy_num = pooled_loy_den = 0.0
pooled_c_num = pooled_c_den = 0.0
for model, md in d['per_model'].items():
    total = md['n']
    n_cell = 100 if total >= 400 else round(total / 4)
    c = md['cells']
    p_pl = c['sx_patient_plain'] / 100.0
    p_ly = c['sx_patient_loyal'] / 100.0
    w_pl = c['sx_ward_plain'] / 100.0
    w_ly = c['sx_ward_loyal'] / 100.0
    n = n_cell
    var = lambda p: p * (1 - p) / n
    scope = ((p_pl + p_ly) - (w_pl + w_ly)) / 2.0
    scope_se = 0.5 * math.sqrt(var(p_pl) + var(p_ly) + var(w_pl) + var(w_ly))
    loy = ((p_ly + w_ly) - (p_pl + w_pl)) / 2.0
    loy_se = 0.5 * math.sqrt(var(p_ly) + var(w_ly) + var(p_pl) + var(w_pl))
    inter = (p_ly - p_pl) - (w_ly - w_pl)
    inter_se = math.sqrt(var(p_ly) + var(p_pl) + var(w_ly) + var(w_pl))
    contrast = p_pl - w_ly
    contrast_se = math.sqrt(var(p_pl) + var(w_ly))
    contrast_z = contrast / contrast_se if contrast_se else float('nan')
    contrast_p = math.erfc(abs(contrast_z) / math.sqrt(2))
    out[model] = {
        'n_total': total, 'n_per_cell': n_cell,
        'cells_pct': {'patient_plain': round(p_pl*100,1), 'patient_loyal': round(p_ly*100,1),
                      'ward_plain': round(w_pl*100,1), 'ward_loyal': round(w_ly*100,1)},
        'scope_pp': round(scope*100,1), 'scope_ci95': [round((scope-Z*scope_se)*100,1), round((scope+Z*scope_se)*100,1)],
        'loyalty_pp': round(loy*100,1), 'loyalty_ci95': [round((loy-Z*loy_se)*100,1), round((loy+Z*loy_se)*100,1)],
        'interaction_pp': round(inter*100,1), 'interaction_ci95': [round((inter-Z*inter_se)*100,1), round((inter+Z*inter_se)*100,1)],
        'scope_gt_loyalty': scope > loy,
        'contrast_scope_minus_wording_pp': round(contrast*100,1),
        'contrast_ci95': [round((contrast-Z*contrast_se)*100,1), round((contrast+Z*contrast_se)*100,1)],
        'contrast_p': round(contrast_p,4),
    }
    w_scope = 1.0 / (scope_se**2); pooled_scope_num += scope * w_scope; pooled_scope_den += w_scope
    w_loy = 1.0 / (loy_se**2); pooled_loy_num += loy * w_loy; pooled_loy_den += w_loy
    w_c = 1.0 / (contrast_se**2); pooled_c_num += contrast * w_c; pooled_c_den += w_c

pooled_scope = pooled_scope_num / pooled_scope_den
pooled_scope_se = math.sqrt(1.0 / pooled_scope_den)
pooled_loy = pooled_loy_num / pooled_loy_den
pooled_loy_se = math.sqrt(1.0 / pooled_loy_den)
pooled_c = pooled_c_num / pooled_c_den
pooled_c_se = math.sqrt(1.0 / pooled_c_den)
pooled_c_p = math.erfc(abs(pooled_c / pooled_c_se) / math.sqrt(2))

for m, r in out.items():
    print(f"{m:20s} scope {r['scope_pp']:+5.1f} CI{r['scope_ci95']}  loyalty {r['loyalty_pp']:+5.1f} CI{r['loyalty_ci95']}  interaction {r['interaction_pp']:+5.1f} CI{r['interaction_ci95']}  scope>loyalty={r['scope_gt_loyalty']}")
print()
print(f"POOLED (inverse-variance): scope {pooled_scope*100:+.1f} pp (95% CI {(pooled_scope-Z*pooled_scope_se)*100:.1f} to {(pooled_scope+Z*pooled_scope_se)*100:.1f})")
print(f"POOLED (inverse-variance): loyalty {pooled_loy*100:+.1f} pp (95% CI {(pooled_loy-Z*pooled_loy_se)*100:.1f} to {(pooled_loy+Z*pooled_loy_se)*100:.1f})")
print(f"scope > loyalty in {sum(1 for r in out.values() if r['scope_gt_loyalty'])}/{len(out)} models")
print(f"POOLED CONTRAST (scope - wording): {pooled_c*100:+.1f} pp (95% CI {(pooled_c-Z*pooled_c_se)*100:.1f} to {(pooled_c+Z*pooled_c_se)*100:.1f}, P = {pooled_c_p:.3f})")
for m, r in out.items():
    print(f"  {m:20s} contrast {r['contrast_scope_minus_wording_pp']:+5.1f} CI{r['contrast_ci95']} P={r['contrast_p']}")

result = {
    'design': d['design'],
    'note': ('Per-model risk-difference main effects and scope x loyalty interaction with 95% CIs. '
             'Cell counts reconstructed from cell rates: exact at 100/cell for the three models with n=400; '
             'gemma4-31b-o2 (n=383) approximated at 96/cell pending exact O2 recount. Pooled = inverse-variance fixed effect.'),
    'per_model': out,
    'pooled_fixed_effect': {
        'scope_pp': round(pooled_scope*100,1), 'scope_ci95': [round((pooled_scope-Z*pooled_scope_se)*100,1), round((pooled_scope+Z*pooled_scope_se)*100,1)],
        'loyalty_pp': round(pooled_loy*100,1), 'loyalty_ci95': [round((pooled_loy-Z*pooled_loy_se)*100,1), round((pooled_loy+Z*pooled_loy_se)*100,1)],
    },
    'contrast_scope_minus_wording': {
        'pooled_pp': round(pooled_c*100,1),
        'ci95': [round((pooled_c-Z*pooled_c_se)*100,1), round((pooled_c+Z*pooled_c_se)*100,1)],
        'p': round(pooled_c_p,4),
        'note': 'direct contrast = scope main effect - wording main effect = (patient-plain cell) - (ward-loyal cell); pooled inverse-variance across the 4 models',
    },
    'scope_gt_loyalty_in': sum(1 for r in out.values() if r['scope_gt_loyalty']),
    'n_models': len(out),
}
json.dump(result, open('experiment1/results/scope_2x2_stats.json','w'), indent=1)
print('\nwrote scope_2x2_stats.json')
