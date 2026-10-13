"""Analyses the task-scope logs of six self-hosted models over four assigned task sets (focal patient only, plus an irrelevant patient, plus the higher-priority rival, whole ward). It computes violation rates with Wilson intervals and paired McNemar changes between sets, and writes experiment1/results/boundary_crossing.json."""
import json, os, math

MODELS = {
    'gemma4-31b-o2':  'output/intermediate/ward_gemma4-31b-o2_boundary_icu.jsonl',
    'gemma4-12b-o2':  'output/intermediate/ward_gemma4-12b-o2_boundary_icu.jsonl',
    'phi4-o2':        'output/intermediate/ward_phi4-o2_boundary_icu.jsonl',
    'qwen35-27b-o2':  'output/intermediate/ward_qwen35-27b-o2_boundary_icu.jsonl',
    'glm47-flash-o2': 'output/intermediate/ward_glm47-flash-o2_boundary_icu.jsonl',
    'granite41-30b-o2':'output/intermediate/ward_granite41-30b-o2_boundary_icu.jsonl',
}
ARMS = ['mem_a', 'mem_a_c', 'mem_a_b', 'mem_ward']
OUT = 'experiment1/results/boundary_crossing.json'


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0.0, c-h), min(1.0, c+h)


def cell(vals):
    k, n = sum(vals), len(vals)
    p, lo, hi = wilson(k, n)
    return {'harm': k, 'n': n, 'apvr': round(p, 4), 'ci95': [round(lo, 4), round(hi, 4)]}


def mcnemar(pairs):
    both = {s: v for s, v in pairs.items() if 'from' in v and 'to' in v}
    n = len(both)
    if n == 0:
        return None
    b = sum(1 for v in both.values() if v['to'] == 1 and v['from'] == 0)
    c = sum(1 for v in both.values() if v['from'] == 1 and v['to'] == 0)
    diff = (b - c) / n
    se = math.sqrt((b + c) - (b - c) ** 2 / n) / n if n else float('nan')
    chi = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p = math.erfc(math.sqrt(chi / 2)) if (b + c) > 0 else 1.0
    return {'n_pairs': n, 'paired_change_pp': round(diff * 100, 1),
            'ci95_pp': [round((diff-1.96*se)*100, 1), round((diff+1.96*se)*100, 1)],
            'mcnemar_p': round(p, 4)}


def _grids(path):
    grid = {a: {} for a in ARMS}
    if not os.path.exists(path):
        return grid
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        a = r.get('arm')
        if a in grid:
            grid[a].setdefault(r.get('sid', ''), 1 if r.get('harm') else 0)
    return grid


per_model = {}
for model, path in MODELS.items():
    grid = _grids(path)
    if not any(grid[a] for a in ARMS):
        continue
    cells = {a: cell(list(grid[a].values())) for a in ARMS}

    def paired(from_arm, to_arm):
        sids = set(grid[from_arm]) & set(grid[to_arm])
        return mcnemar({s: {'from': grid[from_arm][s], 'to': grid[to_arm][s]} for s in sids})

    per_model[model] = {
        'cells': cells,
        'boundary_cross_ac_to_ab': paired('mem_a_c', 'mem_a_b'),
        'breadth_control_a_to_ac': paired('mem_a', 'mem_a_c'),
        'ward_gain_ab_to_ward': paired('mem_a_b', 'mem_ward'),
    }
    print(f'\n=== {model} ===')
    for a in ARMS:
        cc = cells[a]
        print(f'  {a:9s} APVR {cc["apvr"]:.0%} ({cc["harm"]}/{cc["n"]})')
    bc = per_model[model]['boundary_cross_ac_to_ab']; br = per_model[model]['breadth_control_a_to_ac']
    if bc: print(f'  +rival B (mem_a_c->mem_a_b): {bc["paired_change_pp"]:+.0f}pp  P={bc["mcnemar_p"]}')
    if br: print(f'  +irrelevant C (mem_a->mem_a_c): {br["paired_change_pp"]:+.0f}pp  P={br["mcnemar_p"]}')

def pooled_apvr(a):
    vals = [per_model[m]['cells'][a]['apvr'] for m in per_model if not math.isnan(per_model[m]['cells'][a]['apvr'])]
    return round(sum(vals) / len(vals), 4) if vals else float('nan')

def pooled_paired(key):
    vals = [per_model[m][key]['paired_change_pp'] for m in per_model if per_model[m][key]]
    return round(sum(vals) / len(vals), 1) if vals else float('nan')

pooled = {'models': list(per_model.keys()), 'n_models': len(per_model),
          'apvr_by_arm': {a: pooled_apvr(a) for a in ARMS},
          'boundary_cross_ac_to_ab_pp': pooled_paired('boundary_cross_ac_to_ab'),
          'breadth_control_a_to_ac_pp': pooled_paired('breadth_control_a_to_ac')}
print('\n=== POOLED (model-equal) ===')
print('  APVR:', {a: f'{pooled["apvr_by_arm"][a]:.0%}' for a in ARMS})
print(f'  boundary crossing (+B): {pooled["boundary_cross_ac_to_ab_pp"]:+.0f}pp   '
      f'breadth control (+C): {pooled["breadth_control_a_to_ac_pp"]:+.0f}pp')

out = {'design': ('four assigned-task sets over the identical base ICU scenarios: mem_a (focal only), '
                  'mem_a_c (focal + a strictly lower-priority irrelevant patient), mem_a_b (focal + the '
                  'higher-priority rival), mem_ward. Board, rules, contract and tool authority identical; '
                  'the rival is visible in every arm. APVR = deterministic collateral-harm (rival denied).'),
       'decisive_contrast': 'mem_a_c -> mem_a_b (adding the rival) vs mem_a -> mem_a_c (adding an irrelevant patient)',
       'per_model': per_model, 'pooled_model_equal': pooled}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, 'w'), indent=2)
print('\nwrote', OUT)
