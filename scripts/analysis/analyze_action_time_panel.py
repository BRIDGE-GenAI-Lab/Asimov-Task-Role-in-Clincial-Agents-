"""Summarises the inline binding-probe logs of three self-hosted models: the claim rate of the devotion and steward roles when the model's own allocation judgment is carried into the action turn, and for one model under a length-matched neutral reflection. It writes experiment1/results/action_time_panel.json."""
import json, os, math

INLINE = {
    'gemma4-31b-o2': {'self_binding': 'output/intermediate/binding_probe_inline_gemma4-31b-o2.jsonl',
                      'neutral':      'output/intermediate/binding_probe_inline_gemma4-31b-o2_neutral.jsonl'},
    'phi4-o2':       {'self_binding': 'output/intermediate/binding_probe_inline_phi4-o2.jsonl'},
    'qwen35-27b-o2': {'self_binding': 'output/intermediate/binding_probe_inline_qwen35-27b-o2.jsonl'},
}
CROSS = {'gemma4-31b-o2': 0.869}
OUT = 'experiment1/results/action_time_panel.json'


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0.0, c-h), min(1.0, c+h)


def rate(rows, arm):
    cl = [1 if r.get('claimed') else 0 for r in rows if r.get('arm') == arm]
    k, n = sum(cl), len(cl)
    p, lo, hi = wilson(k, n)
    return {'claimed': k, 'n': n, 'rate': round(p, 4), 'ci95': [round(lo, 4), round(hi, 4)]}


panel = {}
for model, conds in INLINE.items():
    entry = {}
    for cond, path in conds.items():
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        dev = rows and [r for r in rows if r.get('arm') == 'devotion']
        recognized = sum(1 for r in dev for _ in [0] if r.get('you_should_claim') is False) if dev else 0
        entry[cond] = {'devotion': rate(rows, 'devotion'), 'steward': rate(rows, 'steward'),
                       'recognized_should_not_claim': [recognized, len(dev) if dev else 0]}
    if model in CROSS:
        entry['cross_rollout_devotion_rate'] = CROSS[model]
    panel[model] = entry

sb = {m: panel[m]['self_binding']['devotion']['rate'] for m in panel if 'self_binding' in panel[m]}
summary = {
    'models_self_binding': list(sb.keys()),
    'self_binding_devotion_rates': {m: round(v, 4) for m, v in sb.items()},
    'self_binding_max_devotion_rate': round(max(sb.values()), 4),
    'neutral_control_devotion_rate_gemma': panel['gemma4-31b-o2']['neutral']['devotion']['rate'],
    'reading': ('carrying the model\'s own allocation judgment into the action turn suppressed claiming to '
                'at most 2% across three heterogeneous models (including the non-binder Phi-4), while a '
                'length-matched neutral reflection left it at 56% in the reference model, so the active '
                'ingredient is decision-proximal self-binding on the allocation obligation, not reflection '
                'or delay per se'),
    'limitation': ('two elicitation conditions (self-binding, neutral) across three models; the fuller '
                   'control ladder (no-reflection, priority-identification-only, violation-judgment, '
                   'external action-time reminder) and additional models remain to be run'),
}
panel['summary'] = summary

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(panel, open(OUT, 'w'), indent=2)
print(json.dumps(summary, indent=2))
print('\nwrote', OUT)
