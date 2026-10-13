"""Summarises the objective-hierarchy logs (local, fair, precedence roles) of four self-hosted models as violation rates, and lists the action-time and live-oversight result files as comparators. It reads ward_<model>_hier_icu.jsonl and writes experiment1/results/hierarchical_objective.json."""
import json, os, math
FS = 'experiment1/results'
MODELS = ['gemma4-31b-o2', 'phi4-o2', 'qwen35-27b-o2', 'gemma4-12b-o2']
ARMS = ['hier_local', 'hier_fair', 'hier_precedence']
OUT = f'{FS}/hierarchical_objective.json'


def apvr(path, arm):
    if not os.path.exists(path):
        return None
    seen = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('arm') == arm:
            seen.setdefault(r.get('sid', ''), 1 if r.get('harm') else 0)
    if not seen:
        return None
    k, n = sum(seen.values()), len(seen)
    return {'apvr': round(k / n, 4), 'harm': k, 'n': n}


lang = {}
for m in MODELS:
    p = f'output/intermediate/ward_{m}_hier_icu.jsonl'
    lang[m] = {a: apvr(p, a) for a in ARMS}
    row = lang[m]
    if any(row.values()):
        print(m, {a: (f'{row[a]["apvr"]:.0%}' if row[a] else 'NA') for a in ARMS})

extra = {}
atp = f'{FS}/action_time_panel.json'
if os.path.exists(atp):
    extra['action_time_replay'] = json.load(open(atp))
lob = f'{FS}/live_oversight_balanced.json'
if os.path.exists(lob):
    extra['structural_oracle_veto'] = json.load(open(lob))

out = {'design': ('language objective-hierarchy ladder (local -> +fairness -> +explicit precedence) run on '
                  'self-hosted models, compared against the existing action-time own-judgment replay and the '
                  'existing structural oracle-veto results; APVR = deterministic collateral-harm'),
       'language_hierarchy_per_model': lang,
       'comparator_artifacts': list(extra.keys())}
os.makedirs(FS, exist_ok=True)
json.dump(out, open(OUT, 'w'), indent=2)
print('wrote', OUT)
