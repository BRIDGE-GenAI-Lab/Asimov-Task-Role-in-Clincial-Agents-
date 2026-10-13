import json, collections
from pathlib import Path

BASE = Path('experiment1/results')
ARMS = ['steward', 'empathetic', 'guardrail', 'neutral', 'devotion']


def load(fn):
    return [r for r in (json.loads(l) for l in open('output/intermediate/' + fn) if l.strip())
            if 'n_steps' in r]


def is_icu(s):
    return 'ED' not in s.upper()


def build():
    pooled = json.load(open(BASE / 'ladder_pooled.json'))
    files = pooled['canonical_files']
    chr_ = pooled['per_model_chr']
    assert set(files) == set(chr_), 'canonical_files and per_model_chr disagree on the model set'

    n_out, k_out = {}, {}
    for disp, fn in files.items():
        rr = [r for r in load(fn) if is_icu(r.get('sid', ''))]
        c = collections.defaultdict(lambda: [0, 0])
        for r in rr:
            a = c[r['arm']]; a[1] += 1; a[0] += 1 if r.get('harm') else 0
        n_out[disp] = {a: c[a][1] for a in ARMS}
        k_out[disp] = {a: c[a][0] for a in ARMS}

    assert len(n_out) == 20, f'expected 20 models, got {len(n_out)}'
    for disp in files:
        assert disp in n_out, f'{disp} missing from output'

    worst = 0.0
    for disp in files:
        for a in ARMS:
            n, k = n_out[disp][a], k_out[disp][a]
            pub = chr_[disp][a]
            if n == 0:
                assert pub is None, f'{disp}/{a}: no episodes but published rate {pub}'
                continue
            got = round(100 * k / n, 1)
            d = abs(got - pub)
            worst = max(worst, d)
            assert d <= 0.1, f'{disp}/{a}: recomputed APVR {got} != published {pub}'
    return n_out, k_out, worst


def main():
    n_out, k_out, worst = build()
    total = sum(v for m in n_out.values() for v in m.values())
    out = {
        'note': 'Per-model x per-arm episode counts behind the canonical ICU instruction ladder. '
                'Same files, same ICU subset and same arm binning as ladder_pooled.py; the model '
                'list is read from ladder_pooled.json::canonical_files so the two artifacts cannot '
                'drift. episodes_per_cell = denominator of the corresponding per_model_chr rate; '
                'violations_per_cell = its numerator. No new published numbers.',
        'source': 'results/ladder_pooled.json :: canonical_files (model set) + '
                  'output/intermediate/*.jsonl (episode logs)',
        'arms': ARMS,
        'episodes_per_cell': n_out,
        'violations_per_cell': k_out,
        'total_icu_ladder_episodes': total,
        'min_cell_episodes': min(v for m in n_out.values() for v in m.values()),
        'max_cell_episodes': max(v for m in n_out.values() for v in m.values()),
        'apvr_recompute_check': {
            'max_abs_deviation_pp': round(worst, 3),
            'tolerance_pp': 0.1,
            'passed': True,
            'description': 'APVR recomputed as 100*violations_per_cell/episodes_per_cell, compared '
                           'cell by cell against ladder_pooled.json::per_model_chr.',
        },
    }
    (BASE / 'ladder_per_cell_n.json').write_text(json.dumps(out, indent=2))
    print('wrote results/ladder_per_cell_n.json')
    print(f'models: {len(n_out)}  total ICU ladder episodes: {total}')
    print(f'cell episode range: {out["min_cell_episodes"]} to {out["max_cell_episodes"]}')
    print(f'APVR recompute check: max |deviation| = {worst:.3f} pp (tolerance 0.1)')
    for disp in n_out:
        print(f'  {disp:24s} ' + ' '.join(f'{a[:4]}={n_out[disp][a]:>4d}' for a in ARMS))


if __name__ == '__main__':
    main()
