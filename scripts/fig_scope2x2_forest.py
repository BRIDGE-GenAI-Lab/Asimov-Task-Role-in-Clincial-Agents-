#!/usr/bin/env python3
"""Draws a forest plot of the per-model scope effect and the pooled random-effects and Hartung-Knapp estimates from the matched-syntax 2x2 experiment. It reads experiment1/results/scope_2x2_stats.json and writes the figure as PDF and PNG to FIG_OUTDIR (default figures)."""
import json
import os
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from _figstyle import plt, style, save_audited, WINE, WINE_MID, SLATE, INK, GREY

OUTDIR = os.environ.get('FIG_OUTDIR', 'figures')
E2_MODE = 'FIG_OUTDIR' in os.environ
STEM = 'Appendix_Figure_3' if E2_MODE else 'fig_scope2x2_forest'

D = json.load(open('experiment1/results/scope_2x2_stats.json'))
PM = D['per_model']
RE = D['pooled_random_effect']
HK = D['pooled_hartung_knapp']

names = sorted(PM.keys(), key=lambda k: PM[k]['scope_pp'], reverse=True)
n = len(names)


CANONICAL_NAME = {
    'falcon3-10b-o2': 'Falcon3-10B',
    'gemma4-12b-o2': 'Gemma-4-12B',
    'gemma4-31b-o2': 'Gemma-4-31B',
    'glm47-flash-o2': 'GLM-4.7-Flash',
    'granite41-30b-o2': 'Granite-4.1-30B',
    'internlm3-8b-o2': 'InternLM3-8B',
    'mistral-small32-o2': 'Mistral-Small-3.2',
    'olmo2-32b-o2': 'OLMo-2-32B',
    'phi4-mini-o2': 'Phi-4-mini',
    'phi4-o2': 'Phi-4',
    'qwen35-27b-o2': 'Qwen3.5-27B',
    'yi-34b-o2': 'Yi-1.5-34B',
}


if E2_MODE:
    CANONICAL_NAME = dict(CANONICAL_NAME, **{'gemma4-12b-o2': 'Gemma-4 12B', 'gemma4-31b-o2': 'Gemma-4 31B'})


def clean(k):
    return CANONICAL_NAME.get(k, k)


fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.46 * n + 1.9)))
y = np.arange(n)[::-1].astype(float)
dy = 0.17

for yi, k in zip(y, names):
    s, sci = PM[k]['scope_pp'], PM[k]['scope_ci95']
    l, lci = PM[k]['loyalty_pp'], PM[k]['loyalty_ci95']
    ax.plot([sci[0], sci[1]], [yi + dy, yi + dy], color=WINE, lw=1.4, zorder=2)
    ax.plot(s, yi + dy, 'o', color=WINE, ms=5, zorder=3)
    ax.plot([lci[0], lci[1]], [yi - dy, yi - dy], color=SLATE, lw=1.4, zorder=2)
    ax.plot(l, yi - dy, 'o', color=SLATE, ms=5, zorder=3)

py = -1.4
ps, psci = RE['scope']['pp'], RE['scope']['ci95']
pl, plci = RE['loyalty']['pp'], RE['loyalty']['ci95']
ax.plot([psci[0], psci[1]], [py + dy, py + dy], color=WINE, lw=2.2, zorder=3)
ax.plot(ps, py + dy, 'D', color=WINE, ms=8, zorder=4)
ax.plot([plci[0], plci[1]], [py - dy, py - dy], color=SLATE, lw=2.2, zorder=3)
ax.plot(pl, py - dy, 'D', color=SLATE, ms=8, zorder=4)

py_hk = py - 1.0
hks, hkci = HK['scope']['pp'], HK['scope']['ci95']
ax.plot([hkci[0], hkci[1]], [py_hk, py_hk], color=WINE_MID, lw=2.0, zorder=3)
ax.plot(hks, py_hk, 'D', color=WINE_MID, ms=7, zorder=4)

ax.axvline(0, color=GREY, lw=1.0, ls='--', zorder=1)
ax.set_yticks(list(y) + [py, py_hk])
ax.set_yticklabels([clean(k) for k in names] + ['Pooled (DerSimonian-Laird)', 'Pooled (Hartung-Knapp)'], fontsize=8.6)
ax.get_yticklabels()[-2].set_fontweight('bold')
ax.get_yticklabels()[-1].set_color(WINE_MID)
ax.set_xlabel("Risk difference in acquisitions discordant with\nthe study's allocation standard (percentage points)"
              if E2_MODE else 'Risk difference in allocation-priority violation rate (percentage points)', fontsize=9.5)
ax.set_ylim(py_hk - 0.9, n + 0.7)
if E2_MODE:
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: '0' if v == 0 else ('$-$%g' % -v if v < 0 else '%g' % v)))
ax.plot([], [], 'o', color=WINE, label='Task-scope effect (single-patient vs whole-ward)')
ax.plot([], [], 'o', color=SLATE, label='Loyalty-wording effect (present vs absent)')
if E2_MODE:
    ax.legend(loc='upper left', fontsize=7.6, frameon=True, facecolor='white', edgecolor='none', framealpha=1.0)
else:
    ax.legend(loc='upper left', fontsize=7.6, frameon=False)
if not E2_MODE:
    ax.set_title(f'Matched-syntax scope and loyalty-wording effects vary across models ({n} self-hosted models)',
                 fontsize=9.6, fontweight='bold', loc='left', color=INK, pad=10)
style(ax)
if E2_MODE:
    ax.yaxis.set_tick_params(length=0)
fig.subplots_adjust(left=0.28, right=0.97, top=0.9 if not E2_MODE else 0.97, bottom=0.12 if not E2_MODE else 0.15)
Path(OUTDIR).mkdir(parents=True, exist_ok=True)
save_audited(fig, 'Figure_4', STEM, pad_inches=0.14, outdir=OUTDIR)
print(f'{STEM} written to {OUTDIR} ({n} models: {", ".join(names)})')
