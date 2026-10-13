"""Plots the violation rate under each of the five roles for the reference model (DeepSeek-V4 Flash) on ICU scenarios and on emergency department scenarios. It reads experiment1_reanalysis.json and ed_reference.json and writes the figure as PDF and PNG to FIG_OUTDIR (default figures)."""
import json
import os
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
import sys; sys.path.insert(0, 'scripts')
from _figstyle import plt, style, save_audited, WINE, SLATE, SAGE_PALE, LIGHT_GREY, GREY, INK

OUTDIR = os.environ.get('FIG_OUTDIR', 'figures')
E2_MODE = 'FIG_OUTDIR' in os.environ
STEM = 'Appendix_Figure_1' if E2_MODE else 'fig4_words_ride_prior'
E2_DIR = 'experiment2'
REFERENCE_MODEL = 'DeepSeek-V4 Flash'


def ed_reference():
    per_role = json.load(open(E2_DIR + '/results/ed_reference.json'))['per_role']
    return {r: per_role[r]['rate_pct'] for r in ROLE_ORDER}


def numeral(v):
    if E2_MODE:
        from decimal import Decimal, ROUND_HALF_UP
        return str(Decimal(repr(float(v))).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return f'{v:.0f}'


if E2_MODE:
    sys.path.insert(0, 'scripts')
    from role_labels import ROLE_ORDER, ROLE_LABEL, one_line
    icu = json.load(open(E2_DIR + '/results/experiment1_reanalysis.json'))['role_rates_original']['per_model'][REFERENCE_MODEL]
    ed = ed_reference()
    ARMS = [(one_line(ROLE_LABEL[r]), icu[r]['harm_rate_pct'], ed[r]) for r in ROLE_ORDER]
else:
    ARMS = [
        ('make the ward its job',                   17.2, 0.0),
        ('"never take from a\nhigher-priority patient"', 38.8, 1.5),
        ('"be fair to other patients"',             35.0, 1.5),
        ('minimal patient assignment',              76.8, 10.5),
        ('do everything for your patient',          88.5, 63.8),
    ]
fig, ax = plt.subplots(figsize=(7.8, 4.4))
y = np.arange(len(ARMS))[::-1]
if not E2_MODE:
    ax.axvspan(0, 17, color=SAGE_PALE, zorder=0)
    ax.axvline(17, color='#91AA99', lw=0.9, ls=(0, (3, 3)), zorder=1)
for yy, (lab, icu_v, ed_v) in zip(y, ARMS):
    ax.plot([ed_v, icu_v], [yy, yy], color=LIGHT_GREY, lw=2.6, zorder=1)
    ax.scatter([ed_v], [yy], s=82, color=SLATE, edgecolors='white', linewidths=1.0, zorder=3)
    ax.scatter([icu_v], [yy], s=98, color=WINE, edgecolors='white', linewidths=1.0, zorder=4)
    ax.text(icu_v + 2.6, yy, numeral(icu_v), va='center', ha='left', fontsize=8.6, color=WINE, fontweight='bold')
    ax.text(ed_v + 2.6, yy - 0.32, numeral(ed_v), va='center', ha='left', fontsize=7.8, color=SLATE, fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels([a[0] for a in ARMS], fontsize=9.2)
ax.set_xlim(0, 100); ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("Acquisitions discordant with the study's allocation standard (%)" if E2_MODE
              else 'A higher-priority patient left without the bed or scan (%)', fontsize=9.6)
ax.set_ylim(-0.7, len(ARMS) - 0.3)
handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=WINE, markeredgecolor='white',
                  markersize=8.4, label='ICU  ·  best-survival-first standard'),
           Line2D([0], [0], marker='o', color='none', markerfacecolor=SLATE, markeredgecolor='white',
                  markersize=8.0, label='ED  ·  most-acute-first standard')]
ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=8.6, handletextpad=0.4)
style(ax)
if E2_MODE:
    ax.yaxis.set_tick_params(length=0)
if not E2_MODE:
    fig.text(0.985, 0.02, 'reference model, n = 2,000  ·  ICU + ED task families', ha='right', va='bottom',
             fontsize=7.2, color=GREY)
fig.subplots_adjust(left=0.28, right=0.975, top=0.96, bottom=0.17)
Path(OUTDIR).mkdir(parents=True, exist_ok=True)
save_audited(fig, 'Extended_Data_Figure_1', STEM, pad_inches=0.14, outdir=OUTDIR)
print(STEM + ' written to ' + OUTDIR)
