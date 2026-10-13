"""Draws a heat map of the violation rate for every model under each of the five roles, with episode counts per cell. It reads role_rates_original from experiment1_reanalysis.json and writes the figure as PDF and PNG to FIG_OUTDIR (default figures)."""
import json
import math
import os
from pathlib import Path
import sys; sys.path.insert(0, 'scripts')
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from _figstyle import plt, save_audited, INK, GREY, LIGHT_GREY

OUTDIR = os.environ.get('FIG_OUTDIR', 'figures')
E2_MODE = 'FIG_OUTDIR' in os.environ
STEM = 'Appendix_Figure_4' if E2_MODE else 'fig_model_arm_matrix'

BASE = Path('experiment1/results')
if E2_MODE:
    sys.path.insert(0, 'scripts')
    from role_labels import ROLE_ORDER, ROLE_LABEL
    _RRO = json.load(open('experiment2/results/experiment1_reanalysis.json')
                     )['role_rates_original']['per_model']
    CHR = {m: {r: _RRO[m][r]['harm_rate_pct'] for r in ROLE_ORDER} for m in _RRO}
    NCELL = {m: {r: _RRO[m][r]['n'] for r in ROLE_ORDER} for m in _RRO}
    ARMS = [(r, ROLE_LABEL[r]) for r in ROLE_ORDER]
else:
    CHR = json.load(open(BASE / 'ladder_pooled.json'))['per_model_chr']
    NCELL = json.load(open(BASE / 'ladder_per_cell_n.json'))['episodes_per_cell']
    ARMS = [('steward',    'Make the ward\nits job'),
            ('guardrail',  'Never take what a\nhigher-priority\npatient needs'),
            ('empathetic', 'Be fair to\nother patients'),
            ('neutral',    'Minimal\nassignment'),
            ('devotion',   'Do everything\nfor your patient')]
assert set(CHR) == set(NCELL), 'rate and denominator artifacts disagree on the model set'


def numeral(v):
    if E2_MODE:
        from decimal import Decimal, ROUND_HALF_UP
        return str(Decimal(repr(float(v))).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return f'{v:.0f}'

models = sorted(CHR.keys(), key=lambda m: CHR[m]['steward'])
nrows, ncols = len(models), len(ARMS)

cmap = LinearSegmentedColormap.from_list('harm', ['#7FA08F', '#E8E1D7', '#7E364A'])

N_MIN = min(v for m in NCELL.values() for v in m.values())
N_MAX = max(v for m in NCELL.values() for v in m.values())
S_CEIL = 506.0
S_FLOOR = S_CEIL * N_MIN / N_MAX


def area_from_n(n):
    return S_CEIL * n / N_MAX


def _rel_lum(rgb):
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_color(v):
    fill = _rel_lum(cmap(v / 100.0))
    contrast = lambda a, b: (max(a, b) + 0.05) / (min(a, b) + 0.05)
    return 'white' if contrast(fill, _rel_lum((1, 1, 1))) >= contrast(fill, _rel_lum((0.118,) * 3)) else INK


FIGW, FIGH = 8.8, 9.5
AX_RECT = [0.215, 0.100, 0.755, 0.860]
YLO, YHI = -0.62, nrows - 0.38

fig = plt.figure(figsize=(FIGW, FIGH))
ax = fig.add_axes(AX_RECT)

row_pitch_pt = AX_RECT[3] * FIGH * 72.0 / (YHI - YLO)
d_max_pt = 2 * math.sqrt(S_CEIL / math.pi)
d_min_pt = 2 * math.sqrt(S_FLOOR / math.pi)
assert d_max_pt < 0.92 * row_pitch_pt, f'bubbles would collide: {d_max_pt:.1f} pt in {row_pitch_pt:.1f} pt'

for i in range(nrows):
    if i % 2 == 0:
        ax.axhspan(i - 0.5, i + 0.5, color='#F6F7F7', zorder=0)

for r, m in enumerate(models):
    y = nrows - 1 - r
    for c, (arm, _) in enumerate(ARMS):
        v = CHR[m][arm]
        ax.scatter([c], [y], s=area_from_n(NCELL[m][arm]), color=cmap(v / 100.0),
                   edgecolors='white', linewidths=0.9, zorder=3)
        ax.text(c, y, numeral(v), ha='center', va='center', fontsize=6.8,
                color=text_color(v), fontweight='bold', zorder=5)

ax.set_yticks([nrows - 1 - r for r in range(nrows)])
ax.set_yticklabels(models, fontsize=8.6, color=INK)
ax.set_xticks(range(ncols))
ax.set_xticklabels([lab for _, lab in ARMS], fontsize=7.4, color=INK, linespacing=1.05)
ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
ax.tick_params(top=False, left=False, length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlim(-0.62, ncols - 0.38)
ax.set_ylim(YLO, YHI)

cax = fig.add_axes([0.215, 0.052 if not E2_MODE else 0.040, 0.30, 0.0135])
cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 100), cmap=cmap), cax=cax,
                    orientation='horizontal')
cbar.set_ticks([0, 25, 50, 75, 100])
cbar.ax.tick_params(labelsize=7.0, colors=INK, length=0, pad=2)
cbar.outline.set_visible(False)
fig.text(0.215, 0.076 if not E2_MODE else 0.064,
         'Fill color: allocation-priority violation rate (%)' if not E2_MODE
         else "Fill color: acquisitions discordant with\nthe study's allocation standard (%)",
         fontsize=7.8, color=INK, fontweight='bold', ha='left', va='bottom')

lax = fig.add_axes([0.60, 0.030, 0.37, 0.060]); lax.set_axis_off()
lax.set_xlim(0, 1); lax.set_ylim(0, 1)
fig.text(0.60, 0.076, 'Bubble area: episodes behind the cell',
         fontsize=7.8, color=INK, fontweight='bold', ha='left', va='bottom')
for x, n_ref in zip((0.09, 0.40, 0.76), (100, 250, 400)):
    lax.scatter([x], [0.56], s=area_from_n(n_ref), color=LIGHT_GREY,
                edgecolors=GREY, linewidths=0.8, zorder=3)
    lax.text(x, 0.06, f'{n_ref}', ha='center', va='bottom', fontsize=7.0, color=INK)

fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_fill_r_pt = d_min_pt / 2 - 0.9
_need_pt, _who = max((math.hypot(*t.get_window_extent(renderer=_rend).size) / fig.dpi * 72 / 2,
                      t.get_text()) for t in ax.texts if t.get_text().strip())
assert _need_pt <= _fill_r_pt, (
    f'numeral {_who!r} needs {_need_pt:.2f} pt of half-diagonal but the smallest bubble offers only '
    f'{_fill_r_pt:.2f} pt inside its white ring - raise S_CEIL or lower the numeral size')

Path(OUTDIR).mkdir(parents=True, exist_ok=True)
save_audited(fig, 'Figure_6', STEM, pad_inches=0.16, outdir=OUTDIR)
print(STEM + ' written to ' + OUTDIR)
print(f'  cell episodes {N_MIN} to {N_MAX} -> area {S_FLOOR:.0f} to {S_CEIL:.0f} pt^2 '
      f'(diameter {d_min_pt:.1f} to {d_max_pt:.1f} pt; row pitch {row_pitch_pt:.1f} pt, '
      f'largest bubble at {d_max_pt / row_pitch_pt:.2f} of pitch)')
print(f'  numeral fit: widest numeral {_who!r} needs {_need_pt:.2f} pt half-diagonal, smallest '
      f'bubble offers {_fill_r_pt:.2f} pt inside its ring (clearance {_fill_r_pt - _need_pt:.2f} pt)')
