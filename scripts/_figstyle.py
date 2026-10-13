"""Shared matplotlib style for the figure scripts: serif fonts, colour constants, a Wilson interval helper, axis styling and save_audited, which writes a figure as PDF and PNG and checks that no text falls below the print size floor at column width. It reads nothing and writes only through save_audited."""
import matplotlib
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

plt.rcParams.update({
    'font.family':'serif','font.serif':['CMU Serif','CMU Serif Roman','DejaVu Serif'],
    'mathtext.fontset':'cm','axes.unicode_minus':False,
    'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':1.0})

WINE='#7E364A'; WINE_MID='#A8606F'; WINE_PALE='#F4ECEF'
SAGE='#7FA08F'; SAGE_PALE='#E5EDE8'
TEAL='#3E6C7B'; TEAL_PALE='#E8F0F2'
SLATE='#7E8C8D'; SLATE_PALE='#EEF2F2'
GREY='#8C8C8C'; LIGHT_GREY='#D8DDDE'; INK='#1E1E1E'; EDGE='#4A2F38'
LILAC='#9A8FA8'; LILAC_PALE='#F0EDF3'; BLUE='#2C5F8A'; BLUE_PALE='#E7EEF5'
YLAB='Allocation-priority violation rate (%)'

def wilson(k,n):
    if n==0: return (0.0,0.0,0.0)
    p=k/n; z=1.96; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*p,100*(c-h),100*(c+h))

PRINT_MAX_IN = 180 / 25.4


def print_size_audit(fig, name, floor_pt=5.2, pad_inches=0.16, bbox_inches='tight', verbose=True):
    fig.canvas.draw()
    w_canvas = fig.get_size_inches()[0]
    if bbox_inches == 'tight':
        w_out = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad_inches
    else:
        w_out = w_canvas
    scale = PRINT_MAX_IN / w_out
    worst, worst_pt = None, float('inf')
    for t in fig.findobj(match=lambda o: hasattr(o, 'get_fontsize') and hasattr(o, 'get_text')):
        s = (t.get_text() or '').strip()
        if not s:
            continue
        eff = t.get_fontsize() * scale
        if eff < worst_pt:
            worst_pt, worst = eff, s
    if verbose:
        print(f'[print-size] {name}: canvas {w_canvas:.2f} in, delivered {w_out:.2f} in, '
              f'scale {scale:.3f}, min {worst_pt:.2f} pt @180mm  '
              f'(worst: {worst[:40]!r})')
    assert worst_pt >= floor_pt, (
        f'{name}: smallest type is {worst_pt:.2f} pt at 180 mm, below the {floor_pt} pt floor '
        f'(worst element: {worst[:60]!r})')
    return worst_pt, worst


def save_audited(fig, name, stem, pad_inches=0.16, floor_pt=5.2, dpi=600, outdir='figures'):
    worst_pt, worst = print_size_audit(fig, name, floor_pt=floor_pt, pad_inches=pad_inches)
    out = Path(outdir)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{stem}.{ext}', dpi=dpi, bbox_inches='tight',
                    pad_inches=pad_inches, facecolor='white')
    return worst_pt, worst


def style(ax):
    ax.spines['left'].set_color(INK); ax.spines['bottom'].set_color(INK)
    ax.tick_params(colors=INK, length=4, width=1.0, labelcolor=INK)
    ax.grid(False); ax.set_axisbelow(True)

def panel_label(ax, label, x=-0.10, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            color=INK, va='bottom', ha='left')

def vdelta(ax, x, y0, y1, label, x_guides=(), guide_color=GREY):
    for (gx0,gx1,gy) in x_guides:
        ax.plot([gx0,gx1],[gy,gy], ls=(0,(2,2)), color=guide_color, lw=0.9, zorder=1)
    ax.annotate('', xy=(x,y1), xytext=(x,y0),
                arrowprops=dict(arrowstyle='<|-|>', color=INK, lw=1.4, shrinkA=0, shrinkB=0), zorder=4)
    ax.text(x, (y0+y1)/2, label, ha='center', va='center', fontsize=9.4, color=INK, zorder=5,
            bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='none'))
