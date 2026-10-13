#!/usr/bin/env python3
"""Builds the main figures from the experiment 2 results files (primary, reanalysis and the experiment 1 result files they cite) and the appendix figure on frontier models, then runs the four standalone figure scripts. It writes PDF and PNG files to experiment2/figures and refuses text below the print size floor."""
import json
import os
import sys

if not os.path.exists("scripts/_figstyle.py"):
    raise SystemExit("run from the repo root: python3 scripts/make_figures.py")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1791244800")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from _figstyle import GREY, INK, LIGHT_GREY, SAGE, SLATE, WINE, save_audited, style
from role_labels import ROLE_LABEL, ROLE_ORDER, one_line

E2_DIR = "experiment2"
FS = "experiment1/results/"
OUT = "figures"

plt.rcParams.update({"font.size": 8.0, "axes.labelsize": 8.5, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
                     "legend.fontsize": 7.5, "axes.titlesize": 9.0, "hatch.linewidth": 0.9})
T_SMALL, T_TICK, T_LAB, T_TITLE, T_NUM = 7.0, 8.0, 8.5, 9.0, 6.9


def darker(hex_color, f=0.40):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    ink = (0x1E, 0x1E, 0x1E)
    return "#%02X%02X%02X" % tuple(int(round(c + (k - c) * f)) for c, k in zip((r, g, b), ink))


def _lum(hex_color):
    c = [int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def num_color(hex_color):
    fill = _lum(hex_color)
    contrast = lambda a, b: (max(a, b) + 0.05) / (min(a, b) + 0.05)
    return "white" if contrast(fill, 1.0) >= contrast(fill, _lum(INK)) else INK


ROLE_COLOR = {"steward": "#CF8697", "guardrail": "#BC7183", "empathetic": "#A75C6F", "neutral": "#93495C",
              "devotion": "#7E364A", "sx_ward_plain": "#5F8E9E", "sx_patient_plain": "#3E6C7B",
              "devotion_precedence": "#7E364A"}
ROLE_MARK = {"sx_ward_plain": "^", "sx_patient_plain": "^", "devotion_precedence": "D"}
OTHER_LINE, OTHER_DOT = "#C9CFD0", "#AAB2B3"


def J(path):
    with open(path) as fh:
        return json.load(fh)


def need(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError("frozen JSON has no key path %s (stopped at %r)" % (" > ".join(map(str, keys)), k))
        cur = cur[k]
    return cur


def runs():
    return J(E2_DIR + "/results/experiment2_primary.json")


def offline():
    return J(E2_DIR + "/results/experiment1_reanalysis.json")


def model_name(R, key):
    return need(R, "roster", key, "name")


def is_frontier(R, key):
    return need(R, "roster", key, "tier") == "frontier"


def wiggle(i, width=0.5):
    return ((i * 0.6180339887498949) % 1.0 - 0.5) * width


def dodge_identical(points, step, xmax=None):
    groups = {}
    for i, (x, y) in enumerate(points):
        groups.setdefault((round(x, 6), round(y, 6)), []).append(i)
    off = [0.0] * len(points)
    for (gx, _), idx in groups.items():
        offs = [(r - (len(idx) - 1) / 2.0) * step for r in range(len(idx))]
        if xmax is not None and gx + max(offs) > xmax:
            shift = gx + max(offs) - xmax
            offs = [o - shift for o in offs]
        for r, i in enumerate(idx):
            off[i] = offs[r]
    return off


def grid_identical(points, ranks, step, origin):
    groups = {}
    for i, (x, y) in enumerate(points):
        groups.setdefault((round(x, 6), round(y, 6)), []).append(i)
    off = [(0.0, 0.0)] * len(points)
    stacks = [(pos, idx) for pos, idx in groups.items() if len(idx) > 1]
    assert len(stacks) <= 1, "more than one group of coincident points: give each its own grid origin"
    for (gx, gy), idx in stacks:
        cols = sorted({ranks[i][0] for i in idx})
        rows = sorted({ranks[i][1] for i in idx})
        for i in idx:
            off[i] = (origin[0] + cols.index(ranks[i][0]) * step - gx, origin[1] - rows.index(ranks[i][1]) * step - gy)
    return off


def assert_clear(ax, patch, discs, name):
    ax.figure.canvas.draw()
    bb = patch.get_window_extent()
    bad = []
    for x, y, size in discs:
        cx, cy = ax.transData.transform((x, y))
        dx = max(bb.x0 - cx, 0.0, cx - bb.x1)
        dy = max(bb.y0 - cy, 0.0, cy - bb.y1)
        gap_pt = ((dx * dx + dy * dy) ** 0.5 - size / 2.0 * ax.figure.dpi / 72.0) * 72.0 / ax.figure.dpi
        if gap_pt < 1.0:
            bad.append(((round(x, 2), round(y, 2)), round(gap_pt, 2)))
    assert not bad, "%s: the outline comes within 1 pt of a marker: %s" % (name, bad)


def assert_separated(ax, pts, ms, name):
    ax.figure.canvas.draw()
    inch = ax.transData.transform(pts) / ax.figure.dpi
    bad = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(((inch[i] - inch[j]) ** 2).sum() ** 0.5)
            if d < ms / 72.0:
                bad.append((pts[i], pts[j], round(d * 72.0, 2)))
    assert not bad, "%s: markers overlap (centre distance in points below the marker size %.1f): %s" % (name, ms, bad)


def minus_fmt(v, _pos=None):
    if v == 0:
        return "0"
    if abs(v - round(v)) < 1e-9:
        v = int(round(v))
    return ("$-$%g" % -v) if v < 0 else "%g" % v


def _chevron(direction):
    arms = {"up": [(-1, -1), (0, 0), (1, -1)], "down": [(-1, 1), (0, 0), (1, 1)],
            "right": [(-1, 1), (0, 0), (-1, -1)], "left": [(1, 1), (0, 0), (1, -1)]}[direction]
    return MPath(arms, [MPath.MOVETO, MPath.LINETO, MPath.LINETO])


CHEVRON = {d: _chevron(d) for d in ("up", "down", "left", "right")}


def chevron(ax, x, y, direction, color, size=5.4, lw=1.0, zorder=3, alpha=1.0):
    ax.plot([x], [y], marker=CHEVRON[direction], ms=size, mfc="none", mec=color, mew=lw, ls="none", zorder=zorder,
            alpha=alpha, clip_on=False)


def clip_to(lo, hi, limits):
    return max(lo, limits[0]), min(hi, limits[1]), lo < limits[0], hi > limits[1]


def hinterval(ax, y, lo, hi, limits, color, lw=1.2, cap=0.28, size=5.4, zorder=3):
    clo, chi, lc, hc = clip_to(lo, hi, limits)
    ax.plot([clo, chi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)
    for x, was_clipped, d in ((clo, lc, "left"), (chi, hc, "right")):
        if was_clipped:
            chevron(ax, x, y, d, color, size=size, lw=max(lw, 0.9), zorder=zorder)
        elif cap:
            ax.plot([x, x], [y - cap / 2, y + cap / 2], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)


def vinterval(ax, x, lo, hi, limits, color, lw=1.0, cap=2.2, size=5.4, zorder=2.5, alpha=1.0):
    clo, chi, lc, hc = clip_to(lo, hi, limits)
    ax.plot([x, x], [clo, chi], color=color, lw=lw, solid_capstyle="butt", zorder=zorder, alpha=alpha)
    for y, was_clipped, d in ((clo, lc, "down"), (chi, hc, "up")):
        if was_clipped:
            chevron(ax, x, y, d, color, size=size, lw=max(lw, 0.9), zorder=zorder, alpha=alpha)
        elif cap:
            ax.plot([x - cap, x + cap], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder, alpha=alpha)


def summary_rows(ax, y_me, y_re, est, limits=(-100, 100)):
    me, re_ = need(est, "model_equal"), need(est, "random_effects")
    hinterval(ax, y_me, me["ci"][0], me["ci"][1], limits, WINE, lw=1.6, cap=0.5)
    ax.plot([me["mean"]], [y_me], marker="D", ms=5.2, color=WINE, mec="white", mew=0.5, ls="none", zorder=4)
    hinterval(ax, y_re, re_["pi"][0], re_["pi"][1], limits, INK, lw=0.9, cap=0.5)
    hinterval(ax, y_re, re_["ci_hk"][0], re_["ci_hk"][1], limits, INK, lw=3.0, cap=0, zorder=3.5)
    ax.plot([re_["mu"]], [y_re], marker="s", ms=5.8, color=INK, mec="white", mew=0.7, ls="none", zorder=4)


SUMMARY_LABELS = ["Model-equal mean,\nt-based 95% CI",
                  "Random-effects mean,\nHartung–Knapp 95% CI,\n95% prediction interval"]


def spread(ys, gap):
    idx = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in idx]
    for _ in range(200):
        moved = False
        for j in range(len(pos) - 1):
            d = pos[j + 1] - pos[j]
            if d < gap - 1e-9:
                shift = (gap - d) / 2.0
                pos[j] -= shift
                pos[j + 1] += shift
                moved = True
        if not moved:
            break
    out = [0.0] * len(ys)
    for p_, i in zip(pos, idx):
        out[i] = p_
    return out


def swarm(vals, step, tol=2.0):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    groups, cur = [], []
    for i in order:
        if cur and vals[i] - vals[cur[-1]] > tol:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    off = [0.0] * len(vals)
    for g in groups:
        for r, i in enumerate(g):
            off[i] = (r - (len(g) - 1) / 2.0) * step
    return off


def callouts(ax, items, x_text, others, gap_in=0.13, ha="right", fontsize=T_SMALL + 0.3):
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    xr, yr = ax.get_xlim(), ax.get_ylim()
    ix = (bb.width / fig.dpi) / (xr[1] - xr[0])
    iy = (bb.height / fig.dpi) / (yr[1] - yr[0])

    def seg_dist(px, py, ax_, ay_, bx_, by_):
        dx, dy = bx_ - ax_, by_ - ay_
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax_) * dx + (py - ay_) * dy) / (dx * dx + dy * dy)))
        return ((px - (ax_ + t * dx)) ** 2 + (py - (ay_ + t * dy)) ** 2) ** 0.5

    prev = None
    for item in sorted(items, key=lambda t: t[1]):
        x, y, text = item[:3]
        if len(item) > 3:
            spec = item[3]
            ax.annotate(text, xy=(x, y), xytext=spec["at"], textcoords="data", fontsize=fontsize, color=INK,
                        ha=spec.get("ha", "center"), va=spec.get("va", "top"), zorder=3.6, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, shrinkA=2, shrinkB=3.5,
                                        relpos=spec.get("relpos", (0.5, 1.0)), connectionstyle="arc3,rad=0"))
            continue
        best = None
        for k in range(-40, 41):
            yl = y + k * (0.045 / iy)
            if prev is not None and (yl - prev) * iy < gap_in:
                continue
            if not (yr[0] + 0.03 / iy <= yl <= yr[1] - 0.03 / iy):
                continue
            cost = abs(yl - y) * iy
            for (ox, oy) in others:
                if (ox, oy) == (x, y):
                    continue
                d = seg_dist(ox * ix, oy * iy, (x_text + (0.5 if ha == "right" else -0.5)) * ix, yl * iy, x * ix, y * iy)
                if d < 0.07:
                    cost += 5.0
            if best is None or cost < best[0]:
                best = (cost, yl)
        yl = best[1] if best else y
        prev = yl
        ax.annotate(text, xy=(x, y), xytext=(x_text, yl), textcoords="data", fontsize=fontsize, color=INK,
                    ha=ha, va="center", zorder=3.6,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, shrinkA=2, shrinkB=3.5,
                                    connectionstyle="arc3,rad=0"))


def assert_no_text_collisions(fig, name, slack_pt=0.6):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    items = []
    for t in fig.findobj(Text):
        s_ = (t.get_text() or "").strip()
        if s_ and t.get_visible():
            bb = Text.get_window_extent(t, rend)
            if bb.width > 0 and bb.height > 0:
                items.append((s_.replace("\n", " / "), bb))
    bad = []
    for i, (n1, b1) in enumerate(items):
        for n2, b2 in items[i + 1:]:
            w = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            h = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if w > slack_pt * px and h > slack_pt * px:
                bad.append((n1[:38], n2[:38], round(w / px, 1), round(h / px, 1)))
    assert not bad, "%s: overlapping text: %s" % (name, bad)


def save(fig, name, stem):
    assert_no_text_collisions(fig, name)
    return save_audited(fig, name, stem, outdir=OUT)


def ax_in(fig, left, bottom, width, height):
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def letter(fig, text, x_in, y_in):
    W, H = fig.get_size_inches()
    fig.text(x_in / W, y_in / H, text, fontsize=14, fontweight="bold", color=INK, va="top", ha="left")


def strip_ticks(ax, which="y"):
    (ax.yaxis if which == "y" else ax.xaxis).set_tick_params(length=0)


def tier_dot(ax, x, y, frontier, ms=5.0, alpha=1.0, zorder=4):
    face = INK if frontier else ("none" if abs(y) < 1e-9 else "white")
    ax.plot([x], [y], marker="o", ms=ms, ls="none", mfc=face, mec=INK, mew=0.8, alpha=alpha, zorder=zorder)


ZERO_BAR = 1.0


def zero_mark(ax, x, width, color, lw=2.6):
    ax.plot([x - width / 2, x + width / 2], [0, 0], color=color, lw=lw, solid_capstyle="butt", zorder=3.4,
            clip_on=False)


def fig2():
    off = offline()
    rro = need(off, "role_rates_original")
    per, mem = need(rro, "per_model"), need(rro, "model_equal_mean")
    est = need(off, "primary_estimands", "devotion_minus_steward")
    models = list(per)
    k = int(need(mem, "steward", "k"))
    assert k == len(models) == int(need(est, "k")), "model counts disagree between the Figure 2 blocks"
    fig = plt.figure(figsize=(9.0, 4.75))
    a = ax_in(fig, 0.78, 1.18, 4.05, 3.05)
    b = ax_in(fig, 6.80, 1.18, 2.05, 3.05)
    xs = list(range(len(ROLE_ORDER)))
    for m in models:
        a.plot(xs, [per[m][r]["harm_rate_pct"] for r in ROLE_ORDER], color=OTHER_LINE, lw=0.9, zorder=1)
    means = [mem[r]["harm_rate_pct"] for r in ROLE_ORDER]
    for x, r in zip(xs, ROLE_ORDER):
        vinterval(a, x, mem[r]["harm_ci95"][0], mem[r]["harm_ci95"][1], (0, 100), WINE, lw=1.5, cap=0.07, zorder=3)
    a.plot(xs, means, color=WINE, lw=2.0, marker="o", ms=5, zorder=4)
    a.set_xticks(xs)
    a.set_xticklabels([ROLE_LABEL[r] for r in ROLE_ORDER], fontsize=7.4)
    a.set_xlim(-0.4, len(xs) - 0.6)
    a.set_ylim(0, 100)
    a.set_ylabel("Acquisitions discordant with the\nstudy's allocation standard (%)")
    a.set_title("Policy not stated · %d models" % k, fontsize=T_TITLE, loc="left", pad=6)
    style(a)

    diff = need(est, "per_model_pp")
    order = sorted(diff, key=diff.get)
    ys = {m: i + 4 for i, m in enumerate(order)}
    b.axvline(0, color=GREY, lw=0.8, ls=":", zorder=0)
    for m in order:
        b.plot([diff[m]], [ys[m]], marker="o", ms=4.8, ls="none", mfc=OTHER_DOT, mec="white", mew=0.5, zorder=3)
    y_me, y_re = 2.4, 0.0
    summary_rows(b, y_me, y_re, est)
    b.axhline(3.4, color=LIGHT_GREY, lw=0.7, zorder=0)
    b.set_yticks([ys[m] for m in order] + [y_me, y_re])
    b.set_yticklabels(order + SUMMARY_LABELS, fontsize=T_SMALL + 0.3)
    b.set_ylim(-0.9, len(order) + 3.8)
    b.set_xlim(-25, 100)
    b.set_xticks([-25, 0, 25, 50, 75, 100])
    b.xaxis.set_major_formatter(FuncFormatter(minus_fmt))
    role1 = one_line(ROLE_LABEL["steward"])
    b.set_xlabel("Strong advocacy minus\n%s%s\n(percentage points)" % (role1[0].lower(), role1[1:]))
    b.set_title("Per-model difference", fontsize=T_TITLE, loc="left", pad=6)
    style(b)
    strip_ticks(b, "y")
    b.spines["left"].set_visible(False)
    letter(fig, "A", 0.10, 4.72)
    letter(fig, "B", 5.02, 4.72)
    handles = [Line2D([], [], color=OTHER_LINE, lw=1.4, label="One model"),
               Line2D([], [], color=WINE, lw=2.0, marker="o", ms=4.5, label="Model-equal mean, t-based 95% CI")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=2.2, fontsize=T_SMALL + 0.5)
    save(fig, "Figure 2", "Figure_2")
    plt.close(fig)


ROLES_A = ("steward", "guardrail", "empathetic", "neutral", "devotion", "sx_ward_plain", "sx_patient_plain",
           "devotion_precedence")
ROLE_MS = {"o": 7.0, "^": 9.0, "D": 6.0}
MEAN_MS = {"o": 6.2, "^": 7.8, "D": 5.4}
NUM_DY = {"o": 0.0, "^": -1.0, "D": 0.0}
FRONT_MS = (3.4, 3.8)
FRONT_STEP, GRID_ORIGIN = 1.40, (4.4, 106.2)


class NumMark:
    def __init__(self, role, number):
        self.role, self.number = role, number


class HandlerNumMark(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        cx, cy = xdescent + width / 2.0, ydescent + height / 2.0
        shape = ROLE_MARK.get(orig_handle.role, "o")
        col = ROLE_COLOR[orig_handle.role]
        mk = Line2D([cx], [cy], marker=shape, ms=ROLE_MS[shape], mfc=col, mec="white", mew=0.5, ls="none",
                    transform=trans)
        tx = Text(cx, cy + NUM_DY[shape], str(orig_handle.number), ha="center", va="center_baseline",
                  fontsize=T_NUM, color=num_color(col), transform=trans)
        return [mk, tx]


def fig3():
    R = runs()
    rr = need(R, "role_rates")
    nr, er = need(rr, "NOT_claimed"), need(rr, "ENT_claimed")
    fn, fe = need(rr, "frontier_NOT_claimed"), need(rr, "frontier_ENT_claimed")
    ref = need(R, "reference_policies")
    sc = need(R, "primary_scope")
    mg = need(R, "margin", "model_equal_by_margin")
    k_all = int(need(nr, "steward", "k"))
    W, H = 8.9, 8.62
    fig = plt.figure(figsize=(W, H))

    P, PX, PY = 2.90, 0.88, 5.20
    a = ax_in(fig, PX, PY, P, P)
    LIM = (-4.0, 112.0)
    a.set_xlim(*LIM)
    a.set_ylim(*LIM)
    a.set_aspect("equal", adjustable="box")
    a.plot([0, 100], [0, 100], color=LIGHT_GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    refs = {}
    for key in ("ref-rule", "ref-always", "ref-never"):
        refs[key] = (need(ref, key, "NOT", "claimed_vent"), need(ref, key, "ENT", "claimed_vent"))
    for key, (x, y) in refs.items():
        a.plot([x], [y], marker="s", ms=6.5, mfc=LIGHT_GREY, mec=INK, mew=0.8, ls="none", zorder=3)
    a.text(refs["ref-rule"][0] + 3.5, refs["ref-rule"][1] + 4.6, "Rule follower", fontsize=T_SMALL, color=INK,
           ha="left", va="bottom")
    a.text(refs["ref-always"][0] - 2.5, refs["ref-always"][1] - 5.2, "Always claims", fontsize=T_SMALL, color=INK,
           ha="right", va="top")
    a.text(refs["ref-never"][0] + 4.0, refs["ref-never"][1], "Never claims", fontsize=T_SMALL, color=INK, ha="left",
           va="center")

    def frontier_points(ax, ms, spread=False):
        pts, ranks, roles = [], [], []
        for ri, role in enumerate(ROLES_A):
            xs, ys_ = need(fn, role, "per_model"), need(fe, role, "per_model")
            for m in sorted(xs):
                pts.append((xs[m], ys_[m]))
                ranks.append((ri, m))
                roles.append(role)
        off = grid_identical(pts, ranks, FRONT_STEP, GRID_ORIGIN) if spread else [(0.0, 0.0)] * len(pts)
        drawn = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(pts, off)]
        for (x, y), role in zip(drawn, roles):
            ax.plot([x], [y], marker=ROLE_MARK.get(role, "o"), ms=ms, mfc=ROLE_COLOR[role], mec=INK, mew=0.6,
                    alpha=0.9, ls="none", zorder=4)
        stack = [i for i, o in enumerate(off) if o != (0.0, 0.0)]
        if not stack:
            return drawn, None
        home = pts[stack[0]]
        assert all(pts[i] == home for i in stack), "the laid-out points do not share one position"
        reach = max({"o": 0.5, "^": 0.5, "D": 0.5 * 2 ** 0.5}[ROLE_MARK.get(roles[i], "o")] for i in stack) * ms + 0.6
        r = reach / (ax.get_window_extent().width / ax.figure.dpi * 72.0 / (ax.get_xlim()[1] - ax.get_xlim()[0]))
        xs_, ys_m = [drawn[i][0] for i in stack], [drawn[i][1] for i in stack]
        x0, x1, y0, y1 = min(xs_) - r - 0.30, max(xs_) + r + 0.25, min(ys_m) - r - 0.25, max(ys_m) + r + 0.30
        outline = ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=GREY, lw=0.7, ls=(0, (1.5, 1.8)),
                                         zorder=3))
        ax.annotate("", xy=home, xytext=(x0, y0), textcoords="data", zorder=3,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, shrinkA=0, shrinkB=4.6))
        ax.text(x1 + 0.9, (y0 + y1) / 2.0, "%d high-capability points at (%g, %g)" % (len(stack), home[0], home[1]),
                ha="left", va="center", fontsize=T_SMALL, color=INK, zorder=3)
        return drawn, outline

    frontier_points(a, FRONT_MS[0])
    for role in ROLES_A:
        shape = ROLE_MARK.get(role, "o")
        a.plot([need(nr, role, "mean")], [need(er, role, "mean")], marker=shape, ms=MEAN_MS[shape],
               mfc=ROLE_COLOR[role], mec="white", mew=0.5, ls="none", zorder=5)
    a.set_xticks([0, 25, 50, 75, 100])
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_xlabel("Inappropriate claims, rival favoured (%)")
    a.set_ylabel("Appropriate claims,\nown patient favoured (%)")
    a.set_title("Policy stated · balanced pairs · %d models" % k_all, fontsize=T_TITLE, loc="left", pad=6)
    style(a)

    ZX = PX + P + 0.60
    ZW = W - 0.42 - ZX
    BX = (-2.5, 66.0)
    BY = (74.0, 110.0)
    ZH = ZW * (BY[1] - BY[0]) / (BX[1] - BX[0])
    ZY = PY + P - ZH
    z = ax_in(fig, ZX, ZY, ZW, ZH)
    z.set_xlim(*BX)
    z.set_ylim(*BY)
    z.set_aspect("equal", adjustable="box")
    a.add_patch(Rectangle((BX[0], BY[0]), BX[1] - BX[0], BY[1] - BY[0], fill=False, ec=INK, lw=0.8, zorder=6))
    for (xa, ya, xb, yb) in ((BX[1], BY[1], BX[0], BY[1]), (BX[1], BY[0], BX[0], BY[0])):
        fig.add_artist(ConnectionPatch(xyA=(xa, ya), coordsA=a.transData, xyB=(xb, yb), coordsB=z.transData,
                                       color=GREY, lw=0.7, zorder=1))
    x0, y0 = refs["ref-rule"]
    z.plot([x0], [y0], marker="s", ms=6.5, mfc=LIGHT_GREY, mec=INK, mew=0.8, ls="none", zorder=3)
    placed, outline = frontier_points(z, FRONT_MS[1], spread=True)
    inside = [p for p in placed if BX[0] <= p[0] <= BX[1] and BY[0] <= p[1] <= BY[1]]
    in_grid = {p for p in placed if p[1] > 100.0}
    assert in_grid and max(y for _, y in in_grid) + 1.0 < BY[1], "the grid does not fit inside the enlargement"
    assert all(0.0 <= x <= 100.0 and 0.0 <= y <= 100.0 for x, y in placed if (x, y) not in in_grid), \
        "a frontier point that is not in the labelled grid lies outside the possible range"
    assert_separated(z, inside, FRONT_MS[1] * 2 ** 0.5, "Figure 3A enlargement: frontier points")
    for role in ROLES_A:
        x, y = need(nr, role), need(er, role)
        hinterval(z, y["mean"], x["ci"][0], x["ci"][1], (0, 100), ROLE_COLOR[role], lw=0.8, cap=1.5, zorder=2)
        vinterval(z, x["mean"], y["ci"][0], y["ci"][1], (0, 100), ROLE_COLOR[role], lw=0.8, cap=1.0, zorder=2)
    discs = []
    for i, role in sorted(enumerate(ROLES_A), key=lambda t: t[1] in ROLE_ORDER):
        shape = ROLE_MARK.get(role, "o")
        x, y = need(nr, role, "mean"), need(er, role, "mean")
        z.plot([x], [y], marker=shape, ms=ROLE_MS[shape], mfc=ROLE_COLOR[role], mec="white", mew=0.5, ls="none",
               zorder=5)
        discs.append((x, y, ROLE_MS[shape]))
        z.annotate(str(i + 1), (x, y), xytext=(0, NUM_DY[shape]), textcoords="offset points",
                   ha="center", va="center_baseline", fontsize=T_NUM, color=num_color(ROLE_COLOR[role]), zorder=6)
    assert_clear(z, outline, discs, "Figure 3A enlargement: outline of the grid")
    z.set_xticks([0, 10, 20, 30, 40, 50, 60])
    z.set_yticks([80, 90, 100])
    z.yaxis.tick_right()
    z.set_title("Enlarged view", fontsize=T_SMALL + 0.5, loc="left", pad=3)
    style(z)
    for s_ in z.spines.values():
        s_.set_visible(True)
        s_.set_color(INK)
        s_.set_linewidth(0.8)

    kx = ax_in(fig, ZX, 5.20 - 0.40, ZW, ZY - 0.38 - (5.20 - 0.40))
    kx.axis("off")
    handles = [NumMark(r, i + 1) for i, r in enumerate(ROLES_A)]
    labels = [one_line(ROLE_LABEL[r]) for r in ROLES_A]
    handles += [Line2D([], [], marker="o", ms=FRONT_MS[1], mfc=LIGHT_GREY, mec=INK, mew=0.6, ls="none"),
                Line2D([], [], marker="s", ms=5.6, mfc=LIGHT_GREY, mec=INK, mew=0.8, ls="none")]
    labels += ["High-capability model (small marker,\none per model and role)", "Scripted reference policy"]
    kx.legend(handles=handles, labels=labels, handler_map={NumMark: HandlerNumMark()}, loc="upper left", ncol=2,
              frameon=False, fontsize=T_SMALL, handletextpad=0.35, labelspacing=0.34, columnspacing=1.0,
              borderaxespad=0.0, borderpad=0.0, handlelength=1.5,
              title="Numbered marker: model-equal mean of each role", title_fontsize=T_SMALL, alignment="left")

    b = ax_in(fig, 1.90, 0.82, 1.62, 3.45)
    pm = need(sc, "per_model")
    order = sorted(pm, key=pm.get)
    ys = {m: i + 4 for i, m in enumerate(order)}
    b.axvline(0, color=GREY, lw=0.8, ls=":", zorder=0)
    for m in order:
        tier_dot(b, pm[m], ys[m], is_frontier(R, m), ms=4.8, zorder=3)
    y_me, y_re = 2.4, 0.0
    summary_rows(b, y_me, y_re, sc)
    b.axhline(3.4, color=LIGHT_GREY, lw=0.7, zorder=0)
    b.set_yticks([ys[m] for m in order] + [y_me, y_re])
    b.set_yticklabels([model_name(R, m) for m in order] + SUMMARY_LABELS, fontsize=T_SMALL + 0.3)
    b.set_ylim(-0.9, len(order) + 3.8)
    b.set_xlim(-12, 36)
    b.set_xticks([-10, 0, 10, 20, 30])
    b.xaxis.set_major_formatter(FuncFormatter(minus_fmt))
    b.set_xlabel("Patient scope minus ward scope,\ninappropriate claims (pp)")
    b.set_title("Matched scope", fontsize=T_TITLE, loc="left", pad=6)
    style(b)
    strip_ticks(b, "y")
    b.spines["left"].set_visible(False)
    hb = [Line2D([], [], marker="o", ms=4.8, mfc=INK, mec=INK, ls="none", label="High-capability model"),
          Line2D([], [], marker="o", ms=4.8, mfc="white", mec=INK, mew=0.8, ls="none", label="Panel model")]
    b.legend(handles=hb, loc="center right", bbox_to_anchor=(1.0, 0.50), frameon=False, fontsize=T_SMALL,
             handletextpad=0.2, labelspacing=0.3, borderaxespad=0.1)

    c = ax_in(fig, 4.65, 0.82, 2.70, 3.45)
    roles_c = ("steward", "neutral", "devotion")
    ends = []
    for role in roles_c:
        for half, ls in (("NOT", "-"), ("ENT", "--")):
            d = need(mg, "%s|%s" % (role, half))
            ks = sorted(d, key=float)
            c.plot([100.0 * float(x) for x in ks], [d[x] for x in ks], ls=ls, marker="o", ms=3.2, lw=1.5,
                   color=ROLE_COLOR[role], zorder=3 if half == "NOT" else 2)
            ends.append((role, half, d[ks[-1]], 100.0 * float(ks[-1])))
    x_end = max(e[3] for e in ends)
    gap = 100.0 * (T_SMALL * 1.25 / 72.0) / 3.45
    for half in ("NOT", "ENT"):
        grp = [e for e in ends if e[1] == half]
        moved = spread([e[2] for e in grp], gap)
        for (role, _, y, x), y_lab in zip(grp, moved):
            c.annotate(one_line(ROLE_LABEL[role]), (x, y), xytext=(x + 1.1, y_lab), textcoords="data",
                       fontsize=T_SMALL, color=darker(ROLE_COLOR[role]), va="center", ha="left",
                       annotation_clip=False)
    c.set_xlim(2.0, x_end + 2.0)
    c.set_xticks(sorted({100.0 * float(x) for x in mg["steward|NOT"]}))
    c.set_ylim(0, 100)
    c.set_yticks([0, 25, 50, 75, 100])
    c.set_xlabel("Survival-probability margin between the two patients (percentage points)")
    c.set_ylabel("Ventilator claims (%)")
    c.set_title("By survival margin", fontsize=T_TITLE, loc="left", pad=6)
    style(c)
    hc = [Line2D([], [], color=INK, lw=1.4, ls="-", label="Rival favoured (inappropriate claims)"),
          Line2D([], [], color=INK, lw=1.4, ls="--", label="Own patient favoured (appropriate claims)")]
    c.legend(handles=hc, loc="center left", bbox_to_anchor=(0.02, 0.62), frameon=False, fontsize=T_SMALL,
             handlelength=2.2, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.1)
    letter(fig, "A", 0.10, H - 0.06)
    letter(fig, "B", 0.10, 4.78)
    letter(fig, "C", 3.78, 4.78)
    save(fig, "Figure 3", "Figure_3")
    plt.close(fig)


C_MODES = ("none", "affirm", "no_affirm", "unrelated")
C_LABEL = {"none": "No\nreflection", "affirm": "Judgment\naffirmed", "no_affirm": "Judgment\nnot affirmed",
           "unrelated": "Unrelated\nquestions"}


def fig4():
    R = runs()
    par = need(R, "parity")
    ident = need(par, "knowledge", "devotion", "parsed", "per_model")
    after = need(par, "claimed_after_correct_devotion", "per_model")
    keys = [m for m in ident if m in after and after[m]["rate"] is not None]
    pts = [(ident[m], after[m]["rate"]) for m in keys]
    dx = dodge_identical(pts, 1.5, xmax=100.0)
    fig = plt.figure(figsize=(8.8, 4.15))
    a = ax_in(fig, 1.12, 1.20, 3.20, 2.70)
    items, seen = [], {}
    for m, (x, y), d in zip(keys, pts, dx):
        fr = is_frontier(R, m)
        tier_dot(a, x + d, y, fr, ms=6.4 if fr else 5.0, alpha=0.55 if after[m]["small_n"] else 1.0, zorder=5 if fr else 4)
        if fr:
            n_here = seen[(round(x, 6), round(y, 6))] = seen.get((round(x, 6), round(y, 6)), 0) + 1
            assert n_here <= 2, "more than two labelled frontier points coincide: place their labels by hand"
            spec = {"at": (x + d, -8.0), "ha": "center", "va": "top"} if n_here == 2 else None
            items.append((x + d, y, model_name(R, m)) + ((spec,) if spec else ()))
    a.set_xlim(48, 104)
    a.set_ylim(-15, 104)
    a.set_xticks([50, 60, 70, 80, 90, 100])
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_xlabel("Named the favoured patient, both halves (%)")
    a.set_ylabel("Claimed the ventilator at step 1\nwhen the rival was favoured,\nof correct answers (%)")
    a.set_title("Policy stated · strong advocacy · %d models" % len(keys), fontsize=T_TITLE, loc="left", pad=6)
    style(a)
    a.spines["left"].set_bounds(0, 100)
    callouts(a, items, x_text=80.0, others=[(x + d, y) for (x, y), d in zip(pts, dx)])
    ha = [Line2D([], [], marker="o", ms=6, mfc=INK, mec=INK, ls="none", label="High-capability model"),
          Line2D([], [], marker="o", ms=5, mfc="white", mec=INK, mew=0.8, ls="none", label="Panel model")]
    a.legend(handles=ha, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=T_SMALL,
             handletextpad=0.2, labelspacing=0.3, borderaxespad=0.2)

    rates = need(R, "jba", "rates")
    b = ax_in(fig, 5.25, 1.20, 3.40, 2.70)
    k_models = max(int(need(rates, "none|NOT", "k")), 1)
    for i, mode in enumerate(C_MODES):
        for h, (half, col) in enumerate((("NOT", WINE), ("ENT", SAGE))):
            v = need(rates, "%s|%s" % (mode, half))
            x = i + (h - 0.5) * 0.40
            b.bar([x], [v["mean"]], width=0.36, color=col, zorder=2)
            ms_ = sorted(v["per_model"])
            offs = swarm([v["per_model"][m] for m in ms_], 0.052)
            for m, o in zip(ms_, offs):
                tier_dot(b, x + o, v["per_model"][m], is_frontier(R, m), ms=3.6, zorder=4)
            if v["mean"] < ZERO_BAR:
                zero_mark(b, x, 0.36, col)
    b.set_xticks(range(len(C_MODES)))
    b.set_xticklabels([C_LABEL[m] for m in C_MODES], fontsize=7.4)
    b.set_xlim(-0.55, len(C_MODES) - 0.45)
    b.set_ylim(0, 105)
    b.set_yticks([0, 25, 50, 75, 100])
    b.set_ylabel("Ventilator claims (%)")
    b.set_title("Policy stated · judgment before action · %d models" % k_models, fontsize=T_TITLE, loc="left",
                pad=6)
    style(b)
    hb = [Patch(facecolor=WINE, label="Inappropriate claims, rival favoured"),
          Patch(facecolor=SAGE, label="Appropriate claims, own patient favoured"),
          Line2D([], [], marker="o", ms=4.4, mfc=INK, mec=INK, ls="none", label="High-capability model"),
          Line2D([], [], marker="o", ms=4.4, mfc="white", mec=INK, mew=0.8, ls="none", label="Panel model")]
    b.legend(handles=hb, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=False, fontsize=T_SMALL,
             handlelength=1.2, handletextpad=0.4, columnspacing=1.6, labelspacing=0.35, borderaxespad=0.0)
    letter(fig, "A", 0.10, 4.12)
    letter(fig, "B", 4.40, 4.12)
    save(fig, "Figure 4", "Figure_4")
    plt.close(fig)


CHECK_A = (("none", "No check"), ("partial_noisy", "Estimated\npriority (noisy)"),
           ("partial_tight", "Estimated\npriority (tight)"), ("oracle", "True priority\n(blocks by design)"))
CHECK_B = (("none", "No check"), ("partial", "Estimated\npriority"), ("protocol", "True priority\n(blocks by design)"))
LIVE_KEYS = (("inappropriate_NOT", WINE), ("appropriate_ENT", SAGE))


def fig5():
    rep = J(FS + "oversight_information_replay.json")
    R = runs()
    fig = plt.figure(figsize=(8.8, 4.25))
    a = ax_in(fig, 0.95, 1.30, 3.45, 2.70)
    pooled = need(rep, "pooled", "per_level")
    pm = need(rep, "per_model")
    for i, (lv, _) in enumerate(CHECK_A):
        v = need(pooled, lv, "chr_pct")
        bar = a.bar([i], [v], width=0.62, color=SLATE, zorder=2)[0]
        bar.set_hatch("////")
        bar.set_edgecolor("white")
        bar.set_linewidth(0)
        for n, m in enumerate(sorted(pm)):
            d = need(pm, m, "per_level", lv, "chr_pct")
            a.plot([i + wiggle(n, 0.34)], [d], marker="o", ms=3.0, ls="none", mfc="none" if d == 0 else "white",
                   mec=INK, mew=0.6, alpha=0.9, zorder=4)
        if v < ZERO_BAR:
            zero_mark(a, i, 0.62, SLATE)
    a.set_xticks(range(len(CHECK_A)))
    a.set_xticklabels([t for _, t in CHECK_A], fontsize=7.4)
    a.set_xlim(-0.6, len(CHECK_A) - 0.4)
    a.set_ylim(0, 102)
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel("Acquisitions discordant with the\nstudy's allocation standard (%)")
    a.set_title("Policy not stated · replay · %d models" % len(pm), fontsize=T_TITLE, loc="left", pad=6)
    style(a)
    ha = [Patch(facecolor=SLATE, hatch="////", edgecolor="white", label="All models pooled"),
          Line2D([], [], marker="o", ms=3.4, mfc="white", mec=INK, mew=0.6, ls="none", label="One model")]
    a.legend(handles=ha, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=False, fontsize=T_SMALL,
             handlelength=1.2, handletextpad=0.4, columnspacing=1.6, borderaxespad=0.0)

    lc = need(R, "live_check")
    per, me = need(lc, "per_model"), need(lc, "model_equal")
    b = ax_in(fig, 5.30, 1.30, 3.30, 2.70)
    for i, (lv, _) in enumerate(CHECK_B):
        for h, (key, col) in enumerate(LIVE_KEYS):
            x = i + (h - 0.5) * 0.40
            v = need(me, "%s|%s" % (lv, key))
            b.bar([x], [v], width=0.36, color=col, zorder=2)
            ms_ = [m for m in sorted(per) if per[m].get(lv, {}).get(key, {}).get("rate") is not None]
            offs = swarm([per[m][lv][key]["rate"] for m in ms_], 0.056)
            for m, o in zip(ms_, offs):
                tier_dot(b, x + o, per[m][lv][key]["rate"], is_frontier(R, m), ms=3.8, zorder=4)
            if v < ZERO_BAR:
                zero_mark(b, x, 0.36, col)
    b.set_xticks(range(len(CHECK_B)))
    b.set_xticklabels([t for _, t in CHECK_B], fontsize=7.4)
    b.set_xlim(-0.55, len(CHECK_B) - 0.45)
    b.set_ylim(0, 105)
    b.set_yticks([0, 25, 50, 75, 100])
    b.set_ylabel("Ventilator acquired (%)")
    b.set_title("Policy stated · live check · %d models" % len(per), fontsize=T_TITLE, loc="left", pad=6)
    style(b)
    hb = [Patch(facecolor=WINE, label="Inappropriate acquisition, rival favoured"),
          Patch(facecolor=SAGE, label="Appropriate acquisition, own patient favoured"),
          Line2D([], [], marker="o", ms=4.4, mfc=INK, mec=INK, ls="none", label="High-capability model"),
          Line2D([], [], marker="o", ms=4.4, mfc="white", mec=INK, mew=0.8, ls="none", label="Panel model")]
    b.legend(handles=hb, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=False, fontsize=T_SMALL,
             handlelength=1.2, handletextpad=0.4, columnspacing=1.6, labelspacing=0.35, borderaxespad=0.0)
    letter(fig, "A", 0.10, 4.22)
    letter(fig, "B", 4.40, 4.22)
    save(fig, "Figure 5", "Figure_5")
    plt.close(fig)


ROLES_O = ("steward", "neutral", "devotion")
MARK_O = ("o", "s", "^", "D")


def appendix_fig2():
    R, off = runs(), offline()
    orig = need(off, "role_rates_original", "per_model")
    od = need(R, "original_design_frontier", "per_model")
    fig = plt.figure(figsize=(6.9, 3.9))
    a = ax_in(fig, 0.95, 0.80, 4.20, 2.80)
    xs = list(range(len(ROLES_O)))
    for m in orig:
        a.plot(xs, [need(orig, m, r, "harm_rate_pct") for r in ROLES_O], color=OTHER_LINE, lw=0.9, zorder=1)
    names = sorted(od, key=lambda m: od[m]["devotion"])
    step = 0.07
    ends = []
    for n, m in enumerate(names):
        off_x = (n - (len(names) - 1) / 2.0) * step
        ys_ = [od[m][r] for r in ROLES_O]
        a.plot([x + off_x for x in xs], ys_, color=INK, lw=1.3, marker=MARK_O[n % len(MARK_O)], ms=5.0, mfc=INK,
               mec="white", mew=0.5, zorder=3)
        ends.append((m, xs[-1] + off_x, ys_[-1]))
    gap = 100.0 * (T_SMALL * 1.45 / 72.0) / 2.80
    moved = spread([e[2] for e in ends], gap)
    for n, ((m, x, y), y_lab) in enumerate(zip(ends, moved)):
        x_lab = xs[-1] + 0.22
        a.plot([x_lab], [y_lab], marker=MARK_O[n % len(MARK_O)], ms=5.0, mfc=INK, mec="white", mew=0.5, ls="none",
               clip_on=False, zorder=3)
        a.annotate(model_name(R, m), (x_lab, y_lab), xytext=(6, 0), textcoords="offset points", fontsize=T_SMALL,
                   color=INK, va="center", ha="left", annotation_clip=False)
    a.annotate("%d experiment 1 models" % len(orig), (xs[0], 4), xytext=(xs[0] - 0.30, 100), textcoords="data",
               fontsize=T_SMALL, color=GREY, va="top", ha="left")
    a.set_xticks(xs)
    a.set_xticklabels([ROLE_LABEL[r] for r in ROLES_O], fontsize=7.6)
    a.set_xlim(-0.4, len(xs) - 0.4)
    a.set_ylim(-9, 100)
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel("Acquisitions discordant with the\nstudy's allocation standard (%)")
    a.set_title("Policy not stated · frontier models", fontsize=T_TITLE, loc="left", pad=6)
    style(a)
    a.spines["left"].set_bounds(0, 100)
    save(fig, "Appendix figure 2", "Appendix_Figure_2")
    plt.close(fig)


ALL = ("fig2", "fig3", "fig4", "fig5", "appendix_fig2")
EXTERNAL = ("scripts/fig1_scope_of_concern.py", "scripts/fig4_words_ride_prior.py", "scripts/fig_scope2x2_forest.py",
            "scripts/fig_model_arm_matrix.py")


def external():
    import subprocess
    env = dict(os.environ, FIG_OUTDIR=OUT)
    for script in EXTERNAL:
        subprocess.run([sys.executable, "-W", "ignore", script], env=env, check=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    chosen = sys.argv[1:] or list(ALL) + ["external"]
    for name in chosen:
        globals()[name]()
