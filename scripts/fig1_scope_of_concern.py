"""Draws the study overview figure: the ward scene under whole-ward and single-patient assignment, the evidence flow and the policy statement line. It reads the experiment 2 primary, secondary and experiment 1 reanalysis results and several experiment 1 result files, and writes the figure as PDF and PNG to the directory in FIG_OUTDIR (default figures)."""
import json
import os
from pathlib import Path
import sys; sys.path.insert(0, 'scripts')
from _figstyle import (plt, panel_label, save_audited, WINE, SAGE, SAGE_PALE, TEAL, TEAL_PALE, SLATE_PALE,
                       SLATE, GREY, LIGHT_GREY, INK)
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, FancyArrowPatch

BLUE = '#2F5D8A'
SAGE_DARK = '#3F5E48'
RESOURCES = ['1 ventilator', '1 imaging slot', 'on-call attending']
OUTDIR = os.environ.get('FIG_OUTDIR', 'figures')
E2_MODE = 'FIG_OUTDIR' in os.environ
STEM = 'Figure_1' if E2_MODE else 'fig1_scope_of_concern'
HEAD_X = 0.075 if E2_MODE else 0.045
Y_LO = 0.14 if E2_MODE else 0.0
E2_DIR = 'experiment2'
FS = 'experiment1/results/' 
if E2_MODE:
    sys.path.insert(0, 'scripts')
    from role_labels import ROLE_LABEL, one_line
R_PATIENT = 0.030

CX, CY = 0.50, 0.39
SEMI = {'patient': (0.2350, 0.1100),
        'ward':    (0.4700, 0.2200)}
ROW_Y = 0.44
PAIR_X = (0.115, 0.195)
PAIR_LAB_X = 0.155
HIGH_X = 0.845
AI_XY, AI_WH = (0.50, 0.735), (0.104, 0.066)
LAB_DY = 0.042
STATUS_DY = 0.120
OUT_Y = 0.085
STRIP_Y0, STRIP_H = 0.882, 0.072


def chip(ax, x, y, text, w=0.235, h=0.050, fs=7.0):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle='round,pad=0.004,rounding_size=0.018', fc='white', ec=GREY, lw=0.9, zorder=5))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=INK, zorder=6)


def resource_strip(ax):
    ax.text(0.5, 0.980, 'Shared, scarce ward resources', ha='center', va='center', fontsize=7.6,
            color=GREY, zorder=3)
    ax.add_patch(FancyBboxPatch((0.09, STRIP_Y0), 0.82, STRIP_H,
                 boxstyle='round,pad=0.004,rounding_size=0.02', fc=LIGHT_GREY, ec=GREY, lw=0.8, zorder=2))
    for i, r in enumerate(RESOURCES):
        chip(ax, 0.235 + i * 0.265, STRIP_Y0 + STRIP_H / 2, r)


def dot(ax, x, y, r, fc, label=None, tcol=None, fs=7.8):
    m = ax.add_patch(Circle((x, y), r, fc=fc, ec='white', lw=1.2, zorder=6))
    t = None
    if label:
        t = ax.text(x, y - LAB_DY, label, ha='center', va='top', fontsize=fs, color=tcol or fc,
                    fontweight='bold', zorder=6, linespacing=1.15)
    return m, t


def scene(ax, scope):
    ax.set_xlim(0, 1); ax.set_ylim(Y_LO, 1); ax.axis('off')
    sx, sy = SEMI[scope]
    single = scope == 'patient'
    bnd = TEAL if single else SAGE
    bnd_fc = TEAL_PALE if single else SAGE_PALE
    bnd_tx = TEAL if single else SAGE_DARK
    head = 'Single-patient task' if single else 'Ward-steward task'
    high_fc = WINE if single else SAGE
    high_tx = WINE if single else SAGE_DARK
    out_tx = 'allocation-priority violation' if single else 'allocation priority preserved'
    out_col = WINE if single else SAGE_DARK
    side = 'outside' if single else 'inside'
    if E2_MODE:
        bnd, bnd_fc, bnd_tx = TEAL, TEAL_PALE, TEAL
        head = one_line(ROLE_LABEL['sx_patient_plain' if single else 'sx_ward_plain'])

    ax.text(HEAD_X, 1.0, head, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10.5, color=bnd_tx, fontweight='bold')
    resource_strip(ax)

    bnd_patch = ax.add_patch(Ellipse((CX, CY), 2 * sx, 2 * sy, fc=bnd_fc, ec=bnd, lw=1.6,
                                     ls=(0, (5, 3)), zorder=2))
    assert (bnd_patch.center, bnd_patch.width, bnd_patch.height) == ((CX, CY), 2 * sx, 2 * sy), \
        'boundary artist disagrees with the constants the containment guard tests against'
    chk = [(ax.text(CX, CY - sy + 0.033, 'TASK BOUNDARY', ha='center', va='center', fontsize=6.3,
                    color=bnd_tx, fontweight='bold', zorder=4,
                    bbox=dict(facecolor=bnd_fc, edgecolor='none', pad=1.5)),
            'inside', 'TASK BOUNDARY')]

    (aix, aiy), (aiw, aih) = AI_XY, AI_WH
    ax.plot([CX, CX], [STRIP_Y0 - 0.004, aiy + aih / 2 + 0.005], color=GREY, lw=1.0, zorder=3)
    chk.append((ax.add_patch(FancyBboxPatch((aix - aiw / 2, aiy - aih / 2), aiw, aih,
                boxstyle='round,pad=0.006,rounding_size=0.02', fc=BLUE, ec='white', lw=1.0, zorder=6)),
                'outside', 'AI box'))
    ax.text(aix, aiy, 'AI', ha='center', va='center', fontsize=9.5, color='white',
            fontweight='bold', zorder=7)

    if E2_MODE:
        for px, where in ((PAIR_LAB_X, 'left'), (HIGH_X, 'right')):
            m, t = dot(ax, px, ROW_Y, R_PATIENT, SLATE, 'other\npatient')
            chk += [(m, side, f'other patient dot ({where})'), (t, side, f'other patient ({where})')]
        m, t = dot(ax, CX, ROW_Y, R_PATIENT, TEAL, 'assigned\npatient')
        chk += [(m, 'inside', 'assigned dot'), (t, 'inside', 'assigned patient')]
    else:
        for i, px in enumerate(PAIR_X):
            m, _ = dot(ax, px, ROW_Y, 0.026, SLATE)
            chk.append((m, side, f'other ward dot {i + 1}'))
        chk.append((ax.text(PAIR_LAB_X, ROW_Y - LAB_DY, 'other ward\npatients', ha='center', va='top',
                            fontsize=7.8, color=SLATE, fontweight='bold', zorder=6, linespacing=1.15),
                    side, 'other ward patients'))
        m, t = dot(ax, CX, ROW_Y, 0.030, TEAL, 'assigned\npatient')
        chk += [(m, 'inside', 'assigned dot'), (t, 'inside', 'assigned patient')]
        m, t = dot(ax, HIGH_X, ROW_Y, 0.032, high_fc, 'higher-priority\npatient', tcol=high_tx)
        chk += [(m, side, 'higher-priority dot'), (t, side, 'higher-priority patient')]
        chk.append((ax.text(HIGH_X, ROW_Y - STATUS_DY, 'denied' if single else 'granted', ha='center',
                            va='top', fontsize=7.2, color=high_tx, style='italic', zorder=6),
                    side, 'higher-priority status'))

    if E2_MODE:
        covered = [(CX, R_PATIENT)] if single else [(px, R_PATIENT) for px in (PAIR_LAB_X, CX, HIGH_X)]
        for tx, r in covered:
            ax.add_patch(FancyArrowPatch((aix, aiy - aih / 2 - 0.004), (tx, ROW_Y + r + 0.010), arrowstyle='-|>',
                         mutation_scale=10, color=BLUE, lw=1.3, zorder=5, connectionstyle='arc3,rad=0'))
    else:
        tgt = (CX, ROW_Y + 0.038) if single else (HIGH_X - 0.036, ROW_Y + 0.024)
        ax.add_patch(FancyArrowPatch((aix, aiy - aih / 2 - 0.004), tgt, arrowstyle='-|>',
                     mutation_scale=13, color=out_col, lw=1.7, zorder=5,
                     connectionstyle='arc3,rad=0' if single else 'arc3,rad=-0.18'))

    if not E2_MODE:
        chk.append((ax.text(CX, OUT_Y, out_tx, ha='center', va='center', fontsize=8.4, color=out_col,
                            fontweight='bold', zorder=6), 'outside', 'outcome line'))
    return chk


def _containment_report(fig, ax, scope, checks, pad_pts=2.0):
    sx, sy = SEMI[scope]
    inv = ax.transData.inverted()
    rend = fig.canvas.get_renderer()
    bad = []
    for art, want, name in checks:
        bb = art.get_window_extent(renderer=rend).expanded(1, 1)
        pad = pad_pts * fig.dpi / 72.0
        (x0, y0), (x1, y1) = inv.transform([[bb.x0 - pad, bb.y0 - pad], [bb.x1 + pad, bb.y1 + pad]])
        norms = [((x - CX) / sx) ** 2 + ((y - CY) / sy) ** 2
                 for x, y in [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]]
        got = ('inside' if all(n < 1.0 for n in norms)
               else 'outside' if all(n > 1.0 for n in norms) else 'STRADDLES')
        if got != want:
            bad.append((scope, name, want, got, [round(n, 3) for n in norms]))
    return bad


def _overlap_report(fig, ax, scope, slack_pts=1.0):
    rend = fig.canvas.get_renderer()
    items = [(t.get_text().replace('\n', ' '), t.get_window_extent(renderer=rend).expanded(1, 1))
             for t in ax.texts if t.get_text().strip()]
    slack = slack_pts * fig.dpi / 72.0
    bad = []
    for i, (n1, b1) in enumerate(items):
        for n2, b2 in items[i + 1:]:
            dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0) - 2 * slack
            dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0) - 2 * slack
            if dx > 0 and dy > 0:
                bad.append((scope, n1, n2, round(dx / fig.dpi * 72, 1), round(dy / fig.dpi * 72, 1)))
    return bad


def thousands(n):
    n = int(n)
    return '{:,}'.format(n).replace(',', '\u2009') if n >= 10000 else str(n)


def flow_counts():
    R = json.load(open(E2_DIR + '/results/experiment2_primary.json'))
    O = json.load(open(E2_DIR + '/results/experiment1_reanalysis.json'))
    facts = json.load(open(E2_DIR + '/results/experiment2_secondary.json'))['facts']
    rep = json.load(open(FS + 'oversight_information_replay.json'))
    two_by_two = json.load(open(FS + 'scope_2x2_stats.json'))['per_model']
    probe = json.load(open(FS + 'comprehension_probe.json'))
    jba = json.load(open(FS + 'action_time_panel.json'))
    live = json.load(open(FS + 'live_oversight_balanced.json'))['per_model']
    comp, ep = R['completeness'], R['episodes']

    def run(name):
        rows = [v['rows'] for v in comp[name].values() if isinstance(v, dict) and 'rows' in v]
        assert sum(rows) == ep[name], 'completeness rows and episodes disagree for run ' + name
        return sum(1 for r in rows if r > 0), ep[name]

    two = [e for e in R['reconciliation']['table'] if e.get('data') == 'original 2x2, all models']
    assert len(two) == 1 and two[0]['policy'] == 'stated in all cells', \
        'the design record of the matched-wording experiment no longer says that its rules were stated in all cells'
    c2 = facts['C2']['values']
    assert c2['scenario_resource']['value']['ward_scenarios_2ward_bed.jsonl'] == {'ventilator': 100}, \
        'the matched-wording scenarios no longer contest a ventilator in every row'
    inc = O['inclusion_matrix']
    wording_n = sum(v['n_total'] for v in two_by_two.values())
    logs = {k: v for k, v in c2['logs']['value'].items() if k.startswith('ward_scope2x2_')}
    assert len(logs) == len(two_by_two) == len(inc['matched_wording_2x2']) and \
        sum(v['rows'] for v in logs.values()) == wording_n, 'matched-wording episodes disagree with the log rows'
    assert all(v['rules_stated'] == {'True': v['rows']} for v in logs.values()), 'a matched-wording row without a rule'
    probe_n = probe['pooled']['n']
    assert probe_n == sum(v['n'] for v in probe['per_model'].values()), 'probe replies disagree with the per-model n'
    jba_models = jba['summary']['models_self_binding']
    jba_n = sum(jba[m][cond][role]['n'] for m in jba_models for cond in ('self_binding', 'neutral') if cond in jba[m]
                for role in ('devotion', 'steward'))
    live_n = sum(d[half]['n'] for v in live.values() for d in v.values()
                 for half in ('APVR_when_not_rightful', 'appropriate_acq_when_rightful'))
    replay = {lv: sum(v['per_level'][lv]['total'] for v in rep['per_model'].values())
              for lv in rep['pooled']['per_level']}
    assert len(set(replay.values())) == 1, 'the replay levels hold different episodes'
    p_run = run('P')
    p_analysed = len(R['parity']['knowledge']['devotion']['parsed']['per_model'])
    p_dropped = [m for m, v in R['exclusions']['P'].items() if set(v['probe_cells_excluded']) >= {'devotion', 'neutral'}]
    assert p_analysed == p_run[0] - len(p_dropped), 'Run P: analysed models disagree with the exclusions'
    c_run = run('C')
    c_scen = {v['rows'] for v in comp['C'].values() if isinstance(v, dict) and v.get('rows')}
    c_modes = {k.split('|')[0] for k in R['jba']['rates']}
    assert len(c_scen) == 1 and c_run[0] * min(c_scen) == c_run[1], 'Run C: rows are not models x scenarios'
    return {'ladder_k': len(inc['primary_ladder']), 'ladder_n': O['flow']['primary_ladder'],
            'wording_k': len(inc['matched_wording_2x2']), 'wording_n': wording_n,
            'probe_k': len(inc['comprehension_probe']), 'probe_n': probe_n,
            'jba_k': len(inc['judgment_before_action']), 'jba_n': jba_n,
            'live_k': len(inc['live_check_balanced']), 'live_n': live_n,
            'replay_k': len(rep['per_model']), 'replay_n': replay['none'],
            'A': run('A'), 'P': p_run, 'P_analysed': p_analysed, 'C': c_run, 'C_scen': min(c_scen),
            'C_modes': len(c_modes), 'D': run('D'), 'D_none': ep['D_none'], 'O': run('O')}


def draw_flow(ax, fc, width, height):
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.axis('off')
    bw, bh, rg = 3.45, 0.50, 0.13
    xl, xr = 0.03, width - 0.03 - bw
    top = height - 0.44
    yc = [top - bh / 2 - i * (bh + rg) for i in range(5)]
    left = [('Five-role experiment', '%d models \u00b7 %s episodes \u00b7 policy not stated'
             % (fc['ladder_k'], thousands(fc['ladder_n']))),
            ('Matched-wording experiment', '%d models \u00b7 %s episodes \u00b7 rule stated, naming an ICU bed'
             % (fc['wording_k'], thousands(fc['wording_n']))),
            ('Priority-identification probe', '%d models \u00b7 %s probe replies \u00b7 results not used'
             % (fc['probe_k'], thousands(fc['probe_n']))),
            ('Judgment before action', '%d models \u00b7 %s single-action replies'
             % (fc['jba_k'], thousands(fc['jba_n']))),
            ('Allocation check', 'replay: %d models, %s recorded episodes \u00b7 live: %d models, %s episodes'
             % (fc['replay_k'], thousands(fc['replay_n']), fc['live_k'], thousands(fc['live_n'])))]
    not_used = {2}
    right = [('Experiment 1 repeated, high-capability models',
              '%d models \u00b7 %s agentic episodes \u00b7 policy not stated' % (fc['O'][0], thousands(fc['O'][1]))),
             ('Role instructions, matched pairs',
              '%d models \u00b7 %s agentic episodes \u00b7 policy stated' % (fc['A'][0], thousands(fc['A'][1]))),
             ('Priority identification',
              '%d models run, %d analysed \u00b7 %s probe replies \u00b7 policy stated'
              % (fc['P'][0], fc['P_analysed'], thousands(fc['P'][1]))),
             ('Judgment before action',
              '%d models $\\times$ %d scenarios $\\times$ %d modes \u00b7 policy stated'
              % (fc['C'][0], fc['C_scen'], fc['C_modes'])),
             ('Allocation check, live',
              '%d models \u00b7 %s agentic episodes, plus %s no-check \u00b7 policy stated'
              % (fc['D'][0], thousands(fc['D'][1]), thousands(fc['D_none'])))]
    lines = []
    for x, items, face, edge, lw, dashed in ((xl, left, SLATE_PALE, SLATE, 0.9, not_used),
                                             (xr, right, 'white', INK, 1.1, set())):
        for i, (yy, (title, sub)) in enumerate(zip(yc, items)):
            ax.add_patch(FancyBboxPatch((x, yy - bh / 2), bw, bh, boxstyle='round,pad=0.0,rounding_size=0.06',
                                        fc=face, ec=edge, lw=lw, zorder=2,
                                        ls=(0, (4, 2.5)) if i in dashed else '-'))
            lines.append((ax.text(x + 0.12, yy + 0.115, title, ha='left', va='center', fontsize=7.9,
                                  fontweight='bold', color=INK, zorder=3), x + bw))
            lines.append((ax.text(x + 0.12, yy - 0.115, sub, ha='left', va='center', fontsize=7.3, color=INK,
                                  zorder=3), x + bw))
    ax.text(xl + bw / 2, height - 0.15, 'Experiment 1', ha='center', va='center', fontsize=8.8, fontweight='bold',
            color='#5B6869')
    ax.text(xr + bw / 2, height - 0.15, 'Experiment 2', ha='center', va='center', fontsize=8.8, fontweight='bold',
            color=INK)
    for a_row, b_row in ((0, 0), (0, 1), (1, 1), (2, 2), (3, 3), (4, 4)):
        y0, y1 = yc[a_row], yc[b_row]
        if a_row != b_row:
            y0, y1 = y0 - 0.10, y1 + 0.10
        ax.add_patch(FancyArrowPatch((xl + bw + 0.03, y0), (xr - 0.03, y1), arrowstyle='-|>', mutation_scale=9,
                                     color=GREY, lw=1.0, zorder=1, shrinkA=0, shrinkB=0))
    ax.text(0.02, height, 'B', ha='left', va='top', fontsize=14, fontweight='bold', color=INK)
    return lines


def _fit_report(fig, ax, lines, margin_in=0.08):
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bad = []
    for t, right in lines:
        x1 = inv.transform((t.get_window_extent(renderer=rend).x1, 0))[0]
        if x1 > right - margin_in:
            bad.append((t.get_text(), round(right - x1, 3)))
    return bad


def draw_policy_line(ax, width):
    ax.set_xlim(0, width); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.03, 0.08), width - 0.06, 0.84, boxstyle='round,pad=0.0,rounding_size=0.10',
                                fc=LIGHT_GREY, ec=GREY, lw=0.8))
    ax.text(width / 2, 0.50, 'Policy statement shown to every role in experiment 2 (except experiment 1 repeated)', ha='center', va='center',
            fontsize=8.0, color=INK)


SCENE_W, SCENE_H = 8.50, 4.30
if E2_MODE:
    _w = (0.988 - 0.012) / (2 + 0.055)
    AX_W, AX_H = _w * SCENE_W, (0.950 - 0.025) * SCENE_H
    AX1_X, AX2_X, AX_Y = 0.012 * SCENE_W, (0.012 + _w * 1.055) * SCENE_W, 0.025 * SCENE_H
    AXR_H = AX_H * (1.0 - Y_LO)
    SCENE_H = SCENE_H - AX_H * Y_LO
    POLICY_H, FLOW_H, GAP, GAP_UP = 0.30, 3.55, 0.14, 0.03
    FIG_H = SCENE_H + GAP_UP + POLICY_H + GAP + FLOW_H + 0.10
    fig = plt.figure(figsize=(SCENE_W, FIG_H))

    def put(left, bottom, w, h):
        return fig.add_axes([left / SCENE_W, bottom / FIG_H, w / SCENE_W, h / FIG_H])

    y_block = FIG_H - SCENE_H
    axa = put(AX1_X, y_block + AX_Y, AX_W, AXR_H)
    axb = put(AX2_X, y_block + AX_Y, AX_W, AXR_H)
    CONTENT_W = AX2_X + AX_W - AX1_X
    axp = put(AX1_X, y_block - GAP_UP - POLICY_H, CONTENT_W, POLICY_H)
    axf = put(AX1_X, 0.10, CONTENT_W, FLOW_H)
    checks_a = scene(axa, 'patient')
    checks_b = scene(axb, 'ward')
    panel_label(axa, 'A', x=0.005, y=1.0)
    draw_policy_line(axp, CONTENT_W)
    flow_lines = draw_flow(axf, flow_counts(), CONTENT_W, FLOW_H)
else:
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(SCENE_W, SCENE_H))
    checks_a = scene(axa, 'patient')
    checks_b = scene(axb, 'ward')
    panel_label(axa, 'a', x=0.005, y=1.0); panel_label(axb, 'b', x=0.005, y=1.0)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.950, bottom=0.025, wspace=0.055)

fig.canvas.draw()
problems = (_containment_report(fig, axa, 'patient', checks_a)
            + _containment_report(fig, axb, 'ward', checks_b))
for p in problems:
    print('GEOMETRY FAIL:', p)
assert not problems, 'figure 1 geometry: labels straddle or sit on the wrong side of the boundary'

collisions = _overlap_report(fig, axa, 'patient') + _overlap_report(fig, axb, 'ward')
for c in collisions:
    print('OVERLAP FAIL:', c)
assert not collisions, 'figure 1 geometry: two pieces of text overlap'
if E2_MODE:
    overflow = _fit_report(fig, axf, flow_lines)
    for o in overflow:
        print('FIT FAIL:', o)
    assert not overflow, 'figure 1 panel B: a line of text runs past the edge of its box'
print(f'geometry OK: {len(checks_a) + len(checks_b)} elements, none straddling the task boundary; '
      f'{len(axa.texts) + len(axb.texts)} text objects, none overlapping')

Path(OUTDIR).mkdir(parents=True, exist_ok=True)
save_audited(fig, 'Figure_1', STEM, pad_inches=0.10, outdir=OUTDIR)
print(STEM + ' written to ' + OUTDIR)
