import numpy as np, pandas as pd
from analysis.stats import (wilson_ci, cramers_v, bh_adjust, cochran_armitage, risk_diff,
                            gee_logit_or, paired_bootstrap, mediation_indirect)


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(20, 100)
    assert 0.12 < lo < 0.20 < hi < 0.30


def test_cramers_v_perfect_association():
    table = np.array([[50, 0], [0, 50]])
    assert cramers_v(table) > 0.99


def test_bh_monotone():
    p = [0.001, 0.02, 0.5]
    adj = bh_adjust(p)
    assert adj[0] <= adj[1] <= adj[2] and adj[2] <= 1.0


def test_cochran_armitage_detects_trend():
    counts = [(5, 95), (20, 80), (60, 40)]  # (events, non-events) per ordered level
    stat, p = cochran_armitage(counts)
    assert p < 0.001


def test_risk_diff_sign():
    rd, lo, hi = risk_diff(events_a=10, n_a=100, events_b=30, n_b=100)
    assert rd < 0 and hi < 0  # group A lower risk


def test_gee_logit_or_recovers_effect():
    rng = np.random.default_rng(0)
    n = 600
    model = rng.integers(0, 2, n)
    cluster = rng.integers(0, 50, n)
    p = 1 / (1 + np.exp(-(-1.0 + 1.2 * model)))
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({"y": y, "model": model, "cluster": cluster})
    res = gee_logit_or(df, outcome="y", predictor="model", cluster="cluster")
    assert res["odds_ratio"] > 1.5 and res["ci_low"] > 1.0


def test_paired_bootstrap_detects_difference():
    a = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1] * 5   # ~0.8
    b = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1] * 5   # ~0.3
    res = paired_bootstrap(a, b)
    assert res["diff"] > 0 and res["ci_low"] > 0 and res["p_value"] < 0.05


def test_mediation_detects_indirect_path():
    rng = np.random.default_rng(1)
    n = 800
    t = rng.integers(0, 2, n)
    m = 0.1 + 0.7 * t + rng.normal(0, 0.2, n)        # M depends on T
    y = 0.05 + 0.0 * t + 0.8 * m + rng.normal(0, 0.2, n)  # Y depends on M only
    df = pd.DataFrame({"T": t, "M": m, "Y": y})
    res = mediation_indirect(df, "Y", "T", "M", n_boot=300)
    assert res["indirect"] > 0 and res["ci_low"] > 0 and res["prop_mediated"] > 0.7


def test_fleiss_kappa_perfect_and_chance():
    from analysis.stats import fleiss_kappa
    assert fleiss_kappa([[1, 1, 1], [0, 0, 0]]) == 1.0          # perfect agreement
    k = fleiss_kappa([[1, 0, 1], [0, 1, 0], [1, 1, 0]])         # mixed -> < 1
    assert -1.0 <= k < 1.0
