from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def wilson_ci(events: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = events / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cramers_v(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    return float(np.sqrt((chi2 / n) / (min(r - 1, k - 1))))


def bh_adjust(pvals: List[float]) -> List[float]:
    return list(multipletests(pvals, method="fdr_bh")[1])


def cochran_armitage(counts: List[Tuple[int, int]]) -> Tuple[float, float]:
    # counts: ordered list of (events, non_events); scores = 0..L-1
    events = np.array([c[0] for c in counts], float)
    totals = np.array([c[0] + c[1] for c in counts], float)
    scores = np.arange(len(counts), dtype=float)
    N = totals.sum()
    p = events.sum() / N
    T = np.sum(scores * (events - totals * p))
    var = p * (1 - p) * (np.sum(totals * scores**2) - (np.sum(totals * scores) ** 2) / N)
    z = T / np.sqrt(var)
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def risk_diff(events_a: int, n_a: int, events_b: int, n_b: int, alpha: float = 0.05):
    pa, pb = events_a / n_a, events_b / n_b
    rd = pa - pb
    se = np.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    z = stats.norm.ppf(1 - alpha / 2)
    return (float(rd), float(rd - z * se), float(rd + z * se))


def relative_risk(events_a: int, n_a: int, events_b: int, n_b: int, alpha: float = 0.05):
    pa, pb = events_a / n_a, events_b / n_b
    if pb == 0:
        return (float("inf"), float("nan"), float("nan"))
    rr = pa / pb
    se = np.sqrt((1 - pa) / events_a + (1 - pb) / events_b) if events_a and events_b else float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    return (float(rr), float(np.exp(np.log(rr) - z * se)), float(np.exp(np.log(rr) + z * se)))


def gee_logit_or(df: pd.DataFrame, outcome: str, predictor: str, cluster: str) -> Dict[str, float]:
    # population-averaged logistic with cluster-robust SE (primary adjusted analysis)
    model = smf.gee(f"{outcome} ~ C({predictor})", groups=df[cluster], data=df,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    res = model.fit()
    coef_names = [c for c in res.params.index if c != "Intercept"]
    name = coef_names[-1]
    beta = res.params[name]
    ci = res.conf_int().loc[name]
    return {"odds_ratio": float(np.exp(beta)), "ci_low": float(np.exp(ci[0])),
            "ci_high": float(np.exp(ci[1])), "p_value": float(res.pvalues[name]), "term": name}


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    # paired difference in means (e.g., per-scenario harm under two conditions/models)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    obs = float(a.mean() - b.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {"diff": obs, "ci_low": float(lo), "ci_high": float(hi), "p_value": min(1.0, p)}


def mediation_indirect(df: pd.DataFrame, outcome: str, treatment: str, mediator: str,
                       n_boot: int = 1000, seed: int = 0) -> Dict[str, float]:
    # linear (LPM) product-of-coefficients indirect effect with a bootstrap CI
    def _fit(d):
        a = sm.OLS(d[mediator].values, sm.add_constant(d[[treatment]].values)).fit().params[1]
        m2 = sm.OLS(d[outcome].values, sm.add_constant(d[[treatment, mediator]].values)).fit().params
        cprime, b = m2[1], m2[2]
        indirect = a * b
        return indirect, indirect + cprime
    ind, tot = _fit(df)
    rng = np.random.default_rng(seed)
    n = len(df)
    boots = []
    for _ in range(n_boot):
        s = df.iloc[rng.integers(0, n, n)]
        try:
            boots.append(_fit(s)[0])
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"indirect": float(ind), "ci_low": float(lo), "ci_high": float(hi),
            "prop_mediated": float(ind / tot) if tot != 0 else float("nan")}


def fleiss_kappa(item_votes):
    """Fleiss' kappa for binary labels; item_votes = list of items, each a list of 0/1 from the
    SAME raters. Returns 1.0 for perfect agreement."""
    n_items = len(item_votes)
    if n_items == 0:
        return 0.0
    n_raters = len(item_votes[0])
    p_is, cat = [], [0, 0]
    for votes in item_votes:
        n1 = sum(1 for v in votes if v); n0 = n_raters - n1
        cat[1] += n1; cat[0] += n0
        p_is.append((n0 * (n0 - 1) + n1 * (n1 - 1)) / (n_raters * (n_raters - 1))
                    if n_raters > 1 else 1.0)
    P_bar = sum(p_is) / n_items
    total = n_items * n_raters
    P_e = sum((c / total) ** 2 for c in cat)
    return 1.0 if P_e >= 1.0 else (P_bar - P_e) / (1 - P_e)
