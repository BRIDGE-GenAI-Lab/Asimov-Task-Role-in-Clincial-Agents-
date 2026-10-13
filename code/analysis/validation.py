from __future__ import annotations
from typing import List
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from statsmodels.stats.inter_rater import fleiss_kappa


def cohen_kappa(a: List[int], b: List[int]) -> float:
    return float(cohen_kappa_score(a, b))


def judge_auroc(truth: List[int], judge_scores: List[float]) -> float:
    return float(roc_auc_score(truth, judge_scores))


def fleiss_kappa_matrix(category_counts) -> float:
    return float(fleiss_kappa(category_counts))
