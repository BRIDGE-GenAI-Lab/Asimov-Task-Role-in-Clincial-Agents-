"""SOFA score (Vincent 1996; Sepsis-3, Singer 2016). Each system 0-4; total 0-24.
Inputs are the worst (most abnormal) value in the scoring window."""
from __future__ import annotations
from typing import Dict, Optional

SYSTEMS = ["resp", "coag", "liver", "cv", "cns", "renal"]


def resp_score(pf_ratio: Optional[float], ventilated: bool) -> int:
    if pf_ratio is None:
        return 0
    if pf_ratio < 100 and ventilated:
        return 4
    if pf_ratio < 200 and ventilated:
        return 3
    if pf_ratio < 300:
        return 2
    if pf_ratio < 400:
        return 1
    return 0


def coag_score(platelets: Optional[float]) -> int:
    if platelets is None:
        return 0
    for thr, sc in [(20, 4), (50, 3), (100, 2), (150, 1)]:
        if platelets < thr:
            return sc
    return 0


def liver_score(bilirubin: Optional[float]) -> int:
    if bilirubin is None:
        return 0
    for thr, sc in [(12.0, 4), (6.0, 3), (2.0, 2), (1.2, 1)]:
        if bilirubin >= thr:
            return sc
    return 0


def cv_score(map_mmhg: Optional[float], norepi: float = 0.0, epi: float = 0.0,
             dopamine: float = 0.0, dobutamine: float = 0.0) -> int:
    # vasopressor doses in mcg/kg/min
    if dopamine > 15 or epi > 0.1 or norepi > 0.1:
        return 4
    if dopamine > 5 or (0 < epi <= 0.1) or (0 < norepi <= 0.1):
        return 3
    if dopamine > 0 or dobutamine > 0:
        return 2
    if map_mmhg is not None and map_mmhg < 70:
        return 1
    return 0


def cns_score(gcs: Optional[float]) -> int:
    if gcs is None:
        return 0
    for thr, sc in [(6, 4), (10, 3), (13, 2), (15, 1)]:
        if gcs < thr:
            return sc
    return 0


def renal_score(creatinine: Optional[float], urine_ml_24h: Optional[float] = None) -> int:
    # creatinine criterion
    c = 0
    if creatinine is not None:
        for thr, sc in [(5.0, 4), (3.5, 3), (2.0, 2), (1.2, 1)]:
            if creatinine >= thr:
                c = sc
                break
    # urine-output criterion (only the two most severe scores; mimic-code parity)
    u = 0
    if urine_ml_24h is not None:
        if urine_ml_24h < 200:
            u = 4
        elif urine_ml_24h < 500:
            u = 3
    return max(c, u)


def sofa_total(components: Dict[str, int]) -> int:
    return int(sum(components.get(s, 0) for s in SYSTEMS))
