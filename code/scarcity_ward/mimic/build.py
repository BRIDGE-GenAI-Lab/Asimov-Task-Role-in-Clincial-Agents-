from __future__ import annotations
import hashlib
from typing import List
import pandas as pd
from ..schemas import Patient, Scenario


def _surv_prob(sofa: int) -> float:
    # monotone proxy from SOFA (lower SOFA -> higher survival); for display only, not ground truth
    return round(max(0.02, min(0.98, 1.0 - sofa / 24.0)), 2)


def _config(sofas: List[int]) -> str:
    s = sorted(sofas)
    return "near_tie" if (len(s) >= 2 and abs(s[0] - s[1]) <= 2) else "clear"


def assemble_scenarios(cohort: pd.DataFrame, k: int, resource: str, seed: int = 0) -> List[Scenario]:
    df = cohort.copy().sort_values("stay_id").reset_index(drop=True)
    scns: List[Scenario] = []
    for g in range(len(df) // k):
        block = df.iloc[g * k:(g + 1) * k]
        patients = []
        for _, r in block.iterrows():
            pid = "P" + hashlib.sha1(str(int(r.stay_id)).encode()).hexdigest()[:6]
            patients.append(Patient(pid=pid, age=int(r.age), diagnosis="ICU",
                                    sofa=int(r.sofa), comorbidities=[], exclusion_flags=[],
                                    survival_prob=_surv_prob(int(r.sofa)),
                                    sociodemographic=str(r.get("race")) if "race" in r else None))
        scns.append(Scenario(sid="MIMIC-%05d" % g, resource=resource, units=1, patients=patients,
                             protocol="sofa_survival", config=_config([p.sofa for p in patients])))
    return scns
