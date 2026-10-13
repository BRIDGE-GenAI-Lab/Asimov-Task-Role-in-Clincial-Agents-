from __future__ import annotations
import pandas as pd
from .paths import MimicPaths


def build_backbone(p: MimicPaths) -> pd.DataFrame:
    stays = pd.read_csv(p.icu("icustays"), usecols=["subject_id", "hadm_id", "stay_id",
                                                    "first_careunit", "intime", "outtime", "los"])
    pats = pd.read_csv(p.hosp("patients"), usecols=["subject_id", "gender", "anchor_age", "dod"])
    adm = pd.read_csv(p.hosp("admissions"), usecols=["hadm_id", "deathtime", "hospital_expire_flag",
                                                     "insurance", "race", "marital_status",
                                                     "discharge_location"])
    df = stays.merge(pats, on="subject_id", how="left").merge(adm, on="hadm_id", how="left")
    df = df.rename(columns={"anchor_age": "age", "gender": "sex"})
    return df
