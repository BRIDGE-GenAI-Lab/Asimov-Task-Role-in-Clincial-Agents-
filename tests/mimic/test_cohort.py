from pathlib import Path
from scarcity_ward.mimic.paths import MimicPaths
from scarcity_ward.mimic.cohort import build_backbone

FIX = Path(__file__).parent / "fixtures"


def test_backbone_joins_age_sex_outcome():
    df = build_backbone(MimicPaths(root=str(FIX), gz=False))
    row1 = df[df.stay_id == 100].iloc[0]
    assert row1.age == 72 and row1.sex == "F" and row1.hospital_expire_flag == 0 and row1.race == "WHITE"
    row2 = df[df.stay_id == 200].iloc[0]
    assert row2.hospital_expire_flag == 1 and row2.los == 1.0
    assert set(["stay_id", "subject_id", "hadm_id", "age", "sex", "race", "insurance",
                "los", "hospital_expire_flag"]).issubset(df.columns)
