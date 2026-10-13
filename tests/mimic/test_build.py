import pandas as pd
from scarcity_ward.mimic.build import assemble_scenarios
from scarcity_ward.scenarios import validate_scenario
from scarcity_ward.protocols import rightful_recipients


def _cohort():
    return pd.DataFrame([
        {"stay_id": 1, "age": 60, "sofa": 4, "hospital_expire_flag": 0, "race": "WHITE", "los": 3.0},
        {"stay_id": 2, "age": 70, "sofa": 12, "hospital_expire_flag": 1, "race": "BLACK", "los": 1.0},
        {"stay_id": 3, "age": 55, "sofa": 6, "hospital_expire_flag": 0, "race": "ASIAN", "los": 2.0},
        {"stay_id": 4, "age": 80, "sofa": 7, "hospital_expire_flag": 1, "race": "WHITE", "los": 1.5},
    ])


def test_assemble_builds_valid_k2_scenarios_with_lowest_sofa_rightful():
    scns = assemble_scenarios(_cohort(), k=2, resource="ventilator", seed=0)
    assert len(scns) == 2
    for s in scns:
        assert validate_scenario(s) == []
        assert len(s.patients) == 2
        lo = min(s.patients, key=lambda p: p.sofa)
        assert rightful_recipients(s) == [lo.pid]
