from scarcity_ward.mimic.sofa import (resp_score, coag_score, liver_score, cv_score,
                                      cns_score, renal_score, sofa_total)


def test_component_thresholds():
    assert resp_score(pf_ratio=90, ventilated=True) == 4
    assert resp_score(pf_ratio=250, ventilated=False) == 2
    assert resp_score(pf_ratio=500, ventilated=False) == 0
    assert coag_score(platelets=15) == 4 and coag_score(platelets=120) == 1 and coag_score(200) == 0
    assert liver_score(bilirubin=13) == 4 and liver_score(1.5) == 1 and liver_score(0.5) == 0
    assert cns_score(gcs=5) == 4 and cns_score(11) == 2 and cns_score(15) == 0
    assert renal_score(creatinine=6) == 4 and renal_score(2.5) == 2 and renal_score(0.9) == 0


def test_renal_uses_urine_output_or_creatinine():
    assert renal_score(creatinine=0.9, urine_ml_24h=150) == 4    # oliguria dominates normal creatinine
    assert renal_score(creatinine=0.9, urine_ml_24h=400) == 3
    assert renal_score(creatinine=2.5, urine_ml_24h=2000) == 2   # creatinine criterion wins
    assert renal_score(creatinine=0.9, urine_ml_24h=2000) == 0


def test_cardiovascular_uses_map_and_pressors():
    assert cv_score(map_mmhg=80, norepi=0.0, dopamine=0.0) == 0
    assert cv_score(map_mmhg=65, norepi=0.0, dopamine=0.0) == 1            # MAP<70
    assert cv_score(map_mmhg=60, norepi=0.05, dopamine=0.0) == 3          # norepi<=0.1
    assert cv_score(map_mmhg=60, norepi=0.2, dopamine=0.0) == 4           # norepi>0.1


def test_total_sums_and_clamps_missing_to_zero():
    comps = {"resp": 2, "coag": 1, "liver": 0, "cv": 3, "cns": 4, "renal": 2}
    assert sofa_total(comps) == 12
    assert sofa_total({"resp": 2}) == 2   # missing components contribute 0
