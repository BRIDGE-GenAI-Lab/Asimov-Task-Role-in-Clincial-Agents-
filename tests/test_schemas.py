from scarcity_ward.schemas import Patient, Scenario, AssertedClaims, AdvocateTurn, ArenaResult


def test_patient_and_scenario_roundtrip():
    p = Patient(pid="P1", age=70, diagnosis="ARDS", sofa=8,
                comorbidities=["CKD"], exclusion_flags=[], survival_prob=0.6)
    s = Scenario(sid="S1", resource="ventilator", units=1, patients=[p],
                 protocol="sofa_survival", config="clear")
    assert s.patients[0].pid == "P1"
    assert s.units == 1


def test_asserted_defaults_to_true_claims():
    c = AssertedClaims.from_patient(
        Patient(pid="P2", age=50, diagnosis="sepsis", sofa=5,
                comorbidities=[], exclusion_flags=["metastatic_ca"], survival_prob=0.7))
    assert c.sofa == 5 and c.exclusion_flags == ["metastatic_ca"]
