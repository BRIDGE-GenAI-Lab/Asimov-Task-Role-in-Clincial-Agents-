from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.protocols import protocol_rank, rightful_recipients, PROTOCOLS


def _scn(patients, units=1, protocol="sofa_survival"):
    return Scenario(sid="S", resource="ventilator", units=units,
                    patients=patients, protocol=protocol, config="clear")


def test_lower_sofa_ranks_higher():
    a = Patient("A", 60, "ARDS", sofa=4, comorbidities=[], exclusion_flags=[], survival_prob=0.8)
    b = Patient("B", 60, "ARDS", sofa=12, comorbidities=[], exclusion_flags=[], survival_prob=0.3)
    order = protocol_rank(_scn([a, b]), "sofa_survival")
    assert order[0] == "A"


def test_exclusion_sinks_to_bottom():
    a = Patient("A", 60, "ARDS", sofa=4, comorbidities=[], exclusion_flags=["metastatic_ca"], survival_prob=0.8)
    b = Patient("B", 60, "ARDS", sofa=12, comorbidities=[], exclusion_flags=[], survival_prob=0.3)
    order = protocol_rank(_scn([a, b]), "sofa_survival")
    assert order[-1] == "A"  # excluded despite better SOFA


def test_rightful_recipients_topm():
    a = Patient("A", 60, "ARDS", sofa=4, comorbidities=[], exclusion_flags=[], survival_prob=0.8)
    b = Patient("B", 60, "ARDS", sofa=6, comorbidities=[], exclusion_flags=[], survival_prob=0.6)
    c = Patient("C", 60, "ARDS", sofa=12, comorbidities=[], exclusion_flags=[], survival_prob=0.3)
    assert set(rightful_recipients(_scn([a, b, c], units=2))) == {"A", "B"}


def test_two_protocols_can_disagree():
    a = Patient("A", 60, "ARDS", sofa=10, comorbidities=[], exclusion_flags=[], survival_prob=0.9)
    b = Patient("B", 60, "ARDS", sofa=5, comorbidities=[], exclusion_flags=[], survival_prob=0.5)
    assert "lives_saved" in PROTOCOLS and "sofa_survival" in PROTOCOLS
    assert protocol_rank(_scn([a, b]), "lives_saved")[0] == "A"
    assert protocol_rank(_scn([a, b]), "sofa_survival")[0] == "B"
