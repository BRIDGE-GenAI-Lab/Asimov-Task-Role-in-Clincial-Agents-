import pytest
from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.hospital.prompts import doctor_move_prompt, ARMS


def _scn():
    ps = [Patient("A", 60, "ICU", 3, [], [], 0.9), Patient("B", 60, "ICU", 12, [], ["metastatic_ca"], 0.3)]
    return Scenario("S", "ICU bed", 1, ps, "sofa_survival", "clear")


def test_arms_constant():
    assert ARMS == ["rule", "empathetic", "devotion"]


def test_devotion_is_single_minded_and_hides_rival_data():
    txt = doctor_move_prompt(_scn(), "A", "devotion")
    assert "everything in your power" in txt
    assert "without disadvantaging" not in txt
    assert "metastatic_ca" not in txt
    assert '"action"' in txt and "request" in txt


def test_empathetic_mentions_other_patients():
    txt = doctor_move_prompt(_scn(), "A", "empathetic")
    assert "without disadvantaging" in txt or "other patients" in txt


def test_rule_arm_doctor_is_truthful_baseline():
    txt = doctor_move_prompt(_scn(), "A", "rule")
    assert "truthfully" in txt and "defer" in txt


def test_unknown_arm_raises():
    with pytest.raises(KeyError):
        doctor_move_prompt(_scn(), "A", "bogus")
