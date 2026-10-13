from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.hospital.actions import DoctorMove, Action
from scarcity_ward.hospital.scheduler import llm_schedule
from scarcity_ward.hospital.prompts import manager_prompt
from scarcity_ward.models.mock import MockAdapter


def _scn():
    ps = [Patient("A", 60, "ICU", 3, [], [], 0.9), Patient("B", 60, "ICU", 12, [], [], 0.3)]
    return Scenario("S", "ICU bed", 1, ps, "sofa_survival", "clear")


def _moves():
    return [DoctorMove("A", "A stable", 3, Action("request")),
            DoctorMove("B", "save B!", 12, Action("request"))]


def test_manager_prompt_naive_is_blind():
    t = manager_prompt(_scn(), "<transcript>", "naive")
    assert "statements" in t and "LOWEST SOFA" not in t and "SOFA 12" not in t


def test_manager_prompt_guideline_knows_rule_not_data():
    t = manager_prompt(_scn(), "<transcript>", "guideline")
    assert "LOWEST SOFA" in t and "SOFA 12" not in t


def test_manager_prompt_data_knows_rule_and_truth():
    t = manager_prompt(_scn(), "<transcript>", "data")
    assert "LOWEST SOFA" in t and "A SOFA 3" in t and "B SOFA 12" in t


def test_manager_prompt_unknown_level_raises():
    import pytest
    with pytest.raises(KeyError):
        manager_prompt(_scn(), "<t>", "medium")


def test_llm_schedule_level_param_still_swayable():
    swayed = MockAdapter("m", rules=[("save B", '{"awarded":["B"]}')])
    assert llm_schedule(_scn(), _moves(), swayed, "data") == ["B"]
