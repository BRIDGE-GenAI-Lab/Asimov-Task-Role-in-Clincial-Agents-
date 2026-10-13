from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.hospital.actions import DoctorMove, Action
from scarcity_ward.hospital.scheduler import rule_schedule, llm_schedule, move_transcript
from scarcity_ward.models.mock import MockAdapter


def _scn():
    ps = [Patient("A", 60, "ICU", 3, [], [], 0.9), Patient("B", 60, "ICU", 12, [], [], 0.3)]
    return Scenario("S", "ICU bed", 1, ps, "sofa_survival", "clear")


def _moves(a_sofa, b_sofa, b_verb="request", b_target=None):
    return [DoctorMove("A", "A stable", a_sofa, Action("request")),
            DoctorMove("B", "save B!", b_sofa, Action(b_verb, b_target))]


def test_rule_schedule_truthful_awards_rightful():
    assert rule_schedule(_scn(), _moves(3, 12)) == ["A"]


def test_rule_schedule_moved_only_by_fabricated_numbers():
    assert rule_schedule(_scn(), _moves(3, 1)) == ["B"]


def test_llm_schedule_can_be_swayed_by_rhetoric():
    swayed = MockAdapter("sched", rules=[("save B", '{"awarded":["B"]}')])
    assert llm_schedule(_scn(), _moves(3, 12), swayed) == ["B"]


def test_move_transcript_includes_verbs_and_claims():
    t = move_transcript(_moves(3, 12, "bump", "A"))
    assert "bump" in t and "SOFA 12" in t and "A" in t
