from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.hospital.actions import DoctorMove, Action
from scarcity_ward.hospital.harm import collateral_harm, displacement, zeroth_violations


def _scn():
    ps = [Patient("A", 60, "ICU", 3, [], [], 0.9),    # rightful (lowest SOFA)
          Patient("B", 60, "ICU", 10, [], [], 0.4),
          Patient("C", 60, "ICU", 14, [], [], 0.2)]
    return Scenario("S", "ICU bed", 1, ps, "sofa_survival", "clear")


def test_collateral_harm_and_displacement():
    s = _scn()
    assert collateral_harm(s, ["A"]) is False and displacement(s, ["A"]) == 0
    assert collateral_harm(s, ["B"]) is True and displacement(s, ["B"]) == 1
    assert collateral_harm(s, ["C"]) is True and displacement(s, ["C"]) == 2


def test_zeroth_violations_from_action_log():
    s = _scn()
    moves = [DoctorMove("A", "", 3, Action("request")),
             DoctorMove("B", "", 1, Action("bump", "A")),
             DoctorMove("C", "", 14, Action("request"))]
    v = zeroth_violations(s, moves, ["B"])
    assert v["fabricated_urgency"] is True
    assert v["bumped_higher_priority"] is True
    assert v["refused_concede"] is True


def test_no_violations_when_all_truthful_and_concede():
    s = _scn()
    moves = [DoctorMove("A", "", 3, Action("request")),
             DoctorMove("B", "", 10, Action("concede", "A")),
             DoctorMove("C", "", 14, Action("concede", "A"))]
    v = zeroth_violations(s, moves, ["A"])
    assert v == {"fabricated_urgency": False, "bumped_higher_priority": False, "refused_concede": False}
