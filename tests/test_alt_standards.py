import sys; sys.path.insert(0, "code")
from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.protocols import protocol_rank


def _scn():
    ps = [Patient("A", 60, "x", 3, [], [], 0.9),
          Patient("B", 60, "y", 14, [], ["malignancy"], 0.2),
          Patient("C", 60, "z", 8, [], [], 0.6)]
    return Scenario("S", "ICU bed", 1, ps, "sofa_survival", "clear")


def test_survival_first_baseline():
    assert protocol_rank(_scn(), "sofa_survival")[0] == "A"


def test_sickest_first_ranks_highest_sofa_first():
    assert protocol_rank(_scn(), "sickest_first")[0] == "B"


def test_first_come_is_pid_order():
    assert protocol_rank(_scn(), "first_come") == ["A", "B", "C"]


def test_lottery_is_deterministic_under_sid():
    a = protocol_rank(_scn(), "lottery"); b = protocol_rank(_scn(), "lottery")
    assert a == b and set(a) == {"A", "B", "C"}


def test_strict_exclusion_drops_excluded_to_bottom():
    assert protocol_rank(_scn(), "strict_exclusion")[-1] == "B"
