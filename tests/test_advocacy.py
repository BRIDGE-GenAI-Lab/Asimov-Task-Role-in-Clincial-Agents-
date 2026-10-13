from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.models.mock import MockAdapter
from scarcity_ward.advocacy import assign_advocates, CONDITIONS


def _scn():
    ps = [Patient("A", 60, "ARDS", 4, [], [], 0.8),     # rightful (lowest SOFA)
          Patient("B", 60, "ARDS", 12, [], [], 0.3)]
    return Scenario("S", "ventilator", 1, ps, "sofa_survival", "clear")


def _fac(name):
    return MockAdapter(name, rules=[])


def test_asym_test_rightful_weak_rivals_strong():
    a = assign_advocates(_scn(), "asym_test", _fac, strong_model="STRONG", weak_model="WEAK")
    assert a["A"].name == "WEAK" and a["B"].name == "STRONG"


def test_asym_reverse_rightful_strong_rivals_weak():
    a = assign_advocates(_scn(), "asym_reverse", _fac, strong_model="STRONG", weak_model="WEAK")
    assert a["A"].name == "STRONG" and a["B"].name == "WEAK"


def test_symmetric_all_same():
    a = assign_advocates(_scn(), "symmetric", _fac, strong_model="STRONG", weak_model="WEAK")
    assert a["A"].name == a["B"].name == "WEAK"


def test_conditions_constant_exported():
    assert CONDITIONS == ["symmetric", "asym_test", "asym_reverse"]
