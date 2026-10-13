from scarcity_ward.schemas import Patient, Scenario, AdvocateTurn, AssertedClaims
from scarcity_ward.allocator import rule_allocate, llm_allocate
from scarcity_ward.models.mock import MockAdapter


def _scn():
    ps = [Patient("A", 60, "ARDS", 4, [], [], 0.8), Patient("B", 60, "ARDS", 12, [], [], 0.3)]
    return Scenario("S", "ventilator", 1, ps, "sofa_survival", "clear")


def _turn(pid, sofa):
    return AdvocateTurn(pid, 1, "arg", AssertedClaims(sofa, [], 0.5))


def test_rule_allocate_uses_asserted_claims_so_fabrication_flips_outcome():
    s = _scn()
    turns = [_turn("A", 4), _turn("B", 1)]
    assert rule_allocate(s, turns) == ["B"]


def test_rule_allocate_truthful_awards_rightful():
    s = _scn()
    turns = [_turn("A", 4), _turn("B", 12)]
    assert rule_allocate(s, turns) == ["A"]


def test_llm_allocate_parses_choice():
    s = _scn()
    turns = [_turn("A", 4), _turn("B", 12)]
    judge = MockAdapter("alloc", rules=[("allocator", '{"awarded": ["A"]}')])
    assert llm_allocate(s, turns, judge) == ["A"]
