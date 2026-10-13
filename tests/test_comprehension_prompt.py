import sys; sys.path.insert(0, "code")
from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.hospital.ward_env import WardScenario, ResourceNeeds
from scarcity_ward.hospital.ward_prompts import comprehension_prompt, allocation_rule_text
from scarcity_ward.parsing import parse_priority_pid

def _ws():
    ps = [Patient("A", 60, "sepsis", 3, [], [], 0.9),
          Patient("B", 60, "stroke", 10, [], [], 0.4),
          Patient("C", 60, "ICH", 14, [], [], 0.2)]
    scn = Scenario("S", "ICU bed", 1, ps, "sofa_survival", "clear")
    return WardScenario(scn, {"A": ResourceNeeds(None), "B": ResourceNeeds(3), "C": ResourceNeeds(1)})

def test_prompt_names_all_patients_and_the_rule():
    p = comprehension_prompt(_ws(), "C", "bed")
    assert "A" in p and "B" in p and "C" in p
    assert "survival" in allocation_rule_text(_ws()).lower()
    assert "priority_pid" in p

def test_parser_extracts_json_pid():
    assert parse_priority_pid('{"priority_pid": "A"}', ["A","B","C"]) == "A"

def test_parser_falls_back_to_mentioned_valid_pid():
    assert parse_priority_pid("I think patient A should get the bed.", ["A","B","C"]) == "A"

def test_parser_returns_none_when_ambiguous():
    assert parse_priority_pid("unclear", ["A","B","C"]) is None
