import pytest

from scarcity_ward.parsing import ParseError
from scarcity_ward.hospital.ward_actions import WardAction, WardMove, parse_ward_move, VERBS


def test_parse_known_verb_and_rationale():
    m = parse_ward_move('{"rationale":"my patient needs the bed","action":{"verb":"claim_bed"}}', "A", 1)
    assert m == WardMove("A", 1, "my patient needs the bed", WardAction("claim_bed", None))


def test_unknown_verb_defaults_to_wait():
    assert parse_ward_move('{"action":{"verb":"nuke"}}', "B", 2).action.verb == "wait"


def test_concede_target_preserved():
    m = parse_ward_move('{"action":{"verb":"concede","target":"A"}}', "C", 1)
    assert (m.action.verb, m.action.target) == ("concede", "A")


def test_all_verbs_present():
    assert set(VERBS) == {"claim_bed", "hold_bed", "release_bed", "order_imaging",
                          "escalate", "withhold", "disclose", "concede", "wait"}


def test_non_json_raises_parseerror():
    with pytest.raises(ParseError):
        parse_ward_move("not json at all", "A", 1)


def test_non_object_json_raises_parseerror_not_crash():
    with pytest.raises(ParseError):
        parse_ward_move('"just a string"', "A", 1)
    m = parse_ward_move('{"rationale":"x","action":"claim_bed"}', "A", 1)  # action not an object
    assert m.action.verb == "wait"
