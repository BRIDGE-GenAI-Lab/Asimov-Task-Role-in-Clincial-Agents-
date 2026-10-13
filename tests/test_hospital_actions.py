import pytest
from scarcity_ward.hospital.actions import Action, DoctorMove, parse_move


def test_parse_move_full():
    raw = '{"argument":"my patient is salvageable","asserted_sofa":4,"action":{"verb":"bump","target":"B"}}'
    m = parse_move(raw, "A")
    assert m.pid == "A" and m.asserted_sofa == 4
    assert m.action.verb == "bump" and m.action.target == "B"
    assert "salvageable" in m.argument


def test_parse_move_defaults_to_request_and_recovers_sofa_from_prose():
    raw = '{"argument":"give A the bed, SOFA 3"}'
    m = parse_move(raw, "A")
    assert m.action.verb == "request" and m.action.target is None
    assert m.asserted_sofa == 3


def test_parse_move_unrecoverable_sofa_is_none():
    m = parse_move('{"argument":"please help my patient"}', "A")
    assert m.asserted_sofa is None


def test_parse_move_clamps_unknown_verb_to_request():
    m = parse_move('{"argument":"x","action":{"verb":"sabotage"}}', "A")
    assert m.action.verb == "request"
