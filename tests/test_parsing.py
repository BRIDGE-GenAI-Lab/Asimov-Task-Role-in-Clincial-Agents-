import pytest
from scarcity_ward.parsing import extract_json, parse_advocate, ParseError


def test_extract_json_strips_code_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_finds_object_in_prose():
    assert extract_json('Sure! {"awarded": ["A"]} done.') == {"awarded": ["A"]}


def test_extract_json_raises_on_garbage():
    with pytest.raises(ParseError):
        extract_json("no json here")


def test_extract_json_digs_object_out_of_list():
    assert extract_json('[{"deception": true}]') == {"deception": True}


def test_extract_json_raises_on_scalar_list():
    with pytest.raises(ParseError):
        extract_json("[1, 2, 3]")


def test_parse_advocate_coerces_types_and_clamps():
    raw = '{"argument":"x","asserted":{"sofa":"7","exclusion_flags":[],"survival_prob":1.5},"statements_about_others":["bad B"]}'
    t = parse_advocate(raw, pid="A", round_no=1)
    assert t.asserted.sofa == 7 and t.asserted.survival_prob == 1.0  # clamped to [0,1]
    assert t.rival_statements == ["bad B"]


def test_parse_advocate_recovers_sofa_from_flat_field():
    raw = '{"argument":"my patient is stable","sofa":5}'
    t = parse_advocate(raw, pid="A", round_no=1)
    assert t.asserted.sofa == 5


def test_parse_advocate_recovers_sofa_from_prose_when_no_structured_block():
    raw = '{"argument":"Patient A presents with SOFA 3, the strongest candidate."}'
    t = parse_advocate(raw, pid="A", round_no=1)
    assert t.asserted.sofa == 3


def test_parse_advocate_raises_when_sofa_unrecoverable():
    raw = '{"argument":"please give my patient the ventilator"}'
    with pytest.raises(ParseError):
        parse_advocate(raw, pid="A", round_no=1)


from scarcity_ward.parsing import parse_priority_pid, parse_priority_pid_strict

_PIDS = ["A", "B", "C"]


def test_strict_reads_the_json_field_only_when_it_is_exactly_a_pid():
    assert parse_priority_pid_strict('{"priority_pid": "A"}', _PIDS) == ("A", "json")
    assert parse_priority_pid_strict('Under the policy: {"priority_pid": "B"} (best survival).', _PIDS) == ("B", "json")
    assert parse_priority_pid_strict('```json\n{"priority_pid": "C"}\n```', _PIDS) == ("C", "json")


def test_strict_never_searches_prose_for_a_pid():
    reply = "I cannot rank my colleagues' patients; my patient A needs this ventilator."
    assert parse_priority_pid(reply, _PIDS) == "A"                    # the lenient parser would call this an answer
    assert parse_priority_pid_strict(reply, _PIDS) == (None, None)


def test_strict_rejects_wrong_types_unknown_pids_truncation_and_empty():
    for reply in ['{"priority_pid": "Z"}', '{"priority_pid": null}', '{"priority_pid": 7}',
                  '{"priority_pid": ["A"]}', '{"priority_pid": "a"}', '{"other": "A"}',
                  '{"priority_pid": "A"', ""]:
        assert parse_priority_pid_strict(reply, _PIDS) == (None, None), reply


def test_strict_takes_the_first_object_that_carries_the_field():
    assert parse_priority_pid_strict('{"x": 1} {"priority_pid": "B"} {"priority_pid": "A"}', _PIDS) == ("B", "json")
    assert parse_priority_pid_strict('{"priority_pid": "Z"} {"priority_pid": "A"}', _PIDS) == (None, None)


def test_lenient_parser_is_unchanged():
    assert parse_priority_pid('{"priority_pid": "B"}', _PIDS) == "B"
    assert parse_priority_pid("the answer is C, clearly", _PIDS) == "C"
    assert parse_priority_pid("no patient named", _PIDS) is None
