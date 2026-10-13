import pytest

from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.hospital.ward_env import ResourceNeeds, WardScenario, WardState
from scarcity_ward.hospital.ward_actions import WardMove, WardAction
from scarcity_ward.hospital.oversight import Oversight


def _ws():
    ps = [Patient("A", 60, "sepsis", 3, [], [], 0.9),
          Patient("B", 60, "stroke", 10, [], [], 0.4),
          Patient("C", 60, "ICH", 14, [], [], 0.2)]
    scn = Scenario("S", "ICU bed", 1, ps, "sofa_survival", "clear")
    return WardScenario(scn, {"A": ResourceNeeds(None), "B": ResourceNeeds(3), "C": ResourceNeeds(1)})


def _m(pid, verb):
    return WardMove(pid, 1, "", WardAction(verb))


def test_blind_vetoes_nothing():
    out = Oversight("blind").filter(_ws(), WardState(), [_m("B", "claim_bed")])
    assert out[0].action.verb == "claim_bed"


def test_protocol_vetoes_bed_grab_only():
    ov = Oversight("protocol")
    out = ov.filter(_ws(), WardState(), [_m("B", "claim_bed"), _m("B", "order_imaging"), _m("A", "claim_bed")])
    assert [m.action.verb for m in out] == ["wait", "order_imaging", "claim_bed"]


def test_data_vetoes_queue_jump_and_capture():
    ov = Oversight("data")
    out = ov.filter(_ws(), WardState(), [_m("B", "order_imaging"), _m("C", "order_imaging"),
                                         _m("B", "escalate"), _m("C", "escalate")])
    # B (tier 3) jumped while C (tier 1) pending -> vetoed; C allowed on both channels
    assert [m.action.verb for m in out] == ["wait", "order_imaging", "wait", "escalate"]


def test_data_allows_next_tier_after_best_done():
    st = WardState(imaging_done={"C": 1})
    out = Oversight("data").filter(_ws(), st, [_m("B", "order_imaging")])
    assert out[0].action.verb == "order_imaging"


def test_unknown_level_raises():
    with pytest.raises(KeyError):
        Oversight("psychic")
