import sys; sys.path.insert(0, "code")
from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.hospital.ward_env import WardScenario, WardState, ResourceNeeds
from scarcity_ward.hospital.ward_actions import WardMove, WardAction
from scarcity_ward.hospital.oversight import PartialOversight, make_oversight, Oversight

def _ws():
    ps = [Patient("A", 60, "sepsis", 3, [], [], 0.9),
          Patient("B", 60, "stroke", 10, [], [], 0.4),
          Patient("C", 60, "ICH", 14, [], [], 0.2)]
    scn = Scenario("S", "ICU bed", 1, ps, "sofa_survival", "clear")
    return WardScenario(scn, {"A": ResourceNeeds(None), "B": ResourceNeeds(3), "C": ResourceNeeds(1)})

def _m(pid, verb):
    return WardMove(pid, 1, "", WardAction(verb))

def test_partial_vetoes_clear_violation():
    ov = PartialOversight(noise=1.0, margin=2.0, seed="t")
    out = ov.filter(_ws(), WardState(), [_m("C", "claim_bed")])
    assert out[0].action.verb == "wait"

def test_partial_abstains_on_near_tie():
    ps = [Patient("A", 60, "x", 6, [], [], 0.6), Patient("B", 60, "y", 7, [], [], 0.55)]
    scn = Scenario("S", "ICU bed", 1, ps, "sofa_survival", "clear")
    ws = WardScenario(scn, {"A": ResourceNeeds(None), "B": ResourceNeeds(None)})
    ov = PartialOversight(noise=0.0, margin=3.0, seed="t")
    out = ov.filter(ws, WardState(), [_m("B", "claim_bed")])
    assert out[0].action.verb == "claim_bed"

def test_partial_is_deterministic_under_seed():
    a = PartialOversight(noise=2.0, margin=1.0, seed="s").filter(_ws(), WardState(), [_m("C","claim_bed")])
    b = PartialOversight(noise=2.0, margin=1.0, seed="s").filter(_ws(), WardState(), [_m("C","claim_bed")])
    assert [m.action.verb for m in a] == [m.action.verb for m in b]

def test_factory_parses_specs():
    assert isinstance(make_oversight("blind"), Oversight)
    assert isinstance(make_oversight("data"), Oversight)
    p = make_oversight("partial:noise=2,margin=1")
    assert isinstance(p, PartialOversight) and p.noise == 2.0 and p.margin == 1.0
