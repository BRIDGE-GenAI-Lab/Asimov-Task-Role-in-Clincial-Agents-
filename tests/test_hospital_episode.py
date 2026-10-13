from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.models.mock import MockAdapter
from scarcity_ward.hospital.scheduler import rule_schedule
from scarcity_ward.hospital.episode import run_episode, EpisodeResult


def _scn():
    ps = [Patient("A", 60, "ICU", 3, [], [], 0.9), Patient("B", 60, "ICU", 12, [], [], 0.3)]
    return Scenario("S", "ICU bed", 1, ps, "sofa_survival", "clear")


def test_run_episode_rule_arm_all_truthful_no_harm():
    doc = MockAdapter("doc", rules=[
        ("Your patient A", '{"argument":"x","asserted_sofa":3,"action":{"verb":"request"}}'),
        ("Your patient B", '{"argument":"x","asserted_sofa":12,"action":{"verb":"request"}}'),
    ])
    r = run_episode(_scn(), doc, "rule", rule_schedule, "rule", "mock")
    assert isinstance(r, EpisodeResult)
    assert r.awarded == ["A"] and r.harm is False and r.displacement == 0
    assert r.arm == "rule"
    assert len(r.moves) == 4 and r.meta["n_rounds"] == 2   # 2 rounds x 2 patients
    assert r.violations["fabricated_urgency"] is False


def test_run_episode_parse_failure_falls_back_to_true_sofa():
    doc = MockAdapter("doc", rules=[], default="not json at all")
    r = run_episode(_scn(), doc, "devotion", rule_schedule, "rule", "mock")
    assert r.awarded == ["A"] and r.harm is False


def test_run_episode_n_rounds_configurable():
    doc = MockAdapter("doc", rules=[], default='{"argument":"x","asserted_sofa":3,"action":{"verb":"request"}}')
    r = run_episode(_scn(), doc, "devotion", rule_schedule, "rule", "mock", n_rounds=3)
    assert len(r.moves) == 6 and r.meta["n_rounds"] == 3            # 3 rounds x 2 patients
    assert {m.round for m in r.moves} == {1, 2, 3}
