from scarcity_ward.config import RunConfig
from scarcity_ward.runner import expand_cells, run_cells
from scarcity_ward.models.mock import MockAdapter


def test_expand_cells_counts():
    cfg = RunConfig(models=["m1", "m2"], pressures=["advocate", "maximal"],
                    k_levels=[2], reps=3, allocators=["rule"])
    cells = expand_cells(cfg, n_scenarios=5)
    assert len(cells) == 2 * 2 * 1 * 1 * 3 * 5  # models*pressure*k*alloc*reps*scenarios


def test_run_cells_produces_one_record_per_cell(tmp_path):
    cfg = RunConfig(models=["m1"], pressures=["advocate"], k_levels=[2], reps=1, allocators=["rule"])
    reply = '{"argument":"t","asserted":{"sofa":5,"exclusion_flags":[],"survival_prob":0.5},"statements_about_others":[]}'
    def adapter_factory(_):
        return MockAdapter("m1", rules=[("Your patient", reply)])
    judge = MockAdapter("judge", rules=[("TRANSCRIPT",
        '{"deception":false,"sabotage":false,"honest_concession":false,"norm_invocation":false,"evidence":""}')])
    out = tmp_path / "run.jsonl"
    n = run_cells(cfg, "data/scenarios/scenarios_seed.jsonl",
                  adapter_factory=adapter_factory, judge_adapter=judge, out_path=str(out),
                  n_scenarios=1)
    assert n == 1
    assert out.exists() and out.read_text().count("\n") == 1


def test_run_cells_records_officer_model(tmp_path):
    cfg = RunConfig(models=["m1"], pressures=["maximal"], k_levels=[2], reps=1, allocators=["llm"])
    reply = '{"argument":"t","asserted":{"sofa":5,"exclusion_flags":[],"survival_prob":0.5},"statements_about_others":[]}'
    def factory(name):
        return MockAdapter(name, rules=[("Your patient", reply), ("triage officer", '{"awarded": ["A"]}')])
    judge = MockAdapter("judge", rules=[("TRANSCRIPT",
        '{"deception":false,"sabotage":false,"honest_concession":false,"norm_invocation":false,"normative_reframing":false,"evidence":""}')])
    out = tmp_path / "r.jsonl"
    n = run_cells(cfg, "data/scenarios/scenarios_seed.jsonl", adapter_factory=factory,
                  judge_adapter=judge, out_path=str(out), n_scenarios=1, officer_model="officerX")
    import json
    rec = json.loads(out.read_text().splitlines()[0])
    assert n == 1 and rec["officer"] == "officerX"


def test_run_cells_concurrency_produces_all_records(tmp_path):
    cfg = RunConfig(models=["m1", "m2"], pressures=["maximal"], k_levels=[2], reps=1, allocators=["rule"])
    reply = '{"argument":"t","asserted":{"sofa":5,"exclusion_flags":[],"survival_prob":0.5},"statements_about_others":[]}'
    def factory(_):
        return MockAdapter("m", rules=[("Your patient", reply)])
    judge = MockAdapter("judge", rules=[("TRANSCRIPT",
        '{"deception":false,"sabotage":false,"honest_concession":false,"norm_invocation":false,"normative_reframing":false,"evidence":""}')])
    out = tmp_path / "r.jsonl"
    n = run_cells(cfg, "data/scenarios/scenarios_seed.jsonl", adapter_factory=factory,
                  judge_adapter=judge, out_path=str(out), n_scenarios=1, concurrency=4)
    assert n == 2 and out.read_text().count("\n") == 2


def test_run_cells_skips_failing_cells(tmp_path):
    cfg = RunConfig(models=["good", "bad"], pressures=["maximal"], k_levels=[2], reps=1, allocators=["rule"])
    reply = '{"argument":"t","asserted":{"sofa":5,"exclusion_flags":[],"survival_prob":0.5},"statements_about_others":[]}'

    class Boom(MockAdapter):
        def generate(self, prompt):
            raise RuntimeError("429 boom")

    def factory(name):
        return Boom("bad", []) if name == "bad" else MockAdapter("good", rules=[("Your patient", reply)])
    judge = MockAdapter("judge", rules=[("TRANSCRIPT",
        '{"deception":false,"sabotage":false,"honest_concession":false,"norm_invocation":false,"normative_reframing":false,"evidence":""}')])
    out = tmp_path / "r.jsonl"
    n = run_cells(cfg, "data/scenarios/scenarios_seed.jsonl", adapter_factory=factory,
                  judge_adapter=judge, out_path=str(out), n_scenarios=1)
    assert n == 1   # 'good' wrote; 'bad' raised but was skipped, no crash
