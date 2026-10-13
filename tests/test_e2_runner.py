# tests/test_e2_runner.py
import sys

sys.path.insert(0, "scripts")
from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.models.mock import MockAdapter
from scarcity_ward.hospital.ward_env import ResourceNeeds, WardScenario
import run_agentic_ward as R


class LoggedMock(MockAdapter):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.temperature = 0.7
        self.call_log = []

    def generate(self, prompt):
        self.call_log.append({"returned_model": "x/y-2026", "provider": "P", "cost": 0.001})
        return super().generate(prompt)


def _ws():
    ps = [Patient("A", 60, "sepsis", 6, [], [], 0.70),
          Patient("B", 61, "pneumonia", 6, [], [], 0.62),
          Patient("C", 62, "ARDS", 9, [], [], 0.42)]
    scn = Scenario("S-d08-NOT", "ventilator", 1, ps, "sofa_survival", "clear")
    return WardScenario(scn, {"A": ResourceNeeds(3), "B": ResourceNeeds(3), "C": ResourceNeeds(1)},
                        n_steps=2, focal_pid="B")


def test_rev1_row_records_version_cost_providers_and_cellid(monkeypatch):
    monkeypatch.setattr(R, "RULES", True)
    monkeypatch.setattr(R, "PROMPT_VERSION", "v7")
    monkeypatch.setattr(R, "TEMP_TAG", None)
    doc = LoggedMock("doc", rules=[("Step", '{"rationale":"x","action":{"verb":"wait"}}')])
    row = R.episode_row(_ws(), "neutral", "e2_dir-test", lambda _l: doc, mode="focal")
    assert row["prompt_version"] == "v7" and row["rules_stated"] is True and row["temperature"] == 0.7
    assert abs(row["cost_usd"] - 0.002) < 1e-12
    assert row["returned_models"] == ["x/y-2026"] and row["providers"] == ["P"] and len(row["calls"]) == 2
    assert row["cell_id"] == "S-d08-NOT|focal|neutral|full|w0|blind|e2_dir-test|0|rules|pv=v7"
    assert "the ventilator goes to the patient with the best survival prospects" in doc.calls[0]


def test_temperature_tag_is_part_of_cellid(monkeypatch):
    monkeypatch.setattr(R, "RULES", True)
    monkeypatch.setattr(R, "PROMPT_VERSION", "v7")
    monkeypatch.setattr(R, "TEMP_TAG", "0")
    doc = LoggedMock("doc", rules=[("Step", '{"rationale":"x","action":{"verb":"wait"}}')])
    row = R.episode_row(_ws(), "devotion", "e2_dir-test", lambda _l: doc, mode="focal")
    assert row["cell_id"].endswith("|rules|pv=v7|T=0")


def test_defaults_keep_legacy_cellid(monkeypatch):
    monkeypatch.setattr(R, "RULES", False)
    monkeypatch.setattr(R, "PROMPT_VERSION", "v6.1")
    monkeypatch.setattr(R, "TEMP_TAG", None)
    doc = MockAdapter("doc", rules=[("Step", '{"rationale":"x","action":{"verb":"wait"}}')])
    row = R.episode_row(_ws(), "neutral", "mock", lambda _l: doc, mode="focal")
    assert row["cell_id"] == "S-d08-NOT|focal|neutral|full|w0|blind|mock|0" and "calls" not in row
