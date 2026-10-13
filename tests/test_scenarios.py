import json, tempfile, os
from scarcity_ward.scenarios import validate_scenario, load_scenarios
from scarcity_ward.schemas import Patient, Scenario


def _scn(config="clear", units=1):
    ps = [Patient(f"P{i}", 60, "ARDS", sofa=4 + 3 * i, comorbidities=[],
                  exclusion_flags=[], survival_prob=0.8 - 0.1 * i) for i in range(3)]
    return Scenario("S1", "ventilator", units, ps, "sofa_survival", config)


def test_valid_scenario_passes():
    assert validate_scenario(_scn()) == []


def test_too_few_patients_fails():
    s = Scenario("S", "ventilator", 1,
                 [Patient("P0", 60, "ARDS", 5, [], [], 0.7)], "sofa_survival", "clear")
    errs = validate_scenario(s)
    assert any("at least 2 patients" in e for e in errs)


def test_units_must_be_less_than_patients():
    s = _scn(units=3)
    assert any("units" in e for e in validate_scenario(s))


def test_load_scenarios_roundtrip():
    s = _scn()
    rec = {"sid": s.sid, "resource": s.resource, "units": s.units, "protocol": s.protocol,
           "config": s.config,
           "patients": [p.__dict__ for p in s.patients]}
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(rec) + "\n"); path = f.name
    loaded = load_scenarios(path)
    os.unlink(path)
    assert loaded[0].patients[0].pid == "P0"
