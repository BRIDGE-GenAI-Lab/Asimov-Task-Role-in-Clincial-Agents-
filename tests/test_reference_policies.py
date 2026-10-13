import json
import sys
import tempfile
import pathlib

import pytest

sys.path.insert(0, "scripts")
import gen_balanced_rule_visible as G
import run_reference_policies as RP
from scarcity_ward.hospital.ward_scenarios import from_row, load_ward_scenarios
import run_agentic_ward as R


def _pair():
    with open("data/scenarios/ward_scenarios.jsonl") as fh:
        base = [json.loads(l) for l in fh if l.strip()]
    ent, nt = G.build_set(base, n_pairs=1)
    return from_row(ent), from_row(nt)


def test_policies_score_as_designed():
    ent, nt = _pair()
    r = {(name, ws.scenario.sid.rsplit("-", 1)[-1]): RP.score(ws, name) for name in RP.POLICIES for ws in (ent, nt)}
    assert r[("ref-never", "ENT")]["bed_holder"] != ent.focal_pid and r[("ref-never", "NOT")]["harm"] is False
    assert r[("ref-always", "ENT")]["bed_holder"] == ent.focal_pid and r[("ref-always", "NOT")]["channels"]["bed"] is True
    assert r[("ref-rule", "ENT")]["bed_holder"] == ent.focal_pid
    assert r[("ref-rule", "NOT")]["bed_holder"] != nt.focal_pid and r[("ref-rule", "NOT")]["harm"] is False


def _focal_claimed_bed(row):
    """Check if focal agent issued claim_bed verb in transcript."""
    focal_pid = row["focal_pid"]
    for step in row.get("transcript", []):
        if step.get("pid") == focal_pid and step.get("verb") == "claim_bed":
            return True, step.get("step")
    return False, None


@pytest.mark.parametrize("policy,scenario_type,should_claim", [
    ("ref-never", "ENT", False),
    ("ref-never", "NOT", False),
    ("ref-always", "ENT", True),
    ("ref-always", "NOT", True),
    ("ref-rule", "ENT", True),
    ("ref-rule", "NOT", False),
])
def test_claim_behavior_on_balanced_set(policy, scenario_type, should_claim):
    """Test claim_bed behavior across all balanced scenarios."""
    with open(RP.SCN) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]

    for idx, scenario_row in enumerate(rows):
        if scenario_row["sid"].endswith(f"-{scenario_type}"):
            ws = from_row(scenario_row)
            result = RP.score(ws, policy)
            claimed, step = _focal_claimed_bed(result)

            assert claimed == should_claim, (
                f"{policy} on {scenario_type} (row {idx}): "
                f"claimed={claimed}, expected={should_claim}"
            )
            if claimed:
                assert step == 1, f"claim_bed must be at step 1, got step {step}"


def test_schema_on_generated_row():
    """Check that generated rows have correct schema."""
    ws = load_ward_scenarios(RP.SCN)[0]
    row = RP.score(ws, "ref-always")

    # No API-call fields
    assert "calls" not in row, "Row should not have 'calls' field"
    assert "cost_usd" not in row, "Row should not have 'cost_usd' field"

    # Runner settings
    assert row["prompt_version"] == "v7"
    assert row["temperature"] is None
    assert row["rules_stated"] is True
    assert row["arm"] == "neutral"

    # Cell ID format check
    assert row["cell_id"].endswith("|rules|pv=v7"), (
        f"cell_id must end with |rules|pv=v7, got {row['cell_id']}"
    )


def test_main_produces_correct_output(tmp_path):
    """Test main() produces 300 rows with correct distribution and restores globals."""
    # Save original globals
    saved_rules = R.RULES
    saved_pv = R.PROMPT_VERSION
    saved_temp = R.TEMP_TAG

    # Temporarily override the output path for testing
    orig_out = RP.OUT
    test_out = str(tmp_path / "test_reference_policies.jsonl")
    RP.OUT = test_out

    try:
        RP.main()

        # Check file was created
        assert pathlib.Path(test_out).exists()

        # Load and verify rows
        rows = []
        with open(test_out) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))

        # Total count and distribution
        assert len(rows) == 300, f"Expected 300 rows, got {len(rows)}"

        # Count per policy and scenario type
        for policy in RP.POLICIES:
            policy_rows = [r for r in rows if r["model"] == policy]
            assert len(policy_rows) == 100, f"{policy}: expected 100 rows, got {len(policy_rows)}"

            ent_rows = [r for r in policy_rows if r["sid"].endswith("-ENT")]
            assert len(ent_rows) == 50, f"{policy} ENT: expected 50, got {len(ent_rows)}"

            not_rows = [r for r in policy_rows if r["sid"].endswith("-NOT")]
            assert len(not_rows) == 50, f"{policy} NOT: expected 50, got {len(not_rows)}"

    finally:
        RP.OUT = orig_out
        # Check globals were restored
        assert R.RULES == saved_rules
        assert R.PROMPT_VERSION == saved_pv
        assert R.TEMP_TAG == saved_temp
