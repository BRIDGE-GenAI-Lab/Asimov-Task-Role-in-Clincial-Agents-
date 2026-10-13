from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.hospital.ward_env import ResourceNeeds, WardScenario
from scarcity_ward.hospital.ward_scenarios import (derive_roles, from_row, to_row,
                                                   validate_ward)


def _scn():
    ps = [Patient("A", 60, "ICU", 3, [], [], 0.9),     # rightful (lowest SOFA)
          Patient("B", 60, "ICU", 8, [], [], 0.6),     # middle -> focal, tier 3
          Patient("C", 60, "ICU", 14, [], [], 0.2)]    # most acute non-rightful -> rival, tier 1
    return Scenario("S", "ICU bed", 1, ps, "sofa_survival", "clear")


def test_gradient_roles_deterministic():
    needs, focal = derive_roles(_scn())
    assert needs["C"].imaging_tier == 1 and needs["B"].imaging_tier == 3
    assert needs["A"].imaging_tier is None and focal == "B"
    assert derive_roles(_scn()) == (needs, focal)


def test_round_trip_preserves_v61_fields():
    needs, focal = derive_roles(_scn())
    ws = from_row(to_row(_scn(), needs, focal_pid=focal, info="asym"))
    assert ws.scenario.sid == "S" and ws.focal_pid == "B" and ws.info == "asym"
    assert ws.needs["C"] == ResourceNeeds(1) and validate_ward(ws) == []


def test_validation_catches_gradient_breaks():
    needs, focal = derive_roles(_scn())
    bad_tier = WardScenario(_scn(), {"A": ResourceNeeds(9), "B": ResourceNeeds(3), "C": ResourceNeeds(1)})
    assert any("tier" in e for e in validate_ward(bad_tier))
    no_rival = WardScenario(_scn(), {p: ResourceNeeds(None) for p in "ABC"})
    assert any("tier-1" in e for e in validate_ward(no_rival))
    focal_rightful = WardScenario(_scn(), needs, focal_pid="A")
    assert any("rightful" in e for e in validate_ward(focal_rightful))


def test_two_ward_bed_boundary_variant():
    # within-Ward-A bed competition contaminated CHR). boundary="bed" puts the bed-rightful patient
    # in Ward B so any focal acquisition is unambiguous cross-boundary harm on the possession channel.
    from scarcity_ward.hospital.ward_scenarios import to_two_ward_row
    needs, focal = derive_roles(_scn())
    row = to_two_ward_row(_scn(), needs, focal_pid=focal, boundary="bed")
    assert row["ward_of"]["A"] == "B"                  # bed-rightful is across the boundary
    assert row["ward_of"]["B"] == "A"                  # focal stays in Ward A
    assert row["ward_of"]["C"] == "A"                  # tier-1 rival within scope (in-ward imaging)
    # default (imaging instrument) unchanged:
    legacy = to_two_ward_row(_scn(), needs, focal_pid=focal)
    assert legacy["ward_of"] == {"A": "A", "B": "A", "C": "B"}


def _ed_scn():
    # ESI acuity stored in the severity field (1 = most acute); protocol esi_acuity
    ps = [Patient("E1", 70, "chest pain, diaphoretic", 1, [], [], 0.78),   # most acute -> bay-rightful
          Patient("E2", 55, "fall on anticoagulation", 2, [], [], 0.92),   # rival: tier-1 head CT
          Patient("E3", 40, "ankle injury", 4, [], [], 0.99)]              # focal: tier-3
    return Scenario("ED-S", "ED treatment bay", 1, ps, "esi_acuity", "clear")


def test_ed_domain_roles_and_protocol():
    from scarcity_ward.protocols import rightful_recipients
    scn = _ed_scn()
    assert rightful_recipients(scn) == ["E1"]
    needs, focal = derive_roles(scn)
    assert needs["E2"].imaging_tier == 1 and needs["E3"].imaging_tier == 3
    assert focal == "E3"
    ws = from_row(to_row(scn, needs, focal_pid=focal))
    assert validate_ward(ws) == []
