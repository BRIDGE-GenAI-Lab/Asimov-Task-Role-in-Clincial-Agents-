from scarcity_ward.mimic.itemids import SOFA_ITEMIDS, label_matches


def test_canonical_map_has_all_systems():
    for key in ["gcs", "map", "platelets", "bilirubin", "creatinine", "pao2", "fio2",
                "norepinephrine"]:
        assert key in SOFA_ITEMIDS and len(SOFA_ITEMIDS[key]) >= 1


def test_label_matches_is_case_insensitive_substring():
    assert label_matches("Arterial Blood Pressure mean", ["blood pressure mean"])
    assert not label_matches("Heart Rate", ["bilirubin"])
