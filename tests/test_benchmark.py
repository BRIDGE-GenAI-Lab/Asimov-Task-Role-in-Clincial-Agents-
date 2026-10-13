from scarcity_ward.benchmark import generate_frozen_set
from scarcity_ward.scenarios import validate_scenario


def test_frozen_set_deterministic_and_valid():
    a = generate_frozen_set(n=20, k=3, seed=42)
    b = generate_frozen_set(n=20, k=3, seed=42)
    assert [p.sofa for s in a for p in s.patients] == [p.sofa for s in b for p in s.patients]
    assert len(a) == 20
    for s in a:
        assert validate_scenario(s) == []
    configs = set(s.config for s in a)
    assert "near_tie" in configs and "clear" in configs


def test_frozen_set_calibrated_sofa_range():
    s = generate_frozen_set(n=100, k=3, seed=1)
    sofas = [p.sofa for sc in s for p in sc.patients]
    assert 3.0 <= (sum(sofas) / len(sofas)) <= 9.0   # near MIMIC mean ~5.7
    assert all(0 <= x <= 24 for x in sofas)
