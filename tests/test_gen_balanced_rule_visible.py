# tests/test_gen_balanced_rule_visible.py
import collections
import copy
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, "scripts")
import gen_balanced_rule_visible as G
from scarcity_ward.hospital.ward_scenarios import from_row
from scarcity_ward.protocols import rightful_recipients

SRC = "data/scenarios/ward_scenarios.jsonl"
OUT_SHA256 = "10d4a9643e34ef2249941541298916435c67324c123ecaafafc70d05fb37a37e"


def _base():
    with open(SRC) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def test_every_pair_label_matches_scorer():
    rows = G.build_set(_base())
    assert len(rows) == 100
    for r in rows:
        ws = from_row(r)
        top = rightful_recipients(ws.scenario)[0]
        if r["_condition"] == "ENT":
            assert top == r["focal_pid"], r["sid"]
        else:
            assert top != r["focal_pid"], r["sid"]


def test_ten_pairs_per_margin_and_matched_ids():
    rows = G.build_set(_base())
    c = collections.Counter((r["_margin"], r["_condition"]) for r in rows)
    assert all(c[(m, k)] == 10 for m in G.MARGINS for k in ("ENT", "NOT"))
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["_orig_sid"], r["_margin"])][r["_condition"]] = r
    assert all(set(v) == {"ENT", "NOT"} for v in by.values())


def test_focal_identical_and_rival_matched_within_pair():
    base = _base()[0]
    ent, nt = G.build_pair(base, 0.16)
    f = lambda r, pid: next(p for p in r["patients"] if p["pid"] == pid)
    assert f(ent, ent["focal_pid"]) == f(nt, nt["focal_pid"])
    rival = [p["pid"] for p in ent["patients"] if p["sofa"] == f(ent, ent["focal_pid"])["sofa"]
             and p["pid"] != ent["focal_pid"]][0]
    assert ent["needs"][rival] == ent["needs"][ent["focal_pid"]]
    assert round(f(nt, rival)["survival_prob"] - f(ent, rival)["survival_prob"], 2) == 0.32


def test_zero_margin_is_refused():
    with pytest.raises(ValueError, match=r"margin must be positive \(delta > 0\)"):
        G.build_set(_base(), margins=(0.0,), n_pairs=5)
    with pytest.raises(ValueError, match=r"margin must be positive"):
        G.build_pair(_base()[0], -0.08)


def test_identity_order_and_priority_are_fully_crossed():
    rows = G.build_set(_base())
    # every focal patient (identity) appears once favoured and once not, with the same labels, so patient-label
    # order (which sets the imaging tie-break order at the tier-3 tie) is crossed with priority
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["_orig_sid"], r["_margin"])][r["_condition"]] = r
    for v in by.values():
        assert v["ENT"]["focal_pid"] == v["NOT"]["focal_pid"] and v["ENT"]["_rival_pid"] == v["NOT"]["_rival_pid"]
    cells = collections.Counter((r["focal_pid"] < r["_rival_pid"], r["_condition"]) for r in rows)
    assert cells == {(True, "ENT"): 25, (True, "NOT"): 25, (False, "ENT"): 25, (False, "NOT"): 25}


def test_focal_label_sorts_before_rival_in_half_the_pairs_of_each_margin():
    # patient-label (pid) order sets the imaging tie-break order; this does not balance visible board position
    rows = [r for r in G.build_set(_base()) if r["_condition"] == "ENT"]
    for m in G.MARGINS:
        first = [r["focal_pid"] < r["_rival_pid"] for r in rows if r["_margin"] == m]
        assert len(first) == 10 and sum(first) == 5


def _pairs(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["_orig_sid"], r["_margin"])][r["_condition"]] = r
    assert len(by) == 50
    return by.values()


def test_ent_and_not_rows_differ_only_in_sid_condition_and_rival_survival():
    for v in _pairs(G.build_set(_base())):
        ent, nt = copy.deepcopy(v["ENT"]), copy.deepcopy(v["NOT"])
        rival = ent["_rival_pid"]
        assert rival == nt["_rival_pid"] and ent["focal_pid"] == nt["focal_pid"]
        # the rival's survival must actually move, and by exactly 2 x margin
        sr_e = next(p for p in ent["patients"] if p["pid"] == rival)["survival_prob"]
        sr_n = next(p for p in nt["patients"] if p["pid"] == rival)["survival_prob"]
        assert round(sr_n - sr_e, 2) == round(2 * ent["_margin"], 2), ent["sid"]
        for row in (ent, nt):
            for k in ("sid", "_condition"):
                row.pop(k)
            next(p for p in row["patients"] if p["pid"] == rival)["survival_prob"] = None
        assert ent == nt, v["ENT"]["sid"]


def test_matched_rival_dominated_third_and_survival_argmax_agrees_with_label():
    for r in G.build_set(_base()):
        pid = {p["pid"]: p for p in r["patients"]}
        f, v = r["focal_pid"], r["_rival_pid"]
        third = [k for k in pid if k not in (f, v)]
        assert len(pid) == 3 and len(third) == 1 and f != v, r["sid"]
        t = third[0]
        assert pid[f]["sofa"] == pid[v]["sofa"], r["sid"]
        assert r["needs"][f] == r["needs"][v], r["sid"]
        assert pid[f]["exclusion_flags"] == pid[v]["exclusion_flags"], r["sid"]
        # third patient strictly dominated: sicker (higher SOFA) and lower survival than both of the pair
        assert pid[t]["sofa"] > pid[f]["sofa"], r["sid"]
        assert pid[t]["survival_prob"] < min(pid[f]["survival_prob"], pid[v]["survival_prob"]), r["sid"]
        assert all(0.0 <= p["survival_prob"] <= 1.0 for p in pid.values()), r["sid"]
        # survival-only argmax (unique) agrees with the label
        ranked = sorted(pid, key=lambda k: pid[k]["survival_prob"], reverse=True)
        top = ranked[0]
        assert pid[ranked[0]]["survival_prob"] > pid[ranked[1]]["survival_prob"], r["sid"]    # no tie
        assert (top == f) == (r["_condition"] == "ENT"), r["sid"]
        assert top in (f, v), r["sid"]


def test_check_pair_flags_survival_outside_unit_interval():
    ent, nt = G.build_pair(_base()[0], 0.04)
    assert G.check_pair(ent, nt) == []
    bad = copy.deepcopy(nt)
    next(p for p in bad["patients"] if p["pid"] == bad["_rival_pid"])["survival_prob"] = 1.04
    assert any("outside [0, 1]" in e for e in G.check_pair(ent, bad))


def test_scenario_file_on_disk_is_the_generator_output_and_hash_is_pinned():
    if not os.path.exists(SRC):
        pytest.skip("base scenario file %s is absent; cannot regenerate the balanced set" % SRC)
    assert os.path.exists(G.OUT), "balanced scenario file missing: " + G.OUT
    with open(G.OUT, "rb") as fh:
        raw = fh.read()
    assert [json.loads(l) for l in raw.decode().splitlines() if l.strip()] == G.build_set(_base())
    regen = "".join(json.dumps(r) + "\n" for r in G.build_set(_base())).encode()
    assert raw == regen
    assert hashlib.sha256(raw).hexdigest() == OUT_SHA256
