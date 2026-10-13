# tests/test_spend.py
import json
import pytest
from scarcity_ward import spend as S


def test_guard_raises_at_cap():
    seq = iter([10.0, 59.0, 60.0])
    g = S.SpendGuard(50, 10.0, lambda: next(seq))
    assert g.check() == 0.0
    assert g.check() == 49.0
    with pytest.raises(S.SpendCapExceeded):
        g.check()


def test_guard_from_file(tmp_path):
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"baseline_usd": 5.0, "cap_usd": 20.0, "key_limit_usd": 100.0, "created": "x"}))
    g = S.guard_from_file(str(f), usage_fn=lambda: 12.5)
    assert g.spent() == 7.5 and g.cap == 20.0


def test_key_limit_none_when_unlimited():
    assert S.key_limit(fetch=lambda: {"usage": 3.0, "limit": None}) is None
    assert S.key_limit(fetch=lambda: {"usage": 3.0, "limit": 80}) == 80.0
    assert S.key_usage(fetch=lambda: {"usage": 3.25, "limit": 80}) == 3.25


def test_halter_stops_after_cap_crossed():
    seq = iter([0.0, 100.0])          # one probe at start, one after the 20th write
    h = S.Halter(S.SpendGuard(50, 0.0, lambda: next(seq)), every=20)
    assert h.ok()
    for _ in range(19):
        h.after_write()
    assert h.ok()                     # no probe yet
    h.after_write()
    assert not h.ok()


# ---- runner hook (scripts/run_agentic_ward.py, WARD_SPEND_FILE) -------------------------------
# Added beyond the brief's four tests so the Step 5 edit to `main` is exercised without network.
import sys

sys.path.insert(0, "scripts")
from scarcity_ward.schemas import Patient, Scenario
from scarcity_ward.models.mock import MockAdapter
from scarcity_ward.hospital.ward_env import ResourceNeeds, WardScenario
import run_agentic_ward as R


def _ws(sid):
    ps = [Patient("A", 60, "sepsis", 6, [], [], 0.70),
          Patient("B", 61, "pneumonia", 6, [], [], 0.62),
          Patient("C", 62, "ARDS", 9, [], [], 0.42)]
    scn = Scenario(sid, "ventilator", 1, ps, "sofa_survival", "clear")
    return WardScenario(scn, {"A": ResourceNeeds(3), "B": ResourceNeeds(3), "C": ResourceNeeds(1)},
                        n_steps=2, focal_pid="B")


def _wire(monkeypatch, tmp_path, usage, n_scn=60, baseline=0.0, cap=50.0):
    """Point the runner at a tmp output, a spend file, and a fake key-usage reading."""
    bf = tmp_path / "spend_baseline.json"
    bf.write_text(json.dumps({"baseline_usd": baseline, "cap_usd": cap, "key_limit_usd": 1000.0, "created": "x"}))
    out = tmp_path / "run.jsonl"
    monkeypatch.setattr(R, "SPEND_FILE", str(bf))
    monkeypatch.setattr(R, "OUT", str(out))
    monkeypatch.setattr(R, "MODE", "focal")
    monkeypatch.setattr(R, "ARMS", ["neutral"])
    monkeypatch.setattr(R, "INFO", ["full"])
    monkeypatch.setattr(R, "WORDINGS", [0])
    monkeypatch.setattr(R, "OVERSIGHT", ["blind"])
    monkeypatch.setattr(R, "DOCTORS", ["mock"])
    monkeypatch.setattr(R, "CONCURRENCY", 1)
    monkeypatch.setattr(R, "RULES", False)
    monkeypatch.setattr(R, "PROMPT_VERSION", "v6.1")
    monkeypatch.setattr(R, "TEMP_TAG", None)
    monkeypatch.setattr(R, "load_ward_scenarios", lambda _p: [_ws("S%02d" % i) for i in range(n_scn)])
    monkeypatch.setattr(R, "build_adapter",
                        lambda _l: MockAdapter("doc", rules=[("Step", '{"rationale":"x","action":{"verb":"wait"}}')]))
    monkeypatch.setattr(S, "key_usage", usage)
    return out


def test_runner_launches_nothing_when_cap_already_reached(monkeypatch, tmp_path, capsys):
    out = _wire(monkeypatch, tmp_path, usage=lambda: 75.0)      # spent 75 >= cap 50 at start
    with pytest.raises(SystemExit) as e:
        R.main("60", "1")
    assert e.value.code == 3
    assert not out.exists() or out.read_text() == ""
    text = capsys.readouterr().out
    assert "spend cap already reached; nothing launched" in text
    assert "final spent $75.00" in text                         # final spent figure is reported


def test_runner_stops_launching_once_cap_crossed_and_exits_3(monkeypatch, tmp_path, capsys):
    readings = iter([0.0, 10.0, 80.0, 80.0])                    # start, after write 20, after write 40, final read
    out = _wire(monkeypatch, tmp_path, usage=lambda: next(readings, 80.0))
    with pytest.raises(SystemExit) as e:
        R.main("60", "1")                                       # 60 cells available, sequential
    assert e.value.code == 3
    n = len([l for l in out.read_text().splitlines() if l.strip()])
    assert n == 40                                              # halted at the 40th write, not 60
    assert "final spent $80.00" in capsys.readouterr().out


def test_runner_normal_completion_returns_without_exit(monkeypatch, tmp_path):
    out = _wire(monkeypatch, tmp_path, usage=lambda: 1.0, n_scn=25)   # far below the cap
    assert R.main("25", "1") is None                            # no SystemExit: exit status 0
    assert len([l for l in out.read_text().splitlines() if l.strip()]) == 25


def test_runner_without_spend_file_warns_and_runs(monkeypatch, tmp_path, capsys):
    out = _wire(monkeypatch, tmp_path, usage=lambda: 1 / 0, n_scn=5)
    monkeypatch.setattr(R, "SPEND_FILE", None)                  # no guard configured: usage_fn never called
    R.main("5", "1")
    assert len([l for l in out.read_text().splitlines() if l.strip()]) == 5
    assert "NO SPEND GUARD (WARD_SPEND_FILE unset)" in capsys.readouterr().err


def test_runner_with_spend_file_does_not_warn(monkeypatch, tmp_path, capsys):
    _wire(monkeypatch, tmp_path, usage=lambda: 1.0, n_scn=3)
    R.main("3", "1")
    assert "NO SPEND GUARD" not in capsys.readouterr().err


def test_skipped_episodes_advance_the_probe(monkeypatch, tmp_path):
    # Every episode raises (it may already have made paid calls). Attempts, not only successful writes,
    # must drive the probe cadence, otherwise a run of failures is never checked against the cap.
    readings = iter([0.0, 100.0])                               # start, then the probe after the 20th attempt
    out = _wire(monkeypatch, tmp_path, usage=lambda: next(readings, 100.0))
    calls = []

    def boom(_label):
        calls.append(1)
        raise RuntimeError("provider error")

    monkeypatch.setattr(R, "build_adapter", boom)
    with pytest.raises(SystemExit) as e:
        R.main("60", "1")
    assert e.value.code == 3
    assert len(calls) == 20                                     # halted at the 20th attempt, not 60
    assert not out.exists() or out.read_text() == ""            # nothing was written: all skipped


def test_concurrent_runner_halts(monkeypatch, tmp_path):
    out = _wire(monkeypatch, tmp_path, usage=lambda: 0.0, n_scn=60)
    state = {"n": 0}

    def usage():
        state["n"] += 1
        return 0.0 if state["n"] == 1 else 999.0                # first probe fine, every later one over the cap

    monkeypatch.setattr(S, "key_usage", usage)
    monkeypatch.setattr(R, "CONCURRENCY", 4)
    with pytest.raises(SystemExit) as e:
        R.main("60", "1")
    assert e.value.code == 3
    n = len([l for l in out.read_text().splitlines() if l.strip()])
    assert 20 <= n < 60                                         # stopped launching after the first due probe


def test_usage_below_baseline_raises():
    g = S.SpendGuard(50, 100.0, lambda: 90.0)                   # key rotated or recreated: usage fell
    with pytest.raises(S.SpendCapExceeded) as e:
        g.spent()
    assert "below" in str(e.value) and "baseline" in str(e.value)
    with pytest.raises(S.SpendCapExceeded):
        g.check()
    assert S.SpendGuard(50, 100.0, lambda: 100.0).spent() == 0.0   # equal to baseline is fine


def test_halter_halts_when_usage_below_baseline(capsys):
    h = S.Halter(S.SpendGuard(50, 100.0, lambda: 3.0), every=20)
    assert not h.ok()
    assert "below" in capsys.readouterr().out


def _flaky(script):
    """usage_fn that replays `script`: a float is returned, an Exception instance is raised."""
    it = iter(script)
    calls = []

    def fn():
        calls.append(1)
        v = next(it)
        if isinstance(v, Exception):
            raise v
        return v

    return fn, calls


def test_probe_failure_sets_stop_after_three_tries_then_reraises():
    sleeps = []
    fn, calls = _flaky([0.0] + [OSError("net down")] * 3)       # start probe ok, then 3 failures
    h = S.Halter(S.SpendGuard(50, 0.0, fn), every=1, sleep=sleeps.append, backoff=0.01)
    assert h.ok() and len(calls) == 1
    with pytest.raises(OSError):
        h.after_write()
    assert len(calls) == 4                                      # exactly 3 tries in the failing probe
    assert not h.ok()                                           # stop flag set before re-raise
    assert len(sleeps) == 2                                     # backoff between tries, none after the last


def test_probe_retries_recover_from_transient_failure():
    fn, calls = _flaky([0.0, OSError("blip"), 5.0])
    h = S.Halter(S.SpendGuard(50, 0.0, fn), every=1, sleep=lambda s: None)
    h.after_write()                                             # first try fails, second succeeds
    assert h.ok() and len(calls) == 3


def test_probe_failure_at_start_fails_closed():
    fn, calls = _flaky([OSError("down")] * 3)
    with pytest.raises(OSError):
        S.Halter(S.SpendGuard(50, 0.0, fn), every=20, sleep=lambda s: None)
    assert len(calls) == 3


def test_probe_failure_stops_other_threads(monkeypatch, tmp_path):
    state = {"n": 0}

    def usage():
        state["n"] += 1
        if state["n"] == 1:
            return 0.0
        raise OSError("down")

    out = _wire(monkeypatch, tmp_path, usage=usage, n_scn=60)
    monkeypatch.setattr(S, "key_usage", usage)
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    with pytest.raises(OSError):
        R.main("60", "1")
    n = len([l for l in out.read_text().splitlines() if l.strip()])
    assert n == 20                                              # sequential run stopped at the failed probe


# ---- shared launcher helpers: halter_from_env / halter_from_path / exit_if_halted ------------------

def _spend_file(tmp_path, baseline=0.0, cap=50.0):
    f = tmp_path / "spend_baseline.json"
    f.write_text(json.dumps({"baseline_usd": baseline, "cap_usd": cap, "key_limit_usd": 1000.0, "created": "x"}))
    return str(f)


@pytest.mark.parametrize("value", [None, ""])
def test_halter_from_env_unset_or_empty_warns_and_returns_none(monkeypatch, capsys, value):
    if value is None:
        monkeypatch.delenv("WARD_SPEND_FILE", raising=False)
    else:
        monkeypatch.setenv("WARD_SPEND_FILE", value)
    monkeypatch.setattr(S, "key_usage", lambda: 1 / 0)            # must never be read without a guard
    assert S.halter_from_env(every=50) is None
    err = capsys.readouterr().err
    assert "NO SPEND GUARD (WARD_SPEND_FILE unset)" in err


def test_halter_from_env_names_the_variable_it_was_given(monkeypatch, capsys):
    monkeypatch.delenv("MY_SPEND_FILE", raising=False)
    assert S.halter_from_env(every=5, env="MY_SPEND_FILE") is None
    assert "NO SPEND GUARD (MY_SPEND_FILE unset)" in capsys.readouterr().err


def test_halter_from_env_builds_a_working_halter(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("WARD_SPEND_FILE", _spend_file(tmp_path))
    monkeypatch.setattr(S, "key_usage", lambda: 5.0)
    h = S.halter_from_env(every=7)
    assert isinstance(h, S.Halter) and h.ok() and h.every == 7
    assert "NO SPEND GUARD" not in capsys.readouterr().err
    S.exit_if_halted(h)                                           # healthy guard: no exit


def test_halter_from_env_exits_3_when_cap_already_reached_at_start(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("WARD_SPEND_FILE", _spend_file(tmp_path))
    monkeypatch.setattr(S, "key_usage", lambda: 75.0)             # spent 75 >= cap 50
    with pytest.raises(SystemExit) as e:
        S.halter_from_env(every=50)
    assert e.value.code == 3
    out = capsys.readouterr().out
    assert "spend cap already reached; nothing launched" in out
    assert "SPEND GUARD HALT: final spent $75.00 of cap $50.00; exiting with status 3" in out


def test_halter_from_env_fails_closed_when_usage_cannot_be_read(monkeypatch, tmp_path):
    monkeypatch.setenv("WARD_SPEND_FILE", _spend_file(tmp_path))
    monkeypatch.setattr(S, "key_usage", lambda: (_ for _ in ()).throw(OSError("down")))
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    with pytest.raises(OSError):
        S.halter_from_env(every=50)


def test_halter_from_path_matches_env_variant(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(S, "key_usage", lambda: 5.0)
    assert S.halter_from_path(None, every=20) is None
    assert "NO SPEND GUARD (WARD_SPEND_FILE unset)" in capsys.readouterr().err
    assert S.halter_from_path(_spend_file(tmp_path), every=20).every == 20


def test_exit_if_halted_is_a_noop_without_a_guard_or_when_ok():
    S.exit_if_halted(None)
    ok = type("H", (), {"ok": lambda self: True})()
    S.exit_if_halted(ok)


def test_exit_if_halted_mid_run_exits_3_with_final_figure(monkeypatch, tmp_path, capsys):
    readings = iter([0.0, 80.0, 80.0])                            # start probe, probe after attempt 2, final read
    monkeypatch.setenv("WARD_SPEND_FILE", _spend_file(tmp_path))
    monkeypatch.setattr(S, "key_usage", lambda: next(readings, 80.0))
    h = S.halter_from_env(every=2)
    h.after_attempt()
    S.exit_if_halted(h)                                           # not due yet: still no exit
    h.after_attempt()                                             # second attempt triggers the probe
    assert not h.ok()
    with pytest.raises(SystemExit) as e:
        S.exit_if_halted(h)
    assert e.value.code == 3
    assert "final spent $80.00 of cap $50.00; exiting with status 3" in capsys.readouterr().out


def test_exit_if_halted_reports_unknown_when_final_figure_unavailable(capsys):
    class H:
        guard = type("G", (), {"cap": 50.0})()
        def ok(self): return False
        def final_spent(self): return None
    with pytest.raises(SystemExit) as e:
        S.exit_if_halted(H())
    assert e.value.code == 3
    assert "final spent unknown of cap $50.00" in capsys.readouterr().out
