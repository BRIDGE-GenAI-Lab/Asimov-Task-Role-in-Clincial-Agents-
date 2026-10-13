"""Runs ward episodes for the models under test. Settings come from WARD_* environment variables (mode, models, roles, information level, oversight, policy shown, concurrency, output path). It loads the scenario file, runs a focal or whole-ward episode for each scenario, role, model and replicate not yet logged, appends one JSON row per episode to WARD_OUT and stops when the spend limit is reached."""
import dataclasses
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from scarcity_ward.envload import load_env
load_env()
from scarcity_ward.models.registry import build_adapter
from scarcity_ward.hospital.ward_scenarios import load_ward_scenarios
from scarcity_ward.hospital.ward_episode import (run_focal_episode,
                                                 run_ward_episode)
from scarcity_ward.hospital.oversight import Oversight, make_oversight
from scarcity_ward.spend import exit_if_halted, halter_from_path

MODE = os.environ.get("WARD_MODE", "focal")
DOCTORS = os.environ.get("WARD_DOCTORS", "gemma3-12b-qat").split(",")
ARMS = os.environ.get("WARD_ARMS", "steward,devotion").split(",")
INFO = os.environ.get("WARD_INFO", "full").split(",")
WORDINGS = [int(w) for w in os.environ.get("WARD_WORDINGS", "0").split(",")]
OVERSIGHT = os.environ.get("WARD_OVERSIGHT", "blind").split(";")
DIALOGUE = os.environ.get("WARD_DIALOGUE", "0") == "1"
RULES = os.environ.get("WARD_RULES", "0") == "1"
CONCURRENCY = int(os.environ.get("WARD_CONCURRENCY", "1"))
OUT = os.environ.get("WARD_OUT", "output/intermediate/agentic_ward_run.jsonl")
SCN = os.environ.get("WARD_SCN", "data/scenarios/ward_scenarios.jsonl")
PROMPT_VERSION = os.environ.get("WARD_PROMPT_VERSION", "v6.1")
TEMP_TAG = os.environ.get("WARD_TEMPERATURE")
SPEND_FILE = os.environ.get("WARD_SPEND_FILE")


def _cell_suffix():
    s = ["rules"] if RULES else []
    if PROMPT_VERSION != "v6.1":
        s.append("pv=" + PROMPT_VERSION)
    if TEMP_TAG:
        s.append("T=" + TEMP_TAG)
    return s


def episode_row(ws, arm, model_label, adapter_factory, mode=MODE, oversight_level="blind",
                info="full", wording=0, rep=0, dialogue=False):
    ws = dataclasses.replace(ws, info=info, seed=f"{ws.scenario.sid}|{rep}", rules_stated=RULES,
                             prompt_version=PROMPT_VERSION)
    doctor = adapter_factory(model_label)
    ov = make_oversight(oversight_level)
    if mode == "focal":
        r = run_focal_episode(ws, ws.focal_pid, doctor, arm, model_label, oversight=ov, wording=wording)
    else:
        r = run_ward_episode(ws, doctor, arm, model_label, oversight=ov, dialogue=dialogue,
                             wording=wording)
    row = {"sid": r.sid, "mode": mode, "arm": r.arm, "model": r.model, "info": info,
            "wording": wording, "oversight": oversight_level, "rep": rep, "dialogue": dialogue,
            "rules_stated": RULES, "harm": r.harm, "channels": r.channels, "rightful_care": r.rightful_care,
            "delay_harm": r.delay_harm, "displacement": r.displacement,
            "bed_holder": r.bed_holder, "parse_failures": r.parse_failures,
            "parse_fail_samples": r.meta.get("parse_fail_samples", []), "ops": r.ops,
            "focal_pid": r.meta["focal_pid"], "K": r.meta["K"], "n_steps": r.meta["n_steps"],
            "transcript": [{"pid": m.pid, "step": m.step, "verb": m.action.verb,
                            "target": m.action.target, "rationale": m.rationale} for m in r.moves],
            "cell_id": "|".join([r.sid, mode, arm, info, f"w{wording}", oversight_level,
                                 model_label, str(rep)] + _cell_suffix()),
            **r.violations}
    row["prompt_version"] = PROMPT_VERSION
    row["temperature"] = getattr(doctor, "temperature", None)
    calls = getattr(doctor, "call_log", None)
    if calls is not None:
        row["calls"] = calls
        row["cost_usd"] = round(sum((c.get("cost") or 0.0) for c in calls), 8)
        row["returned_models"] = sorted({c["returned_model"] for c in calls if c.get("returned_model")})
        row["providers"] = sorted({c["provider"] for c in calls if c.get("provider")})
    return row


def main(n_scn="100", reps="5"):
    n, n_reps = int(n_scn), int(reps)
    scenarios = load_ward_scenarios(SCN)[:n]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["cell_id"] for l in open(OUT) if l.strip()}
    halter = halter_from_path(SPEND_FILE, every=20)
    cells = []
    for ws in scenarios:
        for arm in ARMS:
            for info in INFO:
                for wording in (WORDINGS if arm in ("devotion", "rescue") else [0]):
                    for ov in OVERSIGHT:
                        for model in DOCTORS:
                            for rep in range(n_reps):
                                cid = "|".join([ws.scenario.sid, MODE, arm, info, f"w{wording}",
                                                ov, model, str(rep)] + _cell_suffix())
                                if cid not in done:
                                    cells.append((cid, ws, arm, info, wording, ov, model, rep))
    lock = threading.Lock()
    written = [0]
    fh = open(OUT, "a")

    def _run(cell):
        if halter is not None and not halter.ok():
            return
        cid, ws, arm, info, wording, ov, model, rep = cell
        try:
            row = episode_row(ws, arm, model, build_adapter, MODE, ov, info, wording, rep, DIALOGUE)
        except Exception as exc:
            print("  [skip %s] %s" % (cid, str(exc)[:70]), file=sys.stderr)
            if halter is not None:
                with lock:
                    halter.after_attempt()
            return
        with lock:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            written[0] += 1
            if halter is not None:
                halter.after_attempt()

    try:
        if CONCURRENCY > 1:
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
                list(ex.map(_run, cells))
        else:
            for cell in cells:
                _run(cell)
    finally:
        fh.close()
    print("agentic-ward (%s) wrote %d episodes over %d scenarios" % (MODE, written[0], len(scenarios)))
    exit_if_halted(halter)


if __name__ == "__main__":
    main(*sys.argv[1:])
