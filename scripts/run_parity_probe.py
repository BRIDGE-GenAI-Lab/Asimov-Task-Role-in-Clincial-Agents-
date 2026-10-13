#!/usr/bin/env python3
"""Asks each model in PROBE_MODELS which patient the policy favours for every scenario in WARD_SCN under each role in ARMS, parses the named patient and marks it correct or not. It appends one JSON row per model, role and scenario to PROBE_OUT and resumes from rows already written."""
import dataclasses
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from scarcity_ward.envload import load_env
load_env()
from scarcity_ward.hospital.ward_env import bed_rightful
from scarcity_ward.hospital.ward_prompts import parity_probe_prompt
from scarcity_ward.hospital.ward_scenarios import load_ward_scenarios
from scarcity_ward.models.registry import build_adapter
from scarcity_ward.parsing import parse_priority_pid_strict
from scarcity_ward.spend import exit_if_halted, halter_from_env


def prepare(ws):
    return dataclasses.replace(ws, rules_stated=True, prompt_version="v7", seed=ws.scenario.sid + "|0")


RAW_KEEP = 500


def probe_row(ws, model, arm, adapter):
    ws = prepare(ws)
    pids = [p.pid for p in ws.scenario.patients]
    reply = adapter.generate(parity_probe_prompt(ws, ws.focal_pid, arm))
    raw = reply if isinstance(reply, str) else ("" if reply is None else str(reply))
    stated, via = parse_priority_pid_strict(raw, pids)
    rightful = bed_rightful(ws)[0]
    log = list(getattr(adapter, "call_log", None) or [])
    return {"sid": ws.scenario.sid, "model": model, "arm": arm, "focal": ws.focal_pid,
            "condition": ws.scenario.sid.rsplit("-", 1)[-1], "stated_pid": stated, "rightful": rightful,
            "correct": (stated == rightful) if stated is not None else None,
            "parse_ok": stated is not None, "parse_via": via, "raw": raw[:RAW_KEEP],
            "prompt_version": "v7",
            "calls": log, "cost_usd": round(sum((c.get("cost") or 0.0) for c in log), 8)}


def main():
    models = os.environ["PROBE_MODELS"].split(",")
    scn = load_ward_scenarios(os.environ["WARD_SCN"])
    arms = os.environ.get("ARMS", "neutral,devotion").split(",")
    out = os.environ["PROBE_OUT"]
    conc = int(os.environ.get("PROBE_CONCURRENCY", "8"))
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["sid"], r["model"], r["arm"]))
    tasks = [(ws, m, a) for m in models for a in arms for ws in scn if (ws.scenario.sid, m, a) not in done]
    halter = halter_from_env(every=50)
    lock = threading.Lock()
    fh = open(out, "a")

    def _run(t):
        if halter is not None and not halter.ok():
            return
        ws, m, a = t
        try:
            row = probe_row(ws, m, a, build_adapter(m))
        except Exception as exc:
            print("  [skip %s %s %s] %s" % (ws.scenario.sid, m, a, str(exc)[:80]), file=sys.stderr)
            if halter is not None:
                with lock:
                    halter.after_attempt()
            return
        with lock:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if halter is not None:
                halter.after_attempt()

    try:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            list(ex.map(_run, tasks))
    finally:
        fh.close()
    print("parity probe: attempted %d tasks -> %s" % (len(tasks), out))
    exit_if_halted(halter)


if __name__ == "__main__":
    main()
