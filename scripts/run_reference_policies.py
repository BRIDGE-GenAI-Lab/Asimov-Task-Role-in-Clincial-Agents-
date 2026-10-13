#!/usr/bin/env python3
"""Scores three scripted reference policies (never claim, always claim, claim only when entitled) on the balanced scenarios by running them through the same episode engine as the models. It reads experiment2/runs/ward_scenarios_balanced_rv.jsonl and writes experiment2/runs/reference_policies.jsonl."""
import json
import re
import sys

sys.path.insert(0, "code")
sys.path.insert(0, "scripts")
from scarcity_ward.models.base import ModelAdapter
from scarcity_ward.hospital.ward_env import bed_rightful
from scarcity_ward.hospital.ward_scenarios import load_ward_scenarios
import run_agentic_ward as R

POLICIES = ("ref-never", "ref-always", "ref-rule")
SCN = "experiment2/runs/ward_scenarios_balanced_rv.jsonl"
OUT = "experiment2/runs/reference_policies.jsonl"


class PolicyAdapter(ModelAdapter):
    def __init__(self, name, verb_for_step):
        self.name, self.verb_for_step = name, verb_for_step

    def generate(self, prompt):
        step = int(re.search(r"Step (\d+) of", prompt).group(1))
        return json.dumps({"rationale": self.name, "action": {"verb": self.verb_for_step(step), "target": None}})


def _claim_then_hold(step):
    return "claim_bed" if step == 1 else "hold_bed"


def policy(name, ws):
    entitled = ws.focal_pid in bed_rightful(ws)
    if name == "ref-never":
        return PolicyAdapter(name, lambda s: "wait")
    if name == "ref-always":
        return PolicyAdapter(name, _claim_then_hold)
    if name == "ref-rule":
        return PolicyAdapter(name, _claim_then_hold if entitled else (lambda s: "wait"))
    raise KeyError(name)


def score(ws, name):
    saved = (R.RULES, R.PROMPT_VERSION, R.TEMP_TAG)
    R.RULES, R.PROMPT_VERSION, R.TEMP_TAG = True, "v7", None
    try:
        ad = policy(name, ws)
        return R.episode_row(ws, "neutral", name, lambda _l: ad, mode="focal")
    finally:
        R.RULES, R.PROMPT_VERSION, R.TEMP_TAG = saved


def main():
    with open(OUT, "w") as fh:
        for ws in load_ward_scenarios(SCN):
            entitled = ws.focal_pid in bed_rightful(ws)
            sid_type = ws.scenario.sid.rsplit("-", 1)[-1]
            expected_entitled = (sid_type == "ENT")
            if entitled != expected_entitled:
                raise AssertionError(
                    f"Entitlement mismatch in {ws.scenario.sid}: "
                    f"ws.focal_pid in bed_rightful(ws) = {entitled}, "
                    f"but scenario type is {sid_type} (expected_entitled = {expected_entitled})"
                )
            for name in POLICIES:
                fh.write(json.dumps(score(ws, name)) + "\n")
    print("wrote reference policies ->", OUT)


if __name__ == "__main__":
    main()
