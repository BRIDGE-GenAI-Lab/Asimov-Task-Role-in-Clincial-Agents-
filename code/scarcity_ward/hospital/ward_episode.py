"""v6.1 episode loops.

Exp 1 (headline, causal): run_focal_episode — ONE LLM agent among scripted honest peers; every harm
event is attributable to the focal agent's choices.
Exp 2 (ecology): run_ward_episode — all agents are LLM, same arm; optional dialogue (communication
ablation, GovSim precedent).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..models.base import ModelAdapter
from ..parsing import ParseError
from .ward_actions import WardAction, WardMove, parse_ward_move
from .ward_env import (WardScenario, WardState, apply_step, attending_priority,
                       bed_rightful, imaging_priority)
from .ward_harm import (collateral_harm, delay_weighted_harm, displacement,
                        harm_channels, operational_metrics, rightful_care,
                        zeroth_violations)
from .ward_prompts import public_state, ward_move_prompt


@dataclass
class WardResult:
    sid: str
    arm: str
    model: str
    harm: bool
    channels: Dict[str, bool]
    rightful_care: Dict[str, bool]
    delay_harm: float
    displacement: int
    violations: Dict[str, bool]
    bed_holder: Optional[str]
    parse_failures: int
    ops: Dict[str, int]
    moves: List[WardMove]
    meta: Dict[str, object] = field(default_factory=dict)


def honest_peer_policy(ws: WardScenario, pid: str, step: int, st: WardState) -> WardMove:
    """Scripted steward peer following standard workflow: time-critical imaging immediately; the
    bed-rightful patient is admitted at step 2 (after the step-1 assessment, when the protocol
    allocation happens); the most-acute escalates once. Deterministic and NEVER contests step 1 for
    the bed -> a focal grab at step 1 pre-empts the protocol (VIP queue-jump), and every harm event
    is attributable to the focal agent's choice, with no tie lottery in focal mode."""
    verb = "wait"
    pending = [p for p in imaging_priority(ws) if p not in st.imaging_done and p not in st.imaging_queue]
    if pending and pid == pending[0]:
        verb = "order_imaging"
    elif pid in bed_rightful(ws) and step >= 2 and st.bed_holder in (None, pid):
        verb = "claim_bed" if st.bed_holder is None else "hold_bed"
    elif pid == attending_priority(ws)[0] and pid not in {p for _, p in st.attending_log}:
        verb = "escalate"
    return WardMove(pid, step, "honest peer policy", WardAction(verb))


def score_episode(ws: WardScenario, st: WardState, moves: List[WardMove], arm: str, model: str,
                  parse_failures: int = 0) -> WardResult:
    return WardResult(
        sid=ws.scenario.sid, arm=arm, model=model,
        harm=collateral_harm(ws, st), channels=harm_channels(ws, st),
        rightful_care=rightful_care(ws, st), delay_harm=delay_weighted_harm(ws, st),
        displacement=displacement(ws, st), violations=zeroth_violations(ws, st, moves),
        bed_holder=st.bed_holder, parse_failures=parse_failures,
        ops=operational_metrics(ws, st), moves=moves,
        meta={"K": len(ws.scenario.patients), "n_steps": ws.n_steps, "info": ws.info,
              "focal_pid": ws.focal_pid, "img_slots_per_step": ws.img_slots_per_step,
              "attn_per_step": ws.attn_per_step})


def _llm_move(adapter: ModelAdapter, ws: WardScenario, pid: str, arm: str, step: int,
              pub: str, wording: int, prior_args: str = "") -> tuple:
    raw = adapter.generate(ward_move_prompt(ws, pid, arm, step, pub, wording, prior_args))
    try:
        return parse_ward_move(raw, pid, step), None
    except ParseError:
        return WardMove(pid, step, "", WardAction("wait")), raw[:200]  # keep a sample for diagnosis


def run_focal_episode(ws: WardScenario, focal_pid: str, adapter: ModelAdapter, arm: str,
                      model_label: str, oversight=None, wording: int = 0) -> WardResult:
    st = WardState()
    all_moves: List[WardMove] = []
    fail_samples: List[str] = []
    for step in range(1, ws.n_steps + 1):
        pub = public_state(ws, st)
        moves: List[WardMove] = []
        for p in ws.scenario.patients:
            if p.pid == focal_pid:
                m, fail = _llm_move(adapter, ws, p.pid, arm, step, pub, wording)
                if fail is not None:
                    fail_samples.append(fail)
            else:
                m = honest_peer_policy(ws, p.pid, step, st)
            moves.append(m)
        if oversight is not None:
            moves = oversight.filter(ws, st, moves)
        apply_step(ws, st, moves)
        all_moves.extend(moves)
    res = score_episode(ws, st, all_moves, arm, model_label, len(fail_samples))
    res.meta["parse_fail_samples"] = fail_samples[:3]
    return res


def run_ward_episode(ws: WardScenario, doctor_adapter: ModelAdapter, arm: str, model_label: str,
                     oversight=None, dialogue: bool = False, wording: int = 0) -> WardResult:
    st = WardState()
    all_moves: List[WardMove] = []
    fail_samples: List[str] = []
    prev: Dict[str, str] = {}
    for step in range(1, ws.n_steps + 1):
        pub = public_state(ws, st)
        moves: List[WardMove] = []
        for p in ws.scenario.patients:
            prior = ""
            if dialogue and prev:
                prior = "\n".join(f"{q}: {r}" for q, r in sorted(prev.items()) if q != p.pid and r)
            m, fail = _llm_move(doctor_adapter, ws, p.pid, arm, step, pub, wording, prior)
            if fail is not None:
                fail_samples.append(fail)
            moves.append(m)
        if oversight is not None:
            moves = oversight.filter(ws, st, moves)
        apply_step(ws, st, moves)
        all_moves.extend(moves)
        prev = {m.pid: m.rationale for m in moves}
    res = score_episode(ws, st, all_moves, arm, model_label, len(fail_samples))
    res.meta["parse_fail_samples"] = fail_samples[:3]
    return res
