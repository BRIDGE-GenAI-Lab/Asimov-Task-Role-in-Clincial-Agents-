from __future__ import annotations

import dataclasses
from typing import Dict, List

from scarcity_ward.hospital.ward_actions import WardAction, WardMove
from scarcity_ward.hospital.ward_env import WardScenario, WardState, apply_step
from scarcity_ward.hospital.ward_harm import operational_metrics


def replay_state(ws: WardScenario, transcript: List[dict]) -> WardState:
    by_step: Dict[int, List[dict]] = {}
    for t in transcript:
        by_step.setdefault(t["step"], []).append(t)
    st = WardState()
    for step in sorted(by_step):
        moves = [WardMove(t["pid"], step, "", WardAction(t["verb"], t.get("target")))
                 for t in by_step[step]]
        apply_step(ws, st, moves)
    return st


def recompute_ops(rows: List[dict], ws_by_sid: Dict[str, WardScenario]) -> List[dict]:
    """Return rows with an 'ops' key (existing 'ops' kept; missing ones replayed)."""
    out = []
    for r in rows:
        if "ops" not in r and r["sid"] in ws_by_sid:
            ws = dataclasses.replace(ws_by_sid[r["sid"]], info=r.get("info", "full"),
                                     seed=f"{r['sid']}|{r.get('rep', 0)}")
            r = {**r, "ops": operational_metrics(ws, replay_state(ws, r["transcript"]))}
        out.append(r)
    return out
