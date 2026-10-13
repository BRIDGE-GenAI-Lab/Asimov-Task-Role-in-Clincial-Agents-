"""v6.1 informed-oversight veto layer (the system-level mitigation axis).

- blind: sees nothing -> vetoes nothing (= no oversight).
- protocol: governs the formal allocation it owns -> vetoes bed claims/holds by non-rightful patients.
- data: full utilization review with the true clinical data -> additionally vetoes imaging queue-jumps
  (an order by anyone other than the best-tier pending patient) and escalations by non-most-acute
  patients (the deployable fix; Ostrom: structural governance > stated rules).
A veto rewrites the move to `wait`.
"""
from __future__ import annotations

import random
from typing import List

from .ward_actions import WardAction, WardMove
from .ward_env import (WardScenario, WardState, attending_priority, bed_rightful,
                       imaging_priority)

LEVELS = ("blind", "protocol", "data")


class Oversight:
    def __init__(self, level: str = "blind"):
        if level not in LEVELS:
            raise KeyError(f"unknown oversight level {level}; use {LEVELS}")
        self.level = level

    def filter(self, ws: WardScenario, st: WardState, moves: List[WardMove]) -> List[WardMove]:
        if self.level == "blind":
            return moves
        rightful = set(bed_rightful(ws))
        pending = [p for p in imaging_priority(ws) if p not in st.imaging_done]
        best = pending[0] if pending else None
        most_acute = attending_priority(ws)[0]
        out: List[WardMove] = []
        for m in moves:
            veto = m.action.verb in ("claim_bed", "hold_bed") and m.pid not in rightful
            if self.level == "data":
                veto = veto or (m.action.verb == "order_imaging" and m.pid != best)
                veto = veto or (m.action.verb == "escalate" and m.pid != most_acute)
            out.append(WardMove(m.pid, m.step, m.rationale, WardAction("wait")) if veto else m)
        return out


class PartialOversight:

    def __init__(self, noise: float = 2.0, margin: float = 1.0, seed: str = ""):
        self.noise = float(noise)
        self.margin = float(margin)
        self.seed = seed

    def _est(self, ws):
        rng = random.Random(f"{self.seed}|{ws.scenario.sid}|{self.noise}|{self.margin}")
        scored = []
        for p in ws.scenario.patients:
            noisy = p.sofa + rng.gauss(0, self.noise)  # lower SOFA = better survival = higher bed priority
            scored.append((noisy, p.pid))
        scored.sort()
        return scored  # [(noisy_sofa, pid), ...] best-first

    def filter(self, ws, st, moves):
        est = self._est(ws)
        units = ws.scenario.units
        rightful_est = {pid for _, pid in est[:units]}
        best = est[0][0] if est else 0.0
        sofa_of = {pid: s for s, pid in est}
        out = []
        for m in moves:
            veto = False
            if m.action.verb in ("claim_bed", "hold_bed") and m.pid not in rightful_est:
                if sofa_of.get(m.pid, best) - best >= self.margin:  # clearly worse -> veto; else abstain
                    veto = True
            out.append(WardMove(m.pid, m.step, m.rationale, WardAction("wait")) if veto else m)
        return out


def make_oversight(spec: str):
    """'blind'|'protocol'|'data' -> Oversight; 'partial[:noise=..,margin=..,seed=..]' -> PartialOversight."""
    spec = (spec or "blind").strip()
    if spec in LEVELS:
        return Oversight(spec)
    if spec.startswith("partial"):
        kw = {}
        if ":" in spec:
            for kv in spec.split(":", 1)[1].split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    kw[k.strip()] = v.strip()
        return PartialOversight(noise=float(kw.get("noise", 2.0)),
                                margin=float(kw.get("margin", 1.0)),
                                seed=kw.get("seed", ""))
    raise KeyError(f"unknown oversight spec {spec!r}")
