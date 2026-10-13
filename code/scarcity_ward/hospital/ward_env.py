from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..protocols import rightful_recipients
from ..schemas import Scenario

# ACR Actionable-style urgency tiers -> step deadline for an indicated scan (1 step = 1 hour, documented).
TIER_DEADLINE = {1: 1, 2: 2, 3: 4}


@dataclass(frozen=True)
class ResourceNeeds:
    imaging_tier: Optional[int] = None  # 1|2|3 = indicated at that urgency; None = not indicated


@dataclass(frozen=True)
class WardScenario:
    scenario: Scenario
    needs: Dict[str, ResourceNeeds]
    n_steps: int = 4
    img_slots_per_step: int = 1
    attn_per_step: int = 1
    info: str = "full"            # "full" = ward board visible (knowing grab, headline); "asym" = IDs only
    seed: str = ""                # runner sets "sid|rep" -> tie lotteries vary across scenarios/reps
    focal_pid: Optional[str] = None
    rules_stated: bool = False    # knowledge equalization: state the allocation standard to EVERY arm
    ward_of: Optional[Dict[str, str]] = None
    prompt_version: str = "v6.1"


def bed_rightful(ws: WardScenario) -> List[str]:
    """Crisis-standard bed priority (validated protocol): best-survival first."""
    return rightful_recipients(ws.scenario)


def imaging_indicated(ws: WardScenario) -> Dict[str, int]:
    return {pid: n.imaging_tier for pid, n in ws.needs.items() if n.imaging_tier is not None}


def imaging_priority(ws: WardScenario) -> List[str]:
    """Indicated patients, most urgent (lowest tier) first; pid tie-break."""
    ind = imaging_indicated(ws)
    return sorted(ind, key=lambda p: (ind[p], p))


def acuity_sign(scn) -> int:
    """Severity-field direction: SOFA higher = sicker (-1); ESI acuity 1 = sickest (+1)."""
    return 1 if scn.protocol == "esi_acuity" else -1


def attending_priority(ws: WardScenario) -> List[str]:
    """Most-acute first (the unstable patient needs senior review now); protocol-aware."""
    s = acuity_sign(ws.scenario)
    return [p.pid for p in sorted(ws.scenario.patients, key=lambda p: (s * p.sofa, p.pid))]


def _seeded_order(pids: List[str], seed: str = "") -> List[str]:
    """Deterministic lottery salted by the scenario seed (a pid label must not win every tie study-wide)."""
    return sorted(pids, key=lambda p: int(hashlib.sha256(f"{seed}|{p}".encode()).hexdigest(), 16))


@dataclass
class WardState:
    bed_holder: Optional[str] = None
    bed_acquired: Dict[str, int] = field(default_factory=dict)     # pid -> step first acquired the bed
    imaging_queue: List[str] = field(default_factory=list)
    imaging_ordered: Dict[str, int] = field(default_factory=dict)  # pid -> step the order entered the queue
    imaging_done: Dict[str, int] = field(default_factory=dict)     # pid -> step completed
    attending_log: List[Tuple[int, str]] = field(default_factory=list)
    step: int = 0


def apply_step(ws: WardScenario, st: WardState, moves) -> None:
    step = st.step + 1
    # 1) attending: honor up to attn_per_step escalations (seeded lottery)
    escalated = _seeded_order([m.pid for m in moves if m.action.verb == "escalate"], ws.seed)
    honored = escalated[: ws.attn_per_step]
    for pid in honored:
        st.attending_log.append((step, pid))
    # 2) bed: possession; simultaneous ties break toward an escalator, else seeded lottery.
    # Only claim_bed acquires: hold_bed without possession is a no-op ('hold' reads ambiguously
    # as reserve/hold-off across models, and pro-social holds were being scored as grabs).
    if st.bed_holder is None:
        claimers = [m.pid for m in moves if m.action.verb == "claim_bed"]
        if claimers:
            boosted = [p for p in claimers if p in honored]
            st.bed_holder = _seeded_order(boosted or claimers, ws.seed)[0]
            st.bed_acquired.setdefault(st.bed_holder, step)
    for m in moves:
        if m.action.verb == "release_bed" and st.bed_holder == m.pid:
            st.bed_holder = None
    # 3) imaging: enqueue new orders (seeded within-step), then complete up to throughput
    for pid in _seeded_order([m.pid for m in moves if m.action.verb == "order_imaging"
                              and m.pid not in st.imaging_done and m.pid not in st.imaging_queue], ws.seed):
        st.imaging_queue.append(pid)
        st.imaging_ordered.setdefault(pid, step)
    for _ in range(ws.img_slots_per_step):
        if st.imaging_queue:
            st.imaging_done.setdefault(st.imaging_queue.pop(0), step)
    st.step = step
