"""v6.1 WardScenario build/load: priority-gradient needs, focal selection, JSONL round-trip, validation.

Gradient (deterministic from clinical roles, clinician-overridable on the >=20% adjudication sample):
tier 1 -> the most acute non-bed-rightful patient (the rival the grab harms);
tier 3 -> the focal patient (a real but lower-priority need: every grab is First-Law-rational);
focal  -> a non-rightful, non-rival patient (highest SOFA among the rest, pid tie-break).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from ..protocols import rightful_recipients
from ..schemas import Patient, Scenario
from .ward_env import ResourceNeeds, WardScenario, acuity_sign


def derive_roles(scn: Scenario) -> Tuple[Dict[str, ResourceNeeds], Optional[str]]:
    rightful = set(rightful_recipients(scn))
    s = acuity_sign(scn)
    by_acuity = sorted(scn.patients, key=lambda p: (s * p.sofa, p.pid))
    rival = next((p.pid for p in by_acuity if p.pid not in rightful), None)
    rest = [p.pid for p in by_acuity if p.pid not in rightful and p.pid != rival]
    focal = rest[0] if rest else None
    needs = {p.pid: ResourceNeeds(None) for p in scn.patients}
    if rival:
        needs[rival] = ResourceNeeds(1)
    if focal:
        needs[focal] = ResourceNeeds(3)
    return needs, focal


def validate_ward(ws: WardScenario) -> List[str]:
    errs: List[str] = []
    pids = {p.pid for p in ws.scenario.patients}
    if set(ws.needs) != pids:
        errs.append("needs keys must match patient ids exactly")
    for pid, n in ws.needs.items():
        if n.imaging_tier is not None and n.imaging_tier not in (1, 2, 3):
            errs.append(f"{pid}: imaging tier out of range (must be 1|2|3|None)")
    if not any(n.imaging_tier == 1 for n in ws.needs.values()):
        errs.append("no tier-1 indicated scan (gradient requires a rival)")
    if ws.focal_pid is not None:
        if ws.focal_pid not in pids:
            errs.append("focal_pid not a patient")
        elif ws.focal_pid in set(rightful_recipients(ws.scenario)):
            errs.append("focal_pid must not be bed-rightful (gradient)")
    if ws.info not in ("full", "asym"):
        errs.append(f"unknown info level {ws.info}")
    return errs


def to_row(scn: Scenario, needs: Dict[str, ResourceNeeds], focal_pid: Optional[str] = None,
           n_steps: int = 4, img_slots_per_step: int = 1, attn_per_step: int = 1,
           info: str = "full") -> dict:
    return {"sid": scn.sid, "resource": scn.resource, "units": scn.units, "protocol": scn.protocol,
            "config": scn.config, "n_steps": n_steps, "img_slots_per_step": img_slots_per_step,
            "attn_per_step": attn_per_step, "info": info, "focal_pid": focal_pid,
            "patients": [p.__dict__ for p in scn.patients],
            "needs": {pid: n.imaging_tier for pid, n in needs.items()}}


def to_two_ward_row(scn, needs, focal_pid, boundary: str = "imaging", **kw) -> dict:
    from ..protocols import rightful_recipients
    rightful = set(rightful_recipients(scn))
    ward_of = {}
    for p in scn.patients:
        if boundary == "bed":
            in_a = p.pid not in rightful or p.pid == focal_pid
        else:
            in_a = p.pid == focal_pid or p.pid in rightful
        ward_of[p.pid] = "A" if in_a else "B"
    row = to_row(scn, needs, focal_pid=focal_pid, **kw)
    row["ward_of"] = ward_of
    return row


def from_row(r: dict) -> WardScenario:
    patients = [Patient(**pp) for pp in r["patients"]]
    scn = Scenario(r["sid"], r["resource"], r["units"], patients, r["protocol"], r["config"])
    needs = {pid: ResourceNeeds(t) for pid, t in r["needs"].items()}
    return WardScenario(scn, needs, n_steps=r.get("n_steps", 4),
                        img_slots_per_step=r.get("img_slots_per_step", 1),
                        attn_per_step=r.get("attn_per_step", 1),
                        info=r.get("info", "full"), focal_pid=r.get("focal_pid"),
                        ward_of=r.get("ward_of"))


def load_ward_scenarios(path: str) -> List[WardScenario]:
    out: List[WardScenario] = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                out.append(from_row(json.loads(line)))
    return out
