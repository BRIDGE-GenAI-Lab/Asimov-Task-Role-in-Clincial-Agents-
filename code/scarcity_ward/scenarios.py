from __future__ import annotations
import json
from typing import List
from .schemas import Patient, Scenario
from .protocols import protocol_rank, PROTOCOLS


def validate_scenario(s: Scenario) -> List[str]:
    errs: List[str] = []
    if len(s.patients) < 2:
        errs.append("scenario must have at least 2 patients")
    if not (1 <= s.units < len(s.patients)):
        errs.append("units must satisfy 1 <= units < n_patients")
    if s.protocol not in PROTOCOLS:
        errs.append(f"unknown protocol {s.protocol}")
    pids = [p.pid for p in s.patients]
    if len(set(pids)) != len(pids):
        errs.append("duplicate patient ids")
    for p in s.patients:
        if not (0 <= p.sofa <= 24):
            errs.append(f"{p.pid}: sofa out of range")
        if not (0.0 <= p.survival_prob <= 1.0):
            errs.append(f"{p.pid}: survival_prob out of range")
    # config must match: 'clear' => unique top-1; 'near_tie' => top two within 2 SOFA
    if s.protocol in PROTOCOLS and len(s.patients) >= 2:
        order = protocol_rank(s, s.protocol)
        by = {p.pid: p for p in s.patients}
        gap = abs(by[order[0]].sofa - by[order[1]].sofa)
        if s.config == "clear" and gap < 2:
            errs.append("config 'clear' but top two are within 2 SOFA")
        if s.config == "near_tie" and gap > 2:
            errs.append("config 'near_tie' but top two differ by >2 SOFA")
    return errs


def load_scenarios(path: str) -> List[Scenario]:
    out: List[Scenario] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            patients = [Patient(**pp) for pp in r["patients"]]
            out.append(Scenario(sid=r["sid"], resource=r["resource"], units=r["units"],
                                 patients=patients, protocol=r["protocol"], config=r["config"]))
    return out
