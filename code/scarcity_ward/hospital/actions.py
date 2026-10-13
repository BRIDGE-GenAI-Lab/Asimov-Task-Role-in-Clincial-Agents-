from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from ..parsing import extract_json, _recover_sofa

_VERBS = ("request", "concede", "bump")


@dataclass
class Action:
    verb: str                       # request | concede | bump
    target: Optional[str] = None    # pid, for bump/concede


@dataclass
class DoctorMove:
    pid: str
    argument: str
    asserted_sofa: Optional[int]    # None if unrecoverable -> episode fills with true SOFA
    action: Action
    round: int = 1


def parse_move(raw: str, pid: str, round_no: int = 1) -> DoctorMove:
    d = extract_json(raw)
    # promote top-level asserted_sofa → sofa so _recover_sofa can find it
    if "asserted_sofa" in d and "sofa" not in d:
        d = {**d, "sofa": d["asserted_sofa"]}
    sofa = _recover_sofa(d, raw)
    a = d.get("action") or {}
    verb = str(a.get("verb", "request")).lower().strip()
    if verb not in _VERBS:
        verb = "request"
    target = a.get("target")
    target = str(target) if target not in (None, "", "null") else None
    return DoctorMove(pid=pid, argument=str(d.get("argument", "")),
                      asserted_sofa=sofa, action=Action(verb=verb, target=target),
                      round=round_no)
