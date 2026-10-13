"""v6.1 ward action menu: structured moves parsed leniently (unknown verb -> wait; non-JSON raises)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..parsing import extract_json

VERBS = ("claim_bed", "hold_bed", "release_bed", "order_imaging",
         "escalate", "withhold", "disclose", "concede", "wait")


@dataclass
class WardAction:
    verb: str
    target: Optional[str] = None  # pid for concede; fact id for withhold/disclose


@dataclass
class WardMove:
    pid: str
    step: int
    rationale: str
    action: WardAction


def parse_ward_move(raw: str, pid: str, step: int) -> WardMove:
    d = extract_json(raw)  # raises ParseError on non-JSON (caught + counted by the episode loop)
    if not isinstance(d, dict):
        from ..parsing import ParseError
        raise ParseError(f"top-level JSON is {type(d).__name__}, not an object")
    a = d.get("action") or {}
    if not isinstance(a, dict):
        a = {}
    verb = str(a.get("verb", "wait")).lower().strip()
    if verb not in VERBS:
        verb = "wait"
    target = a.get("target")
    target = str(target) if target not in (None, "", "null") else None
    return WardMove(pid=pid, step=step, rationale=str(d.get("rationale", "")),
                    action=WardAction(verb=verb, target=target))
