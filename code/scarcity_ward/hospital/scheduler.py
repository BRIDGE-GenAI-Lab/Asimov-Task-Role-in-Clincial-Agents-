from __future__ import annotations
from typing import List
from ..schemas import Patient, Scenario
from ..protocols import PROTOCOLS
from .prompts import manager_prompt
from ..parsing import parse_award, ParseError
from ..models.base import ModelAdapter
from .actions import DoctorMove


def _asserted_sofa(move: DoctorMove, true_sofa: int) -> int:
    return true_sofa if move.asserted_sofa is None else move.asserted_sofa


def rule_schedule(scenario: Scenario, moves: List[DoctorMove]) -> List[str]:
    """Urgency-optimal gatekeeper: applies the crisis-standard protocol to the ASSERTED structured
    claims only (ignores rhetoric/actions -> persuasion-invariant). Only fabricating the numbers moves it."""
    by_move = {m.pid: m for m in moves}
    pseudo = []
    for p in scenario.patients:
        m = by_move.get(p.pid)
        sofa = _asserted_sofa(m, p.sofa) if m else p.sofa
        pseudo.append(Patient(p.pid, p.age, p.diagnosis, sofa, p.comorbidities,
                              p.exclusion_flags, p.survival_prob, p.sociodemographic))
    key = PROTOCOLS[scenario.protocol]
    return [q.pid for q in sorted(pseudo, key=key)][: scenario.units]


def move_transcript(moves: List[DoctorMove]) -> str:
    out = []
    for m in moves:
        tgt = ("->" + m.action.target) if m.action.target else ""
        out.append(f"[r{m.round}] {m.pid} [{m.action.verb}{tgt}] asserts SOFA {m.asserted_sofa}: {m.argument}")
    return "\n".join(out)


def llm_schedule(scenario: Scenario, moves: List[DoctorMove], adapter: ModelAdapter,
                 level: str = "naive") -> List[str]:
    """Persuadable bed-manager at a given knowledge level (naive/guideline/data). Falls back to the rule
    schedule on a parse failure or a non-conforming award (mirrors allocator.py)."""
    try:
        awarded = parse_award(adapter.generate(manager_prompt(scenario, move_transcript(moves), level)),
                              [p.pid for p in scenario.patients])
    except ParseError:
        awarded = []
    return awarded if len(awarded) == scenario.units else rule_schedule(scenario, moves)
