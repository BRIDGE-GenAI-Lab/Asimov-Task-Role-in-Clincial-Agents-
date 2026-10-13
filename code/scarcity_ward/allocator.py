from __future__ import annotations
from typing import List
from .schemas import Patient, Scenario, AdvocateTurn
from .protocols import PROTOCOLS
from .prompts import allocator_prompt
from .parsing import parse_award, ParseError
from .models.base import ModelAdapter


def _pseudo_patients(scenario: Scenario, turns: List[AdvocateTurn]) -> List[Patient]:
    # Build patients from the LATEST asserted claims per pid (fabrication propagates here).
    latest = {}
    for t in turns:
        latest[t.pid] = t
    out = []
    for p in scenario.patients:
        t = latest.get(p.pid)
        if t is None:
            out.append(p)
        else:
            out.append(Patient(p.pid, p.age, p.diagnosis, t.asserted.sofa, p.comorbidities,
                               t.asserted.exclusion_flags, t.asserted.survival_prob, p.sociodemographic))
    return out


def rule_allocate(scenario: Scenario, turns: List[AdvocateTurn]) -> List[str]:
    """Guideline-faithful gatekeeper: applies the named crisis-standard protocol to the asserted
    STRUCTURED claims only. It never reads argument_text or rival_statements, so it is
    persuasion-invariant by construction (cannot be rhetorically swayed); the only way to move it
    is to fabricate the structured claims, which the fabrication flag detects."""
    pseudo = _pseudo_patients(scenario, turns)
    key = PROTOCOLS[scenario.protocol]
    return [p.pid for p in sorted(pseudo, key=key)][: scenario.units]


def _transcript(turns: List[AdvocateTurn]) -> str:
    return "\n".join(f"[round {t.round}] {t.pid}: {t.argument_text} "
                     f"(claims SOFA {t.asserted.sofa}, survival {t.asserted.survival_prob})"
                     for t in turns)


def llm_allocate(scenario: Scenario, turns: List[AdvocateTurn], adapter: ModelAdapter) -> List[str]:
    raw = adapter.generate(allocator_prompt(scenario, _transcript(turns)))
    valid = [p.pid for p in scenario.patients]
    try:
        awarded = parse_award(raw, valid)
    except ParseError:
        awarded = []
    if len(awarded) != scenario.units:           # fallback: rule allocation on asserted claims
        return rule_allocate(scenario, turns)
    return awarded[: scenario.units]
