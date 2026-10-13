from __future__ import annotations
from typing import Callable, Dict
from .schemas import Scenario
from .protocols import rightful_recipients
from .models.base import ModelAdapter

CONDITIONS = ["symmetric", "asym_test", "asym_reverse"]


def assign_advocates(scenario: Scenario, condition: str,
                     adapter_factory: Callable[[str], ModelAdapter],
                     strong_model: str, weak_model: str) -> Dict[str, ModelAdapter]:
    """Map pid -> advocate adapter for the rent-a-strong-doctor design.
    symmetric    = every patient gets the weak model (level field; pressure carries the dose).
    asym_test    = the protocol-rightful patient gets the WEAK advocate, rivals get the STRONG one.
    asym_reverse = the rightful patient gets the STRONG advocate, rivals get the WEAK one (control)."""
    if condition not in CONDITIONS:
        raise KeyError(f"unknown condition {condition}; known: {CONDITIONS}")
    rightful = set(rightful_recipients(scenario))
    strong = adapter_factory(strong_model)
    weak = adapter_factory(weak_model)
    out: Dict[str, ModelAdapter] = {}
    for p in scenario.patients:
        is_rightful = p.pid in rightful
        if condition == "symmetric":
            out[p.pid] = weak
        elif condition == "asym_test":
            out[p.pid] = weak if is_rightful else strong
        else:  # asym_reverse
            out[p.pid] = strong if is_rightful else weak
    return out
