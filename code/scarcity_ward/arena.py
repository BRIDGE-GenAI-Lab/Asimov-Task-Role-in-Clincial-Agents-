from __future__ import annotations
from typing import Callable, Dict, List
from .schemas import Scenario, AdvocateTurn, ArenaResult, AssertedClaims
from .models.base import ModelAdapter
from .prompts import advocate_prompt
from .parsing import parse_advocate, ParseError


def run_arena(scenario: Scenario, adapters: Dict[str, ModelAdapter], pressure: str,
              allocator_fn: Callable[[Scenario, List[AdvocateTurn]], List[str]],
              n_rounds: int = 2, guardrail: bool = False,
              model_label: str = "mixed", allocator_label: str = "rule") -> ArenaResult:
    turns: List[AdvocateTurn] = []
    parse_failures = 0
    for r in range(1, n_rounds + 1):
        prior = list(turns)
        for p in scenario.patients:
            prompt = advocate_prompt(scenario, p.pid, pressure=pressure, round_no=r,
                                     prior_turns=prior, guardrail=guardrail)
            raw = adapters[p.pid].generate(prompt)
            try:
                turn = parse_advocate(raw, pid=p.pid, round_no=r)
            except ParseError:
                parse_failures += 1
                # parse failure => treat as a truthful no-op turn (recorded in meta)
                turn = AdvocateTurn(p.pid, r, "", AssertedClaims(p.sofa, list(p.exclusion_flags),
                                                                 p.survival_prob))
            turns.append(turn)
    awarded = allocator_fn(scenario, turns)
    return ArenaResult(sid=scenario.sid, model=model_label, pressure=pressure,
                       allocator=allocator_label, guardrail=guardrail, framing="deployed",
                       awarded_pids=awarded, turns=turns,
                       meta={"n_rounds": n_rounds, "units": scenario.units,
                             "K": len(scenario.patients), "parse_failures": parse_failures})
