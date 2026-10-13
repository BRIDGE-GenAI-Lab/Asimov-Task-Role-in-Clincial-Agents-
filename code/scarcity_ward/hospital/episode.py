from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List
from ..schemas import Scenario
from ..parsing import ParseError
from ..models.base import ModelAdapter
from .actions import DoctorMove, Action, parse_move
from .prompts import doctor_move_prompt
from .harm import collateral_harm, displacement, zeroth_violations


@dataclass
class EpisodeResult:
    sid: str
    arm: str
    model: str
    scheduler: str
    awarded: List[str]
    harm: bool
    displacement: int
    violations: Dict[str, bool]
    moves: List[DoctorMove]
    meta: Dict[str, object] = field(default_factory=dict)


def run_episode(scenario: Scenario, doctor_adapter: ModelAdapter, arm: str,
                schedule_fn: Callable[[Scenario, List[DoctorMove]], List[str]],
                scheduler_label: str, model_label: str, n_rounds: int = 2) -> EpisodeResult:
    true_sofa = {p.pid: p.sofa for p in scenario.patients}
    all_moves: List[DoctorMove] = []
    prev_args = None
    for rnd in range(1, n_rounds + 1):
        round_moves: List[DoctorMove] = []
        for p in scenario.patients:
            raw = doctor_adapter.generate(
                doctor_move_prompt(scenario, p.pid, arm, round_no=rnd, prior_args=prev_args))
            try:
                m = parse_move(raw, p.pid, round_no=rnd)
            except ParseError:
                m = DoctorMove(p.pid, "", None, Action("request"), round=rnd)
            if m.asserted_sofa is None:
                m.asserted_sofa = true_sofa[p.pid]
            round_moves.append(m)
        all_moves.extend(round_moves)
        prev_args = [(m.pid, m.argument) for m in round_moves]
    awarded = schedule_fn(scenario, all_moves)
    return EpisodeResult(
        sid=scenario.sid, arm=arm, model=model_label, scheduler=scheduler_label,
        awarded=awarded, harm=collateral_harm(scenario, awarded),
        displacement=displacement(scenario, awarded),
        violations=zeroth_violations(scenario, all_moves, awarded), moves=all_moves,
        meta={"K": len(scenario.patients), "beds": scenario.units, "n_rounds": n_rounds})
