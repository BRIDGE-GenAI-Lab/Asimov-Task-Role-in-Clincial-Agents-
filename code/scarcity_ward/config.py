from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class RunConfig:
    models: List[str]
    pressures: List[str]
    k_levels: List[int]
    reps: int
    allocators: List[str] = field(default_factory=lambda: ["rule"])
    guardrail_levels: List[bool] = field(default_factory=lambda: [False])
    seed: int = 1234
    max_arenas: int = 0
