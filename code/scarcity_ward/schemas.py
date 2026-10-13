from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Patient:
    pid: str
    age: int
    diagnosis: str
    sofa: int                       # true SOFA, 0-24 (lower = better prognosis)
    comorbidities: List[str]
    exclusion_flags: List[str]      # protocol exclusion criteria the patient meets
    survival_prob: float            # 0-1 prognosis proxy
    sociodemographic: Optional[str] = None  # identity label, or None = no-identifier control


@dataclass
class AssertedClaims:
    sofa: int
    exclusion_flags: List[str]
    survival_prob: float

    @classmethod
    def from_patient(cls, p: "Patient") -> "AssertedClaims":
        return cls(sofa=p.sofa, exclusion_flags=list(p.exclusion_flags),
                   survival_prob=p.survival_prob)


@dataclass(frozen=True)
class Scenario:
    sid: str
    resource: str
    units: int                      # M scarce units
    patients: List[Patient]         # length K
    protocol: str                   # ground-truth protocol name
    config: str                     # 'clear' | 'near_tie'


@dataclass
class AdvocateTurn:
    pid: str
    round: int
    argument_text: str
    asserted: AssertedClaims
    rival_statements: List[str] = field(default_factory=list)


@dataclass
class ArenaResult:
    sid: str
    model: str
    pressure: str
    allocator: str
    guardrail: bool
    framing: str
    awarded_pids: List[str]
    turns: List[AdvocateTurn]
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class Score:
    fabrication: bool = False
    deception: bool = False
    sabotage: bool = False
    withholding: bool = False
    honest_concession: bool = False
    norm_invocation: bool = False
    normative_reframing: bool = False   # argues protocol-discordant allocation via a competing ethical frame
    details: Dict[str, object] = field(default_factory=dict)
