from __future__ import annotations
import hashlib
from typing import Callable, Dict, List, Tuple
from .schemas import Scenario


def _tiebreak(pid: str) -> int:
    # deterministic pseudo-lottery
    return int(hashlib.sha256(pid.encode()).hexdigest(), 16)


def _key_sofa_survival(p) -> Tuple:
    # not-excluded first; then lower SOFA; then higher survival; then lottery
    return (1 if p.exclusion_flags else 0, p.sofa, -p.survival_prob, _tiebreak(p.pid))


def _key_lives_saved(p) -> Tuple:
    return (1 if p.exclusion_flags else 0, -p.survival_prob, p.sofa, _tiebreak(p.pid))


def _key_esi_acuity(p) -> Tuple:
    # ED triage: most acute first (ESI 1 = sickest, stored in the severity field); then lottery
    return (1 if p.exclusion_flags else 0, p.sofa, _tiebreak(p.pid))


def _key_sickest_first(p) -> Tuple:
    # Highest SOFA first (sickest gets the resource); exclusion criteria not applied
    return (-p.sofa, -p.survival_prob, _tiebreak(p.pid))


def _key_first_come(p) -> Tuple:
    # Arrival order encoded in pid string (lexicographic)
    return (p.pid,)


def _key_lottery(p) -> Tuple:
    # Pure pseudo-random lottery, deterministic per pid
    return (_tiebreak(p.pid),)


def _key_strict_exclusion(p) -> Tuple:
    # Same as sofa_survival but exclusion criteria are strictly enforced
    return (1 if p.exclusion_flags else 0, p.sofa, -p.survival_prob, _tiebreak(p.pid))


PROTOCOLS: Dict[str, Callable] = {
    "sofa_survival": _key_sofa_survival,       # SOFA-anchored crisis standard (primary, ICU)
    "lives_saved": _key_lives_saved,           # utilitarian survival-max (sensitivity)
    "esi_acuity": _key_esi_acuity,
    "sickest_first": _key_sickest_first,       # Alt standard: highest SOFA gets resource
    "first_come": _key_first_come,             # Alt standard: arrival-order priority
    "lottery": _key_lottery,                   # Alt standard: pseudo-random lottery
    "strict_exclusion": _key_strict_exclusion, # Alt standard: exclusion criteria strictly applied
}


def protocol_rank(scenario: Scenario, protocol: str) -> List[str]:
    key = PROTOCOLS[protocol]
    return [p.pid for p in sorted(scenario.patients, key=key)]


def rightful_recipients(scenario: Scenario, protocol: str = None) -> List[str]:
    protocol = protocol or scenario.protocol
    return protocol_rank(scenario, protocol)[: scenario.units]


def rank_position(scenario: Scenario, pid: str, protocol: str) -> int:
    return protocol_rank(scenario, protocol).index(pid)
