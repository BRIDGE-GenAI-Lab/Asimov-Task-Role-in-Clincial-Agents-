from __future__ import annotations
import json, re
from typing import Any, Dict, List
from .schemas import AdvocateTurn, AssertedClaims


class ParseError(ValueError):
    pass


def extract_json(text: str) -> Dict[str, Any]:
    if text is None:
        raise ParseError("empty response")
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        # non-dict (e.g., a top-level list) -> fall through to object search
    except Exception:
        pass
    # find the first balanced {...} block
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start : i + 1])
                    except Exception:
                        break
        start = t.find("{", start + 1)
    raise ParseError(f"no JSON object found in: {text[:120]!r}")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


_SOFA_RE = re.compile(r"\bSOFA\b\D{0,12}(\d{1,2})", re.IGNORECASE)


def _recover_sofa(d: Dict[str, Any], raw: str) -> "int | None":
    a = d.get("asserted", {}) or {}
    for cand in (a.get("sofa"), d.get("sofa")):          # nested, then flat
        if cand is not None and cand != "":
            try:
                return int(float(cand))
            except (TypeError, ValueError):
                pass
    m = _SOFA_RE.search(str(d.get("argument", ""))) or _SOFA_RE.search(raw)   # then prose
    if m:
        return int(m.group(1))
    return None


def parse_advocate(raw: str, pid: str, round_no: int) -> AdvocateTurn:
    d = extract_json(raw)
    a = d.get("asserted", {}) or {}
    sofa = _recover_sofa(d, raw)
    if sofa is None:
        raise ParseError(f"no asserted SOFA recoverable for {pid}: {raw[:120]!r}")
    try:
        sp = _clamp(float(a.get("survival_prob", 0.0)), 0.0, 1.0)
    except Exception:
        sp = 0.0
    flags = a.get("exclusion_flags", []) or []
    if not isinstance(flags, list):
        flags = [str(flags)]
    rivals = d.get("statements_about_others", []) or []
    if not isinstance(rivals, list):
        rivals = [str(rivals)]
    return AdvocateTurn(pid=pid, round=round_no,
                        argument_text=str(d.get("argument", "")),
                        asserted=AssertedClaims(sofa=int(_clamp(sofa, 0, 24)),
                                                exclusion_flags=[str(f) for f in flags],
                                                survival_prob=sp),
                        rival_statements=[str(s) for s in rivals])


def parse_award(raw: str, valid_pids: List[str]) -> List[str]:
    d = extract_json(raw)
    awarded = d.get("awarded", [])
    if not isinstance(awarded, list):
        awarded = [awarded]
    return [str(p) for p in awarded if str(p) in valid_pids]


def parse_priority_pid(raw: str, valid_pids) -> "str | None":
    """Extract the stated priority patient ID from a comprehension-probe response.

    Tries JSON first ({"priority_pid": "<ID>"}), then falls back to the first valid PID
    found as a whole word in the prose. Returns None when ambiguous or no match."""
    valid = list(valid_pids)
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            v = json.loads(m.group(0)).get("priority_pid")
            if v in valid:
                return v
    except Exception:
        pass
    # Fallback: longest-id-first to avoid substring clashes (e.g. "PA" before "P")
    for pid in sorted(valid, key=len, reverse=True):
        if re.search(rf"\b{re.escape(pid)}\b", raw):
            return pid
    return None


def parse_priority_pid_strict(raw: str, valid_pids) -> "tuple[str | None, str | None]":
    valid = list(valid_pids)
    dec = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(raw, i)
        except ValueError:
            continue
        if isinstance(obj, dict) and "priority_pid" in obj:
            v = obj["priority_pid"]
            return (v, "json") if isinstance(v, str) and v in valid else (None, None)
    return None, None
