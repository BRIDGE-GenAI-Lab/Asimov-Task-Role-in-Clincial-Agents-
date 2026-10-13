"""Stream the big MIMIC tables once, filtered to SOFA itemids, keeping the first-24h worst value
per stay per component. Never loads the whole table (disk-is-the-constraint lesson)."""
from __future__ import annotations
import csv
import gzip
from typing import Dict, Optional


class WorstReducer:
    """Per (stay, component) keep the worst value, where 'worst' is min or max by policy."""

    def __init__(self, policy: Dict[str, str]):
        self.policy = policy                       # component -> 'min' | 'max'
        self._d: Dict[str, Dict[str, float]] = {}

    def update(self, stay: str, component: str, val: float) -> None:
        cur = self._d.setdefault(stay, {})
        if component not in cur:
            cur[component] = val
            return
        pol = self.policy.get(component, "max")
        if pol == "min":
            cur[component] = min(cur[component], val)
        elif pol == "sum":
            cur[component] = cur[component] + val
        else:
            cur[component] = max(cur[component], val)

    def value(self, stay: str, component: str) -> Optional[float]:
        return self._d.get(stay, {}).get(component)

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return self._d


def stream_table(gz_path: str, itemid_to_component: Dict[int, str], reducer: WorstReducer,
                 stay_window: Dict[str, tuple], stay_col: str = "stay_id",
                 time_col: str = "charttime", value_col: str = "valuenum",
                 itemid_col: str = "itemid", unit_col: Optional[str] = None,
                 allowed_units: Optional[set] = None) -> None:
    """stay_window: stay_id(str) -> (intime_iso, cutoff_iso). Rows outside the window, with an
    itemid we don't track, a non-numeric value, or (when allowed_units is given) a unit not in
    allowed_units, are skipped. Position-free parse via DictReader."""
    wanted = set(itemid_to_component)
    units = set(allowed_units) if allowed_units else None
    with gzip.open(gz_path, "rt", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                iid = int(row[itemid_col])
            except (ValueError, KeyError, TypeError):
                continue
            if iid not in wanted:
                continue
            if units is not None and unit_col is not None and (row.get(unit_col) or "") not in units:
                continue
            stay = row.get(stay_col) or ""
            win = stay_window.get(stay)
            if win is None:
                continue
            t = row.get(time_col, "")
            if not (win[0] <= t <= win[1]):
                continue
            v = row.get(value_col, "")
            try:
                val = float(v)
            except (ValueError, TypeError):
                continue
            reducer.update(stay, itemid_to_component[iid], val)
