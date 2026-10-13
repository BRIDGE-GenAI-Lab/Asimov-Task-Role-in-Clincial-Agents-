"""Canonical MIMIC-IV SOFA itemids (cross-checked against the mimic-code repo). chart_* live in
chartevents (d_items); lab_* live in labevents (d_labitems). discover() re-derives them from the
dictionaries by label so they are verifiable rather than asserted."""
from __future__ import annotations
from typing import Dict, List
import pandas as pd

SOFA_ITEMIDS: Dict[str, List[int]] = {
    # chartevents (icu/d_items)
    "gcs": [220739, 223900, 223901],                 # GCS eye/verbal/motor (sum separately)
    "map": [220052, 220181, 225312],                 # arterial/non-invasive/cuff MAP
    "fio2": [223835],                                # inspired O2 fraction (%)
    "spo2": [220277],
    # labevents (hosp/d_labitems)
    "platelets": [51265],
    "bilirubin": [50885],                            # total bilirubin
    "creatinine": [50912],
    "pao2": [50821],                                 # arterial PO2
    # outputevents (icu/d_items) — urine output sources (summed for 24h total)
    "urine": [226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558],
    # inputevents (icu/d_items) — vasopressors (mcg/kg/min via rate)
    "norepinephrine": [221906],
    "epinephrine": [221289],
    "dopamine": [221662],
    "dobutamine": [221653],
}


def label_matches(label: str, needles: List[str]) -> bool:
    lab = (label or "").lower()
    return any(n.lower() in lab for n in needles)


def discover(d_items_csv: str, d_labitems_csv: str) -> Dict[str, List[int]]:
    """Re-derive itemids from the dictionaries by label (for verification/audit)."""
    di = pd.read_csv(d_items_csv, usecols=["itemid", "label"])
    dl = pd.read_csv(d_labitems_csv, usecols=["itemid", "label"])
    spec = {"gcs": (di, ["gcs"]), "map": (di, ["blood pressure mean", "arterial pressure mean"]),
            "fio2": (di, ["inspired o2", "fio2"]), "platelets": (dl, ["platelet"]),
            "bilirubin": (dl, ["bilirubin, total"]), "creatinine": (dl, ["creatinine"]),
            "pao2": (dl, ["po2"]), "norepinephrine": (di, ["norepinephrine"])}
    out: Dict[str, List[int]] = {}
    for key, (tbl, needles) in spec.items():
        hits = tbl[tbl["label"].apply(lambda x: label_matches(x, needles))]
        out[key] = sorted(hits["itemid"].tolist())
    return out
