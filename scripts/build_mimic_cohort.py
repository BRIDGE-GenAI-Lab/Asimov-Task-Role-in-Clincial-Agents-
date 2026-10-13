"""Builds the three-patient intensive care scenarios of experiment 1 from MIMIC-IV (credentialed access): one stay per hospital admission, SOFA from the first 24 hours, survival estimate 1 - SOFA/24. Reads the local MIMIC-IV files; writes the scenario file used by gen_ward_scenarios.py."""
import sys
import json
sys.path.insert(0, "code")
import pandas as pd
from scarcity_ward.mimic.paths import MimicPaths
from scarcity_ward.mimic.cohort import build_backbone
from scarcity_ward.mimic.itemids import SOFA_ITEMIDS
from scarcity_ward.mimic.extract import WorstReducer, stream_table
from scarcity_ward.mimic.sofa import (resp_score, coag_score, liver_score, cv_score,
                                      cns_score, renal_score, sofa_total)
from scarcity_ward.mimic.build import assemble_scenarios


def main(k="3", max_scenarios="200"):
    p = MimicPaths()
    base = build_backbone(p).dropna(subset=["intime", "outtime", "hadm_id"])
    base = base.sort_values("intime").drop_duplicates("hadm_id", keep="first")
    base["cutoff"] = (pd.to_datetime(base["intime"]) + pd.Timedelta(hours=24)).astype(str)
    stay_window = {str(int(r.stay_id)): (str(r.intime), str(r.cutoff)) for r in base.itertuples()}
    hadm_window = {str(int(r.hadm_id)): (str(r.intime), str(r.cutoff)) for r in base.itertuples()}
    hadm_to_stay = {str(int(r.hadm_id)): str(int(r.stay_id)) for r in base.itertuples()}

    g_eye, g_verbal, g_motor = SOFA_ITEMIDS["gcs"]
    chart_map = {g_eye: "gcs_eye", g_verbal: "gcs_verbal", g_motor: "gcs_motor"}
    for iid in SOFA_ITEMIDS["map"]:
        chart_map[iid] = "map"
    for iid in SOFA_ITEMIDS["fio2"]:
        chart_map[iid] = "fio2"
    chart_red = WorstReducer({"gcs_eye": "min", "gcs_verbal": "min", "gcs_motor": "min",
                              "map": "min", "fio2": "max"})

    lab_map = {iid: c for c in ["platelets", "bilirubin", "creatinine", "pao2"]
               for iid in SOFA_ITEMIDS[c]}
    lab_red = WorstReducer({"platelets": "min", "bilirubin": "max", "creatinine": "max", "pao2": "min"})

    pressor_map = {iid: c for c in ["norepinephrine", "epinephrine", "dopamine", "dobutamine"]
                   for iid in SOFA_ITEMIDS[c]}
    pressor_red = WorstReducer({c: "max" for c in ["norepinephrine", "epinephrine", "dopamine", "dobutamine"]})

    urine_map = {iid: "urine" for iid in SOFA_ITEMIDS["urine"]}
    urine_red = WorstReducer({"urine": "sum"})

    print("streaming chartevents (3.3G)...")
    stream_table(p.icu("chartevents"), chart_map, chart_red, stay_window)
    print("streaming inputevents (383M)...")
    stream_table(p.icu("inputevents"), pressor_map, pressor_red, stay_window,
                 time_col="starttime", value_col="rate",
                 unit_col="rateuom", allowed_units={"mcg/kg/min"})
    print("streaming labevents (2.4G)...")
    stream_table(p.hosp("labevents"), lab_map, lab_red, hadm_window, stay_col="hadm_id")
    print("streaming outputevents (urine)...")
    stream_table(p.icu("outputevents"), urine_map, urine_red, stay_window, value_col="value")

    rows = []
    for hadm, stay in hadm_to_stay.items():
        ch = chart_red.as_dict().get(stay, {})
        lab = lab_red.as_dict().get(hadm, {})
        pr = pressor_red.as_dict().get(stay, {})
        gparts = [ch.get("gcs_eye"), ch.get("gcs_verbal"), ch.get("gcs_motor")]
        gcs = sum(gparts) if all(x is not None for x in gparts) else None
        pf = (lab.get("pao2") / (ch["fio2"] / 100.0)) if (lab.get("pao2") and ch.get("fio2")) else None
        uo = urine_red.as_dict().get(stay, {}).get("urine")
        comps = {"resp": resp_score(pf, ventilated=bool(ch.get("fio2"))),
                 "coag": coag_score(lab.get("platelets")),
                 "liver": liver_score(lab.get("bilirubin")),
                 "cv": cv_score(ch.get("map"), norepi=pr.get("norepinephrine") or 0.0,
                                epi=pr.get("epinephrine") or 0.0,
                                dopamine=pr.get("dopamine") or 0.0,
                                dobutamine=pr.get("dobutamine") or 0.0),
                 "cns": cns_score(gcs), "renal": renal_score(lab.get("creatinine"), urine_ml_24h=uo)}
        rows.append({"stay_id": int(stay), "sofa": sofa_total(comps)})

    sofa_df = pd.DataFrame(rows)
    base = base.assign(stay_id=base["stay_id"].astype(int))
    cohort = base.merge(sofa_df, on="stay_id", how="inner")
    cohort.to_csv("output/intermediate/cohort_sofa.csv", index=False)

    scns = assemble_scenarios(cohort, k=int(k), resource="ventilator")[: int(max_scenarios)]
    with open("output/intermediate/mimic_scenarios.jsonl", "w") as fh:
        for s in scns:
            fh.write(json.dumps({"sid": s.sid, "resource": s.resource, "units": s.units,
                                 "protocol": s.protocol, "config": s.config,
                                 "patients": [pt.__dict__ for pt in s.patients]}) + "\n")

    band = pd.cut(cohort["sofa"], [-1, 3, 7, 11, 24])
    print("mortality by SOFA band (validity check — expect monotone increasing):")
    print(cohort.groupby(band)["hospital_expire_flag"].mean())
    print(f"wrote {len(scns)} scenarios -> output/intermediate/mimic_scenarios.jsonl "
          f"(cohort n={len(cohort)})")


if __name__ == "__main__":
    main(*sys.argv[1:])
