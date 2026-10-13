"""Selects the 100 intensive care wards of experiment 1 from the MIMIC-derived scenarios (the two lowest SOFA scores more than 2 points apart) and assigns imaging needs and the agent's patient. Writes data/scenarios/ward_scenarios.jsonl."""
import json
import sys

sys.path.insert(0, "code")
from scarcity_ward.scenarios import load_scenarios
from scarcity_ward.hospital.ward_scenarios import (derive_roles, from_row,
                                                   to_row, validate_ward)

SRC = "output/intermediate/mimic_scenarios.jsonl"
OUT = "data/scenarios/ward_scenarios.jsonl"


def main(n="100", variant="single"):
    n = int(n)
    clear = [l for l in open(SRC) if json.loads(l)["config"] == "clear"][:n]
    tmp = SRC + ".clear.tmp"
    open(tmp, "w").writelines(clear)
    suffix = {"single": ".jsonl", "two_ward": "_2ward.jsonl", "two_ward_bed": "_2ward_bed.jsonl"}[variant]
    out_path = OUT if variant == "single" else OUT.replace(".jsonl", suffix)
    written = 0
    with open(out_path, "w") as fh:
        for scn in load_scenarios(tmp):
            needs, focal = derive_roles(scn)
            if variant in ("two_ward", "two_ward_bed"):
                from scarcity_ward.hospital.ward_scenarios import to_two_ward_row
                row = to_two_ward_row(scn, needs, focal_pid=focal,
                                      boundary="bed" if variant == "two_ward_bed" else "imaging")
            else:
                row = to_row(scn, needs, focal_pid=focal)
            errs = validate_ward(from_row(row))
            if errs:
                print("  [skip %s] %s" % (scn.sid, errs), file=sys.stderr)
                continue
            fh.write(json.dumps(row) + "\n")
            written += 1
    print("wrote %d ward scenarios -> %s" % (written, out_path))


if __name__ == "__main__":
    main(*sys.argv[1:])
