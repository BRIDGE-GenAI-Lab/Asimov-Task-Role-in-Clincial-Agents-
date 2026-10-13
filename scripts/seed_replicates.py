"""Seeds the temperature 0.7 replicate logs of Run E. For each model key given on the command line it copies the replicate 0 rows of the steward and devotion roles from experiment2/runs/runA_<key>.jsonl into a new runE_T07_<key>.jsonl and refuses to overwrite an existing file."""
import json, os, sys

E2_DIR = "experiment2"
ARMS = {"steward", "devotion"}


def seed(src, dst):
    if os.path.exists(dst):
        raise FileExistsError(dst)
    n = 0
    with open(src) as fi, open(dst, "x") as fo:
        for line in fi:
            if line.strip():
                r = json.loads(line)
                if r.get("arm") in ARMS and r.get("rep") == 0:
                    fo.write(line if line.endswith("\n") else line + "\n")
                    n += 1
    return n


if __name__ == "__main__":
    for key in sys.argv[1:]:
        n = seed("%s/runs/runA_%s.jsonl" % (E2_DIR, key), "%s/runs/runE_T07_%s.jsonl" % (E2_DIR, key))
        print(key, n)
