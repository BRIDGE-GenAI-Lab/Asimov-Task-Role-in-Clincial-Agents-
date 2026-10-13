import json, pathlib
SRC = pathlib.Path("output/intermediate/oversight_information.jsonl")
OUT = pathlib.Path("experiment1/results/oversight_information.json")
import math
def w(k, n):
    if not n: return (0.0,0.0,0.0)
    p=k/n; z=1.96; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(100*p,1), round(100*(c-h),1), round(100*(c+h),1))
ORDER = ["blind","partial:noise=3,margin=2","partial:noise=1,margin=0.5","data"]
LABEL = {"blind":"none","partial:noise=3,margin=2":"partial_noisy",
         "partial:noise=1,margin=0.5":"partial_tight","data":"oracle"}
rows=[json.loads(l) for l in SRC.open() if l.strip()]
by={}
for r in rows:
    by.setdefault(r["model"],{}).setdefault(r.get("oversight"),[]).append(bool(r["harm"]))
out={"per_model_per_level":{}}
for model,lv in by.items():
    d={}
    for spec in ORDER:
        hs=lv.get(spec,[]); k,n=sum(hs),len(hs); pt,lo,hi=w(k,n)
        d[LABEL[spec]]={"chr":pt,"ci":[lo,hi],"n":n}
    blind=d.get("none",{}).get("chr",0); oracle=d.get("oracle",{}).get("chr",0)
    partial=d.get("partial_noisy",{}).get("chr",0)
    denom=(blind-oracle) or 1
    d["partial_recovers_frac"]=round((blind-partial)/denom,3)
    out["per_model_per_level"][model]=d
OUT.write_text(json.dumps(out,indent=2)); print("wrote",OUT)
import pprint; pprint.pprint(out)
