import json, pathlib, statistics, math
SRC=pathlib.Path("output/intermediate/temp_sweep.jsonl")
OUT=pathlib.Path("experiment1/results/temperature_sweep.json")
def w(k,n):
    if not n: return (0.0,0.0,0.0)
    p=k/n; z=1.96; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(100*p,1),round(100*(c-h),1),round(100*(c+h),1))
if not SRC.exists():
    OUT.write_text(json.dumps({"status":"PENDING — sweep not yet run (reference model queued for OpenRouter top-up)","per_temp":{}}, indent=2))
    print("no sweep data yet; wrote PENDING placeholder"); raise SystemExit(0)
rows=[json.loads(l) for l in SRC.open() if l.strip()]
by={}
for r in rows:
    t=str(r.get("temperature", r.get("temp")))
    by.setdefault((t, r["arm"]),{}).setdefault(r.get("rep",0),[]).append(bool(r["harm"]))
out={"per_temp":{}}
for (t,arm),reps in sorted(by.items()):
    rep_chrs=[100*sum(v)/len(v) for v in reps.values() if v]
    allh=[h for v in reps.values() for h in v]; k,n=sum(allh),len(allh); pt,lo,hi=w(k,n)
    out["per_temp"].setdefault(t,{})[arm]={"chr":pt,"ci":[lo,hi],
        "run_to_run_sd":round(statistics.pstdev(rep_chrs),2) if len(rep_chrs)>1 else 0.0,
        "optimistic":round(min(rep_chrs),1) if rep_chrs else pt,
        "pessimistic":round(max(rep_chrs),1) if rep_chrs else pt}
OUT.write_text(json.dumps(out,indent=2)); print("wrote",OUT); import pprint; pprint.pprint(out)
