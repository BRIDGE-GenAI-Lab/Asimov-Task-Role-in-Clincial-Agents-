import csv, json, statistics
from collections import Counter
from pathlib import Path
BASE = Path(""); VAL = BASE/"output/validation"

key = {r["case_id"]: r["deterministic_harm_label"] for r in csv.DictReader(open(VAL/"clinician_packet_key.csv"))}
packet = {r["case_id"]: r["ward_state"] for r in csv.DictReader(open(VAL/"clinician_packet_blinded.csv"))}
img = {c for c,w in packet.items() if "contested this episode: the shared imaging slot" in w}
ids = [str(i) for i in range(1,151)]

def load(rel):
    rows=list(csv.DictReader(open(BASE/rel, encoding="utf-8-sig")))
    vc=next(c for c in rows[0] if "violation" in c.lower())
    return {r["case_id"]:{"yes":"harm","no":"no_harm"}.get((r.get(vc) or "").strip().lower()) for r in rows}

P = {"physician_1": load("phsician_1.csv"), "physician_2": load("physician_2.csv"), "physician_3": load("physician_3.csv")}
VER = {"physician_1":"v2 (final)", "physician_2":"v1 (pre-tier)", "physician_3":"v2 (final)"}

def stats(lab, subset=None):
    cs = subset or ids
    pairs=[(lab[c], key[c]) for c in cs if lab.get(c)]
    n=len(pairs); po=sum(a==b for a,b in pairs)/n
    ra=Counter(a for a,_ in pairs); ka=Counter(b for _,b in pairs)
    pe=sum((ra[x]/n)*(ka[x]/n) for x in ("harm","no_harm")); k=(po-pe)/(1-pe)
    tp=sum(a=="harm"and b=="harm"for a,b in pairs); fn=sum(a=="no_harm"and b=="harm"for a,b in pairs)
    tn=sum(a=="no_harm"and b=="no_harm"for a,b in pairs); fp=sum(a=="harm"and b=="no_harm"for a,b in pairs)
    return dict(n=n, agree=round(po*100,1), kappa=round(k,3),
                sens=round(tp/(tp+fn)*100,1) if tp+fn else None,
                spec=round(tn/(tn+fp)*100,1) if tn+fp else None)

print("PER-PHYSICIAN vs deterministic scorer:")
for name,lab in P.items():
    s=stats(lab); print(f"  {name} [{VER[name]}]: {s}")
print("  physician_2 on NON-imaging cases only:", stats(P["physician_2"], [c for c in ids if c not in img]))

maj={}
for c in ids:
    votes=[P[n][c] for n in P if P[n].get(c)]
    maj[c]=Counter(votes).most_common(1)[0][0]
print("\nMAJORITY-OF-3 vs scorer:", stats(maj))

def fleiss(labels_by_rater, cs):
    cats=["harm","no_harm"]; N=len(cs); nrat=len(labels_by_rater)
    P_i=[]; coltot=Counter()
    for c in cs:
        cnt=Counter(r[c] for r in labels_by_rater)
        for cat in cats: coltot[cat]+=cnt[cat]
        P_i.append((sum(cnt[cat]**2 for cat in cats)-nrat)/(nrat*(nrat-1)))
    Pbar=sum(P_i)/N; pj=[coltot[cat]/(N*nrat) for cat in cats]; Pe=sum(p**2 for p in pj)
    return round((Pbar-Pe)/(1-Pe),3)
print("Inter-rater Fleiss kappa (3 physicians, all 150):", fleiss(list(P.values()), ids))
print("Inter-rater Fleiss kappa (3 physicians, NON-imaging):", fleiss(list(P.values()), [c for c in ids if c not in img]))
print("Inter-rater Fleiss kappa (physician_1 & _3 only, v2):", fleiss([P["physician_1"],P["physician_3"]], ids))

ctr=[c for c in ids if key[c]=="harm" and "contested this episode: the ICU bed" in packet[c]
     and ("held by free" in packet[c] or "release_bed" in packet[c])]
print(f"\n'claim-then-release'-type bed-harm cases in packet (heuristic): {len(ctr)} -> {ctr}")
