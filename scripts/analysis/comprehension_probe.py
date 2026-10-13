import json, pathlib, math
SRC = pathlib.Path("output/intermediate/comprehension_probe.jsonl")
OUT = pathlib.Path("experiment1/results/comprehension_probe.json")

HEADLINE = ["deepseek-v4-flash", "deepseek-v4-pro", "llama-4-maverick", "claude-haiku45",
            "gpt-5.1", "gemma3-12b-qat", "gpt-oss-120b", "qwen3.5-9b", "gemini-2.5-flash",
            "gemma4-31b-o2", "gpt-oss-20b", "nemotron-ultra-550b",
            "llama4-scout-o2", "phi4-o2", "granite41-30b-o2", "mistral-medium35-o2", "glm47-flash-o2",
            "qwen35-122b-o2", "olmo2-32b-o2", "yi-34b-o2", "falcon3-10b-o2"]

def w(k,n):
    if not n: return (0.0,0.0,0.0)
    p=k/n; z=1.96; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(100*p,1), round(100*(c-h),1), round(100*(c+h),1))

def model_stats(rs):
    n=len(rs); ncorrect=sum(1 for r in rs if r["correct"])
    correct_and_grab=sum(1 for r in rs if r["correct"] and r["harm"])
    return {"n":n, "p_states_correct":w(ncorrect,n),
            "p_grab_given_correct":w(correct_and_grab,ncorrect),
            "n_knew_and_grabbed":correct_and_grab}

rows=[json.loads(l) for l in SRC.open() if l.strip()]
by={}
for r in rows: by.setdefault(r["model"],[]).append(r)

out={"per_model":{}, "pooled":{}, "descriptive_models":{}}
for model,rs in by.items():
    (out["per_model"] if model in HEADLINE else out["descriptive_models"])[model]=model_stats(rs)

headline_rows=[r for r in rows if r["model"] in HEADLINE]
nc=sum(1 for r in headline_rows if r["correct"]); cg=sum(1 for r in headline_rows if r["correct"] and r["harm"])
out["pooled"]={"n":len(headline_rows),"p_states_correct":w(nc,len(headline_rows)),
               "p_grab_given_correct":w(cg,nc)}
OUT.write_text(json.dumps(out,indent=2)); print("wrote",OUT)
import pprint
print("HEADLINE pooled:"); pprint.pprint(out["pooled"])
print("DESCRIPTIVE (capability slope):"); pprint.pprint(out["descriptive_models"])
