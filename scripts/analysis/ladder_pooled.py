import json, math, collections, statistics
from pathlib import Path
from scipy.stats import t as tdist

FILES=[('Gemma-3 12B','ward_exp1_v2.jsonl'),('DeepSeek-V4 Flash','ward_ds_ladder_v2.jsonl'),
('DeepSeek-V4 Pro','ward_ds_pro_icu.jsonl'),('Gemini-2.5 Flash','ward_gem25f_ladder.jsonl'),
('GPT-5.1','ward_gpt51_icu.jsonl'),('Claude Haiku 4.5','ward_haiku_icu.jsonl'),
('GPT-OSS 120B','ward_gptoss_icu.jsonl'),('Gemma-4 31B','ward_gemma4_31b_o2_icu.jsonl'),
('GPT-OSS 20B','ward_oss20b_icu.jsonl'),('Nemotron-3-Ultra 550B','ward_nemotron550b_icu.jsonl'),
('Llama-4-Scout','ward_llama4_scout_o2_icu.jsonl'),('Phi-4','ward_phi4_o2_icu.jsonl'),
('Granite-4.1-30B','ward_granite41_30b_o2_icu.jsonl'),('Mistral-Medium-3.5','ward_mistral35_off_o2_icu.jsonl'),
('GLM-4.7-Flash','ward_glm47_flash_o2_icu.jsonl'),('Qwen3.5-122B','ward_qwen35_122b_o2_icu.jsonl'),
('Cohere Command A+','ward_cohere_command_a_o2_icu.jsonl'),('OLMo-2-32B','ward_olmo2_32b_o2_icu.jsonl'),
('Yi-1.5-34B','ward_yi_34b_o2_icu.jsonl'),('Falcon3-10B','ward_falcon3_10b_o2_icu.jsonl')]
ARMS=['steward','empathetic','guardrail','neutral','devotion']
def load(fn): return [r for r in (json.loads(l) for l in open('output/intermediate/'+fn) if l.strip()) if 'n_steps' in r]
def is_icu(s): return 'ED' not in s.upper()
def wilson(k,n):
    if n==0: return (0,0,0)
    p=k/n; z=1.96; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(100*p,1),round(100*(c-h),1),round(100*(c+h),1))
def odds_ratio(k1,n1,k0,n0):
    a,b,c,dd=k1,n1-k1,k0,n0-k0
    if 0 in (a,b,c,dd): a+=.5;b+=.5;c+=.5;dd+=.5
    orr=(a*dd)/(b*c); se=math.sqrt(1/a+1/b+1/c+1/dd)
    return round(orr,1),round(orr*math.exp(-1.96*se),1),round(orr*math.exp(1.96*se),1)

permodel={}; counts={}
for disp,fn in FILES:
    rr=[r for r in load(fn) if is_icu(r.get('sid',''))]
    c=collections.defaultdict(lambda:[0,0])
    for r in rr:
        a=c[r['arm']]; a[1]+=1; a[0]+= 1 if r.get('harm') else 0
    permodel[disp]={a:(round(100*c[a][0]/c[a][1],1) if c[a][1] else None) for a in ARMS}
    counts[disp]={a:tuple(c[a]) for a in ARMS}

def pooled(a):
    vals=[permodel[d][a] for d,_ in FILES if permodel[d][a] is not None]
    m=statistics.mean(vals); sd=statistics.stdev(vals); se=sd/math.sqrt(len(vals))
    t=tdist.ppf(0.975, len(vals)-1)
    return round(m,1), round(m-t*se,1), round(m+t*se,1), len(vals), [min(vals),max(vals)]

pool={a:dict(zip(['mean','ci_lo','ci_hi','n_models','range'], pooled(a))) for a in ARMS}
gaps=[permodel[d]['devotion']-permodel[d]['steward'] for d,_ in FILES]
gm=statistics.mean(gaps); gse=statistics.stdev(gaps)/math.sqrt(len(gaps))
gt=tdist.ppf(0.975, len(gaps)-1)
gap={'mean':round(gm,1),'ci_lo':round(gm-gt*gse,1),'ci_hi':round(gm+gt*gse,1)}

def contrasts(disp):
    c=counts[disp]
    return {
      'devotion_vs_steward': odds_ratio(c['devotion'][0],c['devotion'][1],c['steward'][0],c['steward'][1]),
      'assignment_neutral_vs_steward': odds_ratio(c['neutral'][0],c['neutral'][1],c['steward'][0],c['steward'][1]),
      'instruction_devotion_vs_neutral': odds_ratio(c['devotion'][0],c['devotion'][1],c['neutral'][0],c['neutral'][1]),
      'empathetic_vs_neutral': odds_ratio(c['empathetic'][0],c['empathetic'][1],c['neutral'][0],c['neutral'][1]),
    }
out={
  'note':'Canonical ICU ladder (same files as figures/capability_slope). Crude ORs with logit 95% CI. '
         'Pooled = mean across 20 primary models (... GPT-OSS 20B 9th, Nemotron-3-Ultra 550B 10th [free OR]), t-based 95% CI (df=19).',
  'canonical_files':{d:f for d,f in FILES},
  'per_model_chr': permodel,
  'pooled_per_arm': pool,
  'pooled_devotion_minus_steward_pp': gap,
  'gemma_contrasts': contrasts('Gemma-3 12B'),
  'deepseek_flash_contrasts': contrasts('DeepSeek-V4 Flash'),
  'deepseek_flash_chr_ci':{a:wilson(*counts['DeepSeek-V4 Flash'][a]) for a in ARMS},
  'gemma_chr_ci':{a:wilson(*counts['Gemma-3 12B'][a]) for a in ARMS},
}
Path('experiment1/results/ladder_pooled.json').write_text(json.dumps(out,indent=2))
print('wrote results/ladder_pooled.json')
print('POOLED per arm:', {a:pool[a]['mean'] for a in ARMS})
print('pooled gap dev-stew:', gap)
print('gemma dev-vs-stew OR:', out['gemma_contrasts']['devotion_vs_steward'])
print('flash dev-vs-stew OR:', out['deepseek_flash_contrasts']['devotion_vs_steward'])
print('flash assignment OR:', out['deepseek_flash_contrasts']['assignment_neutral_vs_steward'],
      ' instruction OR:', out['deepseek_flash_contrasts']['instruction_devotion_vs_neutral'],
      ' empa-vs-neut OR:', out['deepseek_flash_contrasts']['empathetic_vs_neutral'])
print('flash ladder CHR:', {a:permodel['DeepSeek-V4 Flash'][a] for a in ARMS})
