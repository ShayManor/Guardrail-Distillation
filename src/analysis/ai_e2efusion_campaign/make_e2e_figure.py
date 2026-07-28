#!/usr/bin/env python3
"""e2e-fusion 5-seed summary bar chart (ACDC shift). Reads campaign eval dirs on cluster."""
import csv, glob, os, collections, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO=os.path.expanduser("~/Guardrail-Distillation")
def f(x):
    try: return float(x)
    except: return None
def auroc(sc,lb):
    if any(s is None for s in sc): return None
    o=sorted(range(len(sc)),key=lambda i:sc[i]);r=[0.0]*len(sc);i=0
    while i<len(o):
        j=i
        while j<len(o) and sc[o[j]]==sc[o[i]]: j+=1
        a=(i+j-1)/2+1
        for k in range(i,j): r[o[k]]=a
        i=j
    pos=[i for i in range(len(sc)) if lb[i]==1]
    if not pos or len(pos)==len(sc): return None
    return (sum(r[i] for i in pos)-len(pos)*(len(pos)+1)/2)/(len(pos)*(len(sc)-len(pos)))
def prank(xs):
    o=sorted(range(len(xs)),key=lambda i:xs[i]);r=[0.0]*len(xs)
    for k,i in enumerate(o): r[i]=k/(len(xs)-1) if len(xs)>1 else .5
    return r
def col(g,c,s=1): return [None if f(x.get(c)) is None else s*f(x[c]) for x in g]
def load(p): return [x for x in csv.DictReader(open(p)) if x['domain'] in ('fog','night','rain','snow') and f(x.get('guardrailpp_utility_dense_gap')) is not None]
def cellmean(path,scorer):
    g=load(path); gr=collections.defaultdict(list)
    for x in g: gr[x['domain']].append(x)
    v=[]
    for dom,gg in gr.items():
        msps=[f(x['student_msp']) for x in gg]; risks=[f(x['student_risk']) for x in gg]
        idx=[i for i in range(len(gg)) if msps[i]>=0.85]
        if len(idx)<10: continue
        cr=sorted(risks[i] for i in idx); cut=cr[int(0.8*len(cr))]
        lb=[1 if risks[i]>=cut else 0 for i in idx]
        sc=scorer(gg); a=auroc([sc[i] for i in idx],lb)
        if a is not None: v.append(a)
    return st.mean(v) if v else None
# gather eval dirs
def evaldirs(pat): return sorted(glob.glob(os.path.join(REPO,pat)))
CF=evaldirs("paper_eval_acdc_b*_guard_dense_multi_conffeat_s*/csv/per_image.csv")
BASEnew=evaldirs("paper_eval_acdc_b*_guard_dense_multi_base_s*/csv/per_image.csv")
# baseline seeds 42/137/256 dirs (non-conffeat dense_multi s42/137/256 fresh evals)
BASEold=[p for p in evaldirs("paper_eval_acdc_b*_guard_dense_multi_s*/csv/per_image.csv") if 'conffeat' not in p and 'base' not in p]
methods={
 'MSP':lambda g:col(g,'student_msp',-1),
 'Temp-scaling':lambda g:col(g,'temp_msp',-1),
 'Baseline\nguardrail':None,  # handled separately (baseline files)
 'Energy':lambda g:col(g,'energy_score',1),
 'e2e-fusion':lambda g:col(g,'guardrailpp_utility_dense_gap',1),
 'e2e-fusion\n+ energy':lambda g:[a+b for a,b in zip(prank(col(g,'guardrailpp_utility_dense_gap',1)),prank(col(g,'energy_score',1)))],
}
res={}
# energy/msp/temp/e2e from CF dirs (5 seeds); baseline from BASE dirs
for label,scorer in methods.items():
    if label=='Baseline\nguardrail':
        vals=[cellmean(p,lambda g:col(g,'guardrailpp_utility_dense_gap',1)) for p in BASEnew+BASEold]
    else:
        vals=[cellmean(p,scorer) for p in CF]
    vals=[v for v in vals if v is not None]
    res[label]=(st.mean(vals),st.pstdev(vals))
order=sorted(res,key=lambda k:res[k][0])
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(figsize=(7,4.2))
names=order; means=[res[n][0] for n in names]; stds=[res[n][1] for n in names]
colors=["#c44e52" if "e2e-fusion" in n else ("#4c72b0" if "guardrail" in n else "#b0b0b0") for n in names]
ax.barh(range(len(names)),means,xerr=stds,color=colors,height=0.66,error_kw=dict(lw=1,capsize=3,ecolor="#555"))
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set_xlim(0.5,0.72); ax.set_xlabel("Confident-failure AUROC @ MSP≥0.85 (ACDC shift)")
ax.set_title("End-to-end fusion: single learned score beats energy\n(3 backbones × 5 seeds)",fontsize=10.5)
for i,(m,s) in enumerate(zip(means,stds)): ax.text(m+s+0.003,i,f"{m:.3f}",va="center",fontsize=8.5,color="#333")
fig.tight_layout()
out=os.path.join(REPO,"src","analysis","ai_e2efusion_campaign","fig_e2e_fusion.png")
fig.savefig(out,dpi=150,bbox_inches="tight")
print("wrote",out)
print({n:round(res[n][0],3) for n in order})
