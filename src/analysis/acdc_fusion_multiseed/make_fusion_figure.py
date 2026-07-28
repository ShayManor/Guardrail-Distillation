#!/usr/bin/env python3
"""Formal multi-seed fusion summary: tidy CSV + one clean figure.
Reads the fresh ACDC per_image.csv from the 9 eval dirs (b0/b1/b2 x seeds 42/137/256).
Outputs fusion_multiseed_results.csv and fig_fusion_summary.png.
"""
import csv, collections, statistics as st, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.expanduser("~/Guardrail-Distillation")
EVAL = {
 ('b0','42'):'paper_eval_acdc_b0_mit-b0_guard_dense_multi_s42_j9277988',
 ('b0','137'):'paper_eval_acdc_b0_mit-b0_guard_dense_multi_s137_j9273623',
 ('b0','256'):'paper_eval_acdc_b0_mit-b0_guard_dense_multi_s256_j9274873',
 ('b1','42'):'paper_eval_acdc_b1_mit-b1_guard_dense_multi_s42_j9299852',
 ('b1','137'):'paper_eval_acdc_b1_mit-b1_guard_dense_multi_s137_j9299853',
 ('b1','256'):'paper_eval_acdc_b1_mit-b1_guard_dense_multi_s256_j9274870',
 ('b2','42'):'paper_eval_acdc_b2_mit-b2_guard_dense_multi_s42_j9277989',
 ('b2','137'):'paper_eval_acdc_b2_mit-b2_guard_dense_multi_s137_j9274868',
 ('b2','256'):'paper_eval_acdc_b2_mit-b2_guard_dense_multi_s256_j9273626',
}
def f(x):
    try: return float(x)
    except: return None
def auroc(scores,labels):
    order=sorted(range(len(scores)),key=lambda i:scores[i]); ranks=[0.0]*len(scores);i=0
    while i<len(order):
        j=i
        while j<len(order) and scores[order[j]]==scores[order[i]]: j+=1
        avg=(i+j-1)/2.0+1
        for k in range(i,j): ranks[order[k]]=avg
        i=j
    pos=[i for i in range(len(scores)) if labels[i]==1]
    if not pos or len(pos)==len(scores): return None
    np_=len(pos);nn=len(scores)-np_
    return (sum(ranks[i] for i in pos)-np_*(np_+1)/2)/(np_*nn)
def prank(xs):
    order=sorted(range(len(xs)),key=lambda i:xs[i]);o=[0.0]*len(xs)
    for r,i in enumerate(order): o[i]=r/(len(xs)-1) if len(xs)>1 else .5
    return o
def load(bb,seed):
    p=os.path.join(REPO,EVAL[(bb,seed)],"csv","per_image.csv")
    return [x for x in csv.DictReader(open(p))
            if x['domain'] in ('fog','night','rain','snow') and f(x.get('guardrailpp_utility_dense_gap')) is not None]
def col(g,c,s=1): return [None if f(x.get(c)) is None else s*f(x[c]) for x in g]
def fuse(g,cs):
    comps=[prank(col(g,c,s)) for c,s in cs]; return [sum(cc[k] for cc in comps) for k in range(len(g))]
METH=[('MSP',[('student_msp',-1)]),('Temp-scaling',[('temp_msp',-1)]),('MC-dropout',[('mc_entropy',1)]),
      ('Energy',[('energy_score',1)]),('MaxLogit',[('max_logit',-1)]),
      ('Guardrail',[('guardrailpp_utility_dense_gap',1)]),
      ('Fusion',[('guardrailpp_utility_dense_gap',1),('energy_score',1)])]
def get(g,cs): return fuse(g,cs) if len(cs)>1 else col(g,cs[0][0],cs[0][1])
seeds=['42','137','256']; BBS=['b0','b1','b2']

# ---- confident-failure AUROC @0.85 : per (bb,seed) mean over conditions ----
auroc_vals=collections.defaultdict(list)
for bb in BBS:
    for seed in seeds:
        g=load(bb,seed); groups=collections.defaultdict(list)
        for x in g: groups[x['domain']].append(x)
        cm=collections.defaultdict(list)
        for dom,gg in groups.items():
            msps=[f(x['student_msp']) for x in gg]; risks=[f(x['student_risk']) for x in gg]
            idx=[i for i in range(len(gg)) if msps[i]>=0.85]
            if len(idx)<10: continue
            cr=sorted(risks[i] for i in idx); cut=cr[int(0.8*len(cr))]
            labels=[1 if risks[i]>=cut else 0 for i in idx]
            for n,cs in METH:
                sc=get(gg,cs); a=auroc([sc[i] for i in idx],labels)
                if a is not None: cm[n].append(a)
        for n,_ in METH:
            if cm[n]: auroc_vals[n].append(st.mean(cm[n]))

# ---- deferral benefit recovered vs budget ----
budgets=[0.05,0.1,0.2,0.3,0.5]
def benr(scores,ben,bud):
    n=len(scores);k=max(1,int(round(bud*n)))
    order=sorted(range(n),key=lambda i:-scores[i])[:k]
    tot=sum(b for b in ben if b>0)
    return sum(max(0,ben[i]) for i in order)/tot if tot>0 else None
defer=collections.defaultdict(lambda: collections.defaultdict(list))
for bb in BBS:
    for seed in seeds:
        g=load(bb,seed); ben=[f(x.get('teacher_benefit')) or 0 for x in g]
        for n,cs in METH:
            sc=get(g,cs)
            if any(s is None for s in sc): continue
            for bud in budgets:
                r=benr(sc,ben,bud)
                if r is not None: defer[n][bud].append(r)

# ---- tidy CSV ----
outdir=os.path.join(REPO,"src","analysis","acdc_fusion_multiseed"); os.makedirs(outdir,exist_ok=True)
with open(os.path.join(outdir,"fusion_multiseed_results.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["metric","method","x","mean","std","n"])
    for n,_ in METH:
        v=auroc_vals[n]; w.writerow(["conffail_auroc@0.85",n,0.85,round(st.mean(v),4),round(st.pstdev(v),4),len(v)])
    for n,_ in METH:
        for bud in budgets:
            v=defer[n][bud]
            if v: w.writerow(["deferral_benefit_recovered",n,bud,round(st.mean(v),4),round(st.pstdev(v),4),len(v)])

# ---- figure: 2 clean panels ----
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig,(axL,axR)=plt.subplots(1,2,figsize=(11,4.2))
names=[n for n,_ in METH]
order=sorted(names,key=lambda n:st.mean(auroc_vals[n]))
means=[st.mean(auroc_vals[n]) for n in order]; stds=[st.pstdev(auroc_vals[n]) for n in order]
colors=["#c44e52" if n=="Fusion" else ("#4c72b0" if n in("Guardrail",) else "#b0b0b0") for n in order]
axL.barh(order,means,xerr=stds,color=colors,height=0.62,error_kw=dict(lw=1,capsize=3,ecolor="#555"))
axL.set_xlim(0.5,0.70); axL.set_xlabel("Confident-failure AUROC @ MSP≥0.85")
axL.set_title("Detection under shift (ACDC, 3 backbones × 3 seeds)",fontsize=10.5)
for i,(m,s) in enumerate(zip(means,stds)): axL.text(m+s+0.003,i,f"{m:.3f}",va="center",fontsize=8.5,color="#333")
style={"Fusion":("#c44e52","-","o"),"Guardrail":("#4c72b0","-","s"),"Energy":("#55a868","--","^"),
       "Temp-scaling":("#8172b3",":","D"),"MSP":("#999999",":","x")}
for n in ["Fusion","Guardrail","Energy","Temp-scaling","MSP"]:
    ys=[st.mean(defer[n][b]) for b in budgets]; c,ls,mk=style[n]
    axR.plot(budgets,ys,ls,marker=mk,color=c,label=n,lw=1.8,ms=5)
axR.set_xlabel("Teacher-call budget (fraction deferred)"); axR.set_ylabel("Benefit recovered")
axR.set_title("Deployment: teacher-deferral efficiency",fontsize=10.5)
axR.legend(frameon=False,fontsize=9,loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(outdir,"fig_fusion_summary.png"),dpi=150,bbox_inches="tight")
print("wrote",outdir)
print("AUROC@0.85:", {n:round(st.mean(auroc_vals[n]),3) for n,_ in METH})
