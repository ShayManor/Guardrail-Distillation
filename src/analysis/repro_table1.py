import csv, math
from collections import defaultdict
CF="combined_all/per_image.csv"
MODES={"dense_multi":"gap","gt_disagree":"bce","gt_risk":"gap","dense_disagree":"bce","dense_gap":"gap"}
DS=["acdc","bdd","idd"]
def f(x):
    try:return float(x)
    except:return float('nan')
# stream + dedup (prefer rows with max_logit populated), keep b1, needed modes/datasets
store={}  # (ds,mode,seed,image_id) -> row dict (subset)
with open(CF) as fh:
    r=csv.DictReader(fh)
    for row in r:
        if row.get("backbone")!="b1": continue
        ds=row.get("dataset"); mode=row.get("supervision_type")
        if ds not in DS or mode not in MODES: continue
        if ds=="acdc" and row.get("domain")!="all": continue
        key=(ds,mode,row.get("seed"),row.get("image_id"))
        has_new = row.get("max_logit","")!=""
        if key in store and not has_new: continue
        store[key]={"cond":row.get("condition"),"msp":f(row.get("student_msp")),
            "risk":f(row.get("student_risk")),"gap":f(row.get("guardrailpp_utility_dense_gap")),
            "bce":f(row.get("guardrailpp_utility_dense_bce")),"maxl":f(row.get("max_logit")),
            "ent":f(row.get("student_entropy")),"mc":f(row.get("mc_entropy"))}
# regroup
by=defaultdict(list)  # (ds,mode,seed)->[rows]
for (ds,mode,seed,img),v in store.items():
    by[(ds,mode,seed)].append(v)

def ranks(x):
    o=sorted(range(len(x)),key=lambda i:x[i]);rk=[0.0]*len(x);i=0
    while i<len(o):
        j=i
        while j+1<len(o) and x[o[j+1]]==x[o[i]]:j+=1
        for k in range(i,j+1):rk[o[k]]=(i+j)/2.0
        i=j+1
    return rk
def auroc(y,s):
    p=sum(y);n=len(y)-p
    if p==0 or n==0:return float('nan')
    rk=ranks(s);sp=sum(rk[i]+1 for i in range(len(y)) if y[i]);return (sp-p*(p+1)/2)/(p*n)
def quantile(vals,qq):
    xs=sorted(vals);h=(len(xs)-1)*qq;lo=int(h);return xs[lo]+(h-lo)*(xs[min(lo+1,len(xs)-1)]-xs[lo])

def cf_auroc(rows,col,hi,thr=0.85,q=0.20):
    s=[r for r in rows if r["msp"]>=thr and not math.isnan(r[col]) and not math.isnan(r["risk"])]
    if len(s)<20:return float('nan')
    lbl=[r["risk"] for r in s]
    k=max(1,round(q*len(s)));cut=sorted(lbl)[len(s)-k]
    y=[1 if v>=cut else 0 for v in lbl]
    if sum(y)==0 or sum(y)==len(y):return float('nan')
    sc=[r[col] for r in s]
    if not hi: sc=[-x for x in sc]
    return auroc(y,sc)
def cf_auroc_strat(rows,col,hi,thr=0.85,q=0.20):
    s=[r for r in rows if r["msp"]>=thr and not math.isnan(r[col]) and not math.isnan(r["risk"])]
    if len(s)<20:return float('nan')
    cond_cut={}
    bycond=defaultdict(list)
    for r in s: bycond[r["cond"]].append(r["risk"])
    for c,v in bycond.items(): cond_cut[c]=quantile(v,1.0-q)
    y=[1 if r["risk"]>=cond_cut[r["cond"]] else 0 for r in s]
    if sum(y)==0 or sum(y)==len(y):return float('nan')
    sc=[r[col] for r in s]
    if not hi: sc=[-x for x in sc]
    return auroc(y,sc)

def metric(ds,rows,col,hi):
    return cf_auroc_strat(rows,col,hi) if ds=="acdc" else cf_auroc(rows,col,hi)

def per_seed_mean(ds,mode,col,hi):
    seeds=sorted({s for (d,m,s) in by if d==ds and m==mode})
    vals=[]
    for sd in seeds:
        v=metric(ds,by[(ds,mode,sd)],col,hi)
        if v==v: vals.append(v)
    return (sum(vals)/len(vals) if vals else float('nan')), vals, seeds

print("=== REPRODUCTION of paper Table 1 (CF-AUROC @ MSP>=0.85, mit-b1, per-seed mean) ===")
print("paper: ACDC T-Multi .604 GT-Dis .589 GT-Gap .599 | IDD .603/.575/.592 | BDD .758/.738/.716")
print()
labels={"dense_multi":"T-Multi","gt_disagree":"GT-Dis","gt_risk":"GT-Gap","dense_gap":"T-Gap","dense_disagree":"T-Dis"}
for ds in DS:
    parts=[]
    for mode in ["dense_multi","gt_disagree","gt_risk","dense_gap","dense_disagree"]:
        m,vals,seeds=per_seed_mean(ds,mode,MODES[mode],True)
        parts.append(f"{labels[mode]}={m:.3f}(n={len(vals)})")
    # post-hoc from dense_multi rows
    msp,_,_=per_seed_mean(ds,"dense_multi","msp_placeholder",False) if False else (None,None,None)
    print(f"{ds.upper():5s}: "+"  ".join(parts))
# post-hoc baselines (use dense_multi rows, 3 seeds)
print("\n=== post-hoc baselines (dense_multi rows, per-seed mean) ===")
for ds in DS:
    out=[]
    for lab,col,hi in [("MSP","msp",False),("MaxL","maxl",True),("MC","mc",True),("Ent","ent",True)]:
        # msp column is 'msp' -> stored as? we stored 'msp' as student_msp; but cf uses student_msp>=thr AND scores. For MSP score use student_msp negated.
        colname={"msp":"msp","maxl":"maxl","mc":"mc","ent":"ent"}[col]
        seeds=sorted({s for (d,m,s) in by if d==ds and m=="dense_multi"})
        vals=[]
        for sd in seeds:
            v=metric(ds,by[(ds,"dense_multi",sd)],colname,hi)
            if v==v: vals.append(v)
        out.append(f"{lab}={sum(vals)/len(vals):.3f}" if vals else f"{lab}=nan")
    print(f"{ds.upper():5s}: "+"  ".join(out))
