import csv, math, random
from collections import defaultdict
random.seed(11)
CF="combined_all/per_image.csv"
DS=["acdc","bdd","idd"]
MODES=["dense_multi","gt_disagree","gt_risk","dense_gap","dense_disagree"]
COL={"dense_multi":"gap","gt_disagree":"bce","gt_risk":"gap","dense_gap":"gap","dense_disagree":"bce"}
def f(x):
    try:return float(x)
    except:return float('nan')
# seed-42 rows, keyed by (ds, image_id) -> {mode: score, cond, msp, risk}
img=defaultdict(dict)
with open(CF) as fh:
    for row in csv.DictReader(fh):
        if row.get("backbone")!="b1" or row.get("seed")!="42": continue
        ds=row.get("dataset"); mode=row.get("supervision_type")
        if ds not in DS or mode not in MODES: continue
        if ds=="acdc" and row.get("domain")!="all": continue
        iid=row.get("image_id"); d=img[(ds,iid)]
        d["cond"]=row.get("condition"); d["msp"]=f(row.get("student_msp")); d["risk"]=f(row.get("student_risk"))
        d[COL[mode]+"_"+mode]=f(row.get("guardrailpp_utility_dense_"+("gap" if COL[mode]=="gap" else "bce")))
        # store per-mode score explicitly
        d[mode]=f(row.get("guardrailpp_utility_dense_"+("gap" if COL[mode]=="gap" else "bce")))
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
def labels(ds, sub, idx):
    risk=[sub[i]["risk"] for i in idx]
    if ds=="acdc":
        bycond=defaultdict(list)
        for i in idx: bycond[sub[i]["cond"]].append(sub[i]["risk"])
        cut={c:quantile(v,0.80) for c,v in bycond.items()}
        return [1 if sub[i]["risk"]>=cut[sub[i]["cond"]] else 0 for i in idx]
    else:
        k=max(1,round(0.20*len(idx)));c=sorted(risk)[len(idx)-k]
        return [1 if r>=c else 0 for r in risk]
def cell_auroc(ds, sub, idx, mode):
    y=labels(ds,sub,idx)
    if sum(y)==0 or sum(y)==len(y): return float('nan')
    return auroc(y,[sub[i][mode] for i in idx])

# build confident subsets per dataset (seed42)
DATA={}
for ds in DS:
    rows=[img[(d,i)] for (d,i) in img if d==ds]
    conf=[r for r in rows if r.get("msp",float('nan'))>=0.85 and all(mode in r and not math.isnan(r[mode]) for mode in MODES) and not math.isnan(r.get("risk",float('nan')))]
    DATA[ds]=conf
    print(f"{ds}: confident n={len(conf)}")

def point(ds,mode):
    sub=DATA[ds];return cell_auroc(ds,sub,list(range(len(sub))),mode)
print("\nseed-42 point AUROC (paper metric):")
for ds in DS:
    print(f" {ds}: "+" ".join(f"{m.split('_')[0][:5]}:{point(ds,m):.3f}" for m in ["dense_multi","gt_disagree","gt_risk","dense_gap","dense_disagree"]))

NB=3000
pairs=[("dense_multi","gt_disagree"),("dense_multi","gt_risk"),("dense_gap","gt_risk"),("dense_disagree","gt_disagree"),("dense_multi","BESTGT")]
name={"dense_multi":"T-Multi","gt_disagree":"GT-Dis","gt_risk":"GT-Gap","dense_gap":"T-Gap","dense_disagree":"T-Dis","BESTGT":"best-GT"}
print(f"\n=== paired image-bootstrap (seed42, paper metric, {NB} reps) ===")
for a,b in pairs:
    print(f"\n {name[a]} vs {name[b]}:")
    pooled=[]
    for ds in DS:
        sub=DATA[ds];n=len(sub);diffs=[]
        for _ in range(NB):
            idx=[random.randrange(n) for _ in range(n)]
            aa=cell_auroc(ds,sub,idx,a)
            if b=="BESTGT":
                bb=max(cell_auroc(ds,sub,idx,"gt_disagree"),cell_auroc(ds,sub,idx,"gt_risk"))
            else:
                bb=cell_auroc(ds,sub,idx,b)
            if aa==aa and bb==bb: diffs.append(aa-bb)
        diffs.sort();md=sum(diffs)/len(diffs);lo=diffs[int(0.025*len(diffs))];hi=diffs[int(0.975*len(diffs))]
        p=2*min(sum(1 for x in diffs if x<=0),sum(1 for x in diffs if x>=0))/len(diffs)
        sig="**" if (lo>0 or hi<0) else ""
        print(f"   {ds:5s} Δ={md:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f} {sig}")
        pooled.append(diffs)
    # pooled across datasets: mean of the 3 per-rep (align by rep index)
    m=min(len(d) for d in pooled)
    pm=sorted(sum(pooled[k][i] for k in range(3))/3 for i in range(m))
    md=sum(pm)/len(pm);lo=pm[int(0.025*len(pm))];hi=pm[int(0.975*len(pm))]
    p=2*min(sum(1 for x in pm if x<=0),sum(1 for x in pm if x>=0))/len(pm)
    print(f"   POOLED Δ={md:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f} {'**' if (lo>0 or hi<0) else ''}")
