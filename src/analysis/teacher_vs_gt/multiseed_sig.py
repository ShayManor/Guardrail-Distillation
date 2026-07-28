import csv, math, random
from collections import defaultdict
random.seed(3)
DS=["acdc","bdd","idd"]; SEEDS=["42","137","256"]
def f(x):
    try:return float(x)
    except:return float('nan')
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
def quantile(v,q):
    xs=sorted(v);h=(len(xs)-1)*q;lo=int(h);return xs[lo]+(h-lo)*(xs[min(lo+1,len(xs)-1)]-xs[lo])
def cf(rows,hi=True,thr=0.85,q=0.20):  # bdd/idd
    s=[r for r in rows if r["msp"]>=thr and not math.isnan(r["gap"]) and not math.isnan(r["risk"])]
    if len(s)<20:return float('nan')
    lbl=[r["risk"] for r in s];k=max(1,round(q*len(s)));cut=sorted(lbl)[len(s)-k]
    y=[1 if v>=cut else 0 for v in lbl]
    if sum(y) in (0,len(y)):return float('nan')
    return auroc(y,[r["gap"] for r in s])
def cf_strat(rows,thr=0.85,q=0.20):  # acdc
    s=[r for r in rows if r["msp"]>=thr and not math.isnan(r["gap"]) and not math.isnan(r["risk"])]
    if len(s)<20:return float('nan')
    byc=defaultdict(list)
    for r in s:byc[r["cond"]].append(r["risk"])
    cut={c:quantile(v,0.80) for c,v in byc.items()}
    y=[1 if r["risk"]>=cut[r["cond"]] else 0 for r in s]
    if sum(y) in (0,len(y)):return float('nan')
    return auroc(y,[r["gap"] for r in s])
def metric(ds,rows): return cf_strat(rows) if ds=="acdc" else cf(rows)

# load seed-42 from combined_all; seeds 137/256 from pulled files
def load_seed42():
    out={}  # (ds,head)->rows
    modecol={"tgap":"dense_gap","gtgap":"gt_risk"}
    tmp=defaultdict(list)
    with open("combined_all/per_image.csv") as fh:
        for row in csv.DictReader(fh):
            if row.get("backbone")!="b1" or row.get("seed")!="42":continue
            ds=row.get("dataset"); m=row.get("supervision_type")
            if ds not in DS or m not in ("dense_gap","gt_risk"):continue
            if ds=="acdc" and row.get("domain")!="all":continue
            head="tgap" if m=="dense_gap" else "gtgap"
            tmp[(ds,head)].append({"cond":row.get("condition"),"msp":f(row.get("student_msp")),
                "risk":f(row.get("student_risk")),"gap":f(row.get("guardrailpp_utility_dense_gap"))})
    return tmp
def load_file(ds,head,seed):
    rows=[]
    for r in csv.DictReader(open(f"teacher_vs_gt/multiseed/{ds}_{head}_{seed}.csv")):
        rows.append({"cond":r.get("condition"),"msp":f(r.get("student_msp")),
            "risk":f(r.get("student_risk")),"gap":f(r.get("guardrailpp_utility_dense_gap"))})
    return rows
s42=load_seed42()
def get(ds,head,seed):
    return s42[(ds,head)] if seed=="42" else load_file(ds,head,seed)

def wilcoxon(d):
    d=[x for x in d if x!=0];n=len(d)
    if n<1:return float('nan')
    ar=sorted(range(n),key=lambda i:abs(d[i]));rk=[0.0]*n;i=0
    while i<n:
        j=i
        while j+1<n and abs(d[ar[j+1]])==abs(d[ar[i]]):j+=1
        for k in range(i,j+1):rk[ar[k]]=(i+j)/2.0+1
        i=j+1
    Wp=sum(rk[i] for i in range(n) if d[i]>0);mu=n*(n+1)/4;sig=math.sqrt(n*(n+1)*(2*n+1)/24)
    from math import erf;z=(Wp-mu)/sig;return 2*(1-0.5*(1+erf(abs(z)/math.sqrt(2))))

print("=== MULTI-SEED matched: T-Gap vs GT-Gap CF-AUROC@0.85 (paper metric, mit-b1) ===")
print("dataset  seed   T-Gap   GT-Gap   Δ(T-GT)")
perseed_delta=[]  # per (ds,seed)
dsmean={}
for ds in DS:
    tg=[];gg=[]
    for sd in SEEDS:
        a=metric(ds,get(ds,"tgap",sd)); b=metric(ds,get(ds,"gtgap",sd))
        print(f"{ds:6s}  {sd:4s}   {a:.3f}   {b:.3f}    {a-b:+.3f}")
        tg.append(a);gg.append(b);perseed_delta.append((ds,sd,a-b))
    mt=sum(tg)/3; mg=sum(gg)/3
    st=(sum((x-mt)**2 for x in tg)/2)**.5; sgd=(sum((x-mg)**2 for x in gg)/2)**.5
    dsmean[ds]=(mt,st,mg,sgd)
    print(f"  -> {ds} 3-seed mean: T-Gap {mt:.3f}±{st:.3f}  GT-Gap {mg:.3f}±{sgd:.3f}  Δ={mt-mg:+.3f}")
print("\n=== seed-level paired significance (T-Gap vs GT-Gap) ===")
deltas=[d for _,_,d in perseed_delta]
print(f" pooled over 9 (seed×dataset): meanΔ={sum(deltas)/len(deltas):+.4f}  wins={sum(1 for d in deltas if d>0)}/9  Wilcoxon p={wilcoxon(deltas):.4f}")
for ds in DS:
    dd=[d for x,_,d in perseed_delta if x==ds]
    print(f"   {ds}: per-seed Δ={[round(x,3) for x in dd]}  mean={sum(dd)/3:+.3f}  wins={sum(1 for x in dd if x>0)}/3")
