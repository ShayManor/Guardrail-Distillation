import csv, math, random
from collections import defaultdict
random.seed(0)
BBS=["b0","b1","b2"]; CONDS=["fog","night","rain","snow"]
def f(x):
    try: return float(x)
    except: return float('nan')
def load(bb): return [r for r in csv.DictReader(open(f"{bb}_per_image.csv")) if r["domain"] in CONDS]

def ranks(x):
    order=sorted(range(len(x)),key=lambda i:x[i]); rk=[0.0]*len(x)
    i=0
    while i<len(order):
        j=i
        while j+1<len(order) and x[order[j+1]]==x[order[i]]: j+=1
        r=(i+j)/2.0
        for k in range(i,j+1): rk[order[k]]=r
        i=j+1
    return rk
def auroc(scores,labels):
    pos=sum(labels); neg=len(labels)-pos
    if pos==0 or neg==0: return float('nan')
    rk=ranks(scores); sp=sum(rk[i]+1 for i in range(len(labels)) if labels[i])
    return (sp - pos*(pos+1)/2)/(pos*neg)

def domain_scores(rows_d):
    # higher = more failure-prone
    gr=[f(r["guardrail_risk"]) for r in rows_d]; en=[f(r["energy_score"]) for r in rows_d]
    rg,re=ranks(gr),ranks(en); fus=[(rg[i]+re[i])/2 for i in range(len(rows_d))]
    S={"guardrail":gr,"energy":en,"max_logit":[-f(r["max_logit"]) for r in rows_d],
       "deep_ensemble":[f(r["ensemble_entropy"]) for r in rows_d],
       "mc_dropout":[f(r["mc_entropy"]) for r in rows_d],
       "temp_msp":[-f(r["temp_msp"]) for r in rows_d],"msp":[-f(r["student_msp"]) for r in rows_d],
       "fusion":fus}
    return S

def cells(bb, thr=0.85):
    rows=load(bb); out={}
    for d in CONDS:
        rd=[r for r in rows if r["domain"]==d]
        risks=sorted(f(r["student_risk"]) for r in rd)
        # numpy-style linear-interp quantile 0.80
        xs=sorted(f(r["student_risk"]) for r in rd); h=(len(xs)-1)*0.80
        lo=int(h); cut=xs[lo]+(h-lo)*(xs[min(lo+1,len(xs)-1)]-xs[lo])
        S=domain_scores(rd)
        conf=[i for i,r in enumerate(rd) if f(r["student_msp"])>=thr]
        lab=[1 if f(rd[i]["student_risk"])>=cut else 0 for i in conf]
        if sum(lab)==0 or sum(lab)==len(lab): continue
        out[d]={m:auroc([S[m][i] for i in conf],lab) for m in S}
    return out

def csv_cells(bb,thr="0.85"):
    o={}
    for r in csv.DictReader(open(f"{bb}_confident_failures.csv")):
        if r["msp_threshold"]==thr and r["domain"] in CONDS:
            o[r["domain"]]={"guardrail":f(r["guardrail_auroc"]),"energy":f(r["energy_auroc"]),
              "deep_ensemble":f(r["deep_ensemble_auroc"]),"fusion":f(r["fusion_guard_energy_auroc"]),
              "max_logit":f(r["max_logit_auroc"]),"msp":f(r["msp_auroc"]),"temp_msp":f(r["temp_msp_auroc"])}
    return o

print("=== VALIDATION: |repro - eval_csv| @0.85 ===")
maxe=0; worst=None
for bb in BBS:
    mine=cells(bb); th=csv_cells(bb)
    for d in CONDS:
        if d not in mine: continue
        for m in ["guardrail","energy","deep_ensemble","fusion","max_logit","msp","temp_msp"]:
            e=abs(mine[d][m]-th[d][m])
            if e>maxe: maxe=e; worst=(bb,d,m,round(mine[d][m],3),round(th[d][m],3))
print(f"  max abs err = {maxe:.4f}  worst={worst}")
print("  ->", "FAITHFUL (proceed)" if maxe<0.02 else "STILL OFF")

# ---------------- SIGNIFICANCE ANALYSIS ----------------
import random as _r
METHODS=["fusion","energy","max_logit","guardrail","deep_ensemble","mc_dropout","temp_msp","msp"]

def build_cells(thr):
    """return list of cells; each = {method:(scores_conf, ), labels} for confident imgs."""
    C=[]
    for bb in BBS:
        rows=load(bb)
        for d in CONDS:
            rd=[r for r in rows if r["domain"]==d]
            xs=sorted(f(r["student_risk"]) for r in rd); h=(len(xs)-1)*0.80
            lo=int(h); cut=xs[lo]+(h-lo)*(xs[min(lo+1,len(xs)-1)]-xs[lo])
            S=domain_scores(rd)
            conf=[i for i,r in enumerate(rd) if f(r["student_msp"])>=thr]
            lab=[1 if f(rd[i]["student_risk"])>=cut else 0 for i in conf]
            if sum(lab)==0 or sum(lab)==len(lab): continue
            cell={m:[S[m][i] for i in conf] for m in METHODS}
            cell["_lab"]=lab
            C.append((bb,d,cell))
    return C

def pooled_auroc(C, m, resample=None):
    vals=[]
    for (_,_,cell) in C:
        lab=cell["_lab"]; sc=cell[m]
        if resample is not None:
            idx=resample[id(cell)]; sc=[sc[i] for i in idx]; lab=[cell["_lab"][i] for i in idx]
        a=auroc(sc,lab)
        if a==a: vals.append(a)
    return sum(vals)/len(vals)

def boot(C, nrep=1000):
    _r.seed(1)
    pooled={m:[] for m in METHODS}
    diffs=defaultdict(list)
    pairs=[("fusion","energy"),("guardrail","energy"),("guardrail","deep_ensemble"),
           ("fusion","deep_ensemble"),("guardrail","temp_msp"),("deep_ensemble","temp_msp")]
    for _ in range(nrep):
        resample={}
        for (_,_,cell) in C:
            n=len(cell["_lab"]); resample[id(cell)]=[_r.randrange(n) for _ in range(n)]
        pa={m:pooled_auroc(C,m,resample) for m in METHODS}
        for m in METHODS: pooled[m].append(pa[m])
        for a,b in pairs: diffs[(a,b)].append(pa[a]-pa[b])
    return pooled,diffs,pairs

def ci(v):
    s=sorted(v); n=len(s); return s[int(0.025*n)], s[int(0.975*n)]

def wilcoxon(deltas):
    d=[x for x in deltas if x!=0]; n=len(d)
    if n<1: return float('nan')
    ar=sorted(range(n), key=lambda i:abs(d[i]))
    rk=[0.0]*n; i=0
    while i<n:
        j=i
        while j+1<n and abs(d[ar[j+1]])==abs(d[ar[i]]): j+=1
        r=(i+j)/2.0+1
        for k in range(i,j+1): rk[ar[k]]=r
        i=j+1
    Wp=sum(rk[i] for i in range(n) if d[i]>0)
    mu=n*(n+1)/4; sig=math.sqrt(n*(n+1)*(2*n+1)/24)
    z=(Wp-mu)/sig
    from math import erf
    p=2*(1-0.5*(1+erf(abs(z)/math.sqrt(2))))
    return p

for thr in [0.85,0.90]:
    C=build_cells(thr)
    print(f"\n================ CONFIDENT-FAILURE AUROC @ MSP>={thr}  (n={len(C)} cells) ================")
    pooled,diffs,pairs=boot(C, nrep=1000)
    print("  method          pooled_AUROC   95%CI")
    for m in sorted(METHODS,key=lambda m:-pooled_auroc(C,m)):
        lo,hi=ci(pooled[m]); print(f"   {m:14s}  {pooled_auroc(C,m):.3f}        [{lo:.3f}, {hi:.3f}]")
    print("  paired  A vs B:  ΔAUROC   95%CI          boot_p   wilcoxon_p(12)  A_wins")
    cellcache={p:[] for p in pairs}
    for (_,_,cell) in C:
        for a,b in pairs:
            cellcache[(a,b)].append(auroc(cell[a],cell["_lab"])-auroc(cell[b],cell["_lab"]))
    for a,b in pairs:
        dl=diffs[(a,b)]; lo,hi=ci(dl); md=sum(dl)/len(dl)
        bp=2*min(sum(1 for x in dl if x<=0),sum(1 for x in dl if x>=0))/len(dl)
        cd=cellcache[(a,b)]; wp=wilcoxon(cd); wins=sum(1 for x in cd if x>0)
        print(f"   {a:>13s} vs {b:<13s} {md:+.3f}  [{lo:+.3f},{hi:+.3f}]  {bp:.3f}   {wp:.3f}          {wins}/{len(cd)}")
