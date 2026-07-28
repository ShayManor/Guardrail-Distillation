import csv, math
from collections import defaultdict
BBS=["b0","b1","b2"]; CONDS=["fog","night","rain","snow"]
def f(x):
    try:return float(x)
    except:return float('nan')
def load(bb):return [r for r in csv.DictReader(open(f"{bb}_per_image.csv")) if r["domain"] in CONDS]
def ranks(x):
    order=sorted(range(len(x)),key=lambda i:x[i]);rk=[0.0]*len(x);i=0
    while i<len(order):
        j=i
        while j+1<len(order) and x[order[j+1]]==x[order[i]]:j+=1
        r=(i+j)/2.0
        for k in range(i,j+1):rk[order[k]]=r
        i=j+1
    return rk
def auroc(s,l):
    p=sum(l);n=len(l)-p
    if p==0 or n==0:return float('nan')
    rk=ranks(s);sp=sum(rk[i]+1 for i in range(len(l)) if l[i]);return (sp-p*(p+1)/2)/(p*n)
def wilcoxon(d):
    d=[x for x in d if x!=0];n=len(d)
    if n<1:return float('nan')
    ar=sorted(range(n),key=lambda i:abs(d[i]));rk=[0.0]*n;i=0
    while i<n:
        j=i
        while j+1<n and abs(d[ar[j+1]])==abs(d[ar[i]]):j+=1
        r=(i+j)/2.0+1
        for k in range(i,j+1):rk[ar[k]]=r
        i=j+1
    Wp=sum(rk[i] for i in range(n) if d[i]>0);mu=n*(n+1)/4;sig=math.sqrt(n*(n+1)*(2*n+1)/24)
    from math import erf;z=(Wp-mu)/sig;return 2*(1-0.5*(1+erf(abs(z)/math.sqrt(2))))

# ---- OPTION 2: threshold sweep ----
def dscores(rd,wgt=0.5):
    gr=[f(r["guardrail_risk"]) for r in rd];en=[f(r["energy_score"]) for r in rd]
    rg,re=ranks(gr),ranks(en);fus=[wgt*rg[i]+(1-wgt)*re[i] for i in range(len(rd))]
    return {"guardrail":gr,"energy":en,"max_logit":[-f(r["max_logit"]) for r in rd],
      "deep_ensemble":[f(r["ensemble_entropy"]) for r in rd],"temp_msp":[-f(r["temp_msp"]) for r in rd],
      "msp":[-f(r["student_msp"]) for r in rd],"fusion":fus}
def pooled_at(thr,wgt=0.5,methods=None):
    per=defaultdict(list)
    for bb in BBS:
        rows=load(bb)
        for d in CONDS:
            rd=[r for r in rows if r["domain"]==d]
            xs=sorted(f(r["student_risk"]) for r in rd);h=(len(xs)-1)*0.80;lo=int(h)
            cut=xs[lo]+(h-lo)*(xs[min(lo+1,len(xs)-1)]-xs[lo])
            S=dscores(rd,wgt);conf=[i for i,r in enumerate(rd) if f(r["student_msp"])>=thr]
            lab=[1 if f(rd[i]["student_risk"])>=cut else 0 for i in conf]
            if sum(lab)==0 or sum(lab)==len(lab):continue
            for m in (methods or S):per[m].append(auroc([S[m][i] for i in conf],lab))
    return {m:sum(v)/len(v) for m,v in per.items()}
print("### OPTION 2 — AUROC vs MSP threshold (pooled 3bb x 4cond)")
print("thr   fusion energy guard  DeepEns temp  msp")
for thr in [0.50,0.70,0.80,0.85,0.90,0.95,0.97]:
    p=pooled_at(thr,methods=["fusion","energy","guardrail","deep_ensemble","temp_msp","msp"])
    print(f"{thr:.2f}  "+" ".join(f"{p[m]:.3f}" for m in ["fusion","energy","guardrail","deep_ensemble","temp_msp","msp"]))

# ---- OPTION 5: fusion weight robustness ----
print("\n### OPTION 5 — fusion weight sensitivity (AUROC@0.85; w=0 pure energy, w=1 pure guardrail)")
print("w      AUROC")
for w in [0.0,0.25,0.5,0.75,1.0]:
    print(f"{w:.2f}   {pooled_at(0.85,wgt=w,methods=['fusion'])['fusion']:.3f}")

# ---- OPTION 3: cost / latency ----
print("\n### OPTION 3 — inference cost (mean latency ms/img, + effective forward passes)")
def meanlat(bb):
    rows=list(csv.DictReader(open(f"{bb}_latency_samples.csv")))
    def mean(c):
        v=[f(r[c]) for r in rows if r[c] not in ("","nan")];return sum(v)/len(v) if v else float('nan')
    return mean("student_latency_ms"),mean("guardrail_latency_ms"),mean("teacher_latency_ms")
print("bb  student  guardrail(=stu+head)  teacher   | passes: MSP/energy/maxlogit=1x(free)  guardrail=1x+head  MCdrop=5x  DeepEns=3x  teacher=1x(big)")
for bb in BBS:
    s,g,t=meanlat(bb);print(f"{bb}  {s:6.2f}   {g:6.2f}                {t:6.2f}    | guardrail overhead vs student = {(g/s-1)*100:+.0f}%   teacher/student = {t/s:.1f}x")

# ---- OPTION 4: AURC significance (lower=better) ----
print("\n### OPTION 4 — AURC paired significance (Wilcoxon over 12 cells; lower=better)")
AM={"energy":"neg_energy","guardrail":"guardrail","deep_ensemble":"deep_ensemble","fusion":"fusion_guard_energy","max_logit":"max_logit","temp_msp":"temp_msp","msp":"msp"}
cell=defaultdict(dict)
for bb in BBS:
    for r in csv.DictReader(open(f"{bb}_risk_coverage.csv")):
        if r["domain"] in CONDS:cell[(bb,r["domain"])][r["method"]]=f(r["aurc"])
keys=list(cell)
def aurc_mean(m):return sum(cell[k][AM[m]] for k in keys)/len(keys)
print("  method        mean_AURC")
for m in sorted(AM,key=aurc_mean):print(f"   {m:13s} {aurc_mean(m):.4f}")
print("  paired A vs B (Δ=A-B, negative=A better):   Δ       wilcoxon_p   A_better_in")
for a,b in [("energy","guardrail"),("guardrail","deep_ensemble"),("fusion","energy"),("guardrail","temp_msp"),("deep_ensemble","temp_msp")]:
    dl=[cell[k][AM[a]]-cell[k][AM[b]] for k in keys];md=sum(dl)/len(dl)
    better=sum(1 for x in dl if x<0);print(f"   {a:>13s} vs {b:<13s} {md:+.4f}  {wilcoxon(dl):.3f}       {better}/{len(dl)}")

# ---- OPTION 6: calibration ECE ----
print("\n### OPTION 6 — Expected Calibration Error (pooled; lower=better calibrated)")
ece=defaultdict(list)
for bb in BBS:
    grp=defaultdict(lambda:defaultdict(list))
    for r in csv.DictReader(open(f"{bb}_calibration_bins.csv")):
        if r["domain"] not in CONDS: continue
        c=f(r["count"]); ag=r["abs_gap"]
        if c and c>0 and ag not in ("","nan"): grp[(r["domain"],r["method"])]["c"].append((c,float(ag)))
    for (d,m),v in grp.items():
        tot=sum(c for c,_ in v["c"]);
        if tot>0:ece[m].append(sum(c*g for c,g in v["c"])/tot)
for m in sorted(ece,key=lambda m:sum(ece[m])/len(ece[m])):
    print(f"   {m:14s} ECE={sum(ece[m])/len(ece[m]):.4f}")
