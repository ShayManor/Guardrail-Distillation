import csv,math,os
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
def auroc(s,l):
    p=sum(l);n=len(l)-p
    if p==0 or n==0:return float('nan')
    rk=ranks(s);sp=sum(rk[i]+1 for i in range(len(l)) if l[i]);return (sp-p*(p+1)/2)/(p*n)
SC={"guardrail":lambda r:f(r["guardrail_risk"]),"energy":lambda r:f(r["energy_score"]),
 "max_logit":lambda r:-f(r["max_logit"]),"deep_ensemble":lambda r:f(r["ensemble_entropy"]),
 "mc_dropout":lambda r:f(r["mc_entropy"]),"temp_msp":lambda r:-f(r["temp_msp"]),"msp":lambda r:-f(r["student_msp"])}
def cell(ds,bb,thr):
    thr=float(thr)
    rows=list(csv.DictReader(open(f"{ds}_{bb}_per_image.csv")))
    xs=sorted(f(r["student_risk"]) for r in rows);h=(len(xs)-1)*0.8;lo=int(h)
    cut=xs[lo]+(h-lo)*(xs[min(lo+1,len(xs)-1)]-xs[lo])
    conf=[i for i,r in enumerate(rows) if f(r["student_msp"])>=thr]
    lab=[1 if f(rows[i]["student_risk"])>=cut else 0 for i in conf]
    if sum(lab)==0 or sum(lab)==len(lab): return None,len(conf),sum(lab)
    gr=[f(rows[i]["guardrail_risk"]) for i in conf];en=[f(rows[i]["energy_score"]) for i in conf]
    rg,re=ranks(gr),ranks(en);fus=[(rg[i]+re[i])/2 for i in range(len(conf))]
    d={m:auroc([SC[m](rows[i]) for i in conf],lab) for m in SC}; d["fusion"]=auroc(fus,lab)
    return d,len(conf),sum(lab)
def csvcell(ds,bb,thr):
    for r in csv.DictReader(open(f"{ds}_{bb}_confident_failures.csv")):
        if r["msp_threshold"]==thr:
            return {"guardrail":f(r["guardrail_auroc"]),"energy":f(r["energy_auroc"]),"deep_ensemble":f(r["deep_ensemble_auroc"]),
                    "fusion":f(r["fusion_guard_energy_auroc"]),"max_logit":f(r["max_logit_auroc"]),"msp":f(r["msp_auroc"]),"temp_msp":f(r["temp_msp_auroc"])}
BBS=[bb for bb in ["b0","b1","b2"]]
avail={ds:[bb for bb in BBS if os.path.exists(f"{ds}_{bb}_per_image.csv")] for ds in ["bdd","idd"]}
print("available:",avail)
# validate
maxe=0
for ds in ["bdd","idd"]:
    for bb in avail[ds]:
        m,nc,nf=cell(ds,bb,"0.85")
        if m is None: continue
        t=csvcell(ds,bb,"0.85")
        for k in ["guardrail","energy","deep_ensemble","fusion","max_logit","msp"]:
            maxe=max(maxe,abs(m[k]-t[k]))
print(f"VALIDATION max abs err @0.85 = {maxe:.4f}  {'OK' if maxe<0.02 else 'MISMATCH'}")
METH=["fusion","energy","max_logit","guardrail","deep_ensemble","mc_dropout","temp_msp","msp"]
for thr in ["0.85","0.9"]:
    print(f"\n### Confident-failure AUROC @ MSP>={thr}  (n_conf, n_fail shown)")
    print("dataset bb   "+" ".join(f"{m[:6]:>6s}" for m in METH))
    for ds in ["bdd","idd"]:
        for bb in avail[ds]:
            m,nc,nf=cell(ds,bb,thr)
            if m is None: print(f"{ds} {bb}: (no valid failures nc={nc} nf={nf})"); continue
            print(f"{ds:6s} {bb}   "+" ".join(f"{m[x]:.3f}" for x in METH)+f"   (nc={nc},nf={nf})")
# AURC
print("\n### AURC (lower=better)")
AM={"energy":"neg_energy","guardrail":"guardrail","deep_ensemble":"deep_ensemble","fusion":"fusion_guard_energy","max_logit":"max_logit","temp_msp":"temp_msp","msp":"msp"}
for ds in ["bdd","idd"]:
    for bb in avail[ds]:
        rc={r["method"]:f(r["aurc"]) for r in csv.DictReader(open(f"{ds}_{bb}_risk_coverage.csv"))}
        print(f"{ds} {bb}: "+" ".join(f"{k}={rc.get(v,float('nan')):.3f}" for k,v in [("energy","neg_energy"),("guard","guardrail"),("DE","deep_ensemble"),("fus","fusion_guard_energy"),("temp","temp_msp"),("msp","msp")]))
