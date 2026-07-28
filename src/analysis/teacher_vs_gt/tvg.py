import csv,math,random
random.seed(7)
HEADS=["tmulti","gtdis","gtgap","tdis","tgap"]
SCORE_COL={"tmulti":"guardrailpp_utility_dense_gap","gtgap":"guardrailpp_utility_dense_gap","tgap":"guardrailpp_utility_dense_gap","gtdis":"guardrailpp_utility_dense_bce","tdis":"guardrailpp_utility_dense_bce"}
ACDC_CONDS=["fog","night","rain","snow"]
def f(x):
    try:return float(x)
    except:return float('nan')
def load(ds,head):
    return list(csv.DictReader(open(f"{ds}_{head}.csv")))
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
def quant80(vals):
    xs=sorted(vals);h=(len(xs)-1)*0.8;lo=int(h);return xs[lo]+(h-lo)*(xs[min(lo+1,len(xs)-1)]-xs[lo])

# validate guardrail_risk == correct aliased column
def validate():
    for ds in ["acdc"]:
        for head,col in [("gtdis","guardrailpp_utility_dense_bce"),("gtgap","guardrailpp_utility_dense_gap"),("tgap","guardrailpp_utility_dense_gap"),("tdis","guardrailpp_utility_dense_bce")]:
            r=load(ds,head)[0]
            gr=f(r["guardrail_risk"]);cc=f(r.get(col,"nan"))
            print(f"  {head}: guardrail_risk={gr:.4f} vs {col}={cc:.4f}  {'OK' if abs(gr-cc)<1e-6 else 'MISMATCH'}")
print("VALIDATION (guardrail_risk == per-head aliased score):"); validate()

# build cells: list of (name, [ {img: {label, scores{head}}} ]) aligned across heads by image_id
def build_cells(thr):
    cells=[]
    # ACDC per condition; BDD/IDD single
    plan=[("acdc",c) for c in ACDC_CONDS]+[("bdd",None),("idd",None)]
    for ds,cond in plan:
        # load all heads keyed by image_id
        byhead={h:{r["image_id"]:r for r in load(ds,h) if (cond is None or r["domain"]==cond)} for h in HEADS}
        ids=set.intersection(*[set(byhead[h]) for h in HEADS])
        ids=sorted(ids)
        # student_risk/msp from tmulti (identical across heads)
        base=byhead["tmulti"]
        cut=quant80([f(base[i]["student_risk"]) for i in ids])
        conf=[i for i in ids if f(base[i]["student_msp"])>=thr]
        lab=[1 if f(base[i]["student_risk"])>=cut else 0 for i in conf]
        if sum(lab)==0 or sum(lab)==len(lab): continue
        scores={h:[f(byhead[h][i][SCORE_COL[h]]) for i in conf] for h in HEADS}
        cells.append((f"{ds}:{cond or 'all'}",lab,scores))
    return cells

def pooled(cells,h,resample=None):
    vs=[]
    for name,lab,sc in cells:
        s=sc[h];l=lab
        if resample is not None:
            idx=resample[name];s=[sc[h][i] for i in idx];l=[lab[i] for i in idx]
        a=auroc(s,l)
        if a==a:vs.append(a)
    return sum(vs)/len(vs)

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

def analyze(thr):
    cells=build_cells(thr)
    print(f"\n================ TEACHER vs GT — CF-AUROC @ MSP>={thr} (mit-b1, seed42, n={len(cells)} cells) ================")
    print("  per-cell AUROC:")
    print("   cell           "+" ".join(f"{h:>7s}" for h in HEADS))
    for name,lab,sc in cells:
        print(f"   {name:14s} "+" ".join(f"{auroc(sc[h],lab):.3f}  " for h in HEADS)+f"  (n={len(lab)},f={sum(lab)})")
    print("  pooled: "+" ".join(f"{h}={pooled(cells,h):.3f}" for h in HEADS))
    # bootstrap paired Δ
    NB=2000
    pairs=[("tmulti","gtdis"),("tmulti","gtgap"),("tgap","gtgap"),("tdis","gtdis"),("tmulti","bestGT")]
    # precompute per-cell bestGT score arrays (max of gtdis,gtgap per image)? For AUROC of "best GT" per cell we take the better GT head's AUROC per cell
    boot={p:[] for p in pairs}
    for _ in range(NB):
        resample={name:[random.randrange(len(lab)) for _ in range(len(lab))] for name,lab,_ in cells}
        for a,b in pairs:
            if b=="bestGT":
                # per bootstrap: pooled(tmulti) - pooled(best-of-gt per cell)
                va=pooled(cells,"tmulti",resample)
                # best GT: for each cell pick max(gtdis,gtgap) auroc under resample
                bg=[]
                for name,lab,sc in cells:
                    idx=resample[name]
                    ad=auroc([sc["gtdis"][i] for i in idx],[lab[i] for i in idx])
                    ag=auroc([sc["gtgap"][i] for i in idx],[lab[i] for i in idx])
                    bg.append(max(ad,ag))
                boot[(a,b)].append(va-sum(bg)/len(bg))
            else:
                boot[(a,b)].append(pooled(cells,a,resample)-pooled(cells,b,resample))
    # point Δ + per-cell wins + wilcoxon
    print("  paired  A vs B:        ΔAUROC  95%CI            wins  wilcoxon_p(n)")
    for a,b in pairs:
        if b=="bestGT":
            perc=[]
            for name,lab,sc in cells:
                bg=max(auroc(sc["gtdis"],lab),auroc(sc["gtgap"],lab))
                perc.append(auroc(sc["tmulti"],lab)-bg)
        else:
            perc=[auroc(sc[a],lab)-auroc(sc[b],lab) for name,lab,sc in cells]
        md=sum(perc)/len(perc);bl=sorted(boot[(a,b)]);lo=bl[int(0.025*len(bl))];hi=bl[int(0.975*len(bl))]
        wins=sum(1 for x in perc if x>0)
        print(f"   {a:>7s} vs {b:<8s}   {md:+.3f}  [{lo:+.3f},{hi:+.3f}]   {wins}/{len(perc)}   {wilcoxon(perc):.3f}")
analyze(0.85)
analyze(0.95)
