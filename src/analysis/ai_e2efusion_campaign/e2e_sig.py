import csv,math,os
from collections import defaultdict
CONDS=["fog","night","rain","snow"]; BBS=["b0","b1","b2"]; SEEDS=["7","42","99","137","256"]
def f(x):
    try:return float(x)
    except:return float('nan')
def ranks(x):
    order=sorted(range(len(x)),key=lambda i:x[i]);rk=[0.0]*len(x);i=0
    while i<len(order):
        j=i
        while j+1<len(order) and x[order[j+1]]==x[order[i]]:j+=1
        for k in range(i,j+1):rk[order[k]]=(i+j)/2.0
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
        for k in range(i,j+1):rk[ar[k]]=(i+j)/2.0+1
        i=j+1
    Wp=sum(rk[i] for i in range(n) if d[i]>0);mu=n*(n+1)/4;sig=math.sqrt(n*(n+1)*(2*n+1)/24)
    from math import erf;z=(Wp-mu)/sig;return 2*(1-0.5*(1+erf(abs(z)/math.sqrt(2))))
def file_auroc(path):
    rows=[r for r in csv.DictReader(open(path)) if r["domain"] in CONDS]
    if not rows: return None
    per={}
    for d in CONDS:
        rd=[r for r in rows if r["domain"]==d]
        if len(rd)<10: continue
        xs=sorted(f(r["student_risk"]) for r in rd);h=(len(xs)-1)*0.8;lo=int(h)
        cut=xs[lo]+(h-lo)*(xs[min(lo+1,len(xs)-1)]-xs[lo])
        conf=[i for i,r in enumerate(rd) if f(r["student_msp"])>=0.85]
        lab=[1 if f(rd[i]["student_risk"])>=cut else 0 for i in conf]
        if sum(lab)==0 or sum(lab)==len(lab): continue
        gr=[f(rd[i]["guardrail_risk"]) for i in conf]; en=[f(rd[i]["energy_score"]) for i in conf]
        rg,re=ranks(gr),ranks(en); fus=[(rg[i]+re[i])/2 for i in range(len(conf))]
        per[d]={"learned":auroc(gr,lab),"energy":auroc(en,lab),"fusion":auroc(fus,lab)}
    if not per: return None
    return {k:sum(per[d][k] for d in per)/len(per) for k in ["learned","energy","fusion"]}

# gather per (seed,bb) mean AUROC for conffeat and baseline
def gather(prefix, dir_):
    D={}
    for bb in BBS:
        for s in SEEDS:
            p=f"{dir_}/{prefix}_{bb}_{s}.csv"
            if os.path.exists(p):
                a=file_auroc(p)
                if a: D[(bb,s)]=a
    return D
cf=gather("cf","acdc_conffeat_per_image")
base=gather("base","acdc_baseline5_per_image")
print(f"conffeat runs: {len(cf)}   baseline runs: {len(base)}")
keys=sorted(set(cf)&set(base))
print(f"paired (seed,bb) cells: {len(keys)}\n")

def report(name, pairs):
    md=sum(pairs)/len(pairs); wins=sum(1 for x in pairs if x>0)
    print(f"  {name:42s} Δ={md:+.4f}  wilcoxon_p={wilcoxon(pairs):.3f}  wins {wins}/{len(pairs)}")

print("### e2e-fusion (conf-features) vs energy — confident-failure AUROC@0.85, paired over 15 seed×bb")
report("conffeat LEARNED score vs energy", [cf[k]["learned"]-cf[k]["energy"] for k in keys])
report("conffeat FUSION(+energy) vs energy", [cf[k]["fusion"]-cf[k]["energy"] for k in keys])
print("\n### MATCHED ABLATION — does adding energy/max-logit input channels help the learned score?")
report("conffeat LEARNED vs baseline LEARNED", [cf[k]["learned"]-base[k]["learned"] for k in keys])
print("\n### means (pooled over 15):")
for tag,D in [("conffeat",cf),("baseline",base)]:
    print(f"  {tag}: learned={sum(D[k]['learned'] for k in keys)/len(keys):.3f}  fusion={sum(D[k]['fusion'] for k in keys)/len(keys):.3f}  energy={sum(D[k]['energy'] for k in keys)/len(keys):.3f}")
