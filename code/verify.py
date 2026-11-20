import os,sys,json,random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, harness as H
from scipy.stats import fisher_exact, chi2_contingency, ks_2samp, kruskal, mannwhitneyu
ok=lambda c,claim,got: print(f"  {'OK ' if c else 'FAIL'}  {claim:<62} got {got}")
NM={"claude-haiku-5-5":"Haiku 5.5","claude-sonnet-5-5":"Sonnet 5.5","claude-opus-5-5":"Opus 5.5","claude-fable-5-1":"Fable 5.1"}
h=pd.DataFrame([json.loads(l) for l in open("exp2_haiku.jsonl")])
fr=[pd.DataFrame([json.loads(l) for l in open(f)]).assign(label=t) for f,t in
    (("exp3_sonnet.jsonl","Sonnet 5.5"),("exp3_opus.jsonl","Opus 5.5"),("exp3_fable.jsonl","Fable 5.1"))]
h3=h[h.variant.isin(["V1_plain","V3_second","V4_keep"])].assign(label="Haiku 5.5")
d=pd.concat(fr+[h3],ignore_index=True)
cost=pd.DataFrame([json.loads(l) for l in open("cost.jsonl")]); cost["label"]=cost.model.map(NM)
e4=pd.DataFrame([json.loads(l) for l in open("exp4.jsonl")]); e4["label"]=e4.model.map(NM)

print("### PAPER A ###")
tot=sum(len(open(f).readlines()) for f in
    ("pilot.jsonl","pilot2.jsonl","main_haiku.jsonl","exp2_haiku.jsonl","exp3_sonnet.jsonl",
     "exp3_opus.jsonl","exp3_fable.jsonl","cost.jsonl","exp4.jsonl"))
ok(tot==6414,"abstract: 6,414 calls total",tot)
R=h[h.arm=="repeat"].dropna(subset=["give"])
w=R.groupby("aid").give.agg(["std","mean","nunique"])
ok(round(w["std"].mean(),2)==3.60,"4.1: run-to-run SD 3.60",round(w["std"].mean(),2))
ok(int((w["nunique"]==1).sum())==16,"4.1: 16 of 30 agents identical every time",int((w["nunique"]==1).sum()))
ok(round(w["mean"].std(),2)==13.93,"4.1: between-agent SD 13.93",round(w["mean"].std(),2))
rm=[round(R[R.rep==r].give.mean(),2) for r in sorted(R.rep.unique())]
ok(rm==[30.17,29.83,28.67,28.50,29.00],"4.1: replicate means",rm)
T=h[h.arm=="tokens"]
loss=[round(T[T.max_tokens==b].give.isna().mean()*100,1) for b in (16,32,64,128,256,512)]
ok(loss==[86.7,84.0,69.3,2.0,0.0,0.0],"4.2: Haiku loss by budget",loss)
lo=T[T.max_tokens<=64]; hi=T[T.max_tokens>=128]
tab=[[lo.give.notna().sum(),lo.give.isna().sum()],[hi.give.notna().sum(),hi.give.isna().sum()]]
c2,p,_,_=chi2_contingency(tab)
ok(tab[0][0]==90 and tab[1][0]==447,"4.2: 90/450 vs 447/450",f"{tab[0][0]}/450 vs {tab[1][0]}/450")
ok(round(c2,1)==585.1,"4.2: chi-square 585.1",round(c2,1))
D=d[d.arm=="tokens"]
l16={l:round(D[(D.label==l)&(D.max_tokens==16)].give.isna().mean()*100,1) for l in NM.values()}
ok(l16["Opus 5.5"]==100.0 and l16["Haiku 5.5"]==97.8 and l16["Sonnet 5.5"]==3.3 and l16["Fable 5.1"]==1.1,
   "4.2: loss at budget 16 by model",l16)
cm={l:(int(cost[cost.label==l].out_tok.median()),int(cost[cost.label==l].out_tok.quantile(.9)),
       int(cost[cost.label==l].out_tok.max())) for l in NM.values()}
ok(cm["Fable 5.1"][0]==3 and cm["Sonnet 5.5"][0]==3 and cm["Opus 5.5"][0]==49 and cm["Haiku 5.5"][0]==78,
   "4.3: median output tokens 3/3/49/78",{k:v[0] for k,v in cm.items()})
ok([cm[l][1] for l in ("Fable 5.1","Sonnet 5.5","Opus 5.5","Haiku 5.5")]==[3,3,84,99],
   "4.3: p90 = 3/3/84/99",[cm[l][1] for l in ("Fable 5.1","Sonnet 5.5","Opus 5.5","Haiku 5.5")])
ok([cm[l][2] for l in ("Fable 5.1","Sonnet 5.5","Opus 5.5","Haiku 5.5")]==[3,59,128,147],
   "4.3: max = 3/59/128/147",[cm[l][2] for l in ("Fable 5.1","Sonnet 5.5","Opus 5.5","Haiku 5.5")])
a=lo.give.dropna(); b=hi.give.dropna()
orr,pf=fisher_exact([[int((a==50).sum()),int((a!=50).sum())],[int((b==50).sum()),int((b!=50).sum())]])
ok(round(np.mean(a==50)*100,1)==63.3 and round(np.mean(b==50)*100,1)==36.0,
   "4.4: share exactly half 63.3 vs 36.0",f"{np.mean(a==50)*100:.1f} vs {np.mean(b==50)*100:.1f}")
ok(round(orr,2)==3.07,"4.4: odds ratio 3.07",round(orr,2))
ks=ks_2samp(a,b); ok(round(ks.statistic,3)==0.273,"4.4: KS D 0.273",round(ks.statistic,3))
ok(a.median()==50 and b.median()==30,"4.4: median 50 vs 30",f"{a.median()} vs {b.median()}")
ok(round(mannwhitneyu(a,b).pvalue,3)==0.149,"4.4: Mann-Whitney p 0.149",round(mannwhitneyu(a,b).pvalue,3))
G=d[d.arm=="grid"]
p0={l:(round(G[(G.label==l)&(G.persona=="P0_none")].give.mean(),2),
       G[(G.label==l)&(G.persona=="P0_none")].give.nunique()) for l in NM.values()}
ok(p0["Haiku 5.5"][0]==50.00 and p0["Opus 5.5"][0]==50.00,"4.5: no-persona mean 50.00 Haiku and Opus",
   {k:v[0] for k,v in p0.items()})
ok(round(p0["Sonnet 5.5"][0],2)==43.56 and round(p0["Fable 5.1"][0],2)==43.78,"4.5: 43.56 and 43.78",
   (p0["Sonnet 5.5"][0],p0["Fable 5.1"][0]))
Gv=h[(h.arm=="grid")&(h.persona=="P2_rich")].groupby("variant").give.mean()
ok(round(Gv.max()-Gv.min(),2)==9.50,"4.5: variant spread 9.50",round(Gv.max()-Gv.min(),2))
kw=kruskal(*[h[(h.arm=="grid")&(h.persona=="P2_rich")&(h.variant==v)].give.dropna().values
             for v in sorted(h.variant.unique())])
ok(round(kw.statistic,2)==7.55 and round(kw.pvalue,2)==0.11,"4.5: Kruskal H 7.55 p 0.11",
   (round(kw.statistic,2),round(kw.pvalue,3)))
Gp=G.groupby("persona").give.mean()
ok(round(Gp.max()-Gp.min(),2)==18.93,"4.5: persona spread 18.93",round(Gp.max()-Gp.min(),2))
Gr=d[(d.arm=="grid")&(d.persona=="P2_rich")].copy()
rng=random.Random(11)
at={i:dict(age=rng.choice(H.AGES),gender=rng.choice(H.GENDERS),country=rng.choice(H.COUNTRIES),
           job=rng.choice(H.JOBS),trait=rng.choice(H.TRAITS)) for i in range(30)}
Gr["trait"]=Gr.aid.map(lambda x: at[x]["trait"])
fa={l:round(Gr[(Gr.label==l)&(Gr.trait=="concerned about fairness")].give.mean(),2) for l in NM.values()}
ok(all(v==50.00 for v in fa.values()),"4.6: fairness trait gives exactly 50.00 in all four",fa)
med={l:D[D.label==l].give.median() for l in NM.values()}
ok(med["Sonnet 5.5"]==20 and med["Opus 5.5"]==50,"4.6: medians 20 Sonnet, 50 Opus",med)

print("\n### PAPER B ###")
ok(len(e4)==720 and e4.give.notna().all(),"methods: 720 calls, all parsed",f"{len(e4)}, nulls {e4.give.isna().sum()}")
A=e4[e4.arm=="format"]
bare=A[A.fmt=="F1_bare"].give
ok(len(bare)==120,"4.1: bare instruction n (manuscript says 180 across four models)",len(bare))
ok(np.mean(bare%5==0)*100==100.0,"4.1: bare 100% multiples of five",np.mean(bare%5==0)*100)
ok(round(np.mean(bare%10==0)*100,1)==98.3,"4.1: bare 98.3% multiples of ten",round(np.mean(bare%10==0)*100,1))
ok(bare.nunique()==8,"4.1: bare 8 distinct values",bare.nunique())
inv=A[A.fmt=="F2_invite"].give
ok(round(np.mean(inv%5==0)*100,1)==95.8 and inv.nunique()==17,"4.2: invite 95.8% and 17 distinct",
   (round(np.mean(inv%5==0)*100,1),inv.nunique()))
ok(int((e4.give%1!=0).sum())==0,"4.2: zero decimal responses in all 720",int((e4.give%1!=0).sum()))
pr=A[A.fmt=="F3_precise"].give
ok(round(np.mean(pr%5==0)*100,1)==66.7 and pr.nunique()==22,"4.3: precise 66.7% and 22 distinct",
   (round(np.mean(pr%5==0)*100,1),pr.nunique()))
tb=[[int((A[A.fmt==f].give%5==0).sum()),int((A[A.fmt==f].give%5!=0).sum())]
    for f in ("F1_bare","F2_invite","F3_precise","F4_decimal")]
c2,p,_,_=chi2_contingency(tb); ok(round(c2,1)==101.6,"4.3: chi-square 101.6",round(c2,1))
P=A[A.fmt=="F3_precise"]
cmp={m:(int((P[P.label==m].give%5==0).sum()),int((P[P.label==m].give%5!=0).sum())) for m in NM.values()}
ok(cmp["Sonnet 5.5"]==(30,0) and cmp["Opus 5.5"]==(8,22) and cmp["Haiku 5.5"]==(14,16) and cmp["Fable 5.1"]==(28,2),
   "4.3: compliance 30/0, 8/22, 14/16, 28/2",cmp)
c2b,pb,_,_=chi2_contingency([list(cmp[m]) for m in NM.values()])
ok(round(c2b,1)==51.6,"4.3: model x compliance chi-square 51.6",round(c2b,1))
B=e4[e4.arm=="uncert"]
ce=B[B.scen=="S1_certain"].give; un=B[B.scen=="S2_uncertain"].give
_,pfu=fisher_exact([[int((ce%10==0).sum()),int((ce%10!=0).sum())],[int((un%10==0).sum()),int((un%10!=0).sum())]])
ok(round(pfu,3)==1.0,"4.4: granularity Fisher p = 1.000",round(pfu,3))
ok(round(ce.mean(),2)==36.42 and round(un.mean(),2)==30.67,"4.4: means 36.42 and 30.67",
   (round(ce.mean(),2),round(un.mean(),2)))
ok(ce.median()==40 and un.median()==30,"4.4: medians 40 and 30",(ce.median(),un.median()))
ok(round(mannwhitneyu(ce,un).pvalue,4)==0.0053,"4.4: Mann-Whitney p 0.0053",round(mannwhitneyu(ce,un).pvalue,5))
ok(e4.give.nunique()==31,"4.5: 31 of 101 values used overall",e4.give.nunique())
only_pr=set(pr[pr%5!=0].unique())
ok(len(only_pr)==18,"4.5: 18 values appear only under the precise instruction",len(only_pr))
