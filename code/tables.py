import os,sys,json,random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, harness as H
NM={"claude-haiku-5-5":"Haiku 5.5","claude-sonnet-5-5":"Sonnet 5.5",
    "claude-opus-5-5":"Opus 5.5","claude-fable-5-1":"Fable 5.1"}
ORD=["Haiku 5.5","Opus 5.5","Sonnet 5.5","Fable 5.1"]
h=pd.DataFrame([json.loads(l) for l in open("exp2_haiku.jsonl")])
fr=[pd.DataFrame([json.loads(l) for l in open(f)]).assign(label=t) for f,t in
    (("exp3_sonnet.jsonl","Sonnet 5.5"),("exp3_opus.jsonl","Opus 5.5"),("exp3_fable.jsonl","Fable 5.1"))]
d=pd.concat(fr+[h[h.variant.isin(["V1_plain","V3_second","V4_keep"])].assign(label="Haiku 5.5")],ignore_index=True)
cost=pd.DataFrame([json.loads(l) for l in open("cost.jsonl")]); cost["label"]=cost.model.map(NM)
e4=pd.DataFrame([json.loads(l) for l in open("exp4.jsonl")]); e4["label"]=e4.model.map(NM)
out=[]
W=out.append

# ---------------- PAPER A ----------------
W("<<<PAPER_A>>>")
W("**Table 1.** Output token budget, response loss and the conclusion it yields, Haiku 5.5, n = 150 per budget.\n")
W("| Output token budget | Responses parsed | Loss, percent | Median of survivors | Share giving exactly half, percent |")
W("|---|---|---|---|---|")
T=h[h.arm=="tokens"]
for b in (16,32,64,128,256,512):
    s=T[T.max_tokens==b]; sv=s.give.dropna()
    W(f"| {b} | {len(sv)} | {s.give.isna().mean()*100:.1f} | {sv.median():.0f} | {np.mean(sv==50)*100:.1f} |")
W("")
W("**Table 2.** Response loss by model and budget, percent, n = 90 per cell.\n")
bud=[16,32,64,128,256]
W("| Model | "+" | ".join(str(b) for b in bud)+" | Median output tokens spent | 90th percentile |")
W("|---|"+"---|"*(len(bud)+2))
D=d[d.arm=="tokens"]
for l in ORD:
    row=[f"{D[(D.label==l)&(D.max_tokens==b)].give.isna().mean()*100:.1f}" for b in bud]
    c=cost[cost.label==l].out_tok
    W(f"| {l} | "+" | ".join(row)+f" | {c.median():.0f} | {c.quantile(.9):.0f} |")
W("")
W("**Table 3.** Variance components on a common scale, Haiku 5.5.\n")
W("| Source | Shift in dollars | How it was computed |")
W("|---|---|---|")
R=h[h.arm=="repeat"].dropna(subset=["give"]); G=h[h.arm=="grid"]
Gv=G[G.persona=="P2_rich"].groupby("variant").give.mean(); Gp=G.groupby("persona").give.mean()
W(f"| Run to run, identical settings | {R.groupby('aid').give.std().mean():.2f} | mean within-agent SD across five replicates |")
W(f"| Prompt wording, five variants | {Gv.max()-Gv.min():.2f} | spread between variant means |")
W(f"| Persona specification, three levels | {Gp.max()-Gp.min():.2f} | spread between persona means |")
W(f"| Output token budget | {abs(T[T.max_tokens<=64].give.median()-T[T.max_tokens>=128].give.median()):.2f} | median at low budgets against high |")
W("")
W("**Table 4.** Persona specification by model, mean dollars given with standard deviation and distinct values used, n = 90 per cell.\n")
W("| Model | No persona | Demographic | Rich |")
W("|---|---|---|---|")
Gd=d[d.arm=="grid"]
for l in ORD:
    cells=[]
    for p in ("P0_none","P1_demo","P2_rich"):
        s=Gd[(Gd.label==l)&(Gd.persona==p)].give.dropna()
        cells.append(f"{s.mean():.2f} ({s.std():.2f}), {s.nunique()} values")
    W(f"| {l} | "+" | ".join(cells)+" |")
W("")

# ---------------- PAPER B ----------------
W("<<<PAPER_B>>>")
W("**Table 1.** Response granularity under four response-format instructions, four models pooled, n = 120 per instruction.\n")
W("| Response instruction | Multiples of five, percent | Multiples of ten, percent | Non-integer responses | Distinct values used |")
W("|---|---|---|---|---|")
A=e4[e4.arm=="format"]
FN={"F1_bare":"Single number, no further guidance","F2_invite":"Any whole number from 0 to 100, with examples",
    "F3_precise":"An exact rather than a rounded amount","F4_decimal":"Decimals permitted, with an example"}
for f in ("F1_bare","F2_invite","F3_precise","F4_decimal"):
    s=A[A.fmt==f].give
    W(f"| {FN[f]} | {np.mean(s%5==0)*100:.1f} | {np.mean(s%10==0)*100:.1f} | {int((s%1!=0).sum())} | {s.nunique()} |")
W("")
W("**Table 2.** Adherence to the round number grid by model, percent of responses on a multiple of five, n = 30 per cell.\n")
W("| Model | Bare instruction | Instructed to avoid rounding | Change |")
W("|---|---|---|---|")
for l in ORD:
    b=A[(A.fmt=="F1_bare")&(A.label==l)].give; p=A[(A.fmt=="F3_precise")&(A.label==l)].give
    bb=np.mean(b%5==0)*100; pp=np.mean(p%5==0)*100
    W(f"| {l} | {bb:.1f} | {pp:.1f} | {pp-bb:+.1f} |")
W("")
W("**Table 3.** Uncertainty about the recipient, granularity against behaviour, four models pooled, n = 120 per condition.\n")
W("| Condition | Multiples of five, percent | Multiples of ten, percent | Distinct values | Mean dollars given | Median |")
W("|---|---|---|---|---|---|")
B=e4[e4.arm=="uncert"]
for s_,nm in (("S1_certain","Recipient known"),("S2_uncertain","Recipient unknown")):
    s=B[B.scen==s_].give
    W(f"| {nm} | {np.mean(s%5==0)*100:.1f} | {np.mean(s%10==0)*100:.1f} | {s.nunique()} | {s.mean():.2f} | {s.median():.0f} |")
W("")
open("tables.md","w").write("\n".join(out))
print("\n".join(out))
