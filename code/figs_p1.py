import os,sys,json,random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, harness as H
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
OUT=os.environ.get("FIGDIR_A","figs/PaperA")
os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,
 "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.2,"axes.linewidth":0.6,
 "xtick.major.width":0.6,"ytick.major.width":0.6,"axes.spines.top":False,"axes.spines.right":False,
 "savefig.dpi":820,"figure.dpi":820,"legend.frameon":False,"axes.grid":True,
 "grid.color":"#E8E8E8","grid.linewidth":0.5,"axes.axisbelow":True})
BLUE="#0072B2"; VERM="#D55E00"; GREEN="#009E73"; PURP="#CC79A7"; GREY="#8C8C8C"; CM=1/2.54
COL={"Haiku 5.5":VERM,"Opus 5.5":PURP,"Sonnet 5.5":BLUE,"Fable 5.1":GREEN}
MK ={"Haiku 5.5":"o","Opus 5.5":"s","Sonnet 5.5":"^","Fable 5.1":"D"}
HUMAN=28.35
def load():
    fr=[]
    for f,t in (("exp3_sonnet.jsonl","Sonnet 5.5"),("exp3_opus.jsonl","Opus 5.5"),("exp3_fable.jsonl","Fable 5.1")):
        fr.append(pd.DataFrame([json.loads(l) for l in open(f)]).assign(label=t))
    h=pd.DataFrame([json.loads(l) for l in open("exp2_haiku.jsonl")]).assign(label="Haiku 5.5")
    h=h[h.variant.isin(["V1_plain","V3_second","V4_keep"])]
    return pd.concat(fr+[h],ignore_index=True)
d=load(); ORD=["Haiku 5.5","Opus 5.5","Sonnet 5.5","Fable 5.1"]
BUD=[16,32,64,128,256]

# ---------- Figure 1 : the same budget, four models ----------
fig,ax=plt.subplots(1,2,figsize=(16*CM,6.6*CM))
T=d[d.arm=="tokens"]
a=ax[0]
for lab in ORD:
    y=[T[(T.label==lab)&(T.max_tokens==b)].give.isna().mean()*100 for b in BUD]
    a.plot(range(len(BUD)),y,marker=MK[lab],ms=4.2,lw=1.8,color=COL[lab],label=lab,clip_on=False)
a.set_xticks(range(len(BUD))); a.set_xticklabels([str(b) for b in BUD])
a.set_ylim(-4,104); a.yaxis.set_major_locator(MultipleLocator(25))
a.set_xlabel("Output token budget"); a.set_ylabel("Responses lost, percent")
a.set_title("The same budget, four models",pad=5); a.legend(loc="upper right",handlelength=1.5)
b=ax[1]
cost=pd.DataFrame([json.loads(l) for l in open("cost.jsonl")])
NM={"claude-haiku-5-5":"Haiku 5.5","claude-sonnet-5-5":"Sonnet 5.5",
    "claude-opus-5-5":"Opus 5.5","claude-fable-5-1":"Fable 5.1"}
cost["label"]=cost.model.map(NM)
xs=[cost[cost.label==l].out_tok.median() for l in ORD]
ys=[T[(T.label==l)&(T.max_tokens==16)].give.isna().mean()*100 for l in ORD]
# hand-placed label offsets so the two low-cost models do not collide
OFF={"Haiku 5.5":(-10,-14),"Opus 5.5":(-12,-15),"Sonnet 5.5":(9,6),"Fable 5.1":(9,-11)}
HA ={"Haiku 5.5":"right","Opus 5.5":"right","Sonnet 5.5":"left","Fable 5.1":"left"}
for i,l in enumerate(ORD):
    b.scatter(xs[i],ys[i],s=58,color=COL[l],marker=MK[l],zorder=3,clip_on=False)
    b.annotate(f"{l}\n{xs[i]:.0f} tokens",(xs[i],ys[i]),xytext=OFF[l],textcoords="offset points",
               fontsize=6.8,color=COL[l],ha=HA[l],linespacing=1.25)
b.set_xlabel("Output tokens spent to produce one number")
b.set_ylabel("Responses lost at budget 16, percent")
b.set_xlim(-8,112); b.set_ylim(-14,118); b.yaxis.set_major_locator(MultipleLocator(25))
b.set_title("Loss follows what the model spends",pad=5)
fig.savefig(os.path.join(OUT,"Figure1.png"),bbox_inches="tight",pad_inches=0.02); plt.close(fig)

# ---------- Figure 2 : selective loss ----------
# Figure 2 is a single-model analysis: use the full Haiku design (all five variants)
H2=pd.DataFrame([json.loads(l) for l in open("exp2_haiku.jsonl")])
H2=H2[H2.arm=="tokens"]
lo=H2[H2.max_tokens.isin([16,32,64])].give.dropna(); hi=H2[H2.max_tokens.isin([128,256])].give.dropna()
fig,ax=plt.subplots(1,2,figsize=(16*CM,6.4*CM))
a=ax[0]
bins=np.arange(-2.5,78,5)
a.hist(hi,bins=bins,density=True,color=BLUE,alpha=.85,label=f"budget 128 and above (n={len(hi)})")
a.hist(lo,bins=bins,density=True,color=VERM,alpha=.58,label=f"budget 64 and below (n={len(lo)})")
ym=a.get_ylim()[1]; a.set_ylim(0,ym*1.32)
a.axvline(HUMAN,color=GREY,lw=1.1,ls=(0,(3,3)))
a.annotate("human mean 28.35",(HUMAN,ym*0.52),xytext=(5,0),textcoords="offset points",
           fontsize=6.8,color=GREY,rotation=90,va="center")
a.set_xlim(-5,75); a.set_xlabel("Dollars given out of 100"); a.set_ylabel("Density")
a.set_title("Which responses survive truncation",pad=5); a.legend(loc="upper left",handlelength=1.3)
b=ax[1]
y1=[np.mean(H2[H2.max_tokens==x].give.dropna()==50)*100 for x in BUD]
y2=[H2[H2.max_tokens==x].give.median() for x in BUD]
b.plot(range(len(BUD)),y1,marker="^",ms=4.2,lw=1.7,color=GREEN,ls=(0,(4,2)),label="Share giving exactly half",clip_on=False)
b.plot(range(len(BUD)),y2,marker="s",ms=4.2,lw=1.8,color=BLUE,label="Median of survivors",clip_on=False)
b.axhline(HUMAN,color=GREY,lw=1.1,ls=(0,(3,3)))
b.annotate("human mean",(len(BUD)-1,HUMAN),xytext=(-2,-11),textcoords="offset points",ha="right",fontsize=6.8,color=GREY)
b.set_xticks(range(len(BUD))); b.set_xticklabels([str(x) for x in BUD])
b.set_ylim(0,88); b.yaxis.set_major_locator(MultipleLocator(20))
b.set_xlabel("Output token budget"); b.set_ylabel("Dollars, or percent")
b.set_title("The conclusion moves with the budget",pad=5); b.legend(loc="upper right",handlelength=1.5)
fig.savefig(os.path.join(OUT,"Figure2.png"),bbox_inches="tight",pad_inches=0.02); plt.close(fig)
print("Figure1, Figure2 written")
