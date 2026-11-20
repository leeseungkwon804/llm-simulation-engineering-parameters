import os,sys,json,random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, harness as H
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
RA=os.environ.get("FIGDIR_A","figs/PaperA")
RB=os.environ.get("FIGDIR_B","figs/PaperB")
for p in (RA,RB): os.makedirs(p,exist_ok=True)
plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,
 "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.2,"axes.linewidth":0.6,
 "xtick.major.width":0.6,"ytick.major.width":0.6,"axes.spines.top":False,"axes.spines.right":False,
 "savefig.dpi":820,"figure.dpi":820,"legend.frameon":False,"axes.grid":True,
 "grid.color":"#E8E8E8","grid.linewidth":0.5,"axes.axisbelow":True})
BLUE="#0072B2"; VERM="#D55E00"; GREEN="#009E73"; PURP="#CC79A7"; GREY="#8C8C8C"; CM=1/2.54
NM={"claude-haiku-5-5":"Haiku 5.5","claude-sonnet-5-5":"Sonnet 5.5",
    "claude-opus-5-5":"Opus 5.5","claude-fable-5-1":"Fable 5.1"}
ORD=["Haiku 5.5","Opus 5.5","Sonnet 5.5","Fable 5.1"]
COL={"Haiku 5.5":VERM,"Opus 5.5":PURP,"Sonnet 5.5":BLUE,"Fable 5.1":GREEN}
h=pd.DataFrame([json.loads(l) for l in open("exp2_haiku.jsonl")])
fr=[pd.DataFrame([json.loads(l) for l in open(f)]).assign(label=t) for f,t in
    (("exp3_sonnet.jsonl","Sonnet 5.5"),("exp3_opus.jsonl","Opus 5.5"),("exp3_fable.jsonl","Fable 5.1"))]
d=pd.concat(fr+[h[h.variant.isin(["V1_plain","V3_second","V4_keep"])].assign(label="Haiku 5.5")],ignore_index=True)
e4=pd.DataFrame([json.loads(l) for l in open("exp4.jsonl")]); e4["label"]=e4.model.map(NM)
HUMAN=28.35

# ===== Paper A Figure 3 : variance decomposition (within Haiku) =====
G=h[h.arm=="grid"]; R=h[h.arm=="repeat"].dropna(subset=["give"])
Gv=G[G.persona=="P2_rich"].groupby("variant").give.mean(); Gp=G.groupby("persona").give.mean()
T=h[h.arm=="tokens"]
vals=[R.groupby("aid").give.std().mean(), Gv.max()-Gv.min(), Gp.max()-Gp.min(),
      abs(T[T.max_tokens<=64].give.median()-T[T.max_tokens>=128].give.median())]
labs=["Run to run\nidentical settings","Prompt wording\nfive variants","Persona\nthree levels","Token budget\nmedian shift"]
fig,ax=plt.subplots(figsize=(16*CM,7.4*CM))
ax.bar(range(4),vals,0.46,color=[GREY,BLUE,VERM,GREEN],edgecolor="white",lw=0.8)
for i,v in enumerate(vals): ax.annotate(f"{v:.2f}",(i,v),xytext=(0,4),textcoords="offset points",ha="center",fontsize=7.6)
ax.set_xticks(range(4)); ax.set_xticklabels(labs,fontsize=6.9)
ax.set_ylabel("Shift in dollars given"); ax.set_ylim(0,24); ax.yaxis.set_major_locator(MultipleLocator(5))
ax.set_title("Design choices move the result more than sampling noise",pad=6)
fig.savefig(f"{RA}/Figure4.png",bbox_inches="tight",pad_inches=0.02); plt.close(fig)

# ===== Paper A Figure 4 : persona and the between-model spread =====
fig,ax=plt.subplots(1,2,figsize=(16*CM,6.6*CM))
a=ax[0]; PL=["P0_none","P1_demo","P2_rich"]; PN=["none","demographic","rich"]
Gd=d[d.arm=="grid"]; wd=0.2
for i,lab in enumerate(ORD):
    y=[Gd[(Gd.label==lab)&(Gd.persona==p)].give.mean() for p in PL]
    a.bar(np.arange(3)+(i-1.5)*wd,y,wd,color=COL[lab],label=lab,edgecolor="white",lw=0.5)
a.axhline(HUMAN,color=GREY,lw=1.1,ls=(0,(3,3)),label="human mean 28.35")
a.set_xticks(range(3)); a.set_xticklabels(PN); a.set_xlabel("Persona specification")
a.set_ylabel("Mean dollars given"); a.set_ylim(0,70); a.yaxis.set_major_locator(MultipleLocator(10))
a.set_title("Persona level sets the mean",pad=5)
a.legend(loc="upper center",ncol=3,handlelength=1.2,columnspacing=1.0,fontsize=6.6)
b=ax[1]
for i,lab in enumerate(ORD):
    y=[Gd[(Gd.label==lab)&(Gd.persona==p)].give.nunique() for p in PL]
    b.bar(np.arange(3)+(i-1.5)*wd,y,wd,color=COL[lab],edgecolor="white",lw=0.5)
b.set_xticks(range(3)); b.set_xticklabels(PN); b.set_xlabel("Persona specification")
b.set_ylabel("Distinct values used"); b.set_ylim(0,8.6); b.yaxis.set_major_locator(MultipleLocator(2))
b.set_title("And how much variety the population has",pad=5)
fig.savefig(f"{RA}/Figure3.png",bbox_inches="tight",pad_inches=0.02); plt.close(fig)

# ===== Paper B Figure 1 : the grid under four instructions =====
A4=e4[e4.arm=="format"]
FK=["F1_bare","F2_invite","F3_precise","F4_decimal"]
FN=["bare","any whole\nnumber invited","avoid\nrounding","decimals\noffered"]
fig,ax=plt.subplots(1,2,figsize=(16*CM,6.6*CM))
a=ax[0]
y5=[np.mean(A4[A4.fmt==f].give%5==0)*100 for f in FK]
yd=[A4[A4.fmt==f].give.nunique() for f in FK]
a.bar(range(4),y5,0.56,color=[GREY,BLUE,VERM,GREEN],edgecolor="white",lw=0.8)
for i,v in enumerate(y5): a.annotate(f"{v:.1f}",(i,v),xytext=(0,4),textcoords="offset points",ha="center",fontsize=7.4)
a.set_xticks(range(4)); a.set_xticklabels(FN,fontsize=6.8)
a.set_ylabel("Responses on a multiple of five, percent"); a.set_ylim(0,118); a.yaxis.set_major_locator(MultipleLocator(25))
a.set_title("The grid under four response instructions",pad=5)
b=ax[1]
b.bar(range(4),yd,0.56,color=[GREY,BLUE,VERM,GREEN],edgecolor="white",lw=0.8)
for i,v in enumerate(yd): b.annotate(str(v),(i,v),xytext=(0,4),textcoords="offset points",ha="center",fontsize=7.4)
b.axhline(101,color=GREY,lw=1.1,ls=(0,(3,3)))
b.annotate("101 values available",(3.45,101),xytext=(0,-11),textcoords="offset points",ha="right",fontsize=6.8,color=GREY)
b.set_xticks(range(4)); b.set_xticklabels(FN,fontsize=6.8)
b.set_ylabel("Distinct values used"); b.set_ylim(0,116); b.yaxis.set_major_locator(MultipleLocator(25))
b.set_title("How much of the scale is used",pad=5)
fig.savefig(f"{RB}/Figure1.png",bbox_inches="tight",pad_inches=0.02); plt.close(fig)

# ===== Paper B Figure 2 : compliance is model dependent =====
fig,ax=plt.subplots(figsize=(16*CM,7.4*CM))
P=A4[A4.fmt=="F3_precise"]; Bq=A4[A4.fmt=="F1_bare"]
x=np.arange(4)
yb=[np.mean(Bq[Bq.label==l].give%5==0)*100 for l in ORD]
yp=[np.mean(P[P.label==l].give%5==0)*100 for l in ORD]
ax.bar(x-0.17,yb,0.30,color=GREY,label="bare instruction",edgecolor="white",lw=0.6)
ax.bar(x+0.17,yp,0.30,color=VERM,label="instructed to avoid rounding",edgecolor="white",lw=0.6)
for i in range(4):
    ax.annotate(f"{yb[i]:.0f}",(x[i]-0.17,yb[i]),xytext=(0,3),textcoords="offset points",ha="center",fontsize=7)
    ax.annotate(f"{yp[i]:.0f}",(x[i]+0.17,yp[i]),xytext=(0,3),textcoords="offset points",ha="center",fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(ORD,fontsize=7.2)
ax.set_ylabel("Responses on a multiple of five, percent"); ax.set_ylim(0,118)
ax.yaxis.set_major_locator(MultipleLocator(25))
ax.set_title("The same instruction, four different degrees of compliance",pad=20)
ax.legend(loc="lower center",bbox_to_anchor=(0.5,1.01),handlelength=1.3,ncol=2)
fig.savefig(f"{RB}/Figure2.png",bbox_inches="tight",pad_inches=0.02); plt.close(fig)

# ===== Paper B Figure 3 : uncertainty dissociation =====
Bu=e4[e4.arm=="uncert"]
fig,ax=plt.subplots(1,2,figsize=(16*CM,6.6*CM))
SK=["S1_certain","S2_uncertain"]; SN=["recipient known","recipient unknown"]
a=ax[0]
g10=[np.mean(Bu[Bu.scen==s].give%10==0)*100 for s in SK]
a.bar(range(2),g10,0.42,color=[BLUE,VERM],edgecolor="white",lw=0.8)
for i,v in enumerate(g10): a.annotate(f"{v:.1f}",(i,v),xytext=(0,4),textcoords="offset points",ha="center",fontsize=7.6)
a.set_xticks(range(2)); a.set_xticklabels(SN,fontsize=7.2)
a.set_ylabel("Responses on a multiple of ten, percent"); a.set_ylim(0,118)
a.yaxis.set_major_locator(MultipleLocator(25))
a.set_title("Granularity does not move  (p = 1.000)",pad=5)
b=ax[1]
data=[Bu[Bu.scen==s].give.values for s in SK]
bp=b.boxplot(data,positions=[0,1],widths=0.42,patch_artist=True,showfliers=False,
             medianprops=dict(color="white",lw=1.6),whiskerprops=dict(lw=0.8),capprops=dict(lw=0.8))
for patch,c in zip(bp["boxes"],[BLUE,VERM]): patch.set_facecolor(c); patch.set_edgecolor("none")
for i,s in enumerate(SK):
    v=Bu[Bu.scen==s].give
    b.scatter(np.full(len(v),i)+np.linspace(-.13,.13,len(v)),v,s=4,color="#404040",alpha=.30,zorder=3)
    b.annotate(f"mean {v.mean():.2f}",(i,96),ha="center",fontsize=7,color=[BLUE,VERM][i])
b.set_xticks([0,1]); b.set_xticklabels(SN,fontsize=7.2)
b.set_ylabel("Dollars given out of 100"); b.set_ylim(-5,108)
b.yaxis.set_major_locator(MultipleLocator(25))
b.set_title("But the decision does  (p = 0.0053)",pad=5)
fig.savefig(f"{RB}/Figure3.png",bbox_inches="tight",pad_inches=0.02); plt.close(fig)
print("written:", *[f for r in (RA,RB) for f in sorted(os.listdir(r))])
