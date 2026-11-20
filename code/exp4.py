import os,sys,json,random,threading,queue,time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import harness as H

# --- response-format instructions: all request one number, differ in how the grid is invited ---
FORMATS={
"F1_bare"   :"\n\nAnswer with a single number only, the number of dollars you give. No words, no symbols.",
"F2_invite" :"\n\nAnswer with a single number only. You may use any whole number from 0 to 100, for example 23, 37, 61 or 84. No words, no symbols.",
"F3_precise":"\n\nAnswer with a single number only. Give an exact amount rather than a rounded one. No words, no symbols.",
"F4_decimal":"\n\nAnswer with a single number only. You may use decimals, for example 27.5. No words, no symbols.",
}
# --- uncertainty manipulation: human round-number bias strengthens under uncertainty ---
SCEN={
"S1_certain"  :"You have been given {amount} dollars. You may give any part of it to another person who has been given nothing. That person cannot refuse and cannot affect you in any way. You know exactly who they are and what they need. How many dollars do you give?",
"S2_uncertain":"You have been given {amount} dollars. You may give any part of it to another person who has been given nothing. That person cannot refuse and cannot affect you in any way. You know nothing at all about them, their situation or what they need. How many dollars do you give?",
}
MODELS=("claude-haiku-5-5","claude-sonnet-5-5","claude-opus-5-5","claude-fable-5-1")
rng=random.Random(11)
agents=[dict(aid=i,age=rng.choice(H.AGES),gender=rng.choice(H.GENDERS),country=rng.choice(H.COUNTRIES),
             job=rng.choice(H.JOBS),trait=rng.choice(H.TRAITS)) for i in range(30)]
jobs=[]
for m in MODELS:
    for fk in FORMATS:                                   # arm A: does inviting a fine grid break it?
        for ag in agents: jobs.append(("format",m,fk,"S0_base",ag))
    for sk in SCEN:                                      # arm B: does uncertainty coarsen the grid?
        for ag in agents: jobs.append(("uncert",m,"F1_bare",sk,ag))
print(f"jobs {len(jobs)}",flush=True)
q=queue.Queue(); [q.put(j) for j in jobs]
lock=threading.Lock(); f=open("exp4.jsonl","a"); done=[0]
def worker():
    while True:
        try: arm,m,fk,sk,ag=q.get_nowait()
        except queue.Empty: return
        base=H.VARIANTS["V1_plain"] if sk=="S0_base" else SCEN[sk]
        prompt=H.PERSONA["P2_rich"].format(**ag)+base.format(amount=100)+FORMATS[fk]
        txt,us=H.call(m,prompt,None,max_tokens=1024)
        val=None
        if txt:
            import re
            mm=re.search(r"-?\d+(?:\.\d+)?",txt.replace(",",""))
            if mm:
                try:
                    v=float(mm.group())
                    if 0<=v<=100: val=v
                except ValueError: pass
        with lock:
            f.write(json.dumps(dict(arm=arm,model=m,fmt=fk,scen=sk,aid=ag["aid"],raw=(txt or "")[:40],
                give=val,out_tok=us.get("output_tokens"),stop=us.get("stop_reason"),http=us.get("error")))+"\n")
            done[0]+=1
            if done[0]%150==0: f.flush(); print(f"  {done[0]}/{len(jobs)}",flush=True)
        q.task_done()
ts=[threading.Thread(target=worker,daemon=True) for _ in range(6)]
t0=time.time(); [t.start() for t in ts]; [t.join() for t in ts]
f.flush(); f.close(); print(f"DONE {done[0]} calls {time.time()-t0:.0f}s")
