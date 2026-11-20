import os,sys,json,time,random,threading,queue
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import harness as H
MODEL=sys.argv[1]; NAG=int(sys.argv[2]); OUT=sys.argv[3]; WORKERS=8
VAR=list(H.VARIANTS); PER=list(H.PERSONA)
rng=random.Random(11)
agents=[dict(aid=i,age=rng.choice(H.AGES),gender=rng.choice(H.GENDERS),country=rng.choice(H.COUNTRIES),
             job=rng.choice(H.JOBS),trait=rng.choice(H.TRAITS)) for i in range(NAG)]
jobs=[]
# arm 1: clean behavioural grid at a generous budget
for pk in PER:
    for vk in VAR:
        for ag in agents: jobs.append(("grid",pk,vk,512,0,ag))
# arm 2: token-budget audit across ALL variants, rich persona
for mt in (16,32,64,128,256,512):
    for vk in VAR:
        for ag in agents: jobs.append(("tokens","P2_rich",vk,mt,0,ag))
# arm 3: run-to-run stochasticity at fixed settings
for rep in range(5):
    for ag in agents: jobs.append(("repeat","P2_rich","V1_plain",512,rep,ag))
print(f"model={MODEL} agents={NAG} jobs={len(jobs)}",flush=True)
q=queue.Queue(); [q.put(j) for j in jobs]
lock=threading.Lock(); done=[0]; tin=[0]; tout=[0]; f=open(OUT,"a")
def worker():
    while True:
        try: arm,pk,vk,mt,rep,ag=q.get_nowait()
        except queue.Empty: return
        prompt=H.PERSONA[pk].format(**ag)+H.VARIANTS[vk].format(amount=100)+H.ASK
        txt,us=H.call(MODEL,prompt,None,max_tokens=mt)
        val=H.parse(txt,100)
        with lock:
            f.write(json.dumps(dict(arm=arm,model=MODEL,persona=pk,variant=vk,max_tokens=mt,rep=rep,
                aid=ag["aid"],age=ag["age"],gender=ag["gender"],country=ag["country"],job=ag["job"],
                trait=ag["trait"],raw=txt,give=val,stop=us.get("stop_reason"),
                api_error=us.get("api_error"),http=us.get("error")),ensure_ascii=False)+"\n")
            done[0]+=1; tin[0]+=us.get("input_tokens",0); tout[0]+=us.get("output_tokens",0)
            if done[0]%200==0: f.flush(); print(f"  {done[0]}/{len(jobs)} out={tout[0]}",flush=True)
        q.task_done()
ths=[threading.Thread(target=worker,daemon=True) for _ in range(WORKERS)]
t0=time.time(); [t.start() for t in ths]; [t.join() for t in ths]
f.flush(); f.close(); print(f"DONE {done[0]} calls {time.time()-t0:.0f}s in={tin[0]} out={tout[0]}")
