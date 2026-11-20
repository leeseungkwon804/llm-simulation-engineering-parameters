import os,sys,json,time,random,threading,queue
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import harness as H
MODEL=sys.argv[1]; NAG=int(sys.argv[2]); OUT=sys.argv[3]; WORKERS=6
VAR=["V1_plain","V3_second","V4_keep"]      # includes the keep/pass framing pair
PER=["P0_none","P1_demo","P2_rich"]
rng=random.Random(11)
agents=[dict(aid=i,age=rng.choice(H.AGES),gender=rng.choice(H.GENDERS),country=rng.choice(H.COUNTRIES),
             job=rng.choice(H.JOBS),trait=rng.choice(H.TRAITS)) for i in range(NAG)]
jobs=[]
for pk in PER:
    for vk in VAR:
        for ag in agents: jobs.append(("grid",pk,vk,1024,ag))
for mt in (16,32,64,128,256,1024):
    for vk in VAR:
        for ag in agents: jobs.append(("tokens","P2_rich",vk,mt,ag))
print(f"model={MODEL} jobs={len(jobs)}",flush=True)
q=queue.Queue(); [q.put(j) for j in jobs]
lock=threading.Lock(); done=[0]; tout=[0]; f=open(OUT,"a")
def worker():
    while True:
        try: arm,pk,vk,mt,ag=q.get_nowait()
        except queue.Empty: return
        prompt=H.PERSONA[pk].format(**ag)+H.VARIANTS[vk].format(amount=100)+H.ASK
        txt,us=H.call(MODEL,prompt,None,max_tokens=mt)
        with lock:
            f.write(json.dumps(dict(arm=arm,model=MODEL,persona=pk,variant=vk,max_tokens=mt,
                aid=ag["aid"],raw=txt,give=H.parse(txt,100),stop=us.get("stop_reason"),
                api_error=us.get("api_error"),http=us.get("error")),ensure_ascii=False)+"\n")
            done[0]+=1; tout[0]+=us.get("output_tokens",0)
            if done[0]%150==0: f.flush(); print(f"  {done[0]}/{len(jobs)} out={tout[0]}",flush=True)
        q.task_done()
ths=[threading.Thread(target=worker,daemon=True) for _ in range(WORKERS)]
t0=time.time(); [t.start() for t in ths]; [t.join() for t in ths]
f.flush(); f.close(); print(f"DONE {done[0]} calls {time.time()-t0:.0f}s out={tout[0]}")
