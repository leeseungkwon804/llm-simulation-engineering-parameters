import os,sys,json,random,threading,queue,time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import harness as H
rng=random.Random(11)
agents=[dict(aid=i,age=rng.choice(H.AGES),gender=rng.choice(H.GENDERS),country=rng.choice(H.COUNTRIES),
             job=rng.choice(H.JOBS),trait=rng.choice(H.TRAITS)) for i in range(30)]
MODELS=("claude-haiku-5-5","claude-sonnet-5-5","claude-opus-5-5","claude-fable-5-1")
jobs=[(m,v,a) for m in MODELS for v in ("V1_plain","V3_second","V4_keep") for a in agents]
print(f"jobs {len(jobs)}",flush=True)
q=queue.Queue(); [q.put(j) for j in jobs]
lock=threading.Lock(); f=open("cost.jsonl","a"); done=[0]
def w():
    while True:
        try: m,v,ag=q.get_nowait()
        except queue.Empty: return
        p=H.PERSONA["P2_rich"].format(**ag)+H.VARIANTS[v].format(amount=100)+H.ASK
        txt,us=H.call(m,p,None,max_tokens=1024)
        with lock:
            f.write(json.dumps(dict(model=m,variant=v,aid=ag["aid"],give=H.parse(txt,100),
                out_tok=us.get("output_tokens"),in_tok=us.get("input_tokens"),
                stop=us.get("stop_reason"),http=us.get("error")))+"\n")
            done[0]+=1
            if done[0]%120==0: f.flush(); print(f"  {done[0]}/{len(jobs)}",flush=True)
        q.task_done()
ts=[threading.Thread(target=w,daemon=True) for _ in range(6)]
t0=time.time(); [t.start() for t in ts]; [t.join() for t in ts]
f.flush(); f.close(); print(f"DONE {done[0]} {time.time()-t0:.0f}s")
