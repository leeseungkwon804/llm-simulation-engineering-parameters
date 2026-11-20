import os, json, re, time, hashlib, random, sys
import httpx
KEY=os.environ["ANTHROPIC_API_KEY"]
URL="https://api.anthropic.com/v1/messages"
HDR={"x-api-key":KEY,"anthropic-version":"2023-06-01","content-type":"application/json"}

# ---- five meaning-preserving statements of the same dictator game ----
VARIANTS={
"V1_plain":"You have been given {amount} dollars. You may give any part of it to another person who has been given nothing. That person cannot refuse and cannot affect you in any way. How many dollars do you give?",
"V2_passive":"{amount} dollars have been allocated to you. Any portion of this sum may be transferred by you to a second person, who received no allocation. No response or retaliation from that person is possible. What amount is transferred?",
"V3_second":"Imagine you receive {amount} dollars. Another participant receives nothing. You decide how much of your {amount} dollars, if any, to pass to them. They have no say and will never meet you. How much do you pass?",
"V4_keep":"You hold {amount} dollars. A second person holds nothing. You choose how much of the {amount} to keep for yourself; whatever you do not keep goes to them. They cannot respond. How much do you give away?",
"V5_formal":"In this task you are the allocator of an endowment of {amount} dollars. A recipient has an endowment of zero. The allocator may assign any amount between 0 and {amount} to the recipient. The recipient has no decision to make. State the amount you assign.",
}
PERSONA={
"P0_none":"",
"P1_demo":"You are a {age}-year-old {gender} from {country}. ",
"P2_rich":"You are a {age}-year-old {gender} from {country}. You work as a {job}. You describe yourself as {trait}. ",
}
ASK="\n\nAnswer with a single number only, the number of dollars you give. No words, no symbols."

AGES=[21,27,34,41,48,55,62]
GENDERS=["man","woman"]
COUNTRIES=["the United States","Germany","South Korea","Brazil","India","Kenya","Poland"]
JOBS=["nurse","bus driver","software engineer","shop assistant","teacher","accountant","farmer"]
TRAITS=["generally trusting of others","fairly cautious with money","practical and direct",
        "concerned about fairness","focused on your own interests","easy-going"]

def extract_text(d):
    """the model may emit a thinking block first; take the text block"""
    for b in d.get("content",[]):
        if b.get("type")=="text":
            return b.get("text")
    return None

def call(model, prompt, temperature, max_tokens=256, retries=4):
    body={"model":model,"max_tokens":max_tokens,
          "messages":[{"role":"user","content":prompt}]}
    if temperature is not None: body["temperature"]=temperature
    for a in range(retries):
        try:
            r=httpx.post(URL,headers=HDR,json=body,timeout=90)
            if r.status_code==200:
                d=r.json()
                u=d.get("usage",{}); u["stop_reason"]=d.get("stop_reason")
                u["n_blocks"]=len(d.get("content",[]))
                return extract_text(d), u
            if r.status_code in (429,529,500,502,503):
                time.sleep(2*(a+1)); continue
            # 4xx are permanent: record them as API refusals, never as agent non-response
            return None,{"error":r.status_code,"api_error":r.json().get("error",{}).get("message","")[:120]}
        except Exception as e:
            time.sleep(2*(a+1))
    return None,{"error":"retries_exhausted"}

def parse(txt, amount):
    if txt is None: return None
    m=re.search(r"-?\d+(?:\.\d+)?", txt.replace(",",""))
    if not m: return None
    v=float(m.group(0))
    if v<0 or v>amount: return None
    return v

def run(model, n_agents, variants, personas, temps, amount=100, seed=11, outfile="runs.jsonl", verbose=True, max_tokens=256):
    rng=random.Random(seed)
    agents=[]
    for i in range(n_agents):
        agents.append(dict(aid=i, age=rng.choice(AGES), gender=rng.choice(GENDERS),
                           country=rng.choice(COUNTRIES), job=rng.choice(JOBS), trait=rng.choice(TRAITS)))
    done=0; tok_in=tok_out=0
    with open(outfile,"a") as f:
        for pk in personas:
            for vk in variants:
                for T in temps:
                    for ag in agents:
                        pre=PERSONA[pk].format(**ag)
                        prompt=pre+VARIANTS[vk].format(amount=amount)+ASK
                        txt,us=call(model,prompt,T,max_tokens=max_tokens)
                        val=parse(txt,amount)
                        tok_in+=us.get("input_tokens",0); tok_out+=us.get("output_tokens",0)
                        f.write(json.dumps(dict(model=model,persona=pk,variant=vk,temp=T,amount=amount,
                            aid=ag["aid"],age=ag["age"],gender=ag["gender"],country=ag["country"],
                            job=ag["job"],trait=ag["trait"],raw=txt,give=val,
                            stop=us.get("stop_reason"),nblocks=us.get("n_blocks"),
                            max_tokens=max_tokens,err=us.get("error")),ensure_ascii=False)+"\n")
                        done+=1
                        if verbose and done%40==0:
                            print(f"   {done} calls  in={tok_in} out={tok_out}",flush=True)
    return done,tok_in,tok_out

if __name__=="__main__":
    n=int(sys.argv[1]) if len(sys.argv)>1 else 6
    d,ti,to=run("claude-haiku-5-5", n, ["V1_plain","V5_formal"], ["P0_none","P2_rich"], [1.0],
                outfile="pilot2.jsonl", max_tokens=256)
    print(f"pilot done: {d} calls, in={ti} out={to}")
