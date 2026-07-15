"""
구조보존 번역증강 (NLLB GPU 배치) — 경모 3대문제 우회.
 실제 sim-TRAIN(history·meta·label 원본유지)에서 5000샘플 → current_prompt만 반대언어 번역.
 val 세션(exp_data_ab seed42 split) 제외 → 누수 방지.
 방향: en/mixed→ko(한국어 보강), ko→en.  코드용어는 NLLB가 대체로 보존.
사용: CUDA_VISIBLE_DEVICES=2 python make_trans_aug.py
출력: trans_aug.jsonl + trans_aug_labels.csv
"""
import os, sys, json, re
import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,"..",".."))
N=5000
def plang(s):
    s=str(s); h=len(re.findall(r"[가-힣]",s)); a=len(re.findall(r"[A-Za-z]",s))
    return "en" if(a and not h) else "ko" if(h and not a) else "mixed"

lab=pd.read_csv(os.path.join(ROOT,"data/raw/train_labels.csv"))
rows=[json.loads(l) for l in open(os.path.join(ROOT,"data/raw/train.jsonl"),encoding="utf-8")]
df=pd.DataFrame(rows).merge(lab,on="id",how="left")
df["prefix"]=np.where(df["id"].str.startswith("sess_au_"),"au","sim")
df["session"]=df["id"].str.replace(r"-step_\d+$","",regex=True)
sim=df[df.prefix=="sim"].reset_index(drop=True)
gss=GroupShuffleSplit(1,test_size=0.1,random_state=42); tri,vai=next(gss.split(sim,groups=sim["session"]))
sim_train=sim.iloc[tri].reset_index(drop=True)
assert not (set(sim_train["session"]) & set(sim.iloc[vai]["session"])), "누수!"
samp=sim_train.sample(n=min(N,len(sim_train)),random_state=7).reset_index(drop=True)
print(f"[trans] {len(samp)}개 샘플 (val 제외 확인)",flush=True)

m="facebook/nllb-200-distilled-600M"
tok=AutoTokenizer.from_pretrained(m)
model=AutoModelForSeq2SeqLM.from_pretrained(m,torch_dtype=torch.float16).to("cuda").eval()
def batch_tr(texts,src,tgt,bs=48):
    tok.src_lang=src; res=[]
    fb=tok.convert_tokens_to_ids(tgt)
    for i in range(0,len(texts),bs):
        enc=tok([t[:400] for t in texts[i:i+bs]],return_tensors="pt",padding=True,truncation=True,max_length=200).to("cuda")
        with torch.inference_mode():
            out=model.generate(**enc,forced_bos_token_id=fb,max_length=220,num_beams=1)
        res+=tok.batch_decode(out,skip_special_tokens=True)
        if i%480==0: print(f"  {src}->{tgt} {i+bs}/{len(texts)}",flush=True)
    return res
samp["lang"]=samp["current_prompt"].apply(plang)
# en/mixed→ko , ko→en
grp_ko=samp[samp.lang.isin(["en","mixed"])]; grp_en=samp[samp.lang=="ko"]
tr_ko=batch_tr(grp_ko["current_prompt"].astype(str).tolist(),"eng_Latn","kor_Hang")
tr_en=batch_tr(grp_en["current_prompt"].astype(str).tolist(),"kor_Hang","eng_Latn")
trans={}
for (idx,r),t in zip(grp_ko.iterrows(),tr_ko): trans[r["id"]]=t
for (idx,r),t in zip(grp_en.iterrows(),tr_en): trans[r["id"]]=t

out_rows=[]; out_lab=[]
for _,r in samp.iterrows():
    t=trans.get(r["id"],"")
    if not t or t.strip()==str(r["current_prompt"]).strip(): continue
    nid=f"aug_trans_{r['id']}"
    out_rows.append({"id":nid,"session_meta":r["session_meta"],"history":r["history"],"current_prompt":t})
    out_lab.append({"id":nid,"action":r["action"]})
with open(os.path.join(HERE,"trans_aug.jsonl"),"w") as f:
    for r in out_rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
pd.DataFrame(out_lab).to_csv(os.path.join(HERE,"trans_aug_labels.csv"),index=False)
from collections import Counter
print(f"\n[done] {len(out_rows)}개. 결과언어: {dict(Counter(plang(r['current_prompt']) for r in out_rows))}",flush=True)
print(f"샘플: {out_rows[0]['current_prompt'][:80]}",flush=True)
PY_END = None
