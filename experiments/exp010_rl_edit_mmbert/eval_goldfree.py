"""gold 없이(테스트-유효) RL 편집 평가.
선택: base_pred가 inspect + inspect-margin<CM (gold 미사용). 편집 루프도 margin으로만 정지."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import config as C
from common import serialize, serialize_config_from, get_history_pairs, serialize_from_pairs
from exemplar_bank import EXPLORE_CLASSES, extract_pooled_embeddings, build_exemplar_bank
from rl_edit import _batched_forward, _margin_of, _action_mask, PolicyNet
from mmbert_fwd import mmbert_pooled_logits

D=os.path.dirname(__file__); MODEL_DIR=os.path.join(D,"../exp006_mmbert_1024/model"); DEV="cuda"; K=8; CM=0.2
C.SER.add_open_files=True; ser_cfg=serialize_config_from(C.SER)
class2id={c:i for i,c in enumerate(C.CLASS_NAMES)}; explore_ids=[class2id[c] for c in EXPLORE_CLASSES]
insp_set=set(explore_ids)

df=pd.read_json(os.path.join(D,"../../data/processed/train_merged.jsonl"),lines=True)
groups=np.array([re.sub(r"-step_\d+$","",str(i)) for i in df["id"].astype(str)]); y=np.array([class2id[a] for a in df["action"]])
tr,va=next(StratifiedGroupKFold(10,shuffle=True,random_state=42).split(np.arange(len(df)),y,groups))
df_tr=df.iloc[tr].reset_index(drop=True); df_va=df.iloc[va].reset_index(drop=True)
gold=np.array([class2id[a] for a in df_va["action"]])

tok=AutoTokenizer.from_pretrained(MODEL_DIR); model=AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEV).eval()
for p in model.parameters(): p.requires_grad=False

@torch.no_grad()
def probs_all(d):
    t=[serialize(r.current_prompt,r.history,r.session_meta,ser_cfg) for r in d.itertuples()]; out=[]
    for i in range(0,len(t),256):
        enc=tok(t[i:i+256],truncation=True,max_length=C.MAX_LENGTH,padding=True,return_tensors="pt").to(DEV)
        _,lg=mmbert_pooled_logits(model,enc); out.append(torch.softmax(lg,-1).float().cpu().numpy())
    return np.concatenate(out)

pr=probs_all(df_va); base_pred=pr.argmax(1); base_f1=f1_score(gold,base_pred,average="macro")

# bank 재구축 + policy 로드
df_tr_i=df_tr[df_tr["action"].isin(EXPLORE_CLASSES)].reset_index(drop=True)
emb=extract_pooled_embeddings(model,tok,[serialize(r.current_prompt,r.history,r.session_meta,ser_cfg) for r in df_tr_i.itertuples()],DEV,256)
bank=build_exemplar_bank(df_tr_i,emb,0.9,K)
ck=torch.load(os.path.join(D,"ckpt","last.pt"),map_location=DEV)
policy=PolicyNet(model.config.hidden_size+8,4*K+1).to(DEV); policy.load_state_dict(ck["policy"]); policy.eval()

# gold-free 선택: base_pred이 inspect + margin<CM
pred_insp=np.array([p in insp_set for p in base_pred]); margins=_margin_of(pr,explore_ids)
sel=np.where(pred_insp & (margins<CM))[0]
print(f"[gold-free 선택] {len(sel)}개 (pred=inspect & margin<{CM})")
rows=df_va.iloc[sel].reset_index(drop=True).to_dict("records")

@torch.no_grad()
def edit(rows):
    N=len(rows); pairs=[get_history_pairs(r["history"],C.SER.max_history_steps) for r in rows]
    L=np.array([len(p) for p in pairs]); pp=L-1
    tf=lambda idx:[serialize_from_pairs(rows[i]["current_prompt"],pairs[i],rows[i]["session_meta"],ser_cfg) for i in idx]
    pooled,probs=_batched_forward(model,tok,tf(range(N)),DEV,C.MAX_LENGTH,256)
    done=(L==0)|(_margin_of(probs,explore_ids)>=CM); active=[i for i in range(N) if not done[i]]; steps=0
    while active and steps<12:
        steps+=1; states,masks=[],[]
        for i in active:
            mi=_margin_of(probs[i:i+1],explore_ids)[0]
            aux=np.concatenate([np.array([pp[i]/max(L[i],1),L[i]/12,(pp[i]+1)/12,mi],dtype=np.float32),probs[i,explore_ids].astype(np.float32)])
            states.append(np.concatenate([pooled[i].astype(np.float32),aux])); masks.append(_action_mask(L[i],bank,K))
        st=torch.tensor(np.stack(states),device=DEV,dtype=torch.float32); mk=torch.tensor(np.stack(masks),device=DEV)
        lg,_=policy(st); lg=lg.masked_fill(~mk,float("-inf")); acts=torch.argmax(torch.softmax(lg,-1),-1)
        ei=[]
        for li,i in enumerate(active):
            a=int(acts[li])
            if a==4*K: done[i]=True
            else:
                ci,ki=divmod(a,K); bucket=bank.get(EXPLORE_CLASSES[ci],{}).get(str(L[i]),[])
                if ki>=len(bucket): done[i]=True; continue
                pairs[i][pp[i]]=bucket[ki]["pairs"][pp[i]]; ei.append(i)
        if ei:
            np2,pr2=_batched_forward(model,tok,tf(ei),DEV,C.MAX_LENGTH,256)
            for j,i in enumerate(ei):
                pooled[i]=np2[j]; probs[i]=pr2[j]; pp[i]-=1
                if _margin_of(probs[i:i+1],explore_ids)[0]>=CM or pp[i]<0: done[i]=True
        active=[i for i in active if not done[i]]
    return probs.argmax(1)

after=edit(rows)
edit_pred=base_pred.copy(); edit_pred[sel]=after
edit_f1=f1_score(gold,edit_pred,average="macro")
changed=int((base_pred[sel]!=after).sum())
print(f"\n{'='*55}\nval 전체 Macro-F1 (gold-free, 테스트-유효)")
print(f"  exp006 base        : {base_f1:.4f}")
print(f"  + RL편집(gold-free) : {edit_f1:.4f}   (delta {edit_f1-base_f1:+.4f})")
print(f"  선택 {len(sel)}개 중 예측 바뀐 것 {changed}개")
print('='*55)
for ci in explore_ids:
    c=C.CLASS_NAMES[ci]; b=f1_score(gold==ci,base_pred==ci); e=f1_score(gold==ci,edit_pred==ci)
    print(f"  {c:15s} F1 {b:.3f} -> {e:.3f} ({e-b:+.3f})")
