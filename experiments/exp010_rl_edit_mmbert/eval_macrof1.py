"""정책 적용 전/후 전체 val Macro-F1 계산.
exp006 예측(전체 val) 중 confusing inspect 샘플만 정책 편집 예측으로 교체 → Macro-F1 비교."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import config as C
from common import serialize, serialize_config_from
from exemplar_bank import EXPLORE_CLASSES, load_bank
from rl_edit import run_vectorized_batch, PolicyNet, _confused_of
from mmbert_fwd import mmbert_pooled_logits

D = os.path.dirname(__file__); MODEL_DIR = os.path.join(D, "../exp006_mmbert_1024/model"); DEV="cuda"
K=8; CM=0.2
C.SER.add_open_files=True
ser_cfg=serialize_config_from(C.SER)
class2id={c:i for i,c in enumerate(C.CLASS_NAMES)}; explore_ids=[class2id[c] for c in EXPLORE_CLASSES]

df=pd.read_json(os.path.join(D,"../../data/processed/train_merged.jsonl"),lines=True)
groups=np.array([re.sub(r"-step_\d+$","",str(i)) for i in df["id"].astype(str)]); y=np.array([class2id[a] for a in df["action"]])
tr,va=next(StratifiedGroupKFold(10,shuffle=True,random_state=42).split(np.arange(len(df)),y,groups))
df_va=df.iloc[va].reset_index(drop=True)
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

# exp006 base 예측 (전체 val)
pr=probs_all(df_va); base_pred=pr.argmax(1)
base_f1=f1_score(gold,base_pred,average="macro")

# confusing inspect (val)
is_insp=df_va["action"].isin(EXPLORE_CLASSES).values
insp_idx=np.where(is_insp)[0]
df_va_i=df_va.iloc[insp_idx].reset_index(drop=True)
pr_i=pr[insp_idx]; gold_i=gold[insp_idx]
conf_mask=_confused_of(pr_i,gold_i,explore_ids,CM)
conf_local=np.where(conf_mask)[0]                 # df_va_i 내 위치
conf_global=insp_idx[conf_local]                  # df_va 내 위치
df_conf=df_va_i.iloc[conf_local].reset_index(drop=True)

# 정책 로드 후 편집 예측
bank=load_bank(os.path.join(D,"ckpt","best.pt".replace("best.pt","bank.json"))) if os.path.exists(os.path.join(D,"ckpt","bank.json")) else None
# bank 저장 안했으면 재구축 필요 → run.py가 저장안함; 여기선 ckpt의 policy만 쓰고 bank 재구축
from exemplar_bank import extract_pooled_embeddings, build_exemplar_bank
df_tr=df.iloc[tr].reset_index(drop=True); df_tr_i=df_tr[df_tr["action"].isin(EXPLORE_CLASSES)].reset_index(drop=True)
emb=extract_pooled_embeddings(model,tok,[serialize(r.current_prompt,r.history,r.session_meta,ser_cfg) for r in df_tr_i.itertuples()],DEV,256)
bank=build_exemplar_bank(df_tr_i,emb,0.9,K)

ck=torch.load(os.path.join(D,"ckpt","last.pt"),map_location=DEV)
policy=PolicyNet(model.config.hidden_size+8,4*K+1).to(DEV); policy.load_state_dict(ck["policy"]); policy.eval()

# val confusing 편집 예측 수집
after_pred_conf=np.empty(len(df_conf),dtype=int); pos=0
rows=df_conf.to_dict("records")
for s in range(0,len(rows),256):
    br=rows[s:s+256]
    bp,ap,_=run_vectorized_batch(model,tok,bank,policy,None,br,DEV,ser_cfg,K,CM,0.0,12,1.0,0.0,256,class2id,explore_ids,train=False)
    after_pred_conf[pos:pos+len(ap)]=ap; pos+=len(ap)

# 전체 val 예측: base에서 confusing inspect만 교체
edit_pred=base_pred.copy()
edit_pred[conf_global]=after_pred_conf
edit_f1=f1_score(gold,edit_pred,average="macro")

print(f"\n{'='*55}")
print(f"val 전체 Macro-F1  (raw, threshold 전)")
print(f"  exp006 base           : {base_f1:.4f}")
print(f"  + RL 입력편집(confusing): {edit_f1:.4f}   (delta {edit_f1-base_f1:+.4f})")
print(f"{'='*55}")
# inspect 클래스별 F1 비교
for ci in explore_ids:
    c=C.CLASS_NAMES[ci]
    b=f1_score(gold==ci,base_pred==ci); e=f1_score(gold==ci,edit_pred==ci)
    print(f"  {c:15s} F1: {b:.3f} -> {e:.3f} ({e-b:+.3f})")
