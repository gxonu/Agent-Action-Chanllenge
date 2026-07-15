"""
Vocab pruning APPLY + 무손실 검증.
 - keep = (train+test 등장 토큰) ∪ special ∪ (안전버퍼 저빈도 id 0..buffer)
 - 임베딩 행을 keep만 slice, remap(old_id→new_id) 저장
 - 검증: pruned 모델로 val 7000 재추론 → 원본 val_probs와 Macro-F1/argmax 일치 확인 (val⊂train이라 무손실이어야 정상)

사용: python src/prune_apply.py --model mmbert --gpu 4 --buffer 30000
출력: experiments/exp_prune/{model}/ (pruned model.safetensors + config + remap.npy + tokenizer)
"""
import os, sys, json, argparse
import numpy as np
from collections import Counter

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,".."))
sys.path.insert(0, os.path.join(ROOT,"team_ref","sanghyuk_v01_lb0.77585"))
import common
SER=dict(prompt_marker="[PROMPT]",history_marker="[HISTORY]",meta_marker="[META]",
         max_history_steps=12,add_symbol_cues=False,add_open_files=True,add_langmix=False,max_open_files=8)

CFG={
 "mmbert":dict(model_dir="experiments/exp006_mmbert_1024/model",
               tok="jhu-clsp/mmBERT-base", emb_key="model.embeddings.tok_embeddings.weight",
               val="experiments/exp006_mmbert_1024/model/val_probs.npz"),
 "bge":dict(model_dir="experiments/exp_encoders/runs/bge_m3",
            tok="BAAI/bge-m3", emb_key="roberta.embeddings.word_embeddings.weight",
            val="experiments/exp_encoders/runs/bge_m3/val_probs.npz"),
 "xlmr":dict(model_dir="experiments/exp_encoders/runs/xlmr_large",
             tok="FacebookAI/xlm-roberta-large", emb_key="roberta.embeddings.word_embeddings.weight",
             val="experiments/exp_encoders/runs/xlmr_large/val_probs.npz"),
}

def serialize_row(r): return common.serialize(r.get("current_prompt"),r.get("history"),r.get("session_meta"),SER)

def main(model, gpu, buffer):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    from sklearn.metrics import f1_score
    c=CFG[model]; md=os.path.join(ROOT,c["model_dir"])
    tok=AutoTokenizer.from_pretrained(c["tok"])

    # 1) 등장 토큰 집계 (train+test)
    print("[1] 토큰 빈도 집계...")
    texts_all=[]
    for fn in ["data/raw/train.jsonl","data/raw/test.jsonl"]:
        p=os.path.join(ROOT,fn)
        if os.path.exists(p):
            for l in open(p,encoding="utf-8"): texts_all.append(serialize_row(json.loads(l)))
    appeared=set()
    for i in range(0,len(texts_all),2000):
        for ids in tok(texts_all[i:i+2000],add_special_tokens=True,truncation=True,max_length=512)["input_ids"]:
            appeared.update(ids)
    keep=sorted(appeared | set(tok.all_special_ids) | set(range(min(buffer, tok.vocab_size))))
    V=tok.vocab_size
    print(f"    등장 {len(appeared):,} + 버퍼 → keep {len(keep):,} / {V:,}  ({len(keep)/V*100:.1f}%)")

    # 2) remap + 임베딩 slice
    print("[2] 임베딩 pruning...")
    sd=load_file(os.path.join(md,"model.safetensors"))
    emb=sd[c["emb_key"]]
    keep_t=torch.tensor(keep,dtype=torch.long)
    new_emb=emb[keep_t].clone()
    remap=np.full(V, tok.unk_token_id if tok.unk_token_id is not None else 0, dtype=np.int64)
    for new_id,old_id in enumerate(keep): remap[old_id]=new_id
    sd[c["emb_key"]]=new_emb
    outdir=os.path.join(ROOT,"experiments","exp_prune",model); os.makedirs(outdir,exist_ok=True)
    save_file(sd, os.path.join(outdir,"model.safetensors"))
    np.save(os.path.join(outdir,"remap.npy"), remap)
    cfg=AutoConfig.from_pretrained(md); cfg.vocab_size=len(keep); cfg.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    import shutil
    orig_mb=os.path.getsize(os.path.join(md,"model.safetensors"))/1e6
    new_mb=os.path.getsize(os.path.join(outdir,"model.safetensors"))/1e6
    print(f"    fp16 크기: {orig_mb:.0f}MB → {new_mb:.0f}MB (int8 시 대략 절반)")

    # 3) 무손실 검증: pruned 모델로 val 재추론
    print("[3] 무손실 검증 (val 7000 재추론)...")
    vp=np.load(os.path.join(ROOT,c["val"]),allow_pickle=True)
    val_ids=[str(x) for x in vp["ids"]]; classes=[str(x) for x in vp["classes"]]
    y=vp["y"];
    if y.dtype.kind in "US": y=np.array([classes.index(str(v)) for v in y])
    orig_pred=vp["probs"].argmax(1)
    # val 텍스트 복원
    id2text={}
    for l in open(os.path.join(ROOT,"data/raw/train.jsonl"),encoding="utf-8"):
        r=json.loads(l)
        if r["id"] in set(val_ids): id2text[r["id"]]=serialize_row(r)
    texts=[id2text[i] for i in val_ids]
    dev="cuda"
    m=AutoModelForSequenceClassification.from_pretrained(outdir,torch_dtype=torch.float16).to(dev).eval()
    remap_t=torch.tensor(remap,dtype=torch.long,device=dev)
    probs=[]
    with torch.no_grad():
        for i in range(0,len(texts),128):
            enc=tok(texts[i:i+128],return_tensors="pt",padding=True,truncation=True,max_length=512)
            enc={k:v.to(dev) for k,v in enc.items()}
            enc["input_ids"]=remap_t[enc["input_ids"]]   # ★ id remap
            logit=m(**enc).logits.float()
            probs.append(torch.softmax(logit,-1).cpu().numpy())
    probs=np.concatenate(probs)
    new_pred=probs.argmax(1)
    from sklearn.metrics import f1_score
    f_orig=f1_score(y,orig_pred,average="macro",zero_division=0)
    f_new=f1_score(y,new_pred,average="macro",zero_division=0)
    agree=(orig_pred==new_pred).mean()*100
    print(f"    원본 Macro-F1 {f_orig:.4f} | pruned {f_new:.4f} | argmax 일치 {agree:.2f}%")
    print(f"    {'✅ 무손실' if abs(f_orig-f_new)<0.002 and agree>99 else '⚠️ 손실 있음 — 버퍼 늘려야'}")
    np.savez(os.path.join(outdir,"val_probs.npz"), ids=np.array(val_ids),probs=probs,y=vp["y"],classes=vp["classes"])

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True,choices=list(CFG))
    ap.add_argument("--gpu",type=int,required=True)
    ap.add_argument("--buffer",type=int,default=30000)
    a=ap.parse_args(); main(a.model,a.gpu,a.buffer)
