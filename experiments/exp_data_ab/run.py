"""
데이터축 honest A/B — au 제거 / 경모 증강 가치 재검증 (S1 예약 2건).
동일한 sim-only held-out val에, TRAIN 구성만 바꿔 mmbert 학습 → sim-only Macro-F1 델타.

conditions (--cond):
  sim   : train = sim만 (au 제외)              ← au-제거
  full  : train = sim + au (전량)              ← 현행(baseline)
  aug   : train = sim + au + 경모증강5299      ← 증강 가치
  augNoau: train = sim + 경모증강 (au 제외)

val: sim 세션 group-split hold-out(seed42, ~10%). au는 train에만, val엔 절대 없음.
사용: python run.py --cond full --gpu 0
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,"..",".."))
sys.path.insert(0, os.path.join(ROOT,"team_ref","sanghyuk_v01_lb0.77585")); import common
CLASSES=["read_file","grep_search","list_directory","glob_pattern","edit_file","write_file",
 "apply_patch","run_bash","run_tests","lint_or_typecheck","ask_user","plan_task","web_search","respond_only"]
C2I={c:i for i,c in enumerate(CLASSES)}
SER=dict(prompt_marker="[PROMPT]",history_marker="[HISTORY]",meta_marker="[META]",
         max_history_steps=12,add_symbol_cues=False,add_open_files=True,add_langmix=False,max_open_files=8)

def load(path_jsonl, path_csv):
    lab=pd.read_csv(path_csv); rows=[json.loads(l) for l in open(path_jsonl,encoding="utf-8")]
    df=pd.DataFrame(rows).merge(lab,on="id",how="left")
    df["ser"]=[common.serialize(p,h,m,SER) for p,h,m in zip(df["current_prompt"],df["history"],df["session_meta"])]
    return df

def main(cond, gpu, epochs):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    import torch
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import f1_score
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, DataCollatorWithPadding)
    tr=load(os.path.join(ROOT,"data/raw/train.jsonl"),os.path.join(ROOT,"data/raw/train_labels.csv"))
    tr["prefix"]=np.where(tr["id"].str.startswith("sess_au_"),"au","sim")
    tr["session"]=tr["id"].str.replace(r"-step_\d+$","",regex=True)
    # sim만으로 group-split → sim-only val 고정 (seed42)
    sim=tr[tr.prefix=="sim"].reset_index(drop=True)
    gss=GroupShuffleSplit(1,test_size=0.1,random_state=42)
    tri,vai=next(gss.split(sim,groups=sim["session"]))
    val=sim.iloc[vai]; sim_train=sim.iloc[tri]
    au=tr[tr.prefix=="au"]
    parts={"sim":[sim_train],"full":[sim_train,au],
           "aug":[sim_train,au],"augNoau":[sim_train],"trans":[sim_train]}[cond]
    if cond in ("aug","augNoau"):
        aug=load(os.path.join(ROOT,"team_ref/augmented_only_clean_after_wrapper_removal_5299.jsonl"),
                 os.path.join(ROOT,"team_ref/augmented_only_clean_after_wrapper_removal_5299_labels.csv"))
        parts.append(aug)
    if cond=="trans":
        tra=load(os.path.join(HERE,"trans_aug_clean.jsonl"),os.path.join(HERE,"trans_aug_clean_labels.csv"))
        parts.append(tra)
    train=pd.concat(parts,ignore_index=True)
    print(f"[{cond}] train {len(train)} (sim {len(sim_train)} + au {len(au) if cond in ('full','aug') else 0} + aug {'Y' if cond in ('aug','augNoau') else 'N'}) | sim-val {len(val)}",flush=True)

    tok=AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    class DS(torch.utils.data.Dataset):
        def __init__(s,texts,ys): s.e=tok(list(texts),truncation=True,max_length=512); s.y=[int(v) for v in ys]
        def __len__(s): return len(s.y)
        def __getitem__(s,i): return {"input_ids":s.e["input_ids"][i],"attention_mask":s.e["attention_mask"][i],"labels":s.y[i]}
    ytr=train["action"].map(C2I).values; yva=val["action"].map(C2I).values
    cnt=np.bincount(ytr,minlength=14).astype(float); cw=torch.tensor(cnt.sum()/(14*cnt),dtype=torch.float)
    class WT(Trainer):
        def compute_loss(s,model,inp,return_outputs=False,**kw):
            lb=inp.pop("labels"); out=model(**inp)
            loss=torch.nn.functional.cross_entropy(out.logits,lb,weight=cw.to(out.logits.device))
            return (loss,out) if return_outputs else loss
    model=AutoModelForSequenceClassification.from_pretrained("jhu-clsp/mmBERT-base",num_labels=14,
        id2label={i:c for i,c in enumerate(CLASSES)},label2id=C2I)
    args=TrainingArguments(output_dir=os.path.join(HERE,f"_tmp_{cond}"),num_train_epochs=epochs,
        per_device_train_batch_size=32,per_device_eval_batch_size=128,learning_rate=2e-5,
        warmup_ratio=0.06,weight_decay=0.01,bf16=torch.cuda.is_bf16_supported(),
        report_to=[],logging_steps=200,save_strategy="no",eval_strategy="no",seed=42)
    t=WT(model=model,args=args,train_dataset=DS(train["ser"],ytr),data_collator=DataCollatorWithPadding(tok))
    t.train()
    logits=t.predict(DS(val["ser"],yva)).predictions
    probs=torch.softmax(torch.tensor(logits),-1).numpy()
    mf1=f1_score(yva,probs.argmax(1),labels=list(range(14)),average="macro",zero_division=0)
    outd=os.path.join(HERE,cond); os.makedirs(outd,exist_ok=True)
    np.savez(os.path.join(outd,"val_probs.npz"),ids=val["id"].values,probs=probs,y=yva,classes=np.array(CLASSES))
    json.dump({"cond":cond,"sim_val_macro_f1":round(float(mf1),4),"n_train":len(train)},
              open(os.path.join(outd,"metrics.json"),"w"),indent=2)
    print(f"\n=== [{cond}] sim-only val Macro-F1 = {mf1:.4f} ===",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--cond",required=True,choices=["sim","full","aug","augNoau","trans"])
    ap.add_argument("--gpu",type=int,required=True); ap.add_argument("--epochs",type=int,default=4)
    a=ap.parse_args(); main(a.cond,a.gpu,a.epochs)
