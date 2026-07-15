"""
exp_lgbm — 구조화 피처 + LightGBM = 앙상블용 diverse 멤버 (S2/데이터 축)

목적: mmbert(트랜스포머)와 '다르게 틀리는' 트리 모델을 만들어 1GB 앙상블에 거의 공짜로 diversity 추가.
- 앙상블 정렬: exp006(mmbert)의 val id를 그대로 미러링 → 같은 fold라 즉시 blend/비교 가능.
- LightGBM은 (last_action × turn × n_open) 같은 상호작용을 자동으로 잡음 = inspect에서 인코더가 약한 부분.

사용:
  python experiments/exp_lgbm/train_lgbm.py            # 학습+평가+앙상블 체크 (CPU, GPU 불필요)
  python experiments/exp_lgbm/train_lgbm.py --seeds 3  # 여러 seed 멤버(추가 diversity)

산출: experiments/exp_lgbm/val_probs.npz (+ metrics 콘솔). mmbert와 disagreement/앙상블 이득 리포트.
"""
import os, sys, json, re, argparse
import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import f1_score
import lightgbm as lgb

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,"..",".."))
CLASSES=["read_file","grep_search","list_directory","glob_pattern","edit_file","write_file",
         "apply_patch","run_bash","run_tests","lint_or_typecheck","ask_user","plan_task","web_search","respond_only"]
C2I={c:i for i,c in enumerate(CLASSES)}
MMBERT_VP=os.path.join(ROOT,"experiments/exp006_mmbert_1024/model/val_probs.npz")

# --- rare/inspect cue 정규식 (EDA v3 V5 + serialize_variants 재사용) ---
RE={
 "dir": re.compile(r"목록|디렉|폴더|구조|안에|트리|리스트|\bls\b|list|structure",re.I),
 "glob":re.compile(r"\*|모든|전부|패턴|매칭|glob|all |every|\.\w+\s*파일",re.I),
 "find":re.compile(r"어디|찾|검색|참조|정의|import|where|find|search|uses?|referenc|grep",re.I),
 "read":re.compile(r"열어|보여|읽|봐|확인|열|show|read|open|check|look",re.I),
 "tc":  re.compile(r"타입\s*체크|타입체크|typecheck|정적\s*분석|린트|\blint\b|mypy|\btsc\b",re.I),
 "newf":re.compile(r"골격|스캐폴드|scaffold|scratch|처음부터|overwrite|새\s*파일|만들어\s*줘요|모듈을",re.I),
 "ws":  re.compile(r"웹\s*검색|검색해\s*줘|최신|공식\s*문서|베스트\s*프랙티스|changelog|릴리스\s*노트",re.I),
}
def g(sm,*k):
    d=sm
    for x in k: d=(d or {}).get(x) if isinstance(d,dict) else None
    return d
def hist_actions(h):
    return [t.get("name") for t in h if isinstance(t,dict) and (t.get("role")=="assistant_action" or t.get("name"))] if isinstance(h,list) else []
def plang(s):
    s=str(s); ho=len(re.findall(r"[가-힣]",s)); al=len(re.findall(r"[A-Za-z]",s))
    return 0 if (ho and al) else 1 if (al and not ho) else 2  # 0=mixed 1=en 2=ko

def build(df):
    rows=[]
    for r in df.itertuples():
        sm=r.session_meta; ws=(sm or {}).get("workspace") or {}
        ha=hist_actions(r.history); p=str(r.current_prompt)
        of=ws.get("open_files") or []
        lm=ws.get("language_mix") or {}
        toplang=max(lm,key=lm.get) if isinstance(lm,dict) and lm else "na"
        cnt=Counter(ha)
        feat={
          "turn":g(sm,"turn_index"), "elapsed":g(sm,"elapsed_session_sec"),
          "budget":g(sm,"budget_tokens_remaining"), "loc":ws.get("loc"),
          "n_open":len(of), "hlen":len(ha),
          "tier":{"free":0,"pro":1,"enterprise":2}.get(g(sm,"user_tier"),-1),
          "lang_pref":{"ko":0,"en":1,"mixed":2}.get(g(sm,"language_pref"),-1),
          "ci":{"passed":0,"failed":1,"none":2}.get(ws.get("last_ci_status"),-1),
          "dirty":int(bool(ws.get("git_dirty"))),
          "toplang":hash(toplang)%50,
          "last_action":C2I.get(ha[-1],-1) if ha else -1,
          "last2_action":C2I.get(ha[-2],-1) if len(ha)>=2 else -1,
          "plen":len(p), "q":int(p.strip().endswith("?")), "short":int(len(p)<25),
          "plang":plang(p),
          # 최근 action별 개수 (workflow 신호)
          **{f"h_{c}":cnt.get(c,0) for c in CLASSES},
        }
        for k,rgx in RE.items(): feat[f"kw_{k}"]=int(bool(rgx.search(p)))
        rows.append(feat)
    X=pd.DataFrame(rows)
    return X
CAT=["tier","lang_pref","ci","toplang","last_action","last2_action","plang"]

def macro(y,pred): return f1_score(y,pred,labels=list(range(14)),average="macro",zero_division=0)

def main(seeds):
    lab=pd.read_csv(os.path.join(ROOT,"data/raw/train_labels.csv"))
    rows=[json.loads(l) for l in open(os.path.join(ROOT,"data/raw/train.jsonl"),encoding="utf-8")]
    df=pd.DataFrame(rows).merge(lab,on="id",how="left")
    y=df["action"].map(C2I).values
    # exp006 val id 미러링 → 같은 fold
    mm=np.load(MMBERT_VP,allow_pickle=True); val_ids=set(str(x) for x in mm["ids"])
    is_val=df["id"].astype(str).isin(val_ids).values
    print(f"[align] val {is_val.sum()} (exp006와 동일), train {(~is_val).sum()}")
    X=build(df)
    for c in CAT: X[c]=X[c].astype("category")
    Xtr,ytr=X[~is_val],y[~is_val]; Xva,yva=X[is_val],y[is_val]

    # 여러 seed로 bagging (안정성 + 약한 diversity)
    proba=np.zeros((is_val.sum(),14))
    for s in range(seeds):
        m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=48,
            subsample=0.8,colsample_bytree=0.8,min_child_samples=50,
            class_weight="balanced",random_state=42+s,n_jobs=8,verbose=-1)  # 8코어 제한(서버 예절)
        print(f"  seed {s} 학습중...",flush=True)
        m.fit(Xtr,ytr,categorical_feature=CAT)
        proba+=m.predict_proba(Xva)
    proba/=seeds
    pred=proba.argmax(1)
    f=macro(yva,pred)
    print(f"\n[LightGBM] val Macro-F1 = {f:.4f}  (seeds={seeds})")
    # 저장 (앙상블용)
    order_ids=[str(x) for x in mm["ids"]]
    pos={id_:i for i,id_ in enumerate(df.loc[is_val,"id"].astype(str).values)}
    ridx=[pos[i] for i in order_ids]
    out=os.path.join(HERE,"val_probs.npz")
    np.savez(out,ids=np.array(order_ids),probs=proba[ridx],y=mm["y"],classes=np.array(CLASSES))
    print(f"[saved] {out}")

    # === mmbert와 diversity/앙상블 ===
    mmp=mm["probs"]; ymm=mm["y"]
    if ymm.dtype.kind in "US": ymm=np.array([C2I[str(v)] for v in ymm])
    lgp=proba[ridx]
    mm_pred=mmp.argmax(1); lg_pred=lgp.argmax(1)
    print(f"\n[vs mmbert] mmbert 단독 {macro(ymm,mm_pred):.4f} | LGBM 단독 {macro(ymm,lg_pred):.4f}")
    print(f"  불일치율(다르게 예측): {(mm_pred!=lg_pred).mean()*100:.1f}%  (klue는 11.8%였음)")
    print(f"  둘 중 하나라도 정답: {((mm_pred==ymm)|(lg_pred==ymm)).mean()*100:.1f}%")
    best=(-1,0)
    for w in np.arange(0,1.001,0.05):
        pp=(w*mmp+(1-w)*lgp).argmax(1); ff=macro(ymm,pp)
        if ff>best[0]: best=(ff,round(w,2))
    print(f"  앙상블 mmbert*{best[1]}+LGBM*{round(1-best[1],2)} = {best[0]:.4f}  (Δ mmbert 대비 {best[0]-macro(ymm,mm_pred):+.4f})")
    # inspect에서 특히 다른가
    INS=[C2I[c] for c in CLASSES[:4]]
    ins=np.isin(ymm,INS)
    print(f"  [inspect만] 불일치율 {(mm_pred[ins]!=lg_pred[ins]).mean()*100:.1f}%")
    # feature importance top10
    imp=pd.Series(m.feature_importances_,index=X.columns).sort_values(ascending=False)
    print("\n  LGBM 상위 피처:", ", ".join(f"{k}({v})" for k,v in imp.head(10).items()))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=1); a=ap.parse_args()
    main(a.seeds)
