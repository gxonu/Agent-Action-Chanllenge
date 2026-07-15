"""
Vocab pruning — 안 쓰는 토큰 어휘를 잘라 임베딩 테이블 축소 (성능 무손실, 1GB 앙상블 enabler).

단계:
 1) train+test(dummy) 전수를 각 모델 serialize+tokenize → 토큰 빈도 집계
 2) keep 집합 = (등장 토큰) ∪ (special tokens). 크기·절약량 projection
 3) --apply 시 각 모델 임베딩을 keep 토큰만 slice + id remap → 저장

이 스크립트는 STAGE 1-2(빈도·projection, CPU, 안전)만. --apply는 별도 확인 후.
XLM-R 계열(bge/xlmr)은 동일 vocab → keep 집합 공유 가능.
"""
import os, sys, json, argparse
import numpy as np
from collections import Counter

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,".."))
sys.path.insert(0, os.path.join(ROOT,"team_ref","sanghyuk_v01_lb0.77585"))
import common

SER_CFG=dict(prompt_marker="[PROMPT]",history_marker="[HISTORY]",meta_marker="[META]",
             max_history_steps=12,add_symbol_cues=False,add_open_files=True,add_langmix=False,max_open_files=8)

def load_texts():
    texts=[]
    for fn in ["data/raw/train.jsonl","data/raw/test.jsonl"]:
        p=os.path.join(ROOT,fn)
        if not os.path.exists(p): continue
        for line in open(p,encoding="utf-8"):
            r=json.loads(line)
            texts.append(common.serialize(r.get("current_prompt"),r.get("history"),r.get("session_meta"),SER_CFG))
    return texts

def analyze(tok_name, texts, model_type, vocab_size, hidden, cur_int8_mb):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(tok_name)
    cnt=Counter()
    B=2000
    for i in range(0,len(texts),B):
        enc=tok(texts[i:i+B],add_special_tokens=True,truncation=True,max_length=512)
        for ids in enc["input_ids"]: cnt.update(ids)
    used=set(cnt.keys())
    # special tokens 항상 유지
    special=set(tok.all_special_ids)
    for t in range(tok.vocab_size):  # 안전: 자주 나오는 것만 보되 special 포함
        pass
    keep=used|special
    # 커버리지: 상위 K개로 자를 때 토큰 커버율
    print(f"\n[{tok_name}]  ({model_type}, vocab {vocab_size}, hidden {hidden})")
    print(f"  등장한 고유 토큰: {len(used):,} / {vocab_size:,}  ({len(used)/vocab_size*100:.1f}%)")
    print(f"  keep(등장+special): {len(keep):,}")
    # 임베딩 비중 & 절약 추정 (int8 = 1byte/param 가정)
    emb_params=vocab_size*hidden
    emb_mb_i8=emb_params/1e6
    new_emb_params=len(keep)*hidden
    saved_mb=(emb_params-new_emb_params)/1e6
    print(f"  임베딩 파라미터: {emb_params/1e6:.0f}M (int8 ~{emb_mb_i8:.0f}MB, 현재모델 {cur_int8_mb}MB의 {emb_mb_i8/cur_int8_mb*100:.0f}%)")
    print(f"  pruning 후 임베딩: {new_emb_params/1e6:.0f}M → 절약 ~{saved_mb:.0f}MB")
    print(f"  → 추정 pruned 크기: {cur_int8_mb - saved_mb:.0f}MB")
    return keep, cnt

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--apply",action="store_true"); a=ap.parse_args()
    texts=load_texts()
    print(f"총 {len(texts):,} 샘플 serialize 완료 (train+test)")
    models=[
      ("jhu-clsp/mmBERT-base","modernbert",256000,768,327),
      ("BAAI/bge-m3","xlm-roberta",250002,1024,565),
      ("FacebookAI/xlm-roberta-large","xlm-roberta",250002,1024,565),
    ]
    keeps={}
    for name,mt,vs,hd,mb in models:
        try:
            keep,_=analyze(name,texts,mt,vs,hd,mb)
            keeps[name]=keep
        except Exception as e:
            print(f"  [{name}] 실패: {str(e)[:120]}")
    # XLM-R 두 모델 keep 합집합(공유 vocab)
    xlmr=[k for k in keeps if "xlm-roberta" in k or "bge" in k]
    if len(xlmr)>=2:
        u=set().union(*[keeps[k] for k in xlmr])
        print(f"\n[공유] XLM-R 계열(bge+xlmr) keep 합집합: {len(u):,} (동일 토크나이저라 공유 가능)")
