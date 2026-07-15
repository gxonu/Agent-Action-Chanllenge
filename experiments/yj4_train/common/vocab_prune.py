"""Vocab 프루닝 — 임베딩 행을 사용 토큰만 남겨 모델 축소 (sentencepiece 무수정).

원리: 토크나이저는 그대로 두고, 토큰화 후 id를 리매핑(full→compact)한다.
- used_ids = train 직렬화 전체 + 특수토큰 + 짧은 피스(len<=2, 미지 단어 분해 대비)
- id_map.npy: (V,) int32, 미사용 id → <unk> compact id
- 모델: 입력 임베딩 weight[used_ids]로 교체, config.vocab_size=K

효과: xlm-r-base 556MB→~280MB, large 1.12GB→~0.75GB (fp16) → 1GB 안에 앙상블 가능.
"""
from __future__ import annotations
import os
import json
import numpy as np


def collect_used_ids(tokenizer, texts, max_len=320, extra_short_piece_len=2, batch=2000):
    """사용 토큰 id 수집: 관측 ∪ 특수 ∪ 짧은피스(안전마진)."""
    used = set(tokenizer.all_special_ids)
    for b in range(0, len(texts), batch):
        enc = tokenizer(texts[b:b + batch], truncation=True, max_length=max_len, padding=False)
        for ids in enc["input_ids"]:
            used.update(ids)
    # 미지 단어 분해 대비: 길이<=N 피스 전부 포함
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    if extra_short_piece_len > 0:
        toks = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
        for i, t in enumerate(toks):
            if t is not None and len(t.replace("▁", "")) <= extra_short_piece_len:
                used.add(i)
    used = sorted(x for x in used if 0 <= x < vocab_size)
    return used


def build_id_map(used_ids, vocab_size, unk_id):
    """full id → compact id 매핑. 미사용 → compact(unk)."""
    used = list(used_ids)
    if unk_id not in used:
        used = sorted(set(used) | {unk_id})
    pos = {t: i for i, t in enumerate(used)}
    id_map = np.full(vocab_size, pos[unk_id], dtype=np.int32)
    for t, i in pos.items():
        id_map[t] = i
    return id_map, used


def prune_model_dir(model_dir_in, model_dir_out, tokenizer, texts, max_len=320):
    """학습된 분류모델 디렉터리 → 프루닝된 디렉터리(+id_map.npy)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoConfig
    used = collect_used_ids(tokenizer, texts, max_len)
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    id_map, used = build_id_map(used, vocab_size, tokenizer.unk_token_id)
    K = len(used)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir_in)
    emb = model.get_input_embeddings()
    W = emb.weight.data  # (V, H)
    newW = W[torch.tensor(used, dtype=torch.long)]
    new_emb = torch.nn.Embedding(K, W.shape[1],
                                 padding_idx=int(id_map[tokenizer.pad_token_id]))
    new_emb.weight.data.copy_(newW)
    model.set_input_embeddings(new_emb)
    model.config.vocab_size = K
    model.config.pad_token_id = int(id_map[tokenizer.pad_token_id])
    os.makedirs(model_dir_out, exist_ok=True)
    model.half().save_pretrained(model_dir_out, safe_serialization=True)
    tokenizer.save_pretrained(model_dir_out)
    np.save(os.path.join(model_dir_out, "id_map.npy"), id_map)
    meta = {"pruned": True, "orig_vocab": int(vocab_size), "kept": int(K)}
    json.dump(meta, open(os.path.join(model_dir_out, "prune_meta.json"), "w"))
    return K, id_map
