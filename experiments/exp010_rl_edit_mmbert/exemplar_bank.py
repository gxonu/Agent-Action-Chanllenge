"""
탐색 4개 클래스(read_file/grep_search/list_directory/glob_pattern) 안에서
'임베딩 공간에서 잘 분리되는(=kNN purity 높은) 샘플'만 골라
(클래스, history 길이 L) 버킷별로 저장해두는 exemplar bank.

주의: "잘 분리됨"을 2D t-SNE 그림으로 눈대중 판단하지 않는다 — t-SNE는 거리를
왜곡하므로, 반드시 t-SNE에 넣기 전의 원본 고차원 pooled 임베딩에서 kNN purity로
정량 판정한다.

전제: 이미 학습된 model(T5GemmaEncoderClassifier, LoRA 포함)과 val DataFrame이
메모리에 있어야 한다 (train.py 학습 스크립트를 먼저 돌린 뒤 사용).
"""
import json
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

import config as C
from common import get_history_pairs

EXPLORE_CLASSES = ["read_file", "grep_search", "list_directory", "glob_pattern"]


@torch.no_grad()
def extract_pooled_embeddings(model, tok, texts, device, batch_size=32, max_length=None):
    """model.forward는 pooled 벡터를 만든 뒤 곧장 classifier를 통과시키므로,
    분류기 직전의 pooled 벡터만 별도로 뽑기 위해 attn-pool 부분만 재현한다."""
    from mmbert_fwd import mmbert_pooled_logits
    max_length = max_length or C.MAX_LENGTH
    model.eval()
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt").to(device)
        pooled, _ = mmbert_pooled_logits(model, enc)
        outs.append(pooled.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def knn_purity(embeddings: np.ndarray, gold_labels, k: int = 15) -> np.ndarray:
    """각 샘플의 k-최근접 이웃(자기 자신 제외) 중 같은 gold 라벨 비율."""
    k_eff = min(k, len(embeddings) - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(embeddings)
    _, idx = nn.kneighbors(embeddings)
    idx = idx[:, 1:]
    labels = np.array(gold_labels)
    neigh_labels = labels[idx]
    return (neigh_labels == labels[:, None]).mean(axis=1)


def build_exemplar_bank(df_explore, embeddings, purity_threshold=0.9,
                        k_per_bucket=8, max_history_steps=None):
    """
    df_explore: 탐색 4클래스만 필터링된 val DataFrame
                (columns: id, current_prompt, history, session_meta, action)
    embeddings: df_explore와 '같은 순서'의 pooled 임베딩 (N, h) — 순서 어긋나면
                purity/버킷이 다 잘못되니 반드시 df_explore.reset_index(drop=True) 후
                동일 순서로 뽑은 embeddings를 넣을 것.
    반환: bank[class][str(L)] = [ {"pairs":[...], "id":..., "purity":...}, ... ]
    """
    max_history_steps = max_history_steps or C.SER.max_history_steps
    df_explore = df_explore.reset_index(drop=True)
    gold = df_explore["action"].tolist()
    purity = knn_purity(embeddings, gold, k=15)

    bank = {c: {} for c in EXPLORE_CLASSES}
    kept, total = 0, 0
    for (_, row), p in zip(df_explore.iterrows(), purity):
        total += 1
        if p < purity_threshold:
            continue
        pairs = get_history_pairs(row["history"], max_history_steps)
        L = len(pairs)
        if L == 0:
            continue  # 바꿔치기할 step 자체가 없으면 exemplar로 못 씀
        cls = row["action"]
        key = str(L)
        bank[cls].setdefault(key, [])
        if len(bank[cls][key]) >= k_per_bucket:
            continue
        bank[cls][key].append({"pairs": pairs, "id": str(row["id"]), "purity": float(p)})
        kept += 1

    print(f"[exemplar_bank] {kept}/{total} 샘플이 purity>={purity_threshold} 기준 통과")
    for c in EXPLORE_CLASSES:
        sizes = {L: len(v) for L, v in bank[c].items()}
        print(f"  {c}: buckets(L->개수) = {sizes}")
    return bank


def save_bank(bank, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=1)


def load_bank(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)