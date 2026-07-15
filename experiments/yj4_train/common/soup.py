"""Model Soup — 동일 아키텍처 fold 가중치 평균 (uniform / greedy).

- fold마다 분류 헤드 init seed 고정 전제(정렬된 head-space).
- greedy: 프로즌 홀드아웃 macro-F1 로 후보 채택(Wortsman et al., ICML 2022).
torch 필요(Colab). 상태사전(state_dict) 단위로 동작.
"""
from __future__ import annotations


def average_state_dicts(state_dicts):
    """가중 평균(단순 평균). 모든 텐서 키 동일 가정."""
    import torch
    keys = state_dicts[0].keys()
    out = {}
    for k in keys:
        if state_dicts[0][k].dtype.is_floating_point:
            out[k] = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)
        else:
            out[k] = state_dicts[0][k].clone()  # 정수 버퍼 등은 첫 것 유지
    return out


def uniform_soup(state_dicts):
    return average_state_dicts(state_dicts)


def greedy_soup(state_dicts, val_scores, eval_fn, tol=1e-4):
    """val_scores 내림차순으로 후보를 하나씩 추가, 홀드아웃 macro-F1 개선 시만 채택.

    eval_fn(state_dict) -> float (macro-F1). 반환: (souped_state_dict, ingredients_idx, best).
    """
    order = sorted(range(len(state_dicts)), key=lambda i: -val_scores[i])
    ingredients = [order[0]]
    cur = average_state_dicts([state_dicts[order[0]]])
    best = eval_fn(cur)
    for idx in order[1:]:
        cand = average_state_dicts([state_dicts[i] for i in ingredients + [idx]])
        s = eval_fn(cand)
        if s >= best - tol:
            ingredients.append(idx)
            cur = cand
            best = s
    return cur, ingredients, best
