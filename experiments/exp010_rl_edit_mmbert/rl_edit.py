"""
탐색 4개 클래스(read/grep/list/glob)에서 헷갈리는 입력을, exemplar bank에서
같은 위치(step)의 잘 분리되는 pair로 바꿔치기하는 정책을 강화학습으로 학습.

- 메인 분류기(model)는 완전히 고정(freeze). 여기서 학습되는 파라미터는 PolicyNet뿐.
- 위치는 정책이 고르지 않는다. 항상 "마지막 step(j)부터 역순"으로 고정
  (self.p = L-1 에서 시작해서 매 스텝 self.p -= 1). 정책이 고르는 건
  "그 위치에 어떤 (도너 클래스, exemplar) 조합을 넣을지"뿐.
- "정답 swap"을 미리 정해서 지도학습으로 흉내내지 않는다. 최종 분류 정오답 +
  margin 개선분을 보상으로 주는 REINFORCE(+baseline)로 학습한다.

성능: 샘플 1개씩 forward하지 않는다. 매 라운드마다 "그 시점에 아직 안 끝난
에피소드 전체"를 하나의 배치로 묶어 frozen 인코더에 한 번만 통과시킨다.
헷갈리는 샘플이 N개, 최대 스텝 12여도 forward 호출 횟수는 대략
(에피소드 배치 수) x 12 회 수준이지, N x 12 회가 아니다.
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from common import serialize_from_pairs, serialize_config_from, get_history_pairs
from exemplar_bank import EXPLORE_CLASSES

STOP_ACTION_NAME = "STOP"


# ---------------------------------------------------------------------------
# 배치 forward 유틸 (여러 텍스트를 한 번에 frozen 모델에 통과시켜 pooled+probs 반환)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _batched_forward(model, tok, texts, device, max_length, batch_forward_size):
    pooled_chunks, probs_chunks = [], []
    for s in range(0, len(texts), batch_forward_size):
        batch = texts[s:s + batch_forward_size]
        enc = tok(batch, truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt").to(device)
        from mmbert_fwd import mmbert_pooled_logits
        pooled, logits = mmbert_pooled_logits(model, enc)
        probs = torch.softmax(logits, dim=-1)
        pooled_chunks.append(pooled.cpu().numpy())
        probs_chunks.append(probs.cpu().numpy())
    return np.concatenate(pooled_chunks, axis=0), np.concatenate(probs_chunks, axis=0)


def _margin_of(probs, explore_ids):
    sub = np.sort(probs[:, explore_ids], axis=1)[:, ::-1]
    if sub.shape[1] < 2:
        return np.ones(len(probs))
    return sub[:, 0] - sub[:, 1]


def _confused_of(probs, gold_ids, explore_ids, confuse_margin):
    pred = probs.argmax(axis=1)
    return (pred != gold_ids) | (_margin_of(probs, explore_ids) < confuse_margin)


def _action_mask(L, bank, K):
    m = np.zeros(4 * K + 1, dtype=bool)
    for ci, cls in enumerate(EXPLORE_CLASSES):
        bucket = bank.get(cls, {}).get(str(L), [])
        n = min(len(bucket), K)
        m[ci * K: ci * K + n] = True
    m[4 * K] = True  # STOP은 항상 가능
    return m


class PolicyNet(nn.Module):
    """작은 MLP. 메인 모델과 별개로 이것만 학습."""

    def __init__(self, state_dim, n_actions, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


# ---------------------------------------------------------------------------
# 벡터화된 한 "메타배치"(여러 에피소드 동시 진행) 실행 (+옵션: 정책 업데이트 1회)
# ---------------------------------------------------------------------------
def run_vectorized_batch(model, tok, bank, policy, opt, rows, device, ser_cfg,
                         k_per_bucket, confuse_margin, step_penalty, max_edits,
                         gamma, entropy_coef, batch_forward_size, class2id, explore_ids,
                         train=True):
    N = len(rows)
    K = k_per_bucket
    max_length = C.MAX_LENGTH

    gold_ids = np.array([class2id[r["action"]] for r in rows])
    pairs_list = [get_history_pairs(r["history"], C.SER.max_history_steps) for r in rows]
    L = np.array([len(p) for p in pairs_list])
    p_pos = L - 1

    def texts_for(idx_list):
        return [serialize_from_pairs(rows[i]["current_prompt"], pairs_list[i],
                                     rows[i]["session_meta"], ser_cfg) for i in idx_list]

    all_idx = list(range(N))
    pooled, probs = _batched_forward(model, tok, texts_for(all_idx), device, max_length, batch_forward_size)
    before_pred = probs.argmax(axis=1).copy()

    done = (L == 0) | (~_confused_of(probs, gold_ids, explore_ids, confuse_margin))
    active = [i for i in range(N) if not done[i]]

    traj = {i: {"log_probs": [], "values": [], "rewards": [], "entropies": []} for i in active}
    diag_n_decisions = 0
    diag_n_forced_stop = 0
    diag_valid_action_sum = 0

    steps = 0
    while active and steps < max_edits:
        steps += 1
        states, masks = [], []
        for i in active:
            margin_i = _margin_of(probs[i:i + 1], explore_ids)[0]
            # margin/max만으로는 "4개 중 정확히 어느 둘이 헷갈리는지"를 정책망이 알 수 없음
            # -> 4클래스 확률을 통째로 줘서 어떤 도너 클래스를 넣을지 판단할 근거를 준다.
            aux = np.concatenate([
                np.array([p_pos[i] / max(L[i], 1), L[i] / 12.0, (p_pos[i] + 1) / 12.0,
                          margin_i], dtype=np.float32),
                probs[i, explore_ids].astype(np.float32),   # 4개 explore 클래스 raw 확률
            ])
            states.append(np.concatenate([pooled[i].astype(np.float32), aux]))
            m = _action_mask(L[i], bank, K)
            masks.append(m)
            diag_n_decisions += 1
            diag_valid_action_sum += int(m.sum())
            if m.sum() <= 1:   # STOP 하나만 가능 = 실질적으로 선택지 없음
                diag_n_forced_stop += 1

        states_t = torch.tensor(np.stack(states), device=device, dtype=torch.float32)
        masks_t = torch.tensor(np.stack(masks), device=device)

        if train:
            logits, values = policy(states_t)
        else:
            with torch.no_grad():
                logits, values = policy(states_t)
        logits = logits.masked_fill(~masks_t, float("-inf"))
        probs_a = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs_a)
        actions = dist.sample() if train else torch.argmax(probs_a, dim=-1)
        logps = dist.log_prob(actions)
        ents = dist.entropy()

        edit_idx = []
        for li, i in enumerate(active):
            a = int(actions[li].item())
            if train:
                traj[i]["log_probs"].append(logps[li])
                traj[i]["values"].append(values[li])
                traj[i]["entropies"].append(ents[li])

            if a == 4 * K:  # STOP
                if train:
                    pred = int(probs[i].argmax())
                    r = 1.0 if pred == gold_ids[i] else -1.0
                    traj[i]["rewards"].append(r)
                done[i] = True
            else:
                cls_idx, k_idx = divmod(a, K)
                cls_name = EXPLORE_CLASSES[cls_idx]
                bucket = bank.get(cls_name, {}).get(str(L[i]), [])
                if k_idx >= len(bucket):
                    if train:
                        traj[i]["rewards"].append(-1.0)
                    done[i] = True
                    continue
                pairs_list[i][p_pos[i]] = bucket[k_idx]["pairs"][p_pos[i]]
                edit_idx.append(i)
                if train:
                    traj[i]["rewards"].append(None)  # 재forward 후 채움

        if edit_idx:
            new_pooled, new_probs = _batched_forward(model, tok, texts_for(edit_idx),
                                                     device, max_length, batch_forward_size)
            for j, i in enumerate(edit_idx):
                prev_margin = _margin_of(probs[i:i + 1], explore_ids)[0]
                pooled[i] = new_pooled[j]
                probs[i] = new_probs[j]
                new_margin = _margin_of(probs[i:i + 1], explore_ids)[0]
                resolved = not _confused_of(probs[i:i + 1], gold_ids[i:i + 1], explore_ids, confuse_margin)[0]
                p_pos[i] -= 1
                finished = resolved or p_pos[i] < 0
                if train:
                    r = (new_margin - prev_margin) - step_penalty
                    if finished:
                        pred = int(probs[i].argmax())
                        r += 1.0 if pred == gold_ids[i] else -1.0
                    traj[i]["rewards"][-1] = r
                if finished:
                    done[i] = True

        active = [i for i in active if not done[i]]

    after_pred = probs.argmax(axis=1)
    diag = {"n_decisions": diag_n_decisions, "n_forced_stop": diag_n_forced_stop,
           "valid_action_sum": diag_valid_action_sum}

    if not train:
        return before_pred, after_pred, diag

    all_log_probs, all_values, all_returns, all_entropies = [], [], [], []
    ep_returns = []
    for i, tr in traj.items():
        rs = tr["rewards"]
        if not rs:
            continue
        R, rets = 0.0, []
        for r in reversed(rs):
            R = r + gamma * R
            rets.insert(0, R)
        ep_returns.append(rets[0])
        all_log_probs.extend(tr["log_probs"])
        all_values.extend(tr["values"])
        all_entropies.extend(tr["entropies"])
        all_returns.extend(rets)

    if not all_log_probs:
        return ep_returns, before_pred, diag

    log_probs_t = torch.stack(all_log_probs)
    values_t = torch.stack(all_values)
    entropies_t = torch.stack(all_entropies)
    returns_t = torch.tensor(all_returns, device=device, dtype=torch.float32)
    advantage = returns_t - values_t.detach()

    policy_loss = -(log_probs_t * advantage).mean()
    value_loss = F.mse_loss(values_t, returns_t)
    loss = policy_loss + 0.5 * value_loss - entropy_coef * entropies_t.mean()

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()

    return ep_returns, before_pred, diag


# ---------------------------------------------------------------------------
# 시간 예산 기반 학습 루프 (관대한 early stopping)
# ---------------------------------------------------------------------------
def train_policy_vectorized(model, tok, bank, df_confusing, device,
                            val_confusing=None,
                            time_budget_hours=3.0, vec_batch_size=256,
                            lr=3e-4, gamma=0.98, entropy_coef=0.01,
                            k_per_bucket=8, confuse_margin=0.2, step_penalty=0.05,
                            max_edits=12, patience_epochs=20, min_epochs=10,
                            batch_forward_size=256, ckpt_dir=None, eval_every=1):
    """
    time_budget_hours: 이 시간을 넘기면 무조건 종료 (3시간 예산 -> 3.0).
    patience_epochs/min_epochs: 관대한 early stop. 최소 min_epochs는 무조건 채우고,
        그 이후로 patience_epochs 연속 개선(모니터링 지표)이 없으면만 조기 종료.
    val_confusing: 학습에 쓰지 않는 held-out 헷갈리는 샘플. 주어지면 매
        eval_every epoch마다 편집 전/후 정답률을 측정해 "best" 판정과 학습곡선에 쓴다.
        없으면 avg_return(학습 신호)으로만 best/early-stop을 판정한다.
    ckpt_dir: 주어지면 매 epoch 'last.pt'(최신), 개선될 때마다 'best.pt', 그리고
        'history.json'(학습곡선 원본 데이터)을 저장한다.
    """
    class2id = {c: i for i, c in enumerate(C.CLASS_NAMES)}
    explore_ids = [class2id[c] for c in EXPLORE_CLASSES]
    state_dim = model.config.hidden_size + 8
    n_actions = 4 * k_per_bucket + 1
    policy = PolicyNet(state_dim, n_actions).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    ser_cfg = serialize_config_from(C.SER)

    rows_all = df_confusing.to_dict("records")
    start = time.time()
    budget_sec = time_budget_hours * 3600
    best_metric, no_improve, epoch = -1e9, 0, 0
    history = {"epoch": [], "avg_return": [], "elapsed_min": [], "n_episodes": [],
               "val_acc_before": [], "val_acc_after": [], "avg_valid_actions": [],
               "forced_stop_ratio": []}

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    print(f"[policy] 학습 대상 {len(rows_all)}개 샘플"
          + (f" / 검증용 {len(val_confusing)}개" if val_confusing is not None else "")
          + f" / 시간예산 {time_budget_hours}h / 메타배치 {vec_batch_size} / "
            f"patience {patience_epochs}epoch (min {min_epochs}epoch)")

    while True:
        epoch += 1
        elapsed = time.time() - start
        if elapsed > budget_sec:
            print(f"[epoch {epoch}] 시간 예산({time_budget_hours}h) 도달 -> 학습 종료")
            break

        rng = np.random.RandomState(epoch)
        perm = rng.permutation(len(rows_all))
        epoch_returns = []
        diag_decisions, diag_forced_stop, diag_valid_sum = 0, 0, 0

        for s in range(0, len(perm), vec_batch_size):
            if time.time() - start > budget_sec:
                break
            idx = perm[s:s + vec_batch_size]
            batch_rows = [rows_all[j] for j in idx]
            rets, _, diag = run_vectorized_batch(
                model, tok, bank, policy, opt, batch_rows, device, ser_cfg,
                k_per_bucket, confuse_margin, step_penalty, max_edits,
                gamma, entropy_coef, batch_forward_size, class2id, explore_ids, train=True,
            )
            epoch_returns.extend(rets)
            diag_decisions += diag["n_decisions"]
            diag_forced_stop += diag["n_forced_stop"]
            diag_valid_sum += diag["valid_action_sum"]

        avg_return = float(np.mean(epoch_returns)) if epoch_returns else 0.0
        elapsed_min = (time.time() - start) / 60
        avg_valid_actions = diag_valid_sum / diag_decisions if diag_decisions else 0.0
        forced_stop_ratio = diag_forced_stop / diag_decisions if diag_decisions else 0.0

        val_acc_before = val_acc_after = None
        if val_confusing is not None and len(val_confusing) > 0 and epoch % eval_every == 0:
            val_acc_before, val_acc_after = evaluate_policy_vectorized(
                policy, model, tok, bank, val_confusing, device,
                k_per_bucket=k_per_bucket, confuse_margin=confuse_margin,
                vec_batch_size=vec_batch_size, batch_forward_size=batch_forward_size,
            )

        history["epoch"].append(epoch)
        history["avg_return"].append(avg_return)
        history["elapsed_min"].append(elapsed_min)
        history["n_episodes"].append(len(epoch_returns))
        history["val_acc_before"].append(val_acc_before)
        history["val_acc_after"].append(val_acc_after)
        history["avg_valid_actions"].append(avg_valid_actions)
        history["forced_stop_ratio"].append(forced_stop_ratio)

        metric = val_acc_after if val_acc_after is not None else avg_return
        log_msg = (f"[epoch {epoch}] t={elapsed_min:.1f}min episodes={len(epoch_returns)} "
                  f"avg_return={avg_return:.3f}")
        if val_acc_after is not None:
            log_msg += f" val_acc(전->후)={val_acc_before:.4f}->{val_acc_after:.4f}"
        log_msg += (f" avg_valid_actions={avg_valid_actions:.2f} "
                   f"forced_stop_ratio={forced_stop_ratio:.2%}")
        log_msg += f" best_metric={best_metric:.4f} no_improve={no_improve}"
        print(log_msg)
        if forced_stop_ratio > 0.8:
            print("  ⚠️  경고: 결정 지점의 80% 이상에서 STOP 외에 선택지가 없습니다. "
                 "exemplar bank가 너무 희박합니다 (purity_threshold를 낮추거나 K를 늘리세요) "
                 "-> 이 상태면 정책이 사실상 학습될 수 없습니다.")

        if ckpt_dir:
            torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(),
                       "epoch": epoch, "history": history},
                      os.path.join(ckpt_dir, "last.pt"))
            with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
                json.dump(history, f, ensure_ascii=False, indent=1)

        if metric > best_metric + 1e-4:
            best_metric = metric
            no_improve = 0
            if ckpt_dir:
                torch.save({"policy": policy.state_dict(), "epoch": epoch, "metric": best_metric},
                          os.path.join(ckpt_dir, "best.pt"))
        else:
            no_improve += 1

        if epoch >= min_epochs and no_improve >= patience_epochs:
            print(f"[epoch {epoch}] {patience_epochs}epoch 연속 개선 없음(관대한 patience 소진) -> 종료")
            break

    return policy, history


@torch.no_grad()
def evaluate_policy_vectorized(policy, model, tok, bank, df_confusing, device,
                               k_per_bucket=8, confuse_margin=0.2, max_edits=12,
                               vec_batch_size=256, batch_forward_size=256):
    """편집 전/후 정답률을 배치로 빠르게 비교 (샘플 1개씩 안 돌림)."""
    class2id = {c: i for i, c in enumerate(C.CLASS_NAMES)}
    explore_ids = [class2id[c] for c in EXPLORE_CLASSES]
    ser_cfg = serialize_config_from(C.SER)
    rows_all = df_confusing.to_dict("records")
    gold_all = np.array([class2id[r["action"]] for r in rows_all])

    before_all, after_all = [], []
    for s in range(0, len(rows_all), vec_batch_size):
        batch_rows = rows_all[s:s + vec_batch_size]
        before_pred, after_pred, _ = run_vectorized_batch(
            model, tok, bank, policy, None, batch_rows, device, ser_cfg,
            k_per_bucket, confuse_margin, step_penalty=0.0, max_edits=max_edits,
            gamma=1.0, entropy_coef=0.0, batch_forward_size=batch_forward_size,
            class2id=class2id, explore_ids=explore_ids, train=False,
        )
        before_all.append(before_pred)
        after_all.append(after_pred)

    before_all = np.concatenate(before_all)
    after_all = np.concatenate(after_all)
    return float((before_all == gold_all).mean()), float((after_all == gold_all).mean())