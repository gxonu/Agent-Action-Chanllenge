"""규중 RL 입력-편집을 exp006 mmbert(frozen) backbone에 적용.
train fold로 exemplar bank + 정책 학습, val fold inspect-4클래스 before/after 평가."""
import os, sys, re, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

import config as C
from common import serialize, serialize_config_from
from exemplar_bank import EXPLORE_CLASSES, extract_pooled_embeddings, build_exemplar_bank
from rl_edit import train_policy_vectorized, evaluate_policy_vectorized, _confused_of
from mmbert_fwd import mmbert_pooled_logits

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../exp006_mmbert_1024/model")
DEV = "cuda"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "../../data/processed/train_merged.jsonl"))
    ap.add_argument("--time_budget", type=float, default=0.1)
    ap.add_argument("--purity", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--confuse_margin", type=float, default=0.2)
    args = ap.parse_args()

    C.SER.add_open_files = True                      # exp006 학습 serialize 매칭
    ser_cfg = serialize_config_from(C.SER)
    class2id = {c: i for i, c in enumerate(C.CLASS_NAMES)}
    explore_ids = [class2id[c] for c in EXPLORE_CLASSES]

    df = pd.read_json(args.data, lines=True)
    groups = np.array([re.sub(r"-step_\d+$", "", str(i)) for i in df["id"].astype(str)])
    y = np.array([class2id[a] for a in df["action"]])
    tr_idx, va_idx = next(StratifiedGroupKFold(10, shuffle=True, random_state=42).split(np.arange(len(df)), y, groups))
    df_tr, df_va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
    print(f"[split] train {len(df_tr)} / val {len(df_va)} (exp006와 동일 seed42 first fold)")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEV).eval()
    for p in model.parameters(): p.requires_grad = False

    df_tr_i = df_tr[df_tr["action"].isin(EXPLORE_CLASSES)].reset_index(drop=True)
    df_va_i = df_va[df_va["action"].isin(EXPLORE_CLASSES)].reset_index(drop=True)
    print(f"[inspect] train {len(df_tr_i)} / val {len(df_va_i)}")

    def texts_of(d): return [serialize(r.current_prompt, r.history, r.session_meta, ser_cfg) for r in d.itertuples()]
    @torch.no_grad()
    def probs_of(d):
        t = texts_of(d); out = []
        for i in range(0, len(t), 256):
            enc = tok(t[i:i+256], truncation=True, max_length=C.MAX_LENGTH, padding=True, return_tensors="pt").to(DEV)
            _, lg = mmbert_pooled_logits(model, enc)
            out.append(torch.softmax(lg, -1).float().cpu().numpy())
        return np.concatenate(out)

    # exemplar bank: train inspect high-purity
    emb_tr = extract_pooled_embeddings(model, tok, texts_of(df_tr_i), DEV, batch_size=256)
    bank = build_exemplar_bank(df_tr_i, emb_tr, purity_threshold=args.purity, k_per_bucket=args.k)

    # confusing = pred 틀림 or inspect-margin < confuse_margin
    def conf(d):
        pr = probs_of(d); gold = np.array([class2id[a] for a in d["action"]])
        return d[_confused_of(pr, gold, explore_ids, args.confuse_margin)].reset_index(drop=True)
    df_tr_c, df_va_c = conf(df_tr_i), conf(df_va_i)
    print(f"[confusing] train {len(df_tr_c)} / val {len(df_va_c)}")

    policy, _ = train_policy_vectorized(model, tok, bank, df_tr_c, DEV, val_confusing=df_va_c,
        time_budget_hours=args.time_budget, k_per_bucket=args.k, confuse_margin=args.confuse_margin,
        min_epochs=3, patience_epochs=10, ckpt_dir=os.path.join(os.path.dirname(__file__), "ckpt"))

    b, a = evaluate_policy_vectorized(policy, model, tok, bank, df_va_c, DEV, k_per_bucket=args.k, confuse_margin=args.confuse_margin)
    print(f"\n[FINAL] val confusing inspect acc: before {b:.4f} -> after {a:.4f}  (delta {a-b:+.4f})")
    print(f"        (val confusing {len(df_va_c)}개 / val inspect {len(df_va_i)}개 / val 전체 {len(df_va)}개)")

if __name__ == "__main__":
    main()
