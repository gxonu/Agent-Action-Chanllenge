"""
제출용 추론 엔트리포인트 — 2모델 평균 앙상블 (수동 per-row int8, bnb 불필요).

각 model/<tag>/ 는 quant.safetensors(int8 가중치 + <key>.__scale__ fp16) + config.json + 토크나이저.
script.py 가 from_config 로 뼈대 생성 → int8 dequant → load_state_dict → fp16 추론.
requirements: transformers==4.48.3 (사전설치 4.46.3 업그레이드, 상혁 검증됨). bitsandbytes 불필요.
serialize() 는 model/common.py 와 KEEP IN SYNC.
"""
import os, json, glob, re

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
DATA_DIR = os.path.join(HERE, "data")
OUT_PATH = os.path.join(HERE, "output", "submission.csv")

SUBMISSION_ID_COL = "id"
SUBMISSION_PRED_COL = "action"
ID_KEY = "id"
BATCH_SIZE = 256


# ===========================================================================
# serialize() — common.py 와 KEEP IN SYNC (fallback)
# ===========================================================================
def _fmt_args(args):
    if not isinstance(args, dict) or not args:
        return ""
    return ",".join(f"{k}={v}" for k, v in args.items())


def _stringify_history(history, max_steps):
    if history is None:
        return ""
    if isinstance(history, str):
        return history.strip()
    if not isinstance(history, list):
        return str(history)
    items = []
    for step in history:
        if not isinstance(step, dict):
            items.append(str(step)); continue
        role = step.get("role")
        if role == "assistant_action" or step.get("name"):
            name = step.get("name", "")
            a = _fmt_args(step.get("args"))
            res = (step.get("result_summary") or "").strip()
            piece = f"[{name}({a})]" + (f" {res}" if res else "")
        else:
            content = step.get("content") or step.get("text") or ""
            piece = f"user: {content}".strip()
        if piece:
            items.append(piece)
    if max_steps is not None and len(items) > max_steps:
        items = items[-max_steps:]
    return " ".join(reversed(items))


def _stringify_meta(meta):
    if meta is None:
        return ""
    if not isinstance(meta, dict):
        return str(meta)
    ws = meta.get("workspace") or {}
    open_files = ws.get("open_files") or []
    lang_mix = ws.get("language_mix") or {}
    top_lang = max(lang_mix, key=lang_mix.get) if isinstance(lang_mix, dict) and lang_mix else "na"
    parts = [
        f"tier={meta.get('user_tier')}", f"lang={meta.get('language_pref')}",
        f"turn={meta.get('turn_index')}", f"budget={meta.get('budget_tokens_remaining')}",
        f"ci={ws.get('last_ci_status')}", f"dirty={ws.get('git_dirty')}",
        f"loc={ws.get('loc')}", f"toplang={top_lang}", f"nopen={len(open_files)}",
    ]
    return " ".join(parts)


def serialize(current_prompt, history, session_meta, cfg):
    meta_s = _stringify_meta(session_meta)
    hist_s = _stringify_history(history, cfg.get("max_history_steps"))
    prompt_s = "" if current_prompt is None else str(current_prompt).strip()
    parts = [f"{cfg['prompt_marker']} {prompt_s}"]
    if hist_s:
        parts.append(f"{cfg['history_marker']} {hist_s}")
    if meta_s:
        parts.append(f"{cfg['meta_marker']} {meta_s}")
    return " ".join(parts)


import sys as _sys
if MODEL_DIR not in _sys.path:
    _sys.path.insert(0, MODEL_DIR)
try:
    from common import serialize  # noqa: F811  model/common.py = 소스오브트루스
    print("[serialize] using bundled model/common.py")
except Exception as _e:
    print(f"[serialize][WARN] common.py import 실패, 로컬 fallback: {_e}")


# ===========================================================================
# 데이터 로딩
# ===========================================================================
def find_test_file():
    for ext in ("jsonl", "json", "csv", "parquet"):
        p = os.path.join(DATA_DIR, f"test.{ext}")
        if os.path.exists(p):
            return p
    for ext in ("*.jsonl", "*.json", "*.csv", "*.parquet"):
        hits = sorted(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No test file under {DATA_DIR}")


def load_test():
    path = find_test_file()
    if path.endswith((".jsonl", ".json")):
        recs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        df = pd.DataFrame(recs)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df, path


# ===========================================================================
# 수동 int8 모델 로딩 (bnb 불필요)
# ===========================================================================
def load_int8_model(model_path, device):
    """quant.safetensors(int8 + <key>.__scale__) -> dequant fp16 -> load_state_dict."""
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(config)
    comp = load_file(os.path.join(model_path, "quant.safetensors"))
    sd = {}
    for k in comp:
        if k.endswith(".__scale__"):
            continue
        sk = k + ".__scale__"
        if sk in comp:
            sd[k] = (comp[k].float() * comp[sk].float()).half()
        else:
            sd[k] = comp[k].half() if comp[k].is_floating_point() else comp[k]
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    if device == "cuda":
        model.half()
    return model


def model_probs(model_path, texts, classes, max_len, device):
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = load_int8_model(model_path, device)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    cls_pos = {c: i for i, c in enumerate(classes)}
    reorder = [0] * len(classes)
    for mid, lab in id2label.items():
        reorder[cls_pos[lab]] = mid

    enc_all = tok(texts, truncation=True, max_length=max_len)
    input_ids = enc_all["input_ids"]
    order = sorted(range(len(texts)), key=lambda j: len(input_ids[j]))

    out = np.zeros((len(texts), len(classes)), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            padded = tok.pad({"input_ids": [input_ids[j] for j in idx]},
                             return_tensors="pt").to(device)
            logits = model(**padded).logits.float()
            p = torch.softmax(logits, dim=1).cpu().numpy()[:, reorder]
            for k, j in enumerate(idx):
                out[j] = p[k]
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ens = json.load(open(os.path.join(MODEL_DIR, "ensemble.json")))
    ser_cfg = json.load(open(os.path.join(MODEL_DIR, "serialize_config.json")))
    classes = ens["classes"]
    bias = np.array(ens.get("bias", [0.0] * len(classes)), dtype=np.float32)
    max_len = int(ens.get("max_length", 512))
    tags = ens["tags"]

    df, path = load_test()
    print(f"[infer] {len(df)} rows from {path}; models={tags}")

    def get(row, key):
        if key not in row:
            return None
        v = row[key]
        if isinstance(v, (list, dict, str)):
            return v
        return None if pd.isna(v) else v
    texts = [
        serialize(get(r, "current_prompt"), get(r, "history"), get(r, "session_meta"), ser_cfg)
        for _, r in df.iterrows()
    ]

    probs = np.zeros((len(texts), len(classes)), dtype=np.float32)
    for tag in tags:
        probs += model_probs(os.path.join(MODEL_DIR, tag), texts, classes, max_len, device)
    probs /= len(tags)

    pred_idx = (probs + bias).argmax(axis=1)
    pred_labels = [classes[i] for i in pred_idx]

    ids = df[ID_KEY] if ID_KEY in df.columns else np.arange(len(df))
    out = pd.DataFrame({SUBMISSION_ID_COL: ids, SUBMISSION_PRED_COL: pred_labels})
    out.to_csv(OUT_PATH, index=False)
    print(f"[done] wrote {OUT_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    main()
