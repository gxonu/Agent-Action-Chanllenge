"""
제출용 추론 엔트리포인트 v2 — N-모델 tri_cond 조건부 앙상블 (수동 int8 + vocab-prune remap).

v007(2모델 mean) 확장:
 - base 모델: 전체 행 추론 (항상)
 - conditional 모델: base margin(top1-top2) 하위 cond_fraction 행에만 재추론 (시간 절약)
   → 검증: 하위 15%만 재추론해도 full 3-way와 동일(고마진행은 conditional이 안 바꿈)
 - vocab pruning: model/<tag>/remap.npy 있으면 tokenize 후 input_ids를 remap (압축 임베딩용)
 - per-class bias(ensemble.json), gen_rescue 훅(선택)

model/ensemble.json (v2):
{
 "method":"tri_cond", "classes":[...14...], "max_length":512, "bias":[...14...],
 "base":[{"tag":"mmbert_i8","weight":0.45},{"tag":"bge_i8","weight":0.4}],
 "conditional":[{"tag":"xlmr_i8","weight":0.15}],
 "cond_fraction":0.27
}
각 model/<tag>/: quant.safetensors(int8 + <key>.__scale__) or model.safetensors(fp16),
                config.json, tokenizer, (선택)remap.npy.
requirements: transformers==4.48.3.
"""
import os, json, glob, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, pandas as pd, torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
DATA_DIR = os.path.join(HERE, "data")
OUT_PATH = os.path.join(HERE, "output", "submission.csv")
BATCH_SIZE = 256

# --- serialize: model/common.py 우선, 실패시 fallback (v007과 동일) ---
def serialize(cp, h, sm, cfg):
    def meta(m):
        if not isinstance(m, dict): return ""
        ws = m.get("workspace") or {}; of = ws.get("open_files") or []
        lm = ws.get("language_mix") or {}
        tl = max(lm, key=lm.get) if isinstance(lm, dict) and lm else "na"
        return " ".join([f"tier={m.get('user_tier')}", f"lang={m.get('language_pref')}",
            f"turn={m.get('turn_index')}", f"budget={m.get('budget_tokens_remaining')}",
            f"ci={ws.get('last_ci_status')}", f"dirty={ws.get('git_dirty')}",
            f"loc={ws.get('loc')}", f"toplang={tl}", f"nopen={len(of)}"])
    def hist(hh, ms):
        if not isinstance(hh, list): return "" if hh is None else str(hh)
        it = []
        for s in hh:
            if not isinstance(s, dict): it.append(str(s)); continue
            if s.get("role") == "assistant_action" or s.get("name"):
                a = ",".join(f"{k}={v}" for k, v in (s.get("args") or {}).items())
                it.append(f"[{s.get('name','')}({a})] " + (s.get("result_summary") or "").strip())
            else:
                it.append(("user: " + (s.get("content") or s.get("text") or "")).strip())
        if ms and len(it) > ms: it = it[-ms:]
        return " ".join(reversed(it))
    ps = "" if cp is None else str(cp).strip()
    parts = [f"{cfg['prompt_marker']} {ps}"]
    hs = hist(h, cfg.get("max_history_steps")); ms_ = meta(sm)
    if hs: parts.append(f"{cfg['history_marker']} {hs}")
    if ms_: parts.append(f"{cfg['meta_marker']} {ms_}")
    return " ".join(parts)

if MODEL_DIR not in sys.path: sys.path.insert(0, MODEL_DIR)
try:
    from common import serialize  # noqa
    print("[serialize] bundled model/common.py")
except Exception as e:
    print(f"[serialize][WARN] fallback: {e}")


def load_model(mp, device):
    cfg = AutoConfig.from_pretrained(mp, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(cfg)
    qp = os.path.join(mp, "quant.safetensors")
    if os.path.exists(qp):                      # 수동 int8
        comp = load_file(qp); sd = {}
        for k in comp:
            if k.endswith(".__scale__"): continue
            sk = k + ".__scale__"
            sd[k] = (comp[k].float() * comp[sk].float()).half() if sk in comp else \
                    (comp[k].half() if comp[k].is_floating_point() else comp[k])
        model.load_state_dict(sd, strict=True)
    else:                                       # plain fp16/fp32
        model.load_state_dict(load_file(os.path.join(mp, "model.safetensors")), strict=True)
    model.to(device).eval()
    if device == "cuda": model.half()
    remap = None
    rp = os.path.join(mp, "remap.npy")
    if os.path.exists(rp):
        remap = torch.tensor(np.load(rp), dtype=torch.long, device=device)
    return model, remap


def model_probs(mp, texts, classes, max_len, device, subset=None):
    """subset: None=전체, 아니면 인덱스 리스트. 반환: (len(texts),C) — subset 밖은 0."""
    tok = AutoTokenizer.from_pretrained(mp, local_files_only=True, use_fast=True)
    model, remap = load_model(mp, device)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    cpos = {c: i for i, c in enumerate(classes)}
    reorder = [0] * len(classes)
    for mid, lab in id2label.items(): reorder[cpos[lab]] = mid
    idxs = list(range(len(texts))) if subset is None else list(subset)
    enc = tok([texts[j] for j in idxs], truncation=True, max_length=max_len)
    iid = enc["input_ids"]
    order = sorted(range(len(idxs)), key=lambda j: len(iid[j]))
    out = np.zeros((len(texts), len(classes)), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(order), BATCH_SIZE):
            sel = order[i:i + BATCH_SIZE]
            pad = tok.pad({"input_ids": [iid[j] for j in sel]}, return_tensors="pt").to(device)
            if remap is not None:
                pad["input_ids"] = remap[pad["input_ids"]]
            p = torch.softmax(model(**pad).logits.float(), 1).cpu().numpy()[:, reorder]
            for k, j in enumerate(sel): out[idxs[j]] = p[k]
    del model
    if device == "cuda": torch.cuda.empty_cache()
    return out


def load_test():
    for ext in ("jsonl", "json", "csv", "parquet"):
        p = os.path.join(DATA_DIR, f"test.{ext}")
        if os.path.exists(p): path = p; break
    else:
        hits = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.*"), recursive=True))
        path = next(h for h in hits if h.endswith((".jsonl", ".json", ".csv", ".parquet")))
    if path.endswith((".jsonl", ".json")):
        df = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
    elif path.endswith(".parquet"): df = pd.read_parquet(path)
    else: df = pd.read_csv(path)
    return df, path


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ens = json.load(open(os.path.join(MODEL_DIR, "ensemble.json")))
    ser = json.load(open(os.path.join(MODEL_DIR, "serialize_config.json")))
    classes = ens["classes"]; C = len(classes)
    bias = np.array(ens.get("bias", [0.0] * C), dtype=np.float32)
    max_len = int(ens.get("max_length", 512))
    base = ens.get("base") or [{"tag": t, "weight": 1.0} for t in ens.get("tags", [])]
    cond = ens.get("conditional", [])
    cond_frac = float(ens.get("cond_fraction", 0.27))

    df, path = load_test()
    def g(r, k):
        v = r.get(k)
        return v if isinstance(v, (list, dict, str)) else (None if pd.isna(v) else v)
    texts = [serialize(g(r, "current_prompt"), g(r, "history"), g(r, "session_meta"), ser)
             for _, r in df.iterrows()]
    print(f"[infer] {len(texts)} rows; base={[b['tag'] for b in base]} cond={[c['tag'] for c in cond]} frac={cond_frac}")

    # 1) base 전체 추론
    probs = np.zeros((len(texts), C), dtype=np.float32)
    for b in base:
        probs += b["weight"] * model_probs(os.path.join(MODEL_DIR, b["tag"]), texts, classes, max_len, device)

    # 2) conditional: base margin 하위 cond_frac 행에만
    if cond:
        srt = np.sort(probs, axis=1); margin = srt[:, -1] - srt[:, -2]
        thr = np.quantile(margin, cond_frac)
        sub = np.where(margin <= thr)[0]
        print(f"[tri_cond] 저마진 {len(sub)}행 ({len(sub)/len(texts)*100:.0f}%) 재추론")
        for c in cond:
            cp = model_probs(os.path.join(MODEL_DIR, c["tag"]), texts, classes, max_len, device, subset=sub)
            probs[sub] += c["weight"] * cp[sub]

    pred = (probs + bias).argmax(1)
    ids = df["id"] if "id" in df.columns else np.arange(len(df))
    pd.DataFrame({"id": ids, "action": [classes[i] for i in pred]}).to_csv(OUT_PATH, index=False)
    print(f"[done] {OUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
