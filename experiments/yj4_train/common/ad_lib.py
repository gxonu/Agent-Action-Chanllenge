"""ad_lib — 학습/추론 공용 단일 소스 (직렬화 + 추론).

이 파일은 (1) Colab 노트북(학습 입력 직렬화), (2) 서버 script.py(오프라인 추론)에서
동일하게 import 된다 → train/inference 직렬화 드리프트 0 (구조적 보장).

- serialize(): torch 없이 동작(표준 라이브러리만).
- predict(): torch/transformers/numpy 지연 import (serialize만 쓸 땐 불필요).
- 서버 배포 시 model/ad_lib.py 로 복사되어 script.py 가 import.
"""
from __future__ import annotations
import json
import os
import re

# ============================ 클래스 ============================
CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = 14

# ============================ 파싱 ============================
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_INT = re.compile(r"-?\d+")


def has_hangul(t): return bool(_HANGUL.search(t or ""))


def lang_of(t):
    t = t or ""
    ko, en = bool(_HANGUL.search(t)), bool(_LATIN.search(t))
    return "mixed" if ko and en else ("ko" if ko else ("en" if en else "other"))


def first_int(s, default=-1):
    m = _INT.search(s or "")
    return int(m.group()) if m else default


def result_status(name, summary):
    s = (summary or "").strip()
    if not s:
        return "na"
    low = s.lower()
    if low.startswith("error") or "did not apply" in low or "command failed" in low:
        return "error"
    if re.search(r"\bfail(ed|ure)?\b", low):
        return "test_fail" if (name == "run_tests" or "test" in low) else "error"
    if name == "run_tests" and ("pass" in low or "ok" in low):
        return "test_pass"
    m = re.search(r"exit\s*=\s*(-?\d+)", low)
    if m:
        return "success" if m.group(1) == "0" else "nonzero_exit"
    if re.search(r"\b0\s+(match|matches|file|files|result|results|occurrence|occurrences|entries|entry)\b", low) \
            or "empty directory" in low or low.startswith("no "):
        return "zero"
    return "success"


def path_ext(path):
    if not path:
        return ""
    base = path.rstrip("/").split("/")[-1]
    if "." in base:
        return base.rsplit(".", 1)[1].lower()
    known = {"dockerfile", "makefile", "readme", "license", "gemfile", "rakefile"}
    return base.lower() if base.lower() in known else ""


def glob_ext(pattern):
    if not pattern:
        return ""
    m = re.search(r"\*?\.?([A-Za-z0-9]+)$", pattern.strip())
    return m.group(1).lower() if m else ""


def action_turns(sample):
    return [t for t in sample.get("history", []) if t.get("role") == "assistant_action"]


def action_sequence(sample):
    return [t.get("name", "") for t in action_turns(sample)]


def last_action(sample):
    acts = action_turns(sample)
    if not acts:
        return None, {}, "", "na"
    t = acts[-1]
    nm = t.get("name")
    return nm, (t.get("args") or {}), (t.get("result_summary") or ""), result_status(nm, t.get("result_summary") or "")


def arg_path_or_pattern(name, args):
    if not isinstance(args, dict):
        return ""
    for k in ("path", "pattern", "target", "cmd", "query", "goal", "question"):
        if k in args and args[k]:
            return str(args[k])
    return ""


def meta_fields(sample):
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    lm = ws.get("language_mix") or {}
    top_lang = max(lm.items(), key=lambda kv: kv[1])[0] if lm else ""
    return {
        "user_tier": sm.get("user_tier", ""), "language_pref": sm.get("language_pref", ""),
        "budget": sm.get("budget_tokens_remaining", -1), "turn_index": sm.get("turn_index", -1),
        "elapsed": sm.get("elapsed_session_sec", -1), "loc": ws.get("loc", -1),
        "git_dirty": bool(ws.get("git_dirty", False)), "last_ci_status": ws.get("last_ci_status", ""),
        "n_open_files": len(ws.get("open_files") or []), "open_files": ws.get("open_files") or [],
        "top_lang": top_lang,
    }


def _gen(sample_id):
    return "au" if str(sample_id).startswith("sess_au_") else "sim"


# ============================ 직렬화 ============================
def _budget_bin(v):
    if v is None or v < 0: return "na"
    return "vlow" if v < 4000 else "low" if v < 16000 else "mid" if v < 64000 else "high"


def _loc_bin(v):
    if v is None or v < 0: return "na"
    return "s" if v < 2000 else "m" if v < 10000 else "l" if v < 40000 else "xl"


def _turn_bin(v):
    if v is None or v < 0: return "na"
    return "1" if v <= 1 else "2-3" if v <= 3 else "4-7" if v <= 7 else "8+"


def _elapsed_bin(v):
    """v7 [PACE]: elapsed_session_sec 8분위 근사 bin (탐색클러스터 판별 신호 실측 기반)."""
    if v is None or v < 0: return "na"
    for e, name in ((120, "e0"), (250, "e1"), (400, "e2"), (550, "e3"), (700, "e4"), (950, "e5")):
        if v <= e: return name
    return "e6"


def _pace_bin(elapsed, turn):
    if elapsed is None or elapsed < 0: return "na"
    p = elapsed / max(float(turn), 1.0) if turn is not None else elapsed
    for e, name in ((60, "p0"), (90, "p1"), (130, "p2"), (200, "p3"), (350, "p4")):
        if p <= e: return name
    return "p5"


def _fmt_action(t, with_args=False, full_args=False):
    nm = t.get("name", "")
    rs = (t.get("result_summary") or "")[:100]
    st = result_status(nm, rs)
    if full_args:
        # v5: 인자 원문(경로/패턴/명령 등) 보존 — 하드클래스(read/grep/glob/list) 구분 신호
        a = arg_path_or_pattern(nm, t.get("args") or {})[:80]
        return f"{nm}({a})[{st}] {rs}" if a else f"{nm}[{st}] {rs}"
    if with_args:
        a = arg_path_or_pattern(nm, t.get("args") or {})
        ext = path_ext(a) or glob_ext(a)
        return f"{nm}{('('+ext+')') if ext else ''}[{st}] {rs}"
    return f"{nm}[{st}] {rs}"


_RE_GLOB = re.compile(r"[*?]|\[[^\]]+\]")
_RE_PATH = re.compile(r"(?:^|[\s'\"`(])(?:[\w.-]+/)+[\w.-]+|\b[\w-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|c|cpp|h|css|html|json|yaml|yml|toml|md|txt|sh|sql|ipynb|cfg|ini|lock)\b")
_RE_SYM = re.compile(r"`[^`]+`|'[A-Za-z_][\w.]*'|\"[A-Za-z_][\w.]*\"|\b[a-z]+[A-Z]\w*\b|\b\w+_\w+\b")
_RE_DIR = re.compile(r"디렉터리|디렉토리|폴더|구조|folder|directory|directories|tree|structure|파일\s*목록|list\s+files?", re.I)


def _prompt_flags(prompt, open_files):
    fl = []
    if _RE_PATH.search(prompt): fl.append("path")
    if _RE_GLOB.search(prompt): fl.append("glob")
    if _RE_SYM.search(prompt): fl.append("sym")
    if _RE_DIR.search(prompt): fl.append("dir")
    if open_files:
        bases = {p.rsplit("/", 1)[-1] for p in open_files if isinstance(p, str)}
        if any(b and b in prompt for b in bases):
            fl.append("inopen")
    return ",".join(fl) if fl else "none"


def serialize(sample, version="v3", max_hist_turns=8):
    prompt = sample.get("current_prompt")
    if not isinstance(prompt, str):
        prompt = "" if prompt is None else str(prompt)
    if version == "v1":
        return prompt
    nm, args, rs, st = last_action(sample)
    if version == "v2":
        la = ""
        if nm:
            a = arg_path_or_pattern(nm, args)
            ext = path_ext(a) or glob_ext(a)
            la = f" [LAST] {nm}{('('+ext+')') if ext else ''} [{st}] {rs[:100]}"
        return f"[CUR] {prompt}{la}"
    m = meta_fields(sample)
    parts = [f"[TIER] {m['user_tier']} [LANG] {m['language_pref']} [TURN] {_turn_bin(m['turn_index'])} "
             f"[CI] {m['last_ci_status']} [GIT] {'dirty' if m['git_dirty'] else 'clean'}"]
    if version in ("v4", "v5", "v6", "v7"):
        parts.append(f"[GEN] {_gen(sample.get('id',''))} [BUDGET] {_budget_bin(m['budget'])} "
                     f"[LOC] {_loc_bin(m['loc'])} [TOPLANG] {m['top_lang']} [NOPEN] {m['n_open_files']}")
        if m["open_files"]:
            if version == "v5":
                # v5: 열린 파일 실제 경로(최대 5) — 다음 행동 대상 힌트
                parts.append(f"[OPEN] {' '.join(p[:60] for p in m['open_files'][:5])}")
            else:
                exts = sorted({path_ext(p) for p in m["open_files"] if path_ext(p)})
                if exts:
                    parts.append(f"[OPENEXT] {','.join(exts)}")
    full_hist = sample.get("history") or []
    hist = full_hist[-max_hist_turns:]
    if hist:
        parts.append("[HIST]")
        for t in hist:
            if t.get("role") == "user":
                parts.append(f"u: {(t.get('content') or '')[:150]}")
            elif t.get("role") == "assistant_action":
                parts.append(f"a: {_fmt_action(t, with_args=(version in ('v4', 'v6', 'v7')), full_args=(version == 'v5'))}")
    if version in ("v6", "v7"):
        # 좌측절단 생존 위치(끝쪽): 압축 액션 트레일 + 세션시작 마커 + 프롬프트 패턴 플래그
        seq = [t.get("name", "") for t in full_hist if t.get("role") == "assistant_action"]
        parts.append(f"[SEQ] {'>'.join(seq[-12:]) if seq else 'none'}")
        if not full_hist:
            parts.append("[NOHIST]")
        parts.append(f"[PFLAG] {_prompt_flags(prompt, m['open_files'])}")
    if version == "v7":
        # v7 = v6 + [PACE]: elapsed_sec(직렬화 이력상 완전 미노출)와 진행속도 bin.
        # 실측: 탐색4class 구성이 elapsed에 단조 이동(list 22%→6%, glob 12%→25%) — meta_probe GO 근거.
        parts.append(f"[PACE] {_elapsed_bin(m['elapsed'])} {_pace_bin(m['elapsed'], m['turn_index'])}")
    parts.append(f"[CUR] {prompt}")
    return " ".join(parts)


# ============================ 스태킹 피처 ============================
_ST_LIST = ["na", "success", "error", "test_fail", "test_pass", "zero", "nonzero_exit"]
_ST2I = {s: i for i, s in enumerate(_ST_LIST)}


def stack_features(sample):
    """LightGBM 메타용 구조 피처 (학습·추론 동일 — 순서 변경 금지)."""
    nm, args, rs, st = last_action(sample)
    m = meta_fields(sample)
    seq = action_sequence(sample)
    f = [CLASS_TO_IDX.get(nm, -1), _ST2I.get(st, 0), len(seq), m["turn_index"],
         {"passed": 0, "failed": 1, "none": 2}.get(m["last_ci_status"], 2),
         int(m["git_dirty"]), m["n_open_files"],
         {"sim": 0, "au": 1}[_gen(sample.get("id", ""))],
         len(sample.get("current_prompt") or ""),
         int(has_hangul(sample.get("current_prompt") or ""))]
    cnt = [0.0] * NUM_CLASSES
    for a in seq:
        if a in CLASS_TO_IDX:
            cnt[CLASS_TO_IDX[a]] += 1
    return f + cnt


# ============================ 후처리 로드/적용 ============================
def load_bias(path):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if obj.get("classes") != CLASSES:
        raise ValueError("postproc.json 클래스 순서 불일치")
    return obj["bias"]


def _to_logprobs_np(logits, np):
    z = logits.astype("float64")
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
    return z - lse


# ============================ 추론 ============================
def _dequant_state_dict(model_dir):
    """qweights.npz(int8 weight-only 양자화) → fp16 state_dict (메모리 내 복원).

    1GB 제한 대응: 대형 2D 가중치는 group-G int8+fp16 scale로 zip에 저장(sim/quantize_member.py).
    디스크에 safetensors 재작성 없이 로드 — 서버 디스크 쿼터/IO 리스크 제거.
    """
    import numpy as np
    import torch
    qp = os.path.join(model_dir, "qweights.npz")
    if not os.path.exists(qp):
        return None
    z = np.load(qp)
    out = {}
    for k in z.files:
        tag, _, name = k.partition("::")
        if tag == "q":
            q = z[k].astype(np.float32)
            s = z[f"s::{name}"].astype(np.float32)
            o, i = q.shape
            if s.ndim == 2 and s.shape[1] > 1:   # group-G 양자화
                w = (q.reshape(o, s.shape[1], -1) * s[:, :, None]).reshape(o, i)
            else:                                 # per-row (구버전 호환)
                w = q * s
            out[name] = torch.from_numpy(w.astype(np.float16))
        elif tag == "f":
            out[name] = torch.from_numpy(z[k])
    return out


def _load_model_maybe_quant(model_dir):
    """qweights.npz 존재 시 in-memory 복원 로드, 아니면 기존 from_pretrained."""
    from transformers import AutoModelForSequenceClassification, AutoConfig
    sd = None
    if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
        sd = _dequant_state_dict(model_dir)
    if sd is None:
        return AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True)
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"양자화 복원 잉여키: {unexpected[:5]}"
    bad = [m for m in missing if not m.endswith((".position_ids", ".token_type_ids"))]
    assert not bad, f"양자화 복원 누락키: {bad[:5]}"
    return model


def predict_logits(model_dir, samples, version="v3", max_len=192, batch_size=64,
                   device=None, max_hist_turns=8, texts=None, return_probs=False):
    """samples(list[dict]) → logits/probs(np.ndarray [N,14]). 길이정렬 배칭 + fp16(GPU).

    - model_dir/id_map.npy 존재 시 vocab-pruned 모델로 간주, 토큰 id 리매핑.
    - model_dir/qweights.npz 존재 시 int8 양자화 멤버 → safetensors 선복원.
    - texts 인자를 주면 직렬화 재계산 생략(앙상블에서 공유).
    """
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tok.truncation_side = "left"   # [CUR] 현재발화가 뒤쪽 → 왼쪽(오래된 history)부터 자름
    model = _load_model_maybe_quant(model_dir)
    model.to(device)
    if device == "cuda":
        model.half()      # T4 fp16 추론
    else:
        model.float()     # CPU는 fp16 미지원 연산 있어 upcast(오프라인 시뮬용)
    model.eval()

    id_map = None
    imp = os.path.join(model_dir, "id_map.npy")
    if os.path.exists(imp):
        id_map = np.load(imp)   # full id -> compact id (pruned vocab)

    if texts is None:
        texts = [serialize(s, version, max_hist_turns) for s in samples]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    logits = np.zeros((len(texts), NUM_CLASSES), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                      max_length=max_len, pad_to_multiple_of=8, return_tensors="pt")
            if id_map is not None:
                enc["input_ids"] = torch.from_numpy(
                    id_map[enc["input_ids"].numpy()]).to(enc["input_ids"].dtype)
            enc = enc.to(device)
            out = model(**enc).logits.float()
            if return_probs:
                out = torch.softmax(out, dim=1)
            out = out.cpu().numpy()
            for j, i in enumerate(idx):
                logits[i] = out[j]
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return logits


def predict_ensemble_probs(model_root, samples, meta, device=None, return_members=False):
    """run_meta의 ensemble 정의에 따라 멤버별 softmax 계산.

    meta 예: {"version":"v4","max_len":320,"batch_size":128,
              "ensemble":[{"dir":"m1"},{"dir":"m2","version":"v5"}]}
    반환: 평균 probs[N,14]  (return_members=True면 (평균, [멤버별 probs...]))
    """
    import numpy as np
    members = []
    base_ver = meta.get("version", "v4")
    cache = {}
    for member in meta["ensemble"]:
        mdir = os.path.join(model_root, member["dir"])
        ver = member.get("version", base_ver)
        if ver not in cache:
            cache[ver] = [serialize(s, ver) for s in samples]
        if member.get("type") == "ngram":
            p = predict_ngram_probs(mdir, cache[ver])
        else:
            p = predict_logits(mdir, samples, version=ver,
                               max_len=member.get("max_len", meta.get("max_len", 320)),
                               batch_size=meta.get("batch_size", 128),
                               device=device, texts=cache[ver], return_probs=True)
        members.append(p)
    w = meta.get("weights")
    if w:
        assert len(w) == len(members), "weights 길이 != 멤버수"
        mean = sum(wi * m for wi, m in zip(w, members)) / sum(w)
    else:
        mean = sum(members) / len(members)
    return (mean, members) if return_members else mean


def predict_conditional_probs(model_root, samples, meta, device=None):
    """조건부 앙상블 — 10분캡 안에서 full 앙상블 이득 대부분 회수 (codex R11 → R12 일반화).

    full 멤버들(cond_members 제외)을 전체에 추론·가중혼합 → top1-top2 마진 < margin_th 인
    행만 cond 멤버 추가 추론·재혼합. LB 실측(07-05): 서버 시간예산 large+230s/base+117s.
    """
    import numpy as np
    ens = meta["ensemble"]
    w = meta.get("weights") or [1.0] * len(ens)
    cond = meta["conditional"]
    th = float(cond["margin_th"])
    cond_idx = set(cond.get("cond_members", [len(ens) - 1]))   # 기본: 마지막 멤버만 조건부
    base_ver = meta.get("version", "v4")
    ml, bs = meta.get("max_len", 320), meta.get("batch_size", 128)

    def run(mi, subset, texts_cache):
        m = ens[mi]
        ver = m.get("version", base_ver)
        if ver not in texts_cache:
            texts_cache[ver] = {}
        tc = texts_cache[ver]
        tx = []
        for s in subset:
            k = id(s)
            if k not in tc:
                tc[k] = serialize(s, ver)
            tx.append(tc[k])
        return predict_logits(os.path.join(model_root, m["dir"]), subset, version=ver,
                              max_len=ml, batch_size=bs, device=device, texts=tx,
                              return_probs=True)

    tcache = {}
    full_idx = [i for i in range(len(ens)) if i not in cond_idx]
    wf = sum(w[i] for i in full_idx)
    p_full = sum(w[i] * run(i, samples, tcache) for i in full_idx) / wf
    srt = np.sort(p_full, axis=1)
    sel = np.where((srt[:, -1] - srt[:, -2]) < th)[0]
    out = p_full.copy()
    if len(sel):
        sub = [samples[i] for i in sel]
        acc = wf * p_full[sel]
        wt = wf
        for mi in sorted(cond_idx):
            acc = acc + w[mi] * run(mi, sub, tcache)
            wt += w[mi]
        out[sel] = acc / wt
    return out


def predict_ngram_probs(model_dir, texts):
    """HashingVectorizer(무상태) + numpy 선형헤드 → softmax. sklearn pickle 없음.

    model_dir: coef.npy(14×K), intercept.npy(14), meta.json(wpar/cpar 해싱 파라미터).
    서버 sklearn으로 HashingVectorizer를 파라미터만으로 재생성(무상태) → X @ coef.T + b.
    """
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer
    from scipy.sparse import hstack
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    coef = np.load(os.path.join(model_dir, "coef.npy"))          # (14, K)
    intercept = np.load(os.path.join(model_dir, "intercept.npy"))  # (14,)
    hvw = HashingVectorizer(**meta["wpar"]); hvc = HashingVectorizer(**meta["cpar"])
    X = hstack([hvw.transform(texts), hvc.transform(texts)]).tocsr()
    z = (X @ coef.T).astype(np.float32) + intercept                # (N,14)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ==================== POSMAP_BLOCK_START (배포시 자동 제거) ====================
# 순차 구조 지도 — 검증 결과 히든테스트 커버리지 0으로 기각됨(07-04). 배포 패키저가 이 블록을 스트립.
def _sid_step(sample_id):
    sid, _, st = str(sample_id).rpartition("-step_")
    try:
        return sid, int(st)
    except ValueError:
        return sid, -1


def build_posmap(samples, labels=None):
    pm = {}
    for i, s in enumerate(samples):
        sid, st = _sid_step(s.get("id", ""))
        if st < 0 or not sid.startswith("sess_"):
            continue
        d = pm.setdefault(sid, {})
        if labels is not None:
            d[st] = labels[i]
        acts = [t.get("name") for t in (s.get("history") or [])
                if t.get("role") == "assistant_action" and t.get("name")]
        for k, a in enumerate(acts):
            pos = st - len(acts) + k
            if pos >= 1:
                d.setdefault(pos, a)
    return pm


def load_posmap(path):
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as f:
        obj = json.load(f)
    return {sid: {int(k): v for k, v in d.items()} for sid, d in obj.items()}


def _posmap_prepass(model_dir, samples, meta):
    pm = load_posmap(os.path.join(model_dir, meta["posmap"]))
    tp = build_posmap(samples)
    for sid, d in tp.items():
        t = pm.setdefault(sid, {})
        for pos, a in d.items():
            t.setdefault(pos, a)
    out, work = {}, []
    for s in samples:
        sid, st = _sid_step(s.get("id", ""))
        a = pm.get(sid, {}).get(st) if st >= 0 else None
        if a in CLASSES:
            out[s["id"]] = a
        else:
            work.append(s)
    print(f"[posmap] direct={len(out)} model={len(work)}", flush=True)
    return out, work
# ==================== POSMAP_BLOCK_END ====================


def predict(model_dir, samples, version="v3", max_len=192, batch_size=64,
            device=None, postproc_path=None, max_hist_turns=8, meta=None):
    """samples → {id: action_str}. postproc.json 있으면 per-class bias 적용.

    meta "ensemble": 다중모델 평균(softmax) 경로.
    meta "posmap": 순차 지도 직독 — train 지도 + test 내부 관측 병합, 커버 샘플은 모델 생략.
    """
    import numpy as np
    out = {}
    work = samples
    if meta and meta.get("posmap") and "_posmap_prepass" in globals():
        out, work = _posmap_prepass(model_dir, samples, meta)
        if not work:
            return out

    if meta and meta.get("conditional") and "ensemble" in meta:
        probs = predict_conditional_probs(model_dir, work, meta, device=device)
        scores = np.log(probs + 1e-9)
    elif meta and "ensemble" in meta:
        mean_p, members = predict_ensemble_probs(model_dir, work, meta, device=device,
                                                 return_members=True)
        if meta.get("stacker"):
            # LightGBM 메타: [멤버별 확률(순서=ensemble 순서), 구조피처] — 학습과 동일 배치
            import lightgbm as lgb
            booster = lgb.Booster(model_file=os.path.join(model_dir, meta["stacker"]))
            F = np.array([stack_features(s) for s in work], dtype=np.float32)
            X = np.concatenate(members + [F], axis=1)
            probs = booster.predict(X)
        else:
            probs = mean_p
        scores = np.log(probs + 1e-9)
    else:
        scores = predict_logits(model_dir, work, version, max_len, batch_size, device, max_hist_turns)
        scores = _to_logprobs_np(scores, np)
    if postproc_path and os.path.exists(postproc_path):
        bias = np.array(load_bias(postproc_path), dtype=np.float64)
        pred = (scores + bias).argmax(1)
    else:
        pred = scores.argmax(1)
    for i in range(len(work)):
        out[work[i]["id"]] = IDX_TO_CLASS[int(pred[i])]
    return out
