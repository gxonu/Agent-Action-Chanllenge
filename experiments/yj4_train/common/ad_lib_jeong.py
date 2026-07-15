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


# ===== v9 rich 직렬화용 canonical 추출 (R21: full path/result 신호 복원, portable) =====
_DIR_ROLES = [
    ("test", re.compile(r'(^|/)(tests?|spec|specs|__tests__|e2e|fixtures?)(/|$)', re.I)),
    ("config", re.compile(r'(^|/)(config|configs|conf|settings|\.github|ci|deploy)(/|$)|\.(ya?ml|toml|ini|cfg|env|lock)$', re.I)),
    ("docs", re.compile(r'(^|/)(docs?)(/|$)|readme|\.(md|rst)$', re.I)),
    ("src", re.compile(r'(^|/)(src|lib|libs|app|apps|pkg|packages|internal|cmd|core|components?|services?|api|utils?|models?)(/|$)', re.I)),
]


def _dir_role(path):
    for role, rx in _DIR_ROLES:
        if rx.search(path):
            return role
    return "root" if "/" not in path.strip("/") else "dir"


def _canon_arg(name, args):
    """경로/패턴을 portable canonical로: basename + dir역할 + depth + ext (원 숫자/id 제거)."""
    a = arg_path_or_pattern(name, args)
    if not a:
        return ""
    a = a.strip()
    if ("*" in a or "?" in a or name in ("grep_search", "glob_pattern")) and "/" not in a[:3]:
        return re.sub(r"\d+", "#", a[:40])          # 패턴/심볼 원문 보존(숫자만 정규화)
    base = a.rstrip("/").split("/")[-1] or a
    depth = a.strip("/").count("/")
    ext = path_ext(a)
    base = re.sub(r"\d{2,}", "#", base)[:28]        # 긴 숫자/버전 정규화
    return f"{base}|{_dir_role(a)}|d{depth}" + (f"|{ext}" if ext else "")


def _canon_result(name, summary):
    """result_summary에서 결정신호 핵심 토큰만: 개수/줄수/무매치/트레이스/테스트."""
    s = (summary or "").strip().lower()
    if not s:
        return ""
    toks = []
    m = re.search(r"(\d+)\s*(match|matches|file|files|result|results|occurrence|occurrences|entr|item)", s)
    if m:
        toks.append(f"n{m.group(1)}")
    m = re.search(r"\((\d+)\s*l", s)
    if m:
        toks.append(f"L{m.group(1)}")
    if "no match" in s or re.search(r"\b0\s+match", s) or s.startswith("no ") or "empty" in s:
        toks.append("empty")
    if "traceback" in s or "exception" in s or "stack trace" in s:
        toks.append("trace")
    if re.search(r"\bfail", s):
        toks.append("fail")
    elif "pass" in s or "ok;" in s or s.startswith("ok"):
        toks.append("ok")
    return " ".join(dict.fromkeys(toks))[:40]


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


def _fmt_action(t, with_args=False, full_args=False, rich=False):
    nm = t.get("name", "")
    rs = (t.get("result_summary") or "")[:100]
    st = result_status(nm, rs)
    if rich:
        # v9: v6의 (ext,status,result[:100])를 그대로 유지 + 경로구조(role/depth/basename) 추가 = 순수 상위집합(R21)
        raw = arg_path_or_pattern(nm, t.get("args") or {}).strip()
        if raw:
            if "*" in raw or "?" in raw or nm in ("grep_search", "glob_pattern"):
                a = re.sub(r"\d+", "#", raw[:40])                       # 패턴/심볼 원문
            else:
                base = re.sub(r"\d{2,}", "#", raw.rstrip("/").split("/")[-1] or raw)[:28]
                a = f"{path_ext(raw) or '-'}|{_dir_role(raw)}|d{raw.strip('/').count('/')}|{base}"
        else:
            a = ""
        return f"{nm}({a})[{st}] {rs}" if a else f"{nm}[{st}] {rs}"
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
    _metaver = ("v4", "v5", "v6", "v7", "v6n", "v8", "v9")
    _seqver = ("v6", "v7", "v6n", "v8", "v9")
    hdr = (f"[TIER] {m['user_tier']} [LANG] {m['language_pref']} [TURN] {_turn_bin(m['turn_index'])} "
           f"[CI] {m['last_ci_status']} [GIT] {'dirty' if m['git_dirty'] else 'clean'}")
    gen_blk = None
    openext = None
    if version in _metaver:
        # v6n = [GEN] 제거(R15). v8 = 메타/[GEN]를 꼬리로 이동(좌측절단 생존, R18 H2)
        gen_part = "" if version == "v6n" else f"[GEN] {_gen(sample.get('id',''))} "
        gen_blk = (f"{gen_part}[BUDGET] {_budget_bin(m['budget'])} "
                   f"[LOC] {_loc_bin(m['loc'])} [TOPLANG] {m['top_lang']} [NOPEN] {m['n_open_files']}")
        if m["open_files"]:
            if version == "v5":
                openext = f"[OPEN] {' '.join(p[:60] for p in m['open_files'][:5])}"
            else:
                exts = sorted({path_ext(p) for p in m["open_files"] if path_ext(p)})
                if exts:
                    openext = f"[OPENEXT] {','.join(exts)}"
    full_hist = sample.get("history") or []
    hist = full_hist[-max_hist_turns:]
    hist_parts = []
    if hist:
        hist_parts.append("[HIST]")
        for t in hist:
            if t.get("role") == "user":
                hist_parts.append(f"u: {(t.get('content') or '')[:150]}")
            elif t.get("role") == "assistant_action":
                hist_parts.append(f"a: {_fmt_action(t, with_args=(version in ('v4', 'v6', 'v7', 'v6n', 'v8')), full_args=(version == 'v5'), rich=(version == 'v9'))}")
    seq_parts = []
    if version in _seqver:
        seq = [t.get("name", "") for t in full_hist if t.get("role") == "assistant_action"]
        seq_parts.append(f"[SEQ] {'>'.join(seq[-12:]) if seq else 'none'}")
        if not full_hist:
            seq_parts.append("[NOHIST]")
    pflag = f"[PFLAG] {_prompt_flags(prompt, m['open_files'])}" if version in _seqver else None

    if version == "v8":
        # 꼬리 재배치: [HIST](가장 오래됨=절단 1순위) → [SEQ] → 메타헤더+[GEN] → [PFLAG] → [CUR].
        # 좌측절단이 [HIST]부터 먹어 [GEN]/메타가 320창서 항상 생존 (v6는 긴세션 10.8%서 [GEN] 소실).
        parts = list(hist_parts) + list(seq_parts) + [hdr]
        if gen_blk:
            parts.append(gen_blk)
        if openext:
            parts.append(openext)
        if pflag:
            parts.append(pflag)
        parts.append(f"[CUR] {prompt}")
        return " ".join(parts)

    # v4~v7·v6n: 기존 순서(메타헤더 앞) — v6 바이트 동일 보장
    parts = [hdr]
    if gen_blk is not None:
        parts.append(gen_blk)
        if openext:
            parts.append(openext)
    parts += hist_parts
    parts += seq_parts
    if pflag:
        parts.append(pflag)
    if version in ("v7", "v9"):
        # v9: elapsed 세션단계 prior (R9서 탐색분포 단조신호 확인, OOF-kill은 오판 가능 — R21 재검토)
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
        elif tag == "p4":                         # int4 nibble-pack (sim/quantize_member_int4.py)
            packed = z[k]
            s = z[f"s4::{name}"].astype(np.float32)   # (o, i/G)
            o = packed.shape[0]; i = packed.shape[1] * 2
            u = np.empty((o, i), np.uint8)
            u[:, 0::2] = packed >> 4
            u[:, 1::2] = packed & 0x0F
            q = u.astype(np.float32) - 8.0
            w = (q.reshape(o, s.shape[1], -1) * s[:, :, None]).reshape(o, i)
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


def _member_mht(member):
    """멤버별 max_hist_turns (R62 m1-T3 서빙 정합). 멤버 레벨만 허용 — meta 레벨 fallback 금지
    (최상위 오기 한 줄이 전 멤버를 오염시키는 footgun, codex R63). 허용값 {8, 12}."""
    mht = member.get("mht", 8)
    if mht not in (8, 12):
        raise ValueError(f"member mht={mht!r} 미허용 (8 또는 12만)")
    return int(mht)


def _gen_rescue_ids(tok, texts, max_len):
    """GEN-rescue(R53): 좌측절단으로 [GEN] 헤더가 삭제될 행만 헤더보존 input_ids 재구성.

    v6 계열 직렬화는 [TIER]~[OPENEXT]/[GEN] 헤더(~59토큰)가 왼쪽 + truncation_side="left"
    → max_len 초과 행의 일부(fold0-val 11.4%)에서 [GEN] 포함 헤더 소실. 대상 행에 한해
    헤더 블록을 보존하고 잔여 예산을 꼬리(최근 history+[CUR])로 채운다.

    대상 판정(자기가드): "[GEN]"이 원문에 있고, 절단 후 유지창(decode)에서 사라지는 행만.
    v8(헤더 꼬리 배치)·v6n([GEN] 없음)·비절단 행은 자동 무대상 → input_ids 불변.
    반환: {row_index: input_ids(스페셜 포함, len<=max_len)}
    """
    out = {}
    n_special = tok.num_special_tokens_to_add(False)
    keep = max_len - n_special
    ids_all = tok(list(texts), add_special_tokens=False)["input_ids"]
    for i, (t, ids) in enumerate(zip(texts, ids_all)):
        if "[GEN]" not in t or len(ids) <= keep:
            continue                                   # [GEN] 무존재 or 비절단
        if "[GEN]" in tok.decode(ids[-keep:]):
            continue                                   # 절단돼도 [GEN] 생존
        pos = -1
        for mark in (" [HIST]", " [SEQ]", " [PFLAG]", " [CUR]"):
            p = t.find(mark)
            if p > 0:
                pos = p
                break
        if pos <= 0:
            continue                                   # 헤더 경계 불명 → 안전 무개입
        head = tok(t[:pos], add_special_tokens=False)["input_ids"]
        budget = keep - len(head)
        if budget <= 0:
            continue
        tail = tok(t[pos:], add_special_tokens=False)["input_ids"][-budget:]
        out[i] = tok.build_inputs_with_special_tokens(head + tail)
    return out


def serialize_compress(sample, tok, keep, u_cap=60, rs_cap=30, max_items=12):
    """CompressView(R55 D1): v6 헤더/[SEQ]/[PFLAG]/[CUR] 유지 + [HIST] 압축 재직렬화.

    u:{u_cap}/rs:{rs_cap} 캡, 최대 {max_items} 아이템, 턴경계 정렬로 keep 토큰 이하 보장.
    행당 반복 토크나이즈 대신 파트별 1회 토크나이즈 + 누적길이 이분탐색(전략가 최적화):
    xlm-r 계열 sentencepiece 는 파트 선행공백이 ▁ 로 안정 — 파트 길이합 ≈ 결합문 길이.
    경계 오차 대비 최종 1회 검증, 초과 시 아이템 1개씩 추가 드랍(희귀).
    """
    prompt = sample.get("current_prompt")
    if not isinstance(prompt, str):
        prompt = "" if prompt is None else str(prompt)
    m = meta_fields(sample)
    head = [f"[TIER] {m['user_tier']} [LANG] {m['language_pref']} [TURN] {_turn_bin(m['turn_index'])} "
            f"[CI] {m['last_ci_status']} [GIT] {'dirty' if m['git_dirty'] else 'clean'}",
            f"[GEN] {_gen(sample.get('id', ''))} [BUDGET] {_budget_bin(m['budget'])} "
            f"[LOC] {_loc_bin(m['loc'])} [TOPLANG] {m['top_lang']} [NOPEN] {m['n_open_files']}"]
    if m["open_files"]:
        exts = sorted({path_ext(p) for p in m["open_files"] if path_ext(p)})
        if exts:
            head.append(f"[OPENEXT] {','.join(exts)}")
    full_hist = sample.get("history") or []
    seq = [t.get("name", "") for t in full_hist if t.get("role") == "assistant_action"]
    tail = [f"[SEQ] {'>'.join(seq[-12:]) if seq else 'none'}"]
    if not full_hist:
        tail.append("[NOHIST]")
    tail.append(f"[PFLAG] {_prompt_flags(prompt, m['open_files'])}")
    tail.append(f"[CUR] {prompt}")

    def turn_str(t):
        if t.get("role") == "user":
            return f"u: {(t.get('content') or '')[:u_cap]}"
        nm = t.get("name", "")
        rs = (t.get("result_summary") or "")[:rs_cap]
        st = result_status(nm, t.get("result_summary") or "")
        a = arg_path_or_pattern(nm, t.get("args") or {})
        ext = path_ext(a) or glob_ext(a)
        return f"a: {nm}{('(' + ext + ')') if ext else ''}[{st}] {rs}"

    hist_items = [turn_str(t) for t in full_hist[-max_items:]]
    # 파트별 토큰길이 1회 계산 (배치 토크나이즈)
    pieces = head + tail + (["[HIST]"] + hist_items if hist_items else [])
    plens = [len(x) for x in tok(pieces, add_special_tokens=False)["input_ids"]]
    n_ht = len(head) + len(tail)
    fixed = sum(plens[:n_ht])
    if hist_items:
        import bisect
        hist_lens = plens[n_ht + 1:]
        budget = keep - fixed - plens[n_ht]          # [HIST] 마커 비용
        # 뒤(최신)에서부터 누적합 — 예산에 드는 최대 suffix 를 이분탐색으로
        csum, acc = [], 0
        for v in reversed(hist_lens):
            acc += v
            csum.append(acc)
        k = bisect.bisect_right(csum, budget)        # 최신 k개 유지
        hist_keep = hist_items[len(hist_items) - k:] if k > 0 else []
    else:
        hist_keep = []
    while True:
        parts = head + (["[HIST]"] + hist_keep if hist_keep else []) + tail
        text = " ".join(parts)
        if len(tok(text, add_special_tokens=False)["input_ids"]) <= keep or not hist_keep:
            return text
        hist_keep = hist_keep[1:]                    # 경계 오차 시 오래된 것부터 드랍


def predict_logits(model_dir, samples, version="v3", max_len=192, batch_size=64,
                   device=None, max_hist_turns=8, texts=None, return_probs=False, return_emb=False,
                   gen_rescue=False, rescue_rows_out=None):
    """samples(list[dict]) → logits/probs(np.ndarray [N,14]). 길이정렬 배칭 + fp16(GPU).

    - model_dir/id_map.npy 존재 시 vocab-pruned 모델로 간주, 토큰 id 리매핑.
    - model_dir/qweights.npz 존재 시 int8 양자화 멤버 → safetensors 선복원.
    - texts 인자를 주면 직렬화 재계산 생략(앙상블에서 공유).
    - gen_rescue(R53, opt-in): [GEN] 삭제 절단 행만 헤더보존 절단 — 비대상 행 input_ids 불변.
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
    rescued = _gen_rescue_ids(tok, texts, max_len) if gen_rescue else None
    if rescue_rows_out is not None and rescued is not None:
        rescue_rows_out["rows"] = sorted(rescued.keys())   # compress_tta 스캔 재사용(R55 D1)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    logits = np.zeros((len(texts), NUM_CLASSES), dtype=np.float32)
    emb = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32) if return_emb else None
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            if rescued and any(i in rescued for i in idx):
                # 대상 행만 재구성 ids, 비대상 행은 동일 토크나이즈(단건==배치 동일) → byte-identity
                seqs = [rescued[i] if i in rescued else
                        tok(texts[i], truncation=True, max_length=max_len)["input_ids"]
                        for i in idx]
                enc = tok.pad({"input_ids": seqs}, padding=True, pad_to_multiple_of=8,
                              return_tensors="pt")
            else:
                enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                          max_length=max_len, pad_to_multiple_of=8, return_tensors="pt")
            if id_map is not None:
                enc["input_ids"] = torch.from_numpy(
                    id_map[enc["input_ids"].numpy()]).to(enc["input_ids"].dtype)
            enc = enc.to(device)
            if return_emb:
                # base encoder 1회 → last_hidden에서 logits(classifier)와 mean-pool emb 동시 추출.
                # (output_hidden_states=True는 전 레이어 materialize로 T4 OOM 위험 → 회피)
                lh = model.base_model(**enc).last_hidden_state           # [B,T,H]
                out = model.classifier(lh).float()
                mask = enc["attention_mask"].unsqueeze(-1).float()
                mp = ((lh * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy()
                for j, i in enumerate(idx):
                    emb[i] = mp[j]
            else:
                out = model(**enc).logits.float()
            if return_probs:
                out = torch.softmax(out, dim=1)
            out = out.cpu().numpy()
            for j, i in enumerate(idx):
                logits[i] = out[j]
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return (logits, emb) if return_emb else logits


def predict_ensemble_probs(model_root, samples, meta, device=None, return_members=False,
                           return_emb_member=None):
    """run_meta의 ensemble 정의에 따라 멤버별 softmax 계산.

    meta 예: {"version":"v4","max_len":320,"batch_size":128,
              "ensemble":[{"dir":"m1"},{"dir":"m2","version":"v5"}]}
    멤버별 "mht"(max_hist_turns, 기본 8) opt-in — R62 m1-T3(hist12 학습) 서빙 정합용.
    반환: 평균 probs[N,14]  (return_members=True면 (평균, [멤버별 probs...]))
    """
    import numpy as np
    members = []
    emb_out = None
    base_ver = meta.get("version", "v4")
    cache = {}
    for mi, member in enumerate(meta["ensemble"]):
        mdir = os.path.join(model_root, member["dir"])
        ver = member.get("version", base_ver)
        mht = _member_mht(member)
        ck = (ver, mht)
        if ck not in cache:
            cache[ck] = [serialize(s, ver, mht) for s in samples]
        if member.get("type") == "ngram":
            p = predict_ngram_probs(mdir, cache[ck])
        elif return_emb_member is not None and mi == int(return_emb_member):
            raw, emb_out = predict_logits(mdir, samples, version=ver,
                                          max_len=member.get("max_len", meta.get("max_len", 320)),
                                          batch_size=meta.get("batch_size", 128),
                                          device=device, texts=cache[ck], return_emb=True,
                                          gen_rescue=bool(member.get("gen_rescue", meta.get("gen_rescue", False))))
            z = raw - raw.max(1, keepdims=True)
            ez = np.exp(z)
            p = ez / ez.sum(1, keepdims=True)
        else:
            p = predict_logits(mdir, samples, version=ver,
                               max_len=member.get("max_len", meta.get("max_len", 320)),
                               batch_size=meta.get("batch_size", 128),
                               device=device, texts=cache[ck], return_probs=True,
                               gen_rescue=bool(member.get("gen_rescue", meta.get("gen_rescue", False))))
        members.append(p)
    w = meta.get("weights")
    if w:
        assert len(w) == len(members), "weights 길이 != 멤버수"
        mean = sum(wi * m for wi, m in zip(w, members)) / sum(w)
    else:
        mean = sum(members) / len(members)
    if return_emb_member is not None:
        return (mean, members, emb_out) if return_members else (mean, emb_out)
    return (mean, members) if return_members else mean


def predict_conditional_probs(model_root, samples, meta, device=None, return_emb_member=None):
    """조건부 앙상블 — 10분캡 안에서 full 앙상블 이득 대부분 회수 (codex R11 → R12 일반화).

    full 멤버들(cond_members 제외)을 전체에 추론·가중혼합 → top1-top2 마진 < margin_th 인
    행만 cond 멤버 추가 추론·재혼합. LB 실측(07-05): 서버 시간예산 large+230s/base+117s.

    conditional["stages"] (R51 tt30, 선택적·하위호환): 다단 조건부 가중.
      [{"th": 0.6, "weights": [...]}, {"th": 0.3, "weights": [...]}] — th 내림차순으로
      적용해 깊은 단이 덮어씀. 모든 stage th ≤ margin_th 필수(게이트 밖 행은 cond 미추론).
      cond 멤버 추론은 margin_th 게이트 1회뿐 — stage 는 가중 재혼합만 하므로 시간 영향 0.
      stage weights 는 멤버 전체 길이 벡터. full 멤버가 2개 이상이면 full 그룹은 top-level
      weights 비율의 p_full 을 유지하고 stage 의 full 가중 합(wf_s)으로 스케일된다.
      "stages" 부재 시 기존 단일-th 경로 그대로(연산 순서 불변 — 기존 패키지 재현 무결).
    """
    import numpy as np
    ens = meta["ensemble"]
    w = meta.get("weights") or [1.0] * len(ens)
    cond = meta["conditional"]
    th = float(cond["margin_th"])
    stages = cond.get("stages")
    if stages:
        stages = sorted(({"th": float(s["th"]), "weights": [float(x) for x in s["weights"]]}
                         for s in stages), key=lambda s: -s["th"])
        for s in stages:
            assert len(s["weights"]) == len(ens), "stage weights 길이 != 멤버수"
            assert s["th"] <= th + 1e-9, "stage th > margin_th (게이트 밖 행은 cond 추론이 없음)"
    member_th = cond.get("member_th")   # R60 S2: cond 멤버별 참여 th {"1":0.95,"2":0.75} — opt-in
    cond_idx = set(cond.get("cond_members", [len(ens) - 1]))   # 기본: 마지막 멤버만 조건부
    if member_th:
        member_th = {int(k): float(v) for k, v in member_th.items()}
        assert set(member_th) <= cond_idx, "member_th 키가 cond_members 밖"
        assert all(v <= th + 1e-9 for v in member_th.values()), "member_th > margin_th"
        assert not stages, "member_th 와 stages 동시 사용 미지원(오늘 스코프)"
        assert not meta.get("compress_tta"), "member_th 와 compress_tta 동시 사용 미지원(오늘 스코프)"
    base_ver = meta.get("version", "v4")
    ml, bs = meta.get("max_len", 320), meta.get("batch_size", 128)
    emb_out = None

    def run(mi, subset, texts_cache):
        nonlocal emb_out
        m = ens[mi]
        ver = m.get("version", base_ver)
        mht = _member_mht(m)   # R62 m1-T3: hist12 학습멤버 서빙 정합 — opt-in
        ck = (ver, mht)
        if ck not in texts_cache:
            texts_cache[ck] = {}
        tc = texts_cache[ck]
        tx = []
        for s in subset:
            k = id(s)
            if k not in tc:
                tc[k] = serialize(s, ver, mht)
            tx.append(tc[k])
        gr = bool(m.get("gen_rescue", meta.get("gen_rescue", False)))
        mml = int(m.get("max_len", ml))   # 멤버별 max_len 오버라이드 (R57 mdeb@384)
        rro = _rescue_rows if (mi == full_gate_mi and len(subset) == len(samples)) else None
        if return_emb_member is not None and mi == int(return_emb_member) and len(subset) == len(samples):
            raw, emb_out = predict_logits(os.path.join(model_root, m["dir"]), subset, version=ver,
                                          max_len=mml, batch_size=bs, device=device, texts=tx,
                                          return_emb=True, gen_rescue=gr, rescue_rows_out=rro)
            z = raw - raw.max(1, keepdims=True)
            ez = np.exp(z)
            return ez / ez.sum(1, keepdims=True)
        return predict_logits(os.path.join(model_root, m["dir"]), subset, version=ver,
                              max_len=mml, batch_size=bs, device=device, texts=tx,
                              return_probs=True, gen_rescue=gr, rescue_rows_out=rro)

    tta = meta.get("compress_tta")   # R55 D1: {"lambda":0.5,"margin_th":0.5,("members":[...])}
    if tta and any(_member_mht(m) != 8 for m in ens):
        raise ValueError("compress_tta 와 멤버 mht!=8 동시 사용 미지원(R62 스코프)")
    _rescue_rows = {} if tta else None
    full_idx_pre = [i for i in range(len(ens)) if i not in cond_idx]
    full_gate_mi = full_idx_pre[0] if full_idx_pre else -1
    tcache = {}
    full_idx = [i for i in range(len(ens)) if i not in cond_idx]
    wf = sum(w[i] for i in full_idx)
    p_full = sum(w[i] * run(i, samples, tcache) for i in full_idx) / wf
    srt = np.sort(p_full, axis=1)
    sel = np.where((srt[:, -1] - srt[:, -2]) < th)[0]

    rows_tta, comp_texts = [], None
    if tta and len(sel):
        # 대상 = 게이트 마진(p_full, TTA 전 고정) < tta.margin_th ∧ 게이트멤버 토크나이저 기준 GEN삭제.
        # 압축뷰는 게이트멤버(첫 full) 토크나이저로 ≤max_len 보장 — 타 멤버 패스는 gen_rescue 가드.
        lam = float(tta.get("lambda", 0.5))
        tth = float(tta.get("margin_th", 0.5))
        assert tth <= th + 1e-9, "compress_tta.margin_th > margin_th (게이트 밖 행 TTA 불가)"
        from transformers import AutoTokenizer
        m0 = ens[full_idx[0]]
        ver0 = m0.get("version", base_ver)
        tok0 = AutoTokenizer.from_pretrained(os.path.join(model_root, m0["dir"]),
                                             local_files_only=True)
        tok0.truncation_side = "left"
        marg_all = srt[:, -1] - srt[:, -2]
        cand = np.where(marg_all < tth)[0]
        ml0 = int(m0.get("max_len", ml))   # 게이트멤버 max_len (멤버별 오버라이드 존중)
        if _rescue_rows and "rows" in _rescue_rows:
            # 게이트멤버 full 패스(gen_rescue)에서 이미 스캔한 대상 행 재사용 — 추가 스캔 0
            tgt_all = set(_rescue_rows["rows"])
            rows_tta = [int(i) for i in cand if int(i) in tgt_all]
        else:
            tc0 = tcache.get((ver0, 8), {})   # mht!=8 은 위 가드로 배제 — 게이트멤버는 항상 hist8
            cand_texts = [tc0.get(id(samples[i])) or serialize(samples[i], ver0) for i in cand]
            tgt_rel = sorted(_gen_rescue_ids(tok0, cand_texts, ml0).keys())
            rows_tta = [int(cand[j]) for j in tgt_rel]
        if rows_tta:
            n_sp = tok0.num_special_tokens_to_add(False)
            comp_texts = [serialize_compress(samples[i], tok0, ml0 - n_sp) for i in rows_tta]

    tta_pos = {r: j for j, r in enumerate(rows_tta)}
    tta_members = set(int(x) for x in tta.get("members", range(len(ens)))) if tta else set()

    def tta_blend(mi, P_rows, rows_local):
        """멤버 mi 확률(P_rows: rows_local 순서)에 압축뷰 2패스 λ-혼합. 비대상 행 불변."""
        if mi not in tta_members:
            return P_rows
        pos = [k for k, r in enumerate(rows_local) if r in rows_tta_set]
        if not pos:
            return P_rows
        sub_t = [samples[rows_local[k]] for k in pos]
        tx_t = [comp_texts[tta_pos[rows_local[k]]] for k in pos]
        m = ens[mi]
        Pt = predict_logits(os.path.join(model_root, m["dir"]), sub_t,
                            version=m.get("version", base_ver),
                            max_len=int(m.get("max_len", ml)), batch_size=bs,
                            device=device, texts=tx_t, return_probs=True, gen_rescue=True)
        lam = float(tta.get("lambda", 0.5))
        P_rows = P_rows.copy()
        P_rows[pos] = (1.0 - lam) * P_rows[pos] + lam * Pt
        return P_rows

    rows_tta_set = set(rows_tta)
    if rows_tta:
        # full 그룹: p_full 재구성 대신 게이트멤버 단일이면 직접 블렌드(가중 소거 동일).
        # 다중 full 멤버는 각 멤버 TTA 후 재혼합이 정확하나 현행 배포는 full=1 — 일반화는 보수적으로 금지.
        assert len(full_idx) == 1, "compress_tta 는 full 멤버 1개 구조 전용 (현행 tri)"
        p_full = tta_blend(full_idx[0], p_full, list(range(len(samples))))

    out = p_full.copy()
    if len(sel) and member_th:
        # R60 S2 비대칭 커버리지: 멤버 mi 는 margin < member_th[mi] 행만 추론·참여.
        # 행별 참여집합으로 재정규 — 예: 0.75<=m<0.95 행은 (w_L*L + w_D*D)/(w_L+w_D).
        marg_sel = (srt[:, -1] - srt[:, -2])[sel]
        acc = wf * p_full[sel]
        wt_row = np.full(len(sel), wf, dtype=p_full.dtype)
        for mi in sorted(cond_idx):
            mth = member_th.get(mi, th)
            rr = np.where(marg_sel < mth)[0]
            if not len(rr):
                continue
            sub_mi = [samples[int(sel[j])] for j in rr]
            p_mi = run(mi, sub_mi, tcache)          # 참여행만 추론 — 시간 절감 원천
            acc[rr] = acc[rr] + w[mi] * p_mi
            wt_row[rr] += w[mi]
        out[sel] = acc / wt_row[:, None]
    elif len(sel):
        sub = [samples[i] for i in sel]
        if stages or rows_tta:
            # dict 경로: cond 멤버 추론 1회 (+ TTA 블렌드) 후 가중 재혼합
            marg_sel = (srt[:, -1] - srt[:, -2])[sel]
            cond_p = {mi: run(mi, sub, tcache) for mi in sorted(cond_idx)}
            if rows_tta:
                sel_list = [int(i) for i in sel]
                cond_p = {mi: tta_blend(mi, P, sel_list) for mi, P in cond_p.items()}
            if stages:
                for st in stages:                  # 넓은 th → 좁은 th (깊은 단이 덮어씀)
                    rr = np.where(marg_sel < st["th"])[0]
                    if not len(rr):
                        continue
                    sw = st["weights"]
                    wf_s = sum(sw[i] for i in full_idx)
                    acc = wf_s * p_full[sel[rr]]
                    wt = wf_s
                    for mi in sorted(cond_idx):
                        acc = acc + sw[mi] * cond_p[mi][rr]
                        wt += sw[mi]
                    out[sel[rr]] = acc / wt
            else:
                acc = wf * p_full[sel]
                wt = wf
                for mi in sorted(cond_idx):
                    acc = acc + w[mi] * cond_p[mi]
                    wt += w[mi]
                out[sel] = acc / wt
        else:
            acc = wf * p_full[sel]
            wt = wf
            for mi in sorted(cond_idx):
                acc = acc + w[mi] * run(mi, sub, tcache)
                wt += w[mi]
            out[sel] = acc / wt
    if return_emb_member is not None:
        return out, emb_out
    return out


def _retrieval_embedding_from_meta(model_root, samples, meta, device=None):
    """앙상블/조건부 경로용 retrieval embedding 추출.

    기본은 첫 transformer 멤버를 사용한다. tri 계열은 m1 large-v6 기준으로 retrieval
    pack을 만들었으므로, 필요하면 run_meta의 retrieval_emb_member(0-index)로 고정한다.
    """
    ens = meta["ensemble"]
    mi = int(meta.get("retrieval_emb_member", 0))
    if mi < 0 or mi >= len(ens) or ens[mi].get("type") == "ngram":
        mi = next((i for i, m in enumerate(ens) if m.get("type") != "ngram"), -1)
    if mi < 0:
        raise ValueError("retrieval embedding을 뽑을 transformer 멤버가 없음")
    m = ens[mi]
    ver = m.get("version", meta.get("version", "v4"))
    texts = [serialize(s, ver) for s in samples]
    _, emb = predict_logits(os.path.join(model_root, m["dir"]), samples, version=ver,
                            max_len=m.get("max_len", meta.get("max_len", 320)),
                            batch_size=meta.get("batch_size", 128),
                            device=device, texts=texts, return_emb=True)
    return emb


def _retrieval_adjust(scores, emb, model_dir, cfg):
    """near-dup prior 보정 (R22/R23). model_dir/retrieval/ 번들 사용.

    train_emb.npy(중심화·정규화 fp16 [Ntr,H]), train_labels.npy, emb_mean.npy.
    cfg: {lambda, margin_th, purity_th, k, sim_th(옵션)}. gated logit 보정.
    """
    import numpy as np
    rdir = os.path.join(model_dir, "retrieval")
    tr = np.load(os.path.join(rdir, "train_emb.npy")).astype(np.float32)   # [Ntr,H] 중심화·정규화됨
    tl = np.load(os.path.join(rdir, "train_labels.npy"))
    mu = np.load(os.path.join(rdir, "emb_mean.npy")).astype(np.float32)
    pp = os.path.join(rdir, "proj.npy")
    proj = np.load(pp).astype(np.float32) if os.path.exists(pp) else None
    lam = float(cfg.get("lambda", 0.3)); mth = float(cfg.get("margin_th", 0.30))
    pth = float(cfg.get("purity_th", 0.70)); K = int(cfg.get("k", 8))
    sth = float(cfg.get("sim_th", -1.0)); gcap = float(cfg.get("gate_cap", 0.12))
    ho_med = float(cfg.get("holdout_top1_median", 0.0))
    q = emb.astype(np.float32) - mu
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    if proj is not None:
        q = q @ proj
        q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    probs = np.exp(scores - scores.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    srt = np.sort(probs, axis=1); mgn = srt[:, -1] - srt[:, -2]
    logpc = np.log(probs.mean(0) + 1e-9)
    out = scores.copy()
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Tt = torch.from_numpy(tr).to(dev)
        top1_all = np.zeros(len(q), np.float32)
        prs = {}                                            # gi -> prior (통과행만)
        for b in range(0, len(q), 2048):
            qb = torch.from_numpy(q[b:b + 2048]).to(dev)
            tsim, tidx = torch.topk(qb @ Tt.T, K, dim=1)
            tsim = tsim.cpu().numpy(); tidx = tidx.cpu().numpy()
            for r in range(qb.shape[0]):
                gi = b + r; top1_all[gi] = tsim[r][0]
                if mgn[gi] >= mth:
                    continue
                lbl = tl[tidx[r]]
                if np.bincount(lbl[:8], minlength=NUM_CLASSES).max() / 8.0 < pth:
                    continue
                if sth > 0 and tsim[r][0] < sth:
                    continue
                w = np.clip(tsim[r], 0, None) ** 4
                pr = np.bincount(lbl, weights=w, minlength=NUM_CLASSES).astype(np.float64)
                prs[gi] = pr / (pr.sum() + 1e-9)
        # ---- OOD guard (R23): 게이트율 상한 + 테스트 유사도 분포 shift 감지 ----
        rate = len(prs) / max(len(q), 1)
        test_med = float(np.median(top1_all))
        eff_lam = lam
        if ho_med > 0 and test_med < ho_med - 0.03:
            eff_lam = min(eff_lam, 0.15)
            print(f"[retrieval] OOD: test top1 med {test_med:.3f} << holdout {ho_med:.3f} → λ={eff_lam}", flush=True)
        if rate > gcap:
            eff_lam = 0.0
            print(f"[retrieval] 게이트율 {rate*100:.1f}%>{gcap*100:.0f}% → 비활성(안전)", flush=True)
        print(f"[retrieval] 게이트율 {rate*100:.2f}% test-top1-med {test_med:.3f} λ={eff_lam}", flush=True)
        if eff_lam > 0:
            for gi, pr in prs.items():
                out[gi] = out[gi] + eff_lam * (np.log(pr + 1e-9) - logpc)
    except Exception as e:
        print(f"[retrieval] skip (오류 폴백): {e}", flush=True)
        return scores
    return out


def _labelshift_em_adjust(scores, cfg):
    """Serve-time EM label-shift correction (Saerens et al.).

    cfg: {"pi_ref": [C], "shrink": 0.5, "iters": 80, "min_n": 512}.
    Operates in probability space, then returns log-probs for downstream bias.
    """
    import numpy as np
    if not cfg:
        return scores
    n = len(scores)
    min_n = int(cfg.get("min_n", 512))
    if n < min_n:
        print(f"[labelshift_em] skip n={n}<min_n={min_n}", flush=True)
        return scores
    eps = float(cfg.get("eps", 1e-9))
    pi0 = np.asarray(cfg.get("pi_ref", []), dtype=np.float64)
    if pi0.shape[0] != NUM_CLASSES:
        print("[labelshift_em] skip (pi_ref missing/bad)", flush=True)
        return scores
    pi0 = np.maximum(pi0, eps)
    pi0 = pi0 / pi0.sum()
    probs = np.exp(scores - scores.max(1, keepdims=True))
    probs = probs / probs.sum(1, keepdims=True)
    pi = pi0.copy()
    iters = int(cfg.get("iters", 80))
    tol = float(cfg.get("tol", 1e-8))
    for _ in range(iters):
        w = probs * (pi / pi0)
        w = w / np.maximum(w.sum(1, keepdims=True), eps)
        new_pi = w.mean(0)
        if np.abs(new_pi - pi).sum() < tol:
            pi = new_pi
            break
        pi = new_pi
    shrink = float(cfg.get("shrink", cfg.get("lambda", 0.5)))
    shrink = max(0.0, min(1.0, shrink))
    pi_use = (1.0 - shrink) * pi0 + shrink * pi
    pi_use = np.maximum(pi_use, eps)
    pi_use = pi_use / pi_use.sum()
    ratio = pi_use / pi0
    clip_ratio = float(cfg.get("clip_ratio", 0.0))
    if clip_ratio > 1.0:
        ratio = np.clip(ratio, 1.0 / clip_ratio, clip_ratio)
    out = probs * ratio
    out = out / np.maximum(out.sum(1, keepdims=True), eps)
    base = scores.argmax(1)
    new = out.argmax(1)
    print(f"[labelshift_em] n={n} shrink={shrink:.2f} "
          f"pi_l1={np.abs(pi - pi0).sum():.3f} changed={(base != new).mean()*100:.2f}%",
          flush=True)
    return np.log(out + eps)


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
        if meta.get("retrieval"):
            probs, emb = predict_conditional_probs(
                model_dir, work, meta, device=device,
                return_emb_member=meta.get("retrieval_emb_member", 0))
        else:
            probs = predict_conditional_probs(model_dir, work, meta, device=device)
            emb = None
        scores = np.log(probs + 1e-9)
    elif meta and "ensemble" in meta:
        if meta.get("retrieval"):
            mean_p, members, emb = predict_ensemble_probs(
                model_dir, work, meta, device=device, return_members=True,
                return_emb_member=meta.get("retrieval_emb_member", 0))
        else:
            mean_p, members = predict_ensemble_probs(model_dir, work, meta, device=device,
                                                     return_members=True)
            emb = None
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
    elif meta and meta.get("retrieval"):
        # 단일모델 로짓 + mean-pool 임베딩 1-forward → near-dup prior 보정 (R22 X1)
        raw, emb = predict_logits(model_dir, work, version, max_len, batch_size, device,
                                  max_hist_turns, return_emb=True,
                                  gen_rescue=bool(meta.get("gen_rescue", False)) if meta else False)
        scores = _to_logprobs_np(raw, np)
        scores = _retrieval_adjust(scores, emb, model_dir, meta["retrieval"])
    else:
        scores = predict_logits(model_dir, work, version, max_len, batch_size, device, max_hist_turns,
                                gen_rescue=bool(meta.get("gen_rescue", False)) if meta else False)
        scores = _to_logprobs_np(scores, np)

    if meta and meta.get("retrieval") and "ensemble" in meta:
        if emb is None:
            emb = _retrieval_embedding_from_meta(model_dir, work, meta, device=device)
        scores = _retrieval_adjust(scores, emb, model_dir, meta["retrieval"])
    if meta and meta.get("labelshift_em"):
        scores = _labelshift_em_adjust(scores, meta["labelshift_em"])
    if postproc_path and os.path.exists(postproc_path):
        with open(postproc_path, encoding="utf-8") as f:
            _pp = json.load(f)
        if _pp.get("classes") != CLASSES:
            raise ValueError("postproc.json 클래스 순서 불일치")
        bias = np.array(_pp["bias"], dtype=np.float64)
        pred = None
        if "bias_au" in _pp or "bias_sim" in _pp:
            # 듀얼 bias(R14): id prefix로 행별 선택. prefix 체계가 확인될 때만 적용,
            # 아니면(au 없음/무표식 id) 글로벌 bias 폴백 = 기존과 동일 동작(무손실).
            _rid = [str(s.get("id", "")) for s in work]
            _au = np.array([r.startswith("sess_au") for r in _rid])
            _sim = np.array([r.startswith("sess_sim") for r in _rid])
            if _au.any() and bool((_au | _sim).all()):
                _b_sim = np.array(_pp.get("bias_sim", _pp["bias"]), dtype=np.float64)
                _b_au = np.array(_pp.get("bias_au", _pp["bias"]), dtype=np.float64)
                _brow = np.where(_au[:, None], _b_au[None, :], _b_sim[None, :])
                pred = (scores + _brow).argmax(1)
        if pred is None:
            pred = (scores + bias).argmax(1)
    else:
        pred = scores.argmax(1)
    for i in range(len(work)):
        out[work[i]["id"]] = IDX_TO_CLASS[int(pred[i])]
    return out
