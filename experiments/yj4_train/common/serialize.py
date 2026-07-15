"""입력 직렬화 — 구조 신호 + 텍스트를 단일 문자열로.

EDA 결론 반영:
- current_prompt(짧음, mean 61자)이 지배적 신호 → 항상 포함, 뒤쪽(가까운 위치)에 배치.
- 구조는 애매한 prompt 를 가르는 보조 신호 → 필드마커로 명시.
- history는 최대 12턴(≤6 action)으로 이미 제한 → v3/v4는 최근 우선 포함.
- 결측/빈 history/비문자 방어.

버전:
  v1: prompt only
  v2: + 마지막 action(name,args핵심,status) + result
  v3: + 최근 history(user/action 교대) + 핵심 meta 토큰
  v4: v3 + 수치 bin + args 확장자/패턴 + generator + open_files

train/inference 동일 함수 사용(제출 script.py에 인라인 복제).
"""
from __future__ import annotations
from . import parse as P


def _budget_bin(v):
    if v is None or v < 0: return "na"
    if v < 4000: return "vlow"
    if v < 16000: return "low"
    if v < 64000: return "mid"
    return "high"


def _loc_bin(v):
    if v is None or v < 0: return "na"
    if v < 2000: return "s"
    if v < 10000: return "m"
    if v < 40000: return "l"
    return "xl"


def _turn_bin(v):
    if v is None or v < 0: return "na"
    if v <= 1: return "1"
    if v <= 3: return "2-3"
    if v <= 7: return "4-7"
    return "8+"


def _fmt_action(t, with_args=False):
    nm = t.get("name", "")
    rs = (t.get("result_summary") or "")[:100]
    st = P.result_status(nm, rs)
    if with_args:
        a = P.arg_path_or_pattern(nm, t.get("args") or {})
        ext = P.path_ext(a) or P.glob_ext(a)
        extra = f"({ext})" if ext else ""
        return f"{nm}{extra}[{st}] {rs}"
    return f"{nm}[{st}] {rs}"


import re

_RE_GLOB = re.compile(r"[*?]|\[[^\]]+\]")
_RE_PATH = re.compile(r"(?:^|[\s'\"`(])(?:[\w.-]+/)+[\w.-]+|\b[\w-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|c|cpp|h|css|html|json|yaml|yml|toml|md|txt|sh|sql|ipynb|cfg|ini|lock)\b")
_RE_SYM = re.compile(r"`[^`]+`|'[A-Za-z_][\w.]*'|\"[A-Za-z_][\w.]*\"|\b[a-z]+[A-Z]\w*\b|\b\w+_\w+\b")
_RE_DIR = re.compile(r"디렉터리|디렉토리|폴더|구조|folder|directory|directories|tree|structure|파일\s*목록|list\s+files?", re.I)


def _prompt_flags(prompt: str, open_files) -> str:
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


def serialize(sample: dict, version: str = "v3", max_hist_turns: int = 8) -> str:
    prompt = sample.get("current_prompt")
    if not isinstance(prompt, str):
        prompt = "" if prompt is None else str(prompt)

    if version == "v1":
        return prompt

    nm, args, rs, st = P.last_action(sample)
    if version == "v2":
        la = ""
        if nm:
            a = P.arg_path_or_pattern(nm, args)
            ext = P.path_ext(a) or P.glob_ext(a)
            la = f" [LAST] {nm}{('('+ext+')') if ext else ''} [{st}] {rs[:100]}"
        return f"[CUR] {prompt}{la}"

    m = P.meta_fields(sample)
    parts = []
    # 메타 토큰
    parts.append(f"[TIER] {m['user_tier']} [LANG] {m['language_pref']} "
                 f"[TURN] {_turn_bin(m['turn_index'])} [CI] {m['last_ci_status']} "
                 f"[GIT] {'dirty' if m['git_dirty'] else 'clean'}")
    if version in ("v4", "v6"):
        parts.append(f"[GEN] {_gen(sample.get('id',''))}"
                     f" [BUDGET] {_budget_bin(m['budget'])} [LOC] {_loc_bin(m['loc'])} "
                     f"[TOPLANG] {m['top_lang']} [NOPEN] {m['n_open_files']}")
        if m["open_files"]:
            exts = sorted({P.path_ext(p) for p in m["open_files"] if P.path_ext(p)})
            if exts:
                parts.append(f"[OPENEXT] {','.join(exts)}")

    full_hist = sample.get("history", []) or []
    # 최근 history (오래된 것 truncate, 최근 우선)
    hist = full_hist[-max_hist_turns:]
    if hist:
        parts.append("[HIST]")
        for t in hist:
            if t.get("role") == "user":
                parts.append(f"u: {(t.get('content') or '')[:150]}")
            elif t.get("role") == "assistant_action":
                parts.append(f"a: {_fmt_action(t, with_args=(version in ('v4', 'v6')))}")
    if version == "v6":
        # 좌측절단 생존 위치(끝쪽): 압축 액션 트레일 + 세션시작 마커 + 프롬프트 패턴 플래그
        seq = [t.get("name", "") for t in full_hist if t.get("role") == "assistant_action"]
        parts.append(f"[SEQ] {'>'.join(seq[-12:]) if seq else 'none'}")
        if not full_hist:
            parts.append("[NOHIST]")
        parts.append(f"[PFLAG] {_prompt_flags(prompt, m['open_files'])}")
    # 현재 발화(지배적 신호) — 마지막(가까운 위치)에
    parts.append(f"[CUR] {prompt}")
    return " ".join(parts)


# generator를 io_utils가 아닌 parse에서도 쓰기 위한 얇은 래퍼 (v4)
def _gen(sample_id: str) -> str:
    return "au" if str(sample_id).startswith("sess_au_") else "sim"
