"""샘플 파싱 유틸 — history/args/result_summary/meta에서 구조 신호 추출.

EDA·features(GBDT)·serialize(트랜스포머)가 공유하는 단일 진실 원천.
표준 라이브러리만 사용 (로컬 3.10 / Colab 3.11 / 서버 3.11 공용).
"""
from __future__ import annotations
import re

# ----------------------------- 텍스트/언어 -----------------------------
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")


def has_hangul(text: str) -> bool:
    return bool(_HANGUL.search(text or ""))


def lang_of(text: str) -> str:
    """텍스트 기반 언어 추정: ko / en / mixed / other."""
    t = text or ""
    ko = bool(_HANGUL.search(t))
    en = bool(_LATIN.search(t))
    if ko and en:
        return "mixed"
    if ko:
        return "ko"
    if en:
        return "en"
    return "other"


# ----------------------------- 숫자/상태 -----------------------------
_INT = re.compile(r"-?\d+")


def first_int(s: str, default: int = -1) -> int:
    m = _INT.search(s or "")
    return int(m.group()) if m else default


def all_ints(s: str) -> list[int]:
    return [int(x) for x in _INT.findall(s or "")]


# result_summary → 정규화된 상태 태그
# 관측 패턴: "ok; ...", "exit=0; ...", "found N occurrences", "N files matched",
#   "N results retrieved", "plan with N steps", "N entries", "empty directory",
#   "0 matches", "ERROR: ...", "... failed", "hunk ... did not apply",
#   "X passed", "X failed" (tests)
def result_status(name: str | None, summary: str | None) -> str:
    """다음 행동 예측에 강한 신호가 되는 결과 상태.

    반환: error / test_fail / test_pass / zero / nonzero_exit / success / empty / na
    """
    s = (summary or "").strip()
    if not s:
        return "na"
    low = s.lower()
    # 명시적 에러/실패
    if low.startswith("error") or "did not apply" in low or "command failed" in low:
        return "error"
    if re.search(r"\bfail(ed|ure)?\b", low):
        # 테스트 실패 구분
        if name == "run_tests" or "test" in low:
            return "test_fail"
        return "error"
    # 테스트 통과
    if name == "run_tests" and ("pass" in low or "ok" in low):
        return "test_pass"
    # exit code
    m = re.search(r"exit\s*=\s*(-?\d+)", low)
    if m:
        return "success" if m.group(1) == "0" else "nonzero_exit"
    # zero-result (검색/목록에서 0건)
    if re.search(r"\b0\s+(match|matches|file|files|result|results|occurrence|occurrences|entries|entry)\b", low) \
            or "empty directory" in low or low.startswith("no "):
        return "zero"
    # 그 외 정상 (ok;, found N, N files matched, plan with N, N entries, N results)
    if low.startswith("ok") or "found" in low or "matched" in low or "retrieved" in low \
            or "drafted" in low or "entries" in low or "wrote" in low or "applied" in low \
            or "patched" in low or "read" in low:
        return "success"
    return "success"  # 기본은 성공 취급(대부분 정상 요약)


# ----------------------------- 경로/확장자 -----------------------------
def path_ext(path: str) -> str:
    """파일 확장자(소문자, 점 제외). 특수 파일명 처리(Dockerfile 등)."""
    if not path:
        return ""
    base = path.rstrip("/").split("/")[-1]
    if "." in base:
        return base.rsplit(".", 1)[1].lower()
    # 확장자 없는 특수 파일
    known = {"dockerfile", "makefile", "readme", "license", "gemfile", "rakefile"}
    return base.lower() if base.lower() in known else ""


def glob_ext(pattern: str) -> str:
    """glob 패턴에서 확장자 추출: '**/*.py' -> 'py'."""
    if not pattern:
        return ""
    m = re.search(r"\*?\.?([A-Za-z0-9]+)$", pattern.strip())
    return m.group(1).lower() if m else ""


# ----------------------------- history 추출 -----------------------------
def action_turns(sample: dict) -> list[dict]:
    """history 중 assistant_action 턴만 (시간순)."""
    return [t for t in sample.get("history", []) if t.get("role") == "assistant_action"]


def action_sequence(sample: dict) -> list[str]:
    """이전 행동 이름 시퀀스 (시간순)."""
    return [t.get("name", "") for t in action_turns(sample)]


def last_action(sample: dict):
    """(name, args, result_summary, status). 없으면 (None, {}, '', 'na')."""
    acts = action_turns(sample)
    if not acts:
        return None, {}, "", "na"
    t = acts[-1]
    nm = t.get("name")
    args = t.get("args") or {}
    rs = t.get("result_summary") or ""
    return nm, args, rs, result_status(nm, rs)


def arg_path_or_pattern(name: str, args: dict) -> str:
    """행동별 대표 경로/패턴/텍스트 인자."""
    if not isinstance(args, dict):
        return ""
    for k in ("path", "pattern", "target", "cmd", "query", "goal", "question"):
        if k in args and args[k]:
            return str(args[k])
    return ""


# ----------------------------- session_meta -----------------------------
def meta_fields(sample: dict) -> dict:
    """session_meta + workspace 평탄화(결측 방어)."""
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    lm = ws.get("language_mix") or {}
    top_lang = max(lm.items(), key=lambda kv: kv[1])[0] if lm else ""
    return {
        "user_tier": sm.get("user_tier", ""),
        "language_pref": sm.get("language_pref", ""),
        "budget": sm.get("budget_tokens_remaining", -1),
        "turn_index": sm.get("turn_index", -1),
        "elapsed": sm.get("elapsed_session_sec", -1),
        "loc": ws.get("loc", -1),
        "git_dirty": bool(ws.get("git_dirty", False)),
        "last_ci_status": ws.get("last_ci_status", ""),
        "n_open_files": len(ws.get("open_files") or []),
        "open_files": ws.get("open_files") or [],
        "top_lang": top_lang,
        "language_mix": lm,
    }
