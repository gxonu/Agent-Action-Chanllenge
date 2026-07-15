"""
serialize A/B 변형 — 상혁 common.serialize()를 기준(control)으로 두고,
두 개의 실험 축을 옵션으로 켜고 끌 수 있게 확장.

축 A) extend_cues  : rare 3클래스(near-deterministic 어휘) CUE 플래그 추가
                     tc(lint/typecheck) · newf(write_file) · ws(web_search)
축 B) reorder_state: last_action / turn_bucket / nopen_bucket 를 [CUE](prompt 바로 옆)로
                     끌어올려 명시 categorical 토큰으로 노출 (지금은 META 맨끝/history 파묻힘)

설계 원칙
- control(=상혁 baseline)은 common.serialize와 **바이트 단위 동일** 하도록 common의 하위 헬퍼를 그대로 재사용.
  (self-test로 검증: python src/serialize_variants.py --selftest)
- 각 축은 독립 토글 → base / +cue / +reorder / +both 4조건 A/B 가능.
- 새 신호는 전부 soft(텍스트 플래그). hard rule/override 아님 — 모델이 학습으로 사용.
"""
import re
import sys
import os
from typing import Any, Dict, List

# 상혁 common.py 재사용 (동일 로직 보장)
_SANGHYUK = os.path.join(os.path.dirname(__file__), "..",
                         "team_ref", "sanghyuk_v01_lb0.77585")
sys.path.insert(0, os.path.abspath(_SANGHYUK))
import common  # _explore_cues, _stringify_history, _stringify_meta, _fmt_args


# ---------------------------------------------------------------------------
# 축 A — rare 3클래스 판별 큐 (EDA v3 §V5: 상대 클래스에 거의 안 나오는 어휘)
#   빈도는 심층 EDA distinctive-token 분석 기준. 정규식은 튜닝 대상.
# ---------------------------------------------------------------------------
_TC_RE = re.compile(
    r"타입\s*체크|타입체크|type\s*-?\s*check|typecheck|정적\s*분석|정적분석|"
    r"\blint\b|린트|mypy|\btsc\b|타입\s*(맞|안\s*맞|에러|오류)", re.I)
_NEWF_RE = re.compile(
    r"골격|스캐폴드|scaffold|scratch|from\s+scratch|처음부터|밑바닥|boilerplate|"
    r"overwrite|새\s*파일|새파일|빈\s*파일|만들어\s*줘요|모듈을\s*(만들|추가|생성)|"
    r"폴더에\s*.{0,6}(만들|생성|추가)|담을|담은", re.I)
_WS_RE = re.compile(
    r"웹\s*검색|검색해\s*줘|구글링|구글에|\bgoogle|최신\s*(버전|정보|문서|릴리스)|"
    r"공식\s*문서|공식\s*(사이트|가이드)|베스트\s*프랙티스|베스트프랙티스|best\s*practice|"
    r"changelog|릴리스\s*노트|release\s*note|stack\s*overflow|스택\s*오버플로", re.I)


def _rare_cues(prompt: str) -> Dict[str, int]:
    p = prompt or ""
    return {
        "tc":   1 if _TC_RE.search(p) else 0,     # -> lint_or_typecheck
        "newf": 1 if _NEWF_RE.search(p) else 0,   # -> write_file
        "ws":   1 if _WS_RE.search(p) else 0,     # -> web_search
    }


# ---------------------------------------------------------------------------
# 축 B — state를 CUE로 끌어올리기 위한 헬퍼
# ---------------------------------------------------------------------------
def _last_action(history: Any) -> str:
    if not isinstance(history, list):
        return "none"
    for t in reversed(history):
        if isinstance(t, dict) and (t.get("role") == "assistant_action" or t.get("name")):
            return t.get("name") or "none"
    return "none"


def _turn_bucket(meta: Any) -> str:
    turn = None
    if isinstance(meta, dict):
        turn = meta.get("turn_index")
    if turn is None:
        return "na"
    if turn <= 1:
        return "1"
    if turn <= 2:
        return "2"
    if turn <= 4:
        return "3-4"
    if turn <= 7:
        return "5-7"
    return "8+"


def _nopen_bucket(meta: Any) -> str:
    ws = (meta or {}).get("workspace") if isinstance(meta, dict) else None
    n = len((ws or {}).get("open_files") or []) if isinstance(ws, dict) else 0
    return str(n) if n <= 2 else "3+"


# ---------------------------------------------------------------------------
# 메인 — control과 동일 구조로 재구성 + 옵션 삽입
# ---------------------------------------------------------------------------
def serialize_variant(current_prompt: Any, history: Any, session_meta: Any,
                      cfg: Dict, extend_cues: bool = False,
                      reorder_state: bool = False) -> str:
    """
    cfg: 상혁 serialize_config (prompt_marker/history_marker/meta_marker/
         max_history_steps/add_symbol_cues/add_open_files/add_langmix/max_open_files)
    extend_cues=False, reorder_state=False  -> common.serialize와 바이트 동일 (control)
    """
    meta_s = common._stringify_meta(session_meta, cfg.get("add_open_files", False),
                                    cfg.get("add_langmix", False),
                                    cfg.get("max_open_files", 8))
    hist_s = common._stringify_history(history, cfg.get("max_history_steps"))
    prompt_s = "" if current_prompt is None else str(current_prompt).strip()

    q = 1 if prompt_s.endswith("?") else 0
    short = 1 if len(prompt_s) < 25 else 0
    cue = f"[CUE] q={q} short={short}"
    if cfg.get("add_symbol_cues", False):
        c = common._explore_cues(prompt_s)
        cue += f" path={c['path']} sym={c['sym']} dir={c['dir']} glob={c['glob']}"

    # --- 축 A: rare cue 플래그 (control에는 없음) ---
    if extend_cues:
        rc = _rare_cues(prompt_s)
        cue += f" tc={rc['tc']} newf={rc['newf']} ws={rc['ws']}"

    # --- 축 B: state를 CUE로 (prompt 바로 옆, 짧은 명시 토큰) ---
    if reorder_state:
        cue += (f" last={_last_action(history)}"
                f" turnb={_turn_bucket(session_meta)}"
                f" nopenb={_nopen_bucket(session_meta)}")

    parts = [f"{cfg['prompt_marker']} {prompt_s}", cue]
    if hist_s:
        parts.append(f"{cfg['history_marker']} {hist_s}")
    if meta_s:
        parts.append(f"{cfg['meta_marker']} {meta_s}")
    return " ".join(parts)


VARIANTS = {
    "base":     dict(extend_cues=False, reorder_state=False),   # = 상혁 baseline
    "cue":      dict(extend_cues=True,  reorder_state=False),   # 축 A만
    "reorder":  dict(extend_cues=False, reorder_state=True),    # 축 B만
    "both":     dict(extend_cues=True,  reorder_state=True),    # A+B
}


def _selftest():
    """control == common.serialize 바이트 동일 검증."""
    import json
    cfg = dict(prompt_marker="[PROMPT]", history_marker="[HISTORY]", meta_marker="[META]",
               max_history_steps=12, add_symbol_cues=True, add_open_files=True,
               add_langmix=False, max_open_files=8)
    path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "train.jsonl")
    n_ok = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3000:
                break
            r = json.loads(line)
            a = common.serialize(r["current_prompt"], r["history"], r["session_meta"], cfg)
            b = serialize_variant(r["current_prompt"], r["history"], r["session_meta"], cfg,
                                  extend_cues=False, reorder_state=False)
            assert a == b, f"MISMATCH at line {i}\nA:{a}\nB:{b}"
            n_ok += 1
    print(f"[selftest] control == common.serialize : {n_ok}/3000 OK (byte-identical)")
    # 변형 예시 출력
    with open(path, encoding="utf-8") as f:
        r = json.loads(f.readline())
    for name, fl in VARIANTS.items():
        s = serialize_variant(r["current_prompt"], r["history"], r["session_meta"], cfg, **fl)
        print(f"\n[{name}] {s[:220]}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
