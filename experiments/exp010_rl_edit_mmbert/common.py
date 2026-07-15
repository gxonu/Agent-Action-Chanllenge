"""
학습/추론이 공유하는 핵심 로직: 입력 직렬화 + 라벨 인코딩.

주의: 추론 코드(script.py)는 제출 zip 구조 안정성을 위해 *standalone*으로 만들었고,
serialize() 함수를 그대로 복사해 둡니다. 이 파일과 script.py의 serialize()는
반드시 동일하게 유지하세요 (KEEP IN SYNC). 동작은 model/serialize_config.json 으로
잠그기 때문에, 함수 시그니처만 같으면 결과는 항상 일치합니다.
"""
import json
import re
from typing import Any, Dict, List

# 탐색 클러스터 판별용 lexical 패턴 (KEEP IN SYNC with script.py)
_PATH_RE = re.compile(r"[\w./-]+\.(py|ts|js|tsx|jsx|java|go|rs|sql|ya?ml|json|md|toml|cfg|ini|sh|rb|php|cpp?|h|hpp)\b", re.I)
_SYM_RE = re.compile(r"어디|정의|찾아|찾을|참조|걸린|import|where|defined|declared|uses?|reference|grep|검색", re.I)
_DIR_RE = re.compile(r"목록|뭐\s*있|어떤\s*파일|리스트|디렉|폴더|구조|ls\b|list|안에\s*뭐|들어\s*?있", re.I)
_GLOB_RE = re.compile(r"\*\.|모든\s|전부|테스트\s*파일|\.\w+\s*파일|패턴|매칭|glob|all\s+\w+\s+files", re.I)


def _explore_cues(prompt: str) -> Dict[str, int]:
    p = prompt or ""
    return {
        "path": 1 if _PATH_RE.search(p) else 0,   # 명시 파일경로 -> read_file
        "sym":  1 if _SYM_RE.search(p) else 0,     # 심볼/개념 쿼리 -> grep_search
        "dir":  1 if _DIR_RE.search(p) else 0,     # 디렉토리 질의 -> list_directory
        "glob": 1 if _GLOB_RE.search(p) else 0,    # 패턴 -> glob_pattern
    }


# ---------------------------------------------------------------------------
# 입력 직렬화: current_prompt + history + session_meta -> 단일 문자열
# ---------------------------------------------------------------------------
def _fmt_args(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    return ",".join(f"{k}={v}" for k, v in args.items())


def get_history_pairs(history: Any, max_steps: int) -> List[str]:
    """_stringify_history 내부 로직을 그대로 쓰되, 최종 문자열이 아니라 시간순
    (오래된 -> 최신) pair 문자열 리스트를 반환한다. RL 입력-편집 모듈이 특정
    위치(step)의 pair 하나만 골라 다른 샘플의 것으로 바꿔치기하려면 이렇게
    쪼개진 리스트가 필요해서 분리했다. serialize() 결과는 이 함수를 거쳐도
    100% 동일하게 유지된다 (아래 _stringify_history 참조)."""
    if history is None:
        return []
    if isinstance(history, str):
        s = history.strip()
        return [s] if s else []
    if not isinstance(history, list):
        return [str(history)]

    def is_action(s):
        return isinstance(s, dict) and (s.get("role") == "assistant_action" or s.get("name"))

    # (user 요청 -> 그 결과 action) 을 한 덩어리로 묶음 = prompt->action 시연
    pairs: List[str] = []
    i, n = 0, len(history)
    while i < n:
        t = history[i]
        if not isinstance(t, dict):
            i += 1; continue
        if is_action(t):                      # user 없는 action
            a = t
            pairs.append(f"({a.get('name','')}({_fmt_args(a.get('args'))}): "
                         f"{(a.get('result_summary') or '').strip()})")
            i += 1
        else:                                 # user turn
            content = (t.get("content") or t.get("text") or "").strip()
            if i + 1 < n and is_action(history[i + 1]):   # 다음 action 과 묶기
                a = history[i + 1]
                pairs.append(f"(user: {content} -> {a.get('name','')}"
                             f"({_fmt_args(a.get('args'))}): {(a.get('result_summary') or '').strip()})")
                i += 2
            else:
                pairs.append(f"(user: {content})")
                i += 1

    if max_steps is not None and len(pairs) > max_steps:
        pairs = pairs[-max_steps:]           # 최근 쌍 우선 보존
    return pairs   # 시간순(오래된 -> 최신). reverse는 호출부에서.


def _stringify_history(history: Any, max_steps: int) -> str:
    """실제 포맷: user{role,content} / assistant_action{role,name,args,result_summary}."""
    pairs = get_history_pairs(history, max_steps)
    return " ".join(reversed(pairs))         # 최근->과거 (truncation 시 오래된 쌍부터)


def _stringify_meta(meta: Any, add_open_files: bool = False,
                    add_langmix: bool = False, max_open: int = 8) -> str:
    """nested session_meta 평탄화. 블록별 토글로 open_files/langmix 개별 추가."""
    if meta is None:
        return ""
    if not isinstance(meta, dict):
        return str(meta)
    ws = meta.get("workspace") or {}
    open_files = ws.get("open_files") or []
    lang_mix = ws.get("language_mix") or {}
    top_lang = max(lang_mix, key=lang_mix.get) if isinstance(lang_mix, dict) and lang_mix else "na"
    # 식별자(파일경로) 대신 강건한 플래그만 (경로는 과적합)
    has_test = 1 if any("test" in str(f).lower() for f in open_files) else 0
    has_cfg = 1 if any(str(f).endswith((".yml", ".yaml", ".toml", ".json", ".cfg", ".ini"))
                       for f in open_files) else 0
    parts = [
        f"tier={meta.get('user_tier')}",
        f"lang={meta.get('language_pref')}",
        f"turn={meta.get('turn_index')}",
        f"budget={meta.get('budget_tokens_remaining')}",
        f"ci={ws.get('last_ci_status')}",
        f"dirty={ws.get('git_dirty')}",
        f"loc={ws.get('loc')}",
        f"toplang={top_lang}",
        f"nopen={len(open_files)}",
        f"hastest={has_test}",
        f"hascfg={has_cfg}",
    ]
    if add_open_files:
        # 빈 상태(cold-start)도 명시 마커(none)로 -> 학습/추론 표현 일관 (absence 금지)
        val = ",".join(str(f) for f in open_files[:max_open]) if open_files else "none"
        parts.append(f"open_files={val}")
    if add_langmix and isinstance(lang_mix, dict) and lang_mix:
        parts.append("langmix=" + ",".join(f"{k}:{v}" for k, v in lang_mix.items()))
    return " ".join(parts)


def serialize(current_prompt: Any, history: Any, session_meta: Any, cfg: Dict) -> str:
    """
    cfg 예시 (model/serialize_config.json 으로 저장/로드):
      {"meta_marker":"[META]","history_marker":"[HISTORY]",
       "prompt_marker":"[PROMPT]","max_history_steps":12}
    """
    meta_s = _stringify_meta(session_meta, cfg.get("add_open_files", False),
                             cfg.get("add_langmix", False), cfg.get("max_open_files", 8))
    hist_s = _stringify_history(history, cfg.get("max_history_steps"))
    prompt_s = "" if current_prompt is None else str(current_prompt).strip()

    # 강건한 신호만 (이진 플래그). plen(정확한 길이)은 과적합이라 제거.
    q = 1 if prompt_s.endswith("?") else 0
    short = 1 if len(prompt_s) < 25 else 0
    cue = f"[CUE] q={q} short={short}"
    # 판별 피처(add_symbol_cues): 탐색 클러스터 경계용 lexical 신호
    if cfg.get("add_symbol_cues", False):
        c = _explore_cues(prompt_s)
        cue += f" path={c['path']} sym={c['sym']} dir={c['dir']} glob={c['glob']}"

    # 순서: 현재발화 -> cue -> 최근history -> meta.
    parts = [f"{cfg['prompt_marker']} {prompt_s}", cue]
    if hist_s:
        parts.append(f"{cfg['history_marker']} {hist_s}")
    if meta_s:
        parts.append(f"{cfg['meta_marker']} {meta_s}")
    return " ".join(parts)


def serialize_from_pairs(current_prompt: Any, pairs: List[str], session_meta: Any, cfg: Dict) -> str:
    """RL 입력-편집 모듈 전용. history를 다시 파싱하지 않고, 이미 만들어진(일부
    step이 다른 샘플 것으로 바꿔치기된) pairs 리스트를 그대로 조립해 직렬화한다.
    serialize()와 완전히 동일한 포맷/순서를 유지한다 (KEEP IN SYNC with serialize())."""
    meta_s = _stringify_meta(session_meta, cfg.get("add_open_files", False),
                             cfg.get("add_langmix", False), cfg.get("max_open_files", 8))
    hist_s = " ".join(reversed(pairs))
    prompt_s = "" if current_prompt is None else str(current_prompt).strip()

    q = 1 if prompt_s.endswith("?") else 0
    short = 1 if len(prompt_s) < 25 else 0
    cue = f"[CUE] q={q} short={short}"
    if cfg.get("add_symbol_cues", False):
        c = _explore_cues(prompt_s)
        cue += f" path={c['path']} sym={c['sym']} dir={c['dir']} glob={c['glob']}"

    parts = [f"{cfg['prompt_marker']} {prompt_s}", cue]
    if hist_s:
        parts.append(f"{cfg['history_marker']} {hist_s}")
    if meta_s:
        parts.append(f"{cfg['meta_marker']} {meta_s}")
    return " ".join(parts)


def serialize_config_from(SER) -> Dict:
    return {
        "meta_marker": SER.meta_marker,
        "history_marker": SER.history_marker,
        "prompt_marker": SER.prompt_marker,
        "max_history_steps": SER.max_history_steps,
        "add_symbol_cues": SER.add_symbol_cues,
        "add_open_files": SER.add_open_files,
        "add_langmix": SER.add_langmix,
        "max_open_files": SER.max_open_files,
    }


# ---------------------------------------------------------------------------
# 라벨 인코딩
# ---------------------------------------------------------------------------
def build_label_maps(labels: List[str], class_names: List[str] = None):
    """label2id, id2label 생성. class_names 주어지면 그 순서 고정, 아니면 정렬해서 사용."""
    if class_names:
        classes = list(class_names)
    else:
        classes = sorted(set(map(str, labels)))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    return label2id, id2label