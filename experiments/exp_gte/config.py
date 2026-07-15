"""
중앙 설정 파일.

내일 데이터를 받으면 여기서 컬럼명/모델명/직렬화 옵션만 바꾸면 됩니다.
대부분의 "데이터 의존적" 값은 PLACEHOLDER 주석으로 표시해 두었습니다.
"""
from dataclasses import dataclass, field
from typing import List


# =============================================================================
# 1) 모델 선택 — 여기 한 줄만 바꾸면 모델 교체됨
# =============================================================================
# ⚠️ 대회 평가서버 transformers==4.46.3 에서 ModernBERT/Qwen3는 "NOT AVAILABLE"(업그레이드 필요).
# EDA 결과: current_prompt 64% 한국어(ko) → 다국어/한국어 모델 필수.
#
# 기본(경량·서버호환·다국어) -> "microsoft/mdeberta-v3-base"
# 대안                       -> "FacebookAI/xlm-roberta-base"
# 업그레이드(한국어+코드 최강) -> "Qwen/Qwen2.5-1.5B" (Qwen2 arch, 4.46.3 OK, LoRA 권장)
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

# history 평균 6.9턴. current_prompt + 최근 history 를 512 안에 담음.
MAX_LENGTH = 512


# =============================================================================
# 2) 데이터 스키마 — 내일 실제 컬럼명으로 교체 (PLACEHOLDER)
# =============================================================================
@dataclass
class Columns:
    id: str = "id"                       # 샘플 식별자
    label: str = "action"                # 정답 클래스 (train_labels.csv 의 'action')
    current_prompt: str = "current_prompt"
    history: str = "history"             # list[dict]: user{role,content} / assistant_action{role,name,args,result_summary}
    session_meta: str = "session_meta"   # nested dict (user_tier, language_pref, workspace{...}, ...)


COLS = Columns()

# 제출 파일 컬럼명 (sample_submission.csv 와 일치)
SUBMISSION_ID_COL = "id"
SUBMISSION_PRED_COL = "action"

# 14개 클래스 (고정 순서). 대소문자까지 정확히 일치해야 함.
CLASS_NAMES: List[str] = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

# 계층 분류용 coarse 그룹 (14 -> 4). 순서 = coarse id.
COARSE_GROUPS = [
    ("inspect",  ["read_file", "grep_search", "list_directory", "glob_pattern"]),
    ("modify",   ["edit_file", "write_file", "apply_patch"]),
    ("execute",  ["run_bash", "run_tests", "lint_or_typecheck"]),
    ("converse", ["ask_user", "plan_task", "web_search", "respond_only"]),
]


# =============================================================================
# 3) 직렬화 마커 — current_prompt + history + meta 를 한 문자열로
# =============================================================================
@dataclass
class SerializeCfg:
    meta_marker: str = "[META]"
    history_marker: str = "[HISTORY]"
    prompt_marker: str = "[PROMPT]"
    # history에서 최근 몇 스텝만 사용할지 (오래된 건 잘라냄). None이면 전부.
    max_history_steps: int = 12
    # --- 모듈식 입력 실험 플래그 (각각 개별 ablation. 기본 전부 off = 압축형 유지) ---
    add_symbol_cues: bool = False   # CUE에 판별 피처: has_path/symbol_query/dir_query/glob
                                    #   -> read↔grep↔list↔glob 경계용 (lexical, 짧은 입력에도 유효)
    add_open_files: bool = False    # meta에 open_files 경로 (상태의존 pair: cold-start read↔list)
    add_langmix: bool = False       # meta에 language_mix 상세
    max_open_files: int = 8


SER = SerializeCfg()


# =============================================================================
# 4) 학습 하이퍼파라미터
# =============================================================================
@dataclass
class TrainCfg:
    output_dir: str = "model"            # 제출 zip의 model/ 디렉토리로 그대로 사용
    epochs: int = 4
    train_batch_size: int = 16
    eval_batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    fp16: bool = True                    # T4에서 학습/추론 모두 fp16
    seed: int = 42
    val_size: float = 0.1                # stratified hold-out 비율

    # 클래스 불균형 대응: "weighted_ce" | "focal" | "none"
    loss_type: str = "weighted_ce"
    focal_gamma: float = 2.0

    # 검증셋에서 클래스별 threshold를 Macro-F1 기준으로 최적화해 저장할지
    tune_thresholds: bool = True


TRAIN = TrainCfg()
