# 데이터 설명

> 출처: https://dacon.io/competitions/official/236694/data

---

## 배포 파일 (`open.zip`)

| 파일명 | 설명 |
|--------|------|
| `baseline_submit.zip` | 베이스라인 코드 기반 제출 파일 예시 |
| `train.jsonl` | 학습 입력 데이터 |
| `train_labels.csv` | 학습 정답 데이터 |
| `test.jsonl` | 평가 입력 데이터 샘플 |
| `sample_submission.csv` | 제출 양식 |

---

## 데이터 크기

- **학습 데이터**: 70,000건
- **평가 데이터**: 30,000건 (비공개, 실제 서버에서 처리)
- **테스트 샘플**: 5건 (형식 확인용)

---

## 데이터 구조

### `train.jsonl` / `test.jsonl` (JSON Lines 형식)

각 줄은 에이전트 세션의 특정 시점을 나타내며 다음 항목을 포함한다:

| 필드 | 설명 |
|------|------|
| `id` | 샘플 식별자 |
| `session_meta` | 세션 메타정보 (요금제, 언어 선호도, 토큰 예산, 턴 번호, 경과 시간) |
| `workspace` | 작업공간 상태 (언어 비율, 코드 라인 수, git 상태, 열린 파일, CI 상태) |
| `history` | 이전 대화·행동 기록 |
| `current_prompt` | 현재 사용자 발화 |

### `train_labels.csv`

| 컬럼 | 설명 |
|------|------|
| `id` | 샘플 식별자 |
| `action` | 14개 클래스 중 하나 |

---

## 14개 클래스 (타깃 변수)

### 파일/검색 (4개)
- `read_file`
- `grep_search`
- `list_directory`
- `glob_pattern`

### 파일 수정 (3개)
- `edit_file`
- `write_file`
- `apply_patch`

### 실행 (3개)
- `run_bash`
- `run_tests`
- `lint_or_typecheck`

### 상호작용 (4개)
- `ask_user`
- `plan_task`
- `web_search`
- `respond_only`

---

## 제출 양식 (`sample_submission.csv`)

| 컬럼 | 설명 |
|------|------|
| `id` | 평가 데이터 식별자 |
| `action` | 예측값 (14개 클래스명 중 정확히 일치해야 함) |

---

## 평가지표

- **Macro-F1** (14개 클래스에 대한)
