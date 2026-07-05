# EDA Report — 코딩 에이전트 다음 행동 예측 (Dacon 236694)

> `open.zip` 제공 데이터(train.jsonl 70,000건 / train_labels.csv)를 직접 분석한 결과.
> 목적: 모델·피처·검증 설계에 필요한 사실을 재현 가능한 형태로 정리. 모든 수치는 train 전수(70,000) 기준.
> 동봉: `transition_matrix.csv` (직전 action → 다음 action 전이확률 전체).

---

## 0. 한눈에 (TL;DR)

- **BC(behavioral cloning) 문제**: 세션 궤적에서 잘라낸 `(state, next_action)` 쌍. 70,000 샘플이 **9,429개 세션**에서 나옴(세션당 평균 7.4스텝).
- **CV는 세션 단위 GroupKFold 필수** — 샘플이 i.i.d.가 아님(같은 세션 스텝끼리 상관).
- **신호 위계**: `current_prompt`(주신호) > `history` 직전 action·전이(phase 신호) > `session_meta`(약한 booster, 단 turn_index는 유효).
- **불균형 8.8배 × Macro-F1** → 소수 클래스(web_search 1.8%, write_file 2.1%, lint 3.3%)가 승부처.
- **혼동 위험쌍**: glob↔grep, edit↔write↔apply_patch, run_tests↔lint.

---

## 1. 데이터 개요

| 항목 | 값 |
|---|---|
| train 샘플 | 70,000 |
| 고유 세션 수 | 9,429 |
| 세션당 스텝 | mean 7.42 / median 7 / max 18 |
| history 길이 | mean 6.93 / median 8 / max 12 (0~12) |
| history 빈 샘플(세션 시작) | 9,000 (12.9%) |
| 실제 평가셋(비공개) | 30,000 |
| 클래스 수 | 14 (대소문자까지 정확 일치 필요) |

**함의 — GroupKFold**: `id = sess_..._-step_NN` 구조상 한 세션이 여러 샘플로 쪼개진다. 일반 KFold는 같은 세션을 train/valid에 분산시켜 **검증 점수를 낙관적으로 왜곡**한다. 반드시 `session_id`(= id에서 `-step` 앞부분) 기준 **GroupKFold**로 분할할 것. 평가셋 30,000건도 별개 세션이므로 이 방식이 리더보드와 정합적.

---

## 2. 타깃(action) 분포

| action | count | 비율 | | action | count | 비율 |
|---|---|---|---|---|---|---|
| edit_file | 11,171 | 15.96% | | list_directory | 4,329 | 6.18% |
| grep_search | 9,912 | 14.16% | | ask_user | 2,701 | 3.86% |
| read_file | 9,257 | 13.22% | | plan_task | 2,679 | 3.83% |
| glob_pattern | 5,284 | 7.55% | | lint_or_typecheck | 2,283 | 3.26% |
| respond_only | 5,178 | 7.40% | | write_file | 1,481 | 2.12% |
| run_bash | 5,068 | 7.24% | | **web_search** | **1,273** | **1.82%** |
| apply_patch | 4,823 | 6.89% | | | | |
| run_tests | 4,561 | 6.52% | | | | |

- **불균형비 8.8배** (edit_file / web_search). Macro-F1이라 소수 3종(web_search, write_file, lint_or_typecheck)의 F1이 총점에 과대 기여.
- 다수 클래스 baseline(전부 edit_file 예측) 시 accuracy 상한 ≈ 16%, Macro-F1은 매우 낮음.

---

## 3. 세션 궤적 구조 & 국면(phase) 신호

### 3.1 history 직전 action → 다음 action 전이

"직전 assistant_action의 최빈 다음 action"으로만 예측 시 **top-1 ≈ 23.3%** (랜덤 7%, 다수클래스 16% 대비 유의미하나 불충분). 주요 전이(사람 코딩 워크플로우와 일치):

| 직전 action | → 최빈 다음 action | 확률 |
|---|---|---|
| write_file | edit_file | **40.4%** (최강) |
| read_file | edit_file | 29.0% |
| list_directory | read_file | 25.1% |
| lint_or_typecheck | apply_patch | 24.9% |
| run_tests | edit_file | 24.5% |
| edit_file | run_tests | 23.0% |
| glob_pattern | grep_search | 22.2% |
| grep_search | edit_file | 21.9% |
| run_bash | run_bash | 20.4% (연쇄) |
| ask_user | grep_search | 20.0% |
| plan_task | apply_patch | 18.0% |
| apply_patch | lint_or_typecheck | 17.1% |
| web_search | edit_file | 19.0% |

→ 루프 구조: **탐색(list→read / glob→grep) → 수정(edit/apply_patch) → 검증(run_tests/lint) → 재수정**. 직전 action은 강한 phase 사전확률. (전이 전체표는 `transition_matrix.csv`.)

### 3.2 turn_index로 본 국면 순서 (매우 유효한 신호)

클래스별 평균 turn_index (작을수록 세션 초반):

```
초반 ─ write_file(2.0) < list_directory(2.8) < plan_task(3.4) < run_bash(4.2)
      < ask_user(4.4) < read_file(4.5) < edit_file(5.2) < grep_search(5.3)
      < glob_pattern(5.7) < web_search(6.0) < run_tests(6.4)
      < lint_or_typecheck(7.2) < respond_only(7.3) < apply_patch(7.5) ─ 후반
```

- **초반**: 파일 생성·디렉터리 탐색·계획 수립(write_file/list_directory/plan_task).
- **후반**: 검증·마무리·응답(run_tests/lint/respond_only), 그리고 apply_patch(모아서 패치 적용).
- → `turn_index`는 session_meta 중 **가장 예측력 있는 수치 피처**. 구간화 or 원시값 둘 다 실험.

### 3.3 history 빈 샘플(세션 시작, 12.9%)의 label

직전 action 신호가 없는 세션 첫 스텝에서는 탐색·계획이 지배:
list_directory 20.2%, read_file 16.5%, plan_task 12.5%, grep_search 11.6%, run_bash 11.2%, write_file 7.9%.
→ 이 12.9%는 `current_prompt` + meta로만 풀어야 함. 별도 처리(예: history 없음 플래그) 권장.

---

## 4. session_meta 피처 분석

### 4.1 범주형 분포
- `user_tier`: pro 53.9% / free 29.9% / enterprise 16.2%
- `language_pref`: **ko 64.3%** / en 25.4% / mixed 10.2% → **한국어가 지배적**. 다국어/한국어 인코더 유리.
- `last_ci_status`: passed 40.1% / failed 32.3% / none 27.6%
- `git_dirty`: True 76.7% / False 23.3% (대부분 미커밋 변경 존재)

### 4.2 수치 피처 요약
| 피처 | min | median | mean | max |
|---|---|---|---|---|
| budget_tokens_remaining | 55 | 94,450 | 92,439 | 199,741 |
| turn_index | 1 | 5 | 5 | 18 |
| elapsed_session_sec | 30 | 498 | 511 | 1,530 |
| loc | 206 | 18,059 | 22,702 | 90,000 |
| open_files (개수) | 0 | 1 | 1 | 5 |
| current_prompt 길이(char) | 2 | 56 | 61 | 346 |

### 4.3 meta의 marginal 신호는 대체로 약함 (정직한 평가)
- `last_ci_status=failed` → edit_file 18% vs passed 15%: 수정으로 기우는 경향 있으나 약함.
- `budget < 20k` → respond_only 7.0%→11.8%, ask_user 3.8%→4.7%: **예산 압박 시 값싼 행동**으로. 방향 맞지만 효과작고 해당 구간 5k건.
- → 단변량으로는 약하고, `current_prompt`·`turn_index`와의 **상호작용**에서 값이 나옴. 트리 모델(교호작용 자동 포착) 또는 fusion 헤드에 넣어 활용.

---

## 5. 텍스트(current_prompt) 특성

- 길이: median 56자, 최대 346자로 **짧음**(한 문장 발화 수준). → 인코더 max_length 128 내외로 충분, 추론 속도 유리.
- **클래스별 길이 차가 신호**: 대화·계획성(plan_task 88 / ask_user 86 / web_search 85자)은 길고, 실행성(run_bash 39 / lint 40 / run_tests 40 / respond_only 43자)은 짧음. → 프롬프트 길이 자체가 보조 피처.
- 언어: ko 지배 + en/mixed 혼재 → 표면형 TF-IDF(=baseline)는 한계. subword 다국어 인코더 권장.

---

## 6. 혼동 위험쌍 (오분류 집중 예상)

의미가 겹쳐 모델이 헷갈릴 구간. 학습 중 이 쌍들의 confusion을 별도 모니터링.

- **glob_pattern ↔ grep_search**: 파일 이름 검색 vs 파일 내용 검색.
- **edit_file ↔ write_file ↔ apply_patch**: 기존 수정 vs 신규 작성 vs diff 적용.
- **run_tests ↔ lint_or_typecheck**: 실행 검증 vs 정적 검증.
- **respond_only ↔ ask_user**: 응답 종료 vs 되물음(둘 다 도구 미사용, 짧은 프롬프트).

---

## 7. 모델링 시사점 (설계 체크리스트)

1. **검증**: `session_id` GroupKFold. ✅ 완료 — `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`, 상혁 v01과 정렬. 상세: `notebooks/05_cv_setup.md`.
2. **입력 설계 (fusion)**:
   - 텍스트: `current_prompt` (+ 직전 `result_summary` / 최근 user content) → 다국어 인코더 임베딩.
   - 시퀀스: 직전 action(및 최근 k개) 임베딩, action n-gram.
   - 수치/범주: turn_index, budget, elapsed, loc, open_files 수, tier, language_pref, ci_status, git_dirty, prompt_len.
   - `history_empty` 플래그.
3. **불균형**: class-weighted CE 또는 focal loss; 소수 클래스(web_search/write_file/lint) 오버샘플 실험; 추론 시 클래스별 threshold/logit 보정.
4. **경량·속도(본선 10%)**: 프롬프트 짧으니 base~small 인코더 + fp16로 충분. 무거운 앙상블은 속도 손해.
5. **baseline 대비 ablation**: (a) current_prompt only(=baseline) → (b) +직전 action → (c) +turn_index/meta → (d) +focal/weight. 각 단계 Macro-F1 기록.

---

## 8. 추가 관측 팩트 (2026-07-02 검증)

### 8.1 id prefix 두 종류

train.jsonl과 train_labels.csv의 id 문자열을 정규식으로 파싱한 결과:

| prefix | 샘플 수 | 세션 수 | 세션당 평균 스텝 | id 형식 |
|---|---:|---:|---:|---|
| `sess_sim_` | 64,975 (92.8%) | 8,330 | 7.80 | `sess_sim_YYYYMMDD_XXXXXX-step_NN` (8자리 날짜 + 6자리 번호) |
| `sess_au_` | 5,025 (7.2%) | 1,099 | 4.57 | `sess_au_XXXXXX_XXX-step_NN` (날짜 없음, 6자리 + 3자리) |

두 prefix의 라벨 분포 차이 (전체 클래스 중 큰 격차만):

| 클래스 | sess_sim % | sess_au % | 비율(au/sim) |
|---|---:|---:|---:|
| read_file | 12.26 | 25.69 | 2.10× |
| glob_pattern | 8.00 | 1.77 | 0.22× |
| list_directory | 6.49 | 2.17 | 0.33× |
| ask_user | 4.01 | 1.87 | 0.47× |
| lint_or_typecheck | 3.08 | 5.59 | 1.81× |
| apply_patch | 7.13 | 3.80 | 0.53× |

두 prefix의 어휘 차이 (sim/au 각 5,000건 샘플 TF-IDF log-odds 상위):

- `sess_au`에서 상대적으로 많이 등장: "타입체크", "여기까지 하자", "고마워", "요약", "typecheck", "됐다", "깔끔하네"
- `sess_sim`에서 상대적으로 많이 등장: "가능하면", "잠시만", "여기부터", "혹시나 해서", "when you", "you can"

두 prefix의 session_meta 필드 차이:

| 필드 | sess_sim | sess_au |
|---|---|---|
| user_tier enterprise | 14.7% | 35.0% |
| git_dirty True | 78.8% | 49.8% |
| last_ci_status passed | 38.9% | 55.4% |
| elapsed_session_sec (평균) | 526.7 | 307.6 |
| history 아이템 수 (평균) | 7.08 | 5.04 |

**prefix 의미(sim/au가 무엇을 뜻하는지)는 대회 docs·팀 카톡·상혁 코드 어디에도 정의 없음. 미확인.**

### 8.2 history truncation 규칙 (실측)

전체 70,000 샘플의 history 길이를 스텝별로 세어 확인:

| 샘플 스텝 | history 길이 | 담긴 이전 스텝 |
|---|---|---|
| 1 | 0개 | 없음 (첫 턴) |
| 2 | 2개 | 스텝 1 |
| 3 | 4개 | 스텝 1, 2 |
| 4 | 6개 | 스텝 1, 2, 3 |
| 5 | 8개 | 스텝 1~4 |
| 6 | 10개 | 스텝 1~5 |
| 7 | 12개 | 스텝 1~6 |
| **8** | **12개 (잘림)** | **스텝 2~7** (스텝 1 잘림) |
| 9 | 12개 | 스텝 3~8 |
| 14 | 12개 | 스텝 8~13 |
| 18 | 12개 | 스텝 12~17 |

즉 스텝 N의 history 길이 = `min(2*(N-1), 12)`. 스텝 8 이상에서는 최근 6개 스텝의 (user, action) 쌍만 유지, 오래된 것은 잘려나감. 스텝별 unique 길이 확인 결과 편차 없음(예: 스텝 8인 4,222개 샘플 모두 history 길이 12).

### 8.3 라벨과 history 사이 무결성 (100%)

`train_labels.csv`의 스텝 K 라벨 vs 같은 세션 스텝 M(M>K)의 history 안 스텝 K 자리에 있는 tool name 비교:

- 전체 세션 9,429개, 70,000 샘플 다 훑음
- 총 231,664 (스텝K, 스텝M) 쌍 비교
- **일치: 231,664 (100.0000%), 불일치: 0**

### 8.4 세션 스텝 갭

train.jsonl에도 train_labels.csv에도 없는 중간 스텝 존재:

- 9,429 세션 중 1,679 세션(17.81%)에 갭 있음
- 갭 스텝 총 3,205개 (max_step까지 봤을 때 빠진 자리 합)
- 갭 있는 세션 중 단일 갭 1,035개, 다중 갭 644개
- prefix별 갭 비율: `sess_sim` 18.70%, `sess_au` 11.01%
- 갭 스텝의 원본 정보(사용자 발화, 행동)는 다른 스텝의 history에서 회수 가능: 3,181/3,205 = 99.3%
- 회수된 갭 스텝의 라벨 분포: 대부분 전체 분포 대비 lift 0.9~1.2 (편향 미미). 예외 하나: `respond_only`는 갭에 0건 (전체 7.4% 대비)
- 갭 위치 base rate 정규화 결과: 스텝 1~10 정규화된 갭 비율 4.0~4.9%, 스텝 11+ 3.16~3.5% (거의 균등)

### 8.5 데이터 leak (같은 세션 안)

- 세션 안 스텝 K의 라벨을 그 세션의 스텝 M(M>K)의 history에서 조회 가능 (§8.3 결과의 응용)
- 로컬 test.jsonl 5샘플 = train.jsonl과 train_labels.csv에 동일 id로 존재 (완전 중복)
- 로컬 5샘플 lookup 시도 5/5 성공, sample_submission.csv 정답값과 100% 일치
- train에서 스텝 하나 제외하고 lookup 시도한 로컬 시뮬: 972 시도 중 793 (81.6%) 성공, 성공한 것 100% 정답

### 8.6 위 발견 관련 해석·활용은 미검증 (별도 격리)

위 §8.1~8.5는 관측 팩트. 다음 항목들은 **미검증 가설·활용 후보**:

- prefix (`sim`/`au`)가 무엇을 뜻하는지는 미확인
- 갭 스텝이 서버 test 30,000건에 있는지 여부 미확인
- 서버 30k에서 lookup 트릭이 통하는지 여부 미확인 (1회 제출로 확인 가능)

---

## 9. 재현 방법

```python
import json, pandas as pd, numpy as np
lab = pd.read_csv("data/train_labels.csv").set_index("id")["action"].to_dict()
rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
# session_id = id.rsplit("-step",1)[0]  → GroupKFold 그룹키
# 위 섹션들의 집계는 rows를 1-pass 순회하며 Counter로 산출
```
