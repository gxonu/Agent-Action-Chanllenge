# EDA #01 — 데이터셋 기초 분석

**작성**: 2026-07-02 (P3 초기)
**데이터**: `data/raw/train.jsonl` (70,000건) + `data/raw/train_labels.csv` (70,000건, id join 100% 매칭)
**목적**: 클래스 분포·필드별 신호·시퀀스 편향 파악. Rare class Macro-F1 개선 여지 스캔.

---

## 1. 클래스 분포 (전체)

| 클래스 | 건수 | 비중 |
|---|---:|---:|
| edit_file | 11,171 | 15.96% |
| grep_search | 9,912 | 14.16% |
| read_file | 9,257 | 13.22% |
| glob_pattern | 5,284 | 7.55% |
| respond_only | 5,178 | 7.40% |
| run_bash | 5,068 | 7.24% |
| apply_patch | 4,823 | 6.89% |
| run_tests | 4,561 | 6.52% |
| list_directory | 4,329 | 6.18% |
| ask_user | 2,701 | 3.86% |
| plan_task | 2,679 | 3.83% |
| **lint_or_typecheck** | 2,283 | **3.26%** |
| **write_file** | 1,481 | **2.12%** |
| **web_search** | 1,273 | **1.82%** |

- **imbalance ratio 8.8:1** (edit_file 15.96% vs web_search 1.82%)
- **Macro-F1이라 rare 3개(`web_search`, `write_file`, `lint_or_typecheck`)가 총점 결정에 3배 영향** (각 1/14 가중치)

---

## 2. Prefix 발견 (팩트 vs 가설 명확 분리)

### 팩트 (raw id에서 관측)

`id` 컬럼에 두 종류 prefix 실존:

| prefix | 샘플 | 세션 | 세션당 avg step |
|---|---:|---:|---:|
| `sess_sim_` | 64,975 | 8,330 | 7.80 |
| `sess_au_` | 5,025 | 1,099 | 4.57 |

포맷 차이:
- `sim`: `sess_sim_YYYYMMDD_XXXXXX-step_NN` (8자리 날짜 + 6자리 번호)
- `au`: `sess_au_XXXXXX_XXX-step_NN` (**날짜 없음**, 6자리 + 3자리)

### 가설 (추론, 미검증)

- `sim` = simulation, `au` = augmented/actual/authentic — **모두 추측**. 대회 docs·팀 카톡·상혁 코드 어디에도 정의 없음
- 통계 패턴 관찰 결과 `au`가 실제 사용자 로그 특성에 가까워 보이지만 미확정

### Prefix별 클래스 분포 격차 (팩트)

| class | sess_sim % | sess_au % | ratio (au/sim) |
|---|---:|---:|---:|
| edit_file | 15.84 | 17.55 | 1.11× |
| grep_search | 14.41 | 10.95 | 0.76× |
| **read_file** | 12.26 | **25.69** | **2.10×** |
| glob_pattern | 8.00 | 1.77 | **0.22×** |
| respond_only | 7.44 | 6.89 | 0.93× |
| run_bash | 7.04 | 9.87 | 1.40× |
| apply_patch | 7.13 | 3.80 | 0.53× |
| run_tests | 6.51 | 6.59 | 1.01× |
| list_directory | 6.49 | 2.17 | **0.33×** |
| ask_user | 4.01 | 1.87 | **0.47×** |
| plan_task | 3.86 | 3.36 | 0.87× |
| lint_or_typecheck | 3.08 | 5.59 | 1.81× |
| write_file | 2.16 | 1.55 | 0.72× |
| web_search | 1.78 | 2.35 | 1.32× |

**핵심 격차**:
- `au`는 `read_file` 2배, `lint_or_typecheck` 1.8배, `glob_pattern`/`list_directory`/`ask_user` 반토막~1/5

### 활용 아이디어

- **팩트 기반**: prefix를 categorical feature로 넣거나, `au`를 downweight/제외한 학습 실험
- **주의**: test 5샘플 모두 `sess_sim_`. 평가 서버 30k도 `sess_sim_`일 가능성 → **`au`가 학습에 노이즈일 수 있음**. 확인 필요.
- **팀 공유 시**: "통계 패턴은 사실, 의미는 모름" 프레이밍 유지

---

## 3. Step 순서 편향

### step별 총 샘플 & top 클래스

| step | n | top1 | top2 | top3 |
|---|---:|---|---|---|
| 1 | 9,000 | list_directory 20.2% | read_file 16.5% | plan_task 12.5% |
| 2 | 8,797 | read_file 20.4% | edit_file 18.0% | grep_search 14.7% |
| 3 | 8,446 | edit_file 24.3% | read_file 15.7% | grep_search 14.0% |
| 4 | 8,001 | edit_file 23.6% | grep_search 16.3% | read_file 12.5% |
| 5 | 7,526 | edit_file 19.0% | grep_search 15.1% | read_file 11.5% |
| 6 | 6,644 | edit_file 16.0% | grep_search 14.4% | read_file 11.0% |
| 7 | 5,435 | respond_only 14.6% | edit_file 14.0% | grep_search 13.5% |
| 8 | 4,222 | respond_only 14.5% | grep_search 14.3% | edit_file 12.5% |
| 9 | 3,240 | respond_only 14.0% | edit_file 13.8% | grep_search 13.0% |
| 10 | 2,488 | edit_file 14.1% | respond_only 13.5% | grep_search 13.4% |

**시퀀스 패턴**:
- **초기 (step 1~4)**: 탐색·수정 (list/read/edit/grep)
- **중반 (step 5~7)**: `edit_file` 지배 (19→14%)
- **후반 (step 7+)**: `respond_only` 14%+ 유지 → 세션 마무리 톤

**box: 박용욱 관찰 검증** — "step_01에 read_file 15%" → 실측 16.53%. 근데 **1위는 `list_directory` 20.2%**. read보다 list가 더 자연스러운 첫 액션 (탐색). 다만 첫 턴부터 `plan_task`(12.5%)/`run_bash`(11.2%)가 그렇게 많은 건 여전히 부자연스러움.

### prefix별 step 분포 (fact)

| step | sim % | au % |
|---:|---:|---:|
| 01~03 | 36 | **56** |
| 04~07 | 39 | 35 |
| 08+ | 25 | **8** |

- **`au`는 짧은 세션 지배적** (step_10+ 거의 없음). fact.
- 활용: `au`가 test와 다른 분포임을 뒷받침. train weighted split 실험 가치.

---

## 4. 상혁 "64% 한국어" 검증

### session_meta.language_pref (배포된 값)

- **ko 64.3%**, en 25.4%, mixed 10.2%
- → 상혁이 참조한 그 값. **정확히 매칭**.

### current_prompt 실제 텍스트 (heuristic 감지)

- **mixed 56.4%** (한글+영문 기술용어 동시), en 27.4%, ko 16.1%
- → 사용자가 선언한 선호(`language_pref`)와 실제 prompt 언어는 다름
- 실제 prompt는 한/영 혼재 지배적 → **다국어 인코더 필수 판단은 유효**. 상혁 mDeBERTa/xlm-roberta 라인 정당화됨

---

## 5. session_meta 필드 시그널

### user_tier
- pro 53.9% / free 29.9% / enterprise 16.2%
- 클래스 편향 분석은 후속 (여기까지는 분포만)

### workspace.last_ci_status × class top3 (조건부 편향)

| ci status | n | top3 |
|---|---:|---|
| passed | 28,035 | edit 15.2, grep 13.6, read 13.6 |
| **failed** | 22,623 | **edit 18.0**, grep 14.0, read 12.1 |
| none | 19,342 | grep 15.1, edit 14.7, read 13.9 |

- **CI failed 시 edit_file 18%** ← 실패 후 즉시 편집 대응. 조건부 신호.

---

## 6. turn_index / history 길이 vs class (강한 predictor)

### 클래스별 평균 turn_index (오름차순)

| class | mean turn | median | class 비중 |
|---|---:|---:|---:|
| **write_file** | **2.04** | 2 | 2.12% (rare) |
| list_directory | 2.83 | 2 | 6.18% |
| plan_task | 3.44 | 2 | 3.83% |
| run_bash | 4.19 | 3 | 7.24% |
| ask_user | 4.40 | 4 | 3.86% |
| read_file | 4.47 | 4 | 13.22% |
| edit_file | 5.23 | 4 | 15.96% |
| grep_search | 5.29 | 5 | 14.16% |
| glob_pattern | 5.71 | 5 | 7.55% |
| web_search | 6.02 | 6 | 1.82% (rare) |
| run_tests | 6.38 | 6 | 6.52% |
| **lint_or_typecheck** | **7.19** | 7 | 3.26% (rare) |
| respond_only | 7.26 | 7 | 7.40% |
| **apply_patch** | **7.45** | 7 | 6.89% |

**Rare class 특화 시사**:
- **`write_file`은 세션 초반 지배** (median turn 2) — 새 파일 생성은 셋업 액션
- **`lint_or_typecheck`은 후반** (median 7) — 마무리 검증
- **`apply_patch`도 후반** (median 7) — 여러 파일 편집은 정리 단계
- `web_search`는 중반 (turn 6) — 정보 부족 발생 시점

**모델링 액션**: `turn_index`를 categorical bucket(예: [0,1] / [2,4] / [5,7] / [8+])으로 넣으면 rare class 개선 여지. 상혁 코드는 flat 값으로 넣음 — 여기 개선 여지 있음.

### 클래스별 평균 history 길이

| class | mean hist | median |
|---|---:|---:|
| write_file | 2.02 | 2 |
| list_directory | 3.16 | 2 |
| plan_task | 4.02 | 2 |
| run_bash | 5.50 | 4 |
| ask_user | 5.62 | 6 |
| read_file | 5.81 | 6 |
| grep_search | 7.03 | 8 |
| edit_file | 7.15 | 6 |
| glob_pattern | 7.62 | 8 |
| web_search | 8.14 | 10 |
| run_tests | 8.66 | 10 |
| lint_or_typecheck | 9.58 | 12 |
| respond_only | 9.87 | 12 |
| apply_patch | 9.98 | 12 |

- `history` 길이는 `turn_index`와 강상관 → 정보 중복. 하나만 잘 써도 됨.

---

## 7. 라벨 노이즈 관찰 (정성 검증만, 정량화 pending)

- 무작위 20건 훑을 때 **명백 mismatch 4~5건** (약 20~25%)
- 대표 예: prompt "lint도 깔끔한지 간단히" → label `run_tests` (`lint_or_typecheck` 자연)
- 대표 예: prompt "단계 좀 짜줘" → label `web_search` (`plan_task` 자연)
- **팀도 동일 관찰**: 박용욱 "real data 아닌 것 같다", 상혁 "학습에 노이즈 많음. 프롬프트랑 안 맞는 애들이 많긴 해"
- 정량화는 후속 EDA (예: baseline TF-IDF로 self-predict → confidence 낮은 case 뽑기)

---

## 8. 즉시 활용 가능한 모델링 액션

1. **`turn_index` categorical bucket** — rare class 개선 여지. 상혁 코드에 추가 실험 가치
2. **`prefix` (sim/au) feature 또는 sample weight** — au sample downweight 실험. 팩트 기반, 의미 해석 미확정 상태로도 유효
3. **`last_ci_status = failed` → edit_file 편향** — 조건부 rule 앙상블 재료
4. **다국어 인코더 필수** (mDeBERTa/xlm-roberta/bge-m3) — 상혁 판단 뒷받침. Prompt는 mixed 지배
5. **Label smoothing / focal loss / KD** — 노이즈 강건 학습 (상혁이 이미 반영). 사용자는 pseudo-labeling/Cleanlab 같은 다른 축 시도 여지

---

## 9. History 전이 매트릭스 (EDA-A 결과)

### 9.1 History 없는 경우 (n=9,000 = 12.9%)

step_01 분포와 거의 동일:
- `list_directory` 20.20%, `read_file` 16.53%, `plan_task` 12.46%, `grep_search` 11.62%, `run_bash` 11.23%, `write_file` 7.87%
- `apply_patch`는 **0.07% (6건)** — 후반 액션 확정 (`history=[]` 상황에서는 사실상 안 나옴)

### 9.2 강한 전이 (lift = P(next|last)/P(next) 상위 20)

| last_action | next_action | P(next\|last) | lift |
|---|---|---:|---:|
| list_directory | **write_file** | 8.71% | **6.88×** |
| ask_user | plan_task | 13.00% | 5.09× |
| apply_patch | **lint_or_typecheck** | 17.14% | **4.73×** |
| web_search | web_search | 8.92% | 4.68× |
| write_file | run_bash | 28.70% | 4.32× |
| plan_task | list_directory | 16.60% | 4.03× |
| edit_file | run_tests | 23.01% | 3.16× |
| lint_or_typecheck | apply_patch | 24.90% | 3.15× |
| run_bash | run_bash | 20.39% | 3.07× |
| plan_task | plan_task | 6.85% | 2.68× |
| ask_user | list_directory | 10.90% | 2.65× |
| lint_or_typecheck | lint_or_typecheck | 8.78% | 2.42× |
| plan_task | apply_patch | 18.03% | 2.28× |
| glob_pattern | glob_pattern | 17.56% | 2.27× |
| write_file | edit_file | 40.39% | 2.25× |
| run_tests | respond_only | 18.21% | 2.18× |
| list_directory | list_directory | 8.43% | 2.05× |
| apply_patch | run_tests | 14.83% | 2.04× |
| web_search | ask_user | 6.82% | 2.01× |
| apply_patch | apply_patch | 15.60% | 1.98× |

### 9.3 last_action → top1 argmax 정확도

- **Naive Markov baseline (last_action → argmax next) top1 accuracy: 22.94%**
- Majority class baseline (edit_file 15.96%) 대비 **+6.98%p**
- **last_action이 강한 단일 predictor** — 상혁 코드는 문자열 내 embed. 별도 categorical feature 추가 시 boost 여지

### 9.4 last_action → top1 클래스 요약

| last_action | n | top1 | top2 | top3 |
|---|---:|---|---|---|
| read_file | 8,887 | edit_file 29.0% | grep_search 15.0% | read_file 14.0% |
| grep_search | 9,412 | edit_file 21.9% | read_file 19.3% | grep_search 18.3% |
| list_directory | 4,223 | read_file 25.1% | grep_search 21.5% | glob_pattern 10.5% |
| glob_pattern | 4,967 | grep_search 22.2% | glob_pattern 17.6% | read_file 15.5% |
| edit_file | 10,620 | run_tests 23.0% | edit_file 15.4% | apply_patch 10.1% |
| write_file | 1,446 | edit_file 40.4% | run_bash 28.7% | read_file 7.3% |
| apply_patch | 4,417 | **lint_or_typecheck 17.1%** | respond_only 16.1% | apply_patch 15.6% |
| run_bash | 4,797 | run_bash 20.4% | edit_file 18.3% | read_file 11.5% |
| run_tests | 4,251 | edit_file 24.5% | respond_only 18.2% | grep_search 12.9% |
| lint_or_typecheck | 2,016 | apply_patch 24.9% | edit_file 21.6% | respond_only 9.6% |
| ask_user | 2,192 | grep_search 20.0% | read_file 15.3% | edit_file 14.1% |
| plan_task | 2,584 | apply_patch 18.0% | read_file 16.9% | list_directory 16.6% |
| web_search | 1,188 | edit_file 19.0% | grep_search 15.7% | respond_only 12.2% |

### 9.5 Rare class 커버리지 (전이 rule로 잡히는 비율)

**`write_file`** (rare 1,481건):
- `history=[]` → 708건 (rare 48%)
- `list_directory → write_file` (8.71%): 4,223 × 8.71% ≈ 368건 (rare 25%)
- **두 조건으로 rare의 73%** 커버 → rule-based booster 강력

**`lint_or_typecheck`** (rare 2,283건):
- `apply_patch → lint`: 4,417 × 17.14% ≈ 757건 (rare 33%)
- `edit_file → lint`: 10,620 × 5.7% ≈ ~600건 (rare 26%)
- **두 조건으로 rare의 60%+** 커버

### 9.6 활용 아이디어 (사용자 담당 방향)

1. **`last_action`을 강한 categorical feature로**: 상혁 문자열 embed와 별도로 one-hot / label-encoded 카테고리로 넣기 (특히 LightGBM 재료로 적합)
2. **Rare class booster rules**:
   - `last_action=list_directory` + prompt에 "새로/만들/create" → `write_file` boost
   - `last_action=apply_patch` + prompt에 "확인/체크/lint/typecheck" → `lint_or_typecheck` boost
3. **다른 model family 앙상블**: 이런 categorical × lexical feature 조합은 **LightGBM/CatBoost**에 강함. 상혁 encoder 라인과 diverse. **사용자 담당인 "쓸만한 모델 test" 방향에 딱**
4. **Self-transition 활용**: 반복 액션(run_bash, plan_task, lint 등)은 사용자 workflow의 debug loop 시사. 특정 workflow 감지 후 조건부 boost 가능

---

## 10. 후속 EDA 후보

- **B**: baseline TF-IDF로 self-predict → 노이즈 case 정량화
- **C**: `sess_au` 세션 정성 관찰 (뷰어에서 필터해 훑기)
- **D**: prompt 키워드 → class 강신호 (상혁 CUE 4개 외 lexical 신호 발굴)
- **E**: prefix 의미 확인 시도 (DACON 게시판 문의?)
- **F**: 상혁 코드 `_stringify_history`가 pair(user→action)로 묶는 방식이 last_action-only보다 얼마나 강한지 정량 비교
