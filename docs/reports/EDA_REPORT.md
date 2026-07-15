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
- `language_pref`: **ko 64.3%** / en 25.4% / mixed 10.2% → **한국어 선호 선언이 지배적**. 다국어/코드 인코더 유리.
  - ⚠️ **§7.6 V4·V4-b 참조 (선언값 ≠ 실제, 그리고 무신호)**: (1) 선언 ko여도 실제 prompt는 78.7%가 한+영 혼재 → 순수 한국어(klue)보다 **다국어+코드**(mDeBERTa/mmBERT) 적합. (2) `language_pref` 자체는 **클래스·메타·코드베이스와 거의 완전 독립**(어떤 action에도 |lift-1|>0.2 없음, tier/git/ci/turn/loc/toplang 세 값이 사실상 동일) = 생성기의 독립 knob. **모델 categorical feature로는 잉여** — 언어 신호는 선언값이 아니라 실제 prompt 텍스트에 있고 인코더가 직접 읽음. 실질 차이는 (a)prompt 언어 (b)길이(en 86자 > mixed 60 > ko 51)뿐.
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

1. **검증**: `session_id` GroupKFold(5-fold). 지표는 Macro-F1 + 클래스별 F1 표 상시 확인.
2. **입력 설계 (fusion)**:
   - 텍스트: `current_prompt` (+ 직전 `result_summary` / 최근 user content) → 다국어 인코더 임베딩.
   - 시퀀스: 직전 action(및 최근 k개) 임베딩, action n-gram.
   - 수치/범주: turn_index, budget, elapsed, loc, open_files 수, tier, language_pref, ci_status, git_dirty, prompt_len.
   - `history_empty` 플래그.
3. **불균형**: class-weighted CE 또는 focal loss; 소수 클래스(web_search/write_file/lint) 오버샘플 실험; 추론 시 클래스별 threshold/logit 보정.
4. **경량·속도(본선 10%)**: 프롬프트 짧으니 base~small 인코더 + fp16로 충분. 무거운 앙상블은 속도 손해.
5. **baseline 대비 ablation**: (a) current_prompt only(=baseline) → (b) +직전 action → (c) +turn_index/meta → (d) +focal/weight. 각 단계 Macro-F1 기록.

---

## 7.5 정밀 EDA v2 (7/6) — 노이즈 상한·피처 발견·leak 정량화

**F1. 탐색(inspect) 4클래스 라벨은 프롬프트 의미와 크게 분리돼 있다 (구조적).**
- glob cue(`**/`,`*.ext`) 있어도 glob_pattern 20%뿐 (grep 25%, read 23%). dir cue → list_directory 13%뿐 (read 28%, grep 27%).
- 생성기가 프롬프트 의미와 반쯤 무관하게 inspect action을 배정 → t-SNE inspect 혼탁, list_directory 저정밀, LLM 제로샷 붕괴(Claude 0.205)의 공통 원인.
- 단 meta가 일부 가름: inspect 서브셋에서 (직전action, n_open, turn) 조건부 다수결 34.4%→**42.6%**. 의미가 아니라 **meta·이력 조건부 패턴**으로 풀어야 하는 클러스터.

**F2. 환원 불가능 노이즈 직접 추정.** 동일 prompt 중복 10,498샘플 중 라벨 충돌 5,795(55%). 중복 내 다수결 상한 **74.9%**. converse도 충돌(같은 "찾아봐줘"가 web_search/plan_task/ask_user). → 남은 이득은 의미 이해가 아니라 조건부 패턴+threshold에서 나옴. inspect 정밀도에 무한 투자 금지.

**F3. anti-stickiness.** history 최빈 action==정답 8.6%, 직전==정답 13.7% — 둘 다 기저(16%)보다 **낮음**. 생성기는 최근 action 반복을 회피. 최근 action들은 **negative evidence**.

**F4. open_files = 최강급 meta 피처.** n_open=2 → **apply_patch 24%**(기저 6.9%의 3.5배). 3+ → respond_only 24%+lint 10%. 0 → read 19%/list 14%(콜드스타트). 범주형(0/1/2/3+) 필수.

**F5. lookup leak 정량화.** 같은 세션 "더 늦은 step" 샘플 history에서 내 정답 복구 가능 비율 = **86.5%**. dummy test 5세션 **전부 train에 존재**. 실 테스트가 train 세션과 겹친다면 lookup으로 대규모 정답 조회 가능. → 무해한 프로브 제출(세션 매칭 시 lookup override, 아니면 모델 폴백)로 겹침 여부 판정 가능. 규정 적합성은 Q&A 문의.

**F6. CV(0.775)–LB(0.75) 격차 0.025.** 후보: 팀별 fold 불일치 / threshold를 단일 val에 과적합 / fold 분산. → 공통 group 5-fold OOF로 재측정, threshold는 OOF 전체에서 튜닝.

---

## 7.6 v3 (7/5) — 검증·신규발견 (raw 전수 재검증 + 심층 EDA)

> 본 리포트의 수치 주장 ~60개를 `data/raw` 원본에서 전수 재검증하고, 미답 질문 5개를 추가 조사.
> 분석 환경: `intern_aichallenge` env, 상혁 v01 `common.serialize()` + `jhu-clsp/mmBERT-base` tokenizer.

### 검증 결과 — 본 리포트는 사실상 전 항목 정확

- §1~§5, §7.5 F2/F4/F5: **전부 실측 일치** (클래스 count, 전이확률 13개, turn 순서 14개, meta 분포, 중복 10,498/충돌 5,795/상한 74.9%, leak 86.5% 등 숫자 단위 재현).
- §7.5 F3 (anti-stickiness 8.6/13.7%): 분모를 "history 있는 행"으로 두면 일치 (전체 70k 기준은 7.8/12.1%). 방향(기저 16%보다 낮음)은 동일.
- §7.5 F1 (cue 배반 %): cue 정규식 정의에 의존해 정확 재현 불가. dir-cue 근사 일치(list 12/read 23/grep 24 vs claim 13/28/27), **방향은 확인됨**.
- 추가 정합: `step == turn_index` 100% 일치 (중복 정보). 이전 step의 prompt가 다음 step history에 100% 포함 (순수 누적 구조 = leak의 근원).

### V1. 라벨 충돌은 "환원 불가 노이즈"가 아니라 **상태-조건부 규칙** (F2 프레이밍 수정)

중복 prompt 충돌에 컨텍스트를 조건으로 걸면 (n≥2 그룹만 남겨 정직하게 계산):

| 조건 | 충돌 잔존 샘플 | n≥2 그룹 내 majority 상한 |
|---|---:|---:|
| prompt only | 55% | 74.9% |
| + last_action | 30% | 85.5% |
| + turn bucket | 23% | 89.2% |
| + n_open | 22% | **89.7%** |

같은 prompt라도 (직전 action, 턴, 열린 파일 수)가 다르면 라벨이 거의 결정된다 → **생성기가 state를 보고 액션을 배정**했다는 뜻. **0.79 벽은 데이터 상한이 아니라 모델이 상태-조건부 패턴을 덜 뽑아낸 것일 가능성**. "노이즈에 피팅하는 게 맞다"는 팀 실측과 정합하며, 파야 할 곳이 상태-조건부 신호임을 가리킴. (단 89.7%는 train 내 memorization 상한이지 일반화 보장은 아님)

#### V1-b. 용욱 "user instruction ↔ tool 불일치" 세트와의 관계 (상충 아님, 상보적)

용욱이 노션/카톡에 정리한 contradiction id 리스트(dedup 207개)를 직접 대조. **모순이 아니라 같은 원인의 다른 단면**임을 확인.

- **용욱 세트의 정체**: 라벨이 **inspect(탐색) 클러스터 100%** (grep_search 83 / read_file 80 / glob_pattern 37 / list_directory 6 / apply_patch 1). 예: "파이썬 파일 **목록**부터 뽑아줘"(=list 의미) → 라벨 grep_search / "**폴더 안에 뭐뭐** 있는지"(=list) → 라벨 read_file. **prompt 의미와 라벨이 inspect 내부에서 어긋난 케이스만 골라낸 것** = 본 리포트 §7.5 **F1**과 동일 현상.
- **state로 풀리나? 안 풀림**: state-조건부 예측 정확도가 용욱 세트 **15.9%** vs 전체 train 23.5% vs 나머지 23.5%. 207개 중 state로 설명 16% / 미설명 84%. → **V1의 "+state로도 22% 잔존(환원불가)" 구역이 바로 용욱 세트.** 서로 다른 부분집합을 본 것이지 반대 주장이 아님.
- **하나의 뿌리**: 생성기가 액션을 *prompt 의미*가 아니라 *(state + coarse 클러스터)*로 배정. → (a) prompt→coarse 클러스터, (b) state→클러스터 내부 tilt는 살아있는 신호(V1의 90%)지만, (c) **inspect 내부에서 read/grep/list/glob 판별은 prompt로도 state로도 near-random** (F1 실측 상한 42.6%). 용욱 세트 = 이 (c) 구역(노이즈 바닥).
- **실전 함의**:
  1. **버리면 안 됨** — test도 같은 inspect 스크램블 분포라, 제거 시 "list-prompt→grep-label"을 못 배워 감점. 팀 실측 "노이즈 뺄수록 하락"의 정확한 원인.
  2. **의미 추론 금지** — Claude zero-shot 0.205의 원인. "목록 뽑아줘→list_directory"라 추론하면 정답(grep)과 틀림. inspect는 추론이 아니라 암기+state 통계로.
  3. **레버 분리** — V1의 state 신호는 **inspect 바깥**(modify/exec/converse ~59%)과 해소 가능 충돌에서 점수. **inspect 내부(용욱 세트)는 ~43% 천장으로 수용, 정밀도 무한투자 금지** (F2 경고와 일치).

### V2. dummy test 5건 = train 행과 id·내용 완전 동일 (해석 주의)

- **팩트**: test.jsonl 5건 모두 train.jsonl에 동일 id + 동일 내용으로 존재. 라벨도 train_labels에서 그대로 조회됨 (read_file / web_search / run_bash / grep_search / edit_file).
- **해석 (정정됨)**: 공식 설명에 "형식 확인용 5건"이라 명시 → 주최측이 형식 예시로 train에서 복사한 것일 뿐, **실제 서버 test 30k의 성격(train 세션 겹침 여부)에 대한 근거가 아님**. F5의 probe 아이디어는 우선순위 하락.
- **남는 것**: train↔test 겹침과 무관하게, 서버 test 30k가 세션당 여러 step을 포함한다면(train 구조상 30k ≈ 4천 세션) **test 내부에서** 늦은 step의 history로 이른 step 정답 조회 가능 (train 구조 기준 86.5% 복구). 규정 문의 실익은 유지, 제출은 보류(상혁 "불공정 소지").

### V3. "512 truncation 41.5%" 주장 재현 실패 — max_len 1024 lever 재검토 필요

상혁 serialize + mmBERT tokenizer 실측 (70k 전수):

| 설정 | len>512 | 비고 |
|---|---:|---|
| 기본 (open_files O, cues X) | **3.4%** | p50 298 / p95 492 / max 682 |
| + symbol_cues | 4.3% | |
| + langmix | 7.3% | |

- 41.5%가 나오는 설정 없음. **max_len 512→1024의 기대효과는 전체 3~4% 샘플에 한정** → 최우선 lever로 삼기 전에 **본인(건우) mmbert 학습코드의 실제 serialize로 재측정 선행 필수** (본인 코드가 history를 더 길게/raw로 넣으면 41.5%가 맞을 수 있음).
- 단, 잘리는 샘플은 **web_search lift 3.74×** (respond_only 1.44×, plan_task 1.63×) — rare class가 가장 잘림 → 1024 확장이 rare-F1에는 비례 이상 기여 가능.
- **serialize 순서 결함 발견**: `[META]`가 문자열 맨 끝 → truncation 시 META(open_files 포함)부터 통째 소실 (@512에서 3.3% 손상). SOTA 기여 feature가 정확히 잘리는 샘플에서 사라지는 구조.

### V4. 언어와 클래스가 강하게 얽혀 있음 — 번역 증강 경고

- 실제 prompt 언어 (한글/영문자 감지): **mixed 57.7% / en 27.4% / ko_pure 14.8%**
- `language_pref=en` → 실제 prompt **100% 영어**, `pref=ko` → 영어 0% (mixed 78.7 + ko_pure 21.3). **language_pref는 생성기의 결정적 입력**.
- **ko_pure prompt는 실행형 클래스 극단 편향** (en 대비 lift): respond_only 3.29× / lint 2.63× / run_tests 2.49× / run_bash 2.43×, 반대로 edit_file 0.28× / apply_patch 0.25×. 짧은 한국어 명령("다시", "돌려")이 실행 액션과 결합.
- → **ko↔en 번역 증강은 실제 언어별 클래스 prior와 다른 쌍을 주입하는 분포 왜곡 위험**. 언어별 LoRA가 eval↑/LB 0.75로 과적합한 실측과 정합. 증강 시 언어별 클래스 prior 보존 확인 필수.

#### V4-b. `language_pref`(선언값)는 무신호 독립 knob (모델 feature로 잉여)

- **선언 language_pref는 클래스와 무관**: ko/en/mixed 세 값 모두 어떤 action에도 |lift-1|>0.2 없음. top5 클래스 판박이(edit16/grep14/read13...).
- **메타·코드베이스와도 독립**: tier(ent 15.8/16.3/18.5)·git_dirty(76.x 동일)·ci_failed·turn(5.2)·budget·loc·n_open·**코드베이스 top언어 분포(py40/java10/...)** 세 값이 사실상 동일 → 생성기가 language_pref를 **다른 모든 것과 독립**으로 뿌림.
- **유일한 실질 차이 2개**: (a) 실제 prompt 언어(pref=en→영어100%, pref=ko→혼재79%+한21%), (b) prompt 길이(en 86자 > mixed 60 > ko 51).
- **함의**: `lang=<pref>`를 categorical/fusion feature로 넣어도 이득 없음(용욱 fusion +0.01과 정합). **신호는 선언 라벨이 아니라 실제 prompt 텍스트에 있고 인코더가 직접 읽는다.** V4의 클래스-언어 커플링도 "실제 언어" 축이지 선언값 축이 아님.

#### V4-c. 언어 정규화(normalization) 아이디어 — 증강과 구분, 이 대회엔 부적합

> "코딩용어 빼고 전부 한국어로" 또는 "전부 영어로" 통일 = **정규화**(입력 언어 다양성 축소), **증강 아님**(샘플 추가 아님). 두 가지 하드 블로커.

1. **추론 시 test 번역 불가(치명적)**: 정규화는 train뿐 아니라 **test 30k도 같은 규칙으로 번역**해야 일관. 그런데 제출은 오프라인·10분·1GB·T4 → 번역모델을 예산 안에 넣어 30k 실시간 번역 = 사실상 불가. 상혁 07-05 실측 지적("채점할때도 실시간 번역해야 해서")과 동일.
2. **실제 신호 파괴**: V4에서 **실제 언어가 클래스 신호**(ko_pure "다시/돌려"→실행형 2.4~3.3×). 전부 영어로 정규화하면 이 커플링이 소멸 → 오히려 성능 손해 예상.
- **증강 방향(train만 번역본 추가)도 팀 실측 부정적**: 경모 1000개 증강 효과 없음/하락, 용욱 언어별 LoRA LB 0.75 과적합. 근본 이유 = **라벨이 언어에 의존**하므로 언어-불변을 강제하면 틀림.
- **유일하게 안전한 활용 = 진단 probe**(증강 아님): val 일부를 en으로 번역해 예측 변화를 관찰 → 모델의 언어 민감도 측정용. 제출 파이프라인엔 미포함.

### V5. rare 3클래스에 완전 판별 어휘 실존 — CUE 확장 근거

| 클래스 | 판별 토큰 (vs 혼동 상대 빈도) |
|---|---|
| lint_or_typecheck (vs run_tests/run_bash) | **"타입체크"(39:0), "typecheck"(20:0)** — 완전 분리 |
| write_file (vs edit_file/apply_patch) | **"골격"(53:1), scratch(30:0), scaffold, overwrite, brand, "만들어줘요"** |
| web_search (vs plan_task/ask_user/respond_only) | "최신", "공식", "검색해줘", "베스트프랙티스" (강하지만 완전분리 아님) |

### v3 액션 제안 (근거 순위)

1. **[즉시·무료] CUE 확장** — V5 토큰을 `_explore_cues`에 추가해 A/B. write_file "골격" 계열은 거의 무료 점수.
2. **[즉시·무료] serialize 순서 변경** — `[META]`를 `[CUE]` 직후로 이동 + `last=<last_action>`을 CUE에 명시 (V1·V3). last_action 하나로 충돌 55%→30% 해소되는데 현재 HISTORY 문자열에 묻혀 있음.
3. **[선행 필수] exp006(mmbert@1024) 착수 전 본인 serialize로 truncation 재측정** (V3) — 3.4%면 강등, 41.5%면 진행.
4. **[팀 공유] 번역 증강 경고** (V4) — 경모 증강 착수 전 언어별 클래스 prior 보존 체크 공유.
5. **[전략] 상태-조건부 피처를 텍스트 앞쪽에 강조하는 재직렬화 실험** (V1) — MLP fusion이 죽은 건 정보 부재가 아니라 위치·형태 문제일 가능성. 조건부 상한 89.7% vs 현 LB 0.776의 갭.
6. **[보류 유지] lookup probe** — V2 정정으로 우선순위 하락. DACON 규정 문의 결과 대기.
7. **[중기] TAPT** — mixed(한글+코드 식별자) 57.7% 도메인 특수성 → train 텍스트 MLM 적응 (MODEL_ZOO C-4 뒷받침).

---

## 8. 재현 방법

```python
import json, pandas as pd, numpy as np
lab = pd.read_csv("data/train_labels.csv").set_index("id")["action"].to_dict()
rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
# session_id = id.rsplit("-step",1)[0]  → GroupKFold 그룹키
# 위 섹션들의 집계는 rows를 1-pass 순회하며 Counter로 산출
```
