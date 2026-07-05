# 2026 AI·SW중심대학 디지털 경진대회 : AI부문 — 프로젝트 Context

> 이 문서는 Claude Code가 대회 규칙·제약·목표를 이해하고 코드를 작성하도록 돕는 context 파일이다.
> **모든 코드/설계 결정은 아래 "하드 제약(Hard Constraints)"을 반드시 만족해야 한다.**
> 출처: DACON 대회 236694 (개요/평가/규칙 페이지). 데이터 컬럼 상세는 로그인 후 데이터 탭에서 재확인 필요.

---

## 1. 한 줄 요약

AI 코딩 에이전트 세션의 특정 시점 상태를 입력받아, **에이전트가 다음에 수행할 행동(action)을 14개 클래스 중 하나로 분류**한다. 평가지표는 **Macro-F1**. **코드 제출 대회**(추론 코드 + 모델 가중치 제출)이며 오프라인·저사양 서버에서 실행된다.

---

## 2. 문제 정의 (Task)

- **유형**: 다중 클래스 단일 라벨 분류 (14 classes, single-label)
- **입력 (샘플당 3요소)**
  - `current_prompt` : 현재 사용자 발화(텍스트)
  - `history` : 직전까지의 대화·행동 이력(텍스트/시퀀스)
  - `session_meta` : 세션 메타정보 (요금제 tier, 잔여 토큰 예산, 작업공간 상태 등 구조화 피처)
- **출력**: 14개 action 클래스 중 하나
- **예측 대상 action 성격**: 파일 읽기·검색·수정, 셸 명령 실행, 테스트 실행, 사용자 질문 등 (탐색/수정/실행/대화 계열)
- **성격**: 텍스트(current_prompt/history) + 구조화 피처(session_meta)가 섞인 **하이브리드 분류** 문제

> ✅ 아래 "2-bis. 실제 데이터 명세"는 제공 데이터(open.zip)를 직접 EDA해 확정한 내용이다.

---

## 2-bis. 실제 데이터 명세 (EDA 확정)

### 파일 구성 (open.zip)
```
data/
├── train.jsonl            # 학습 입력, 70,000 샘플 (한 줄 = 한 샘플, ~103MB)
├── train_labels.csv       # 학습 정답, 컬럼 = (id, action), 70,000행
├── test.jsonl             # 더미 샘플 5건 (실제 평가셋은 비공개, 구조만 동일)
└── sample_submission.csv  # 제출 형식, 컬럼 = (id, action)
baseline_submit.zip        # 공식 베이스라인 제출물 (아래 4-bis 참고)
```
- **입력 형식은 JSONL** (CSV 아님). **정답은 별도 CSV**로 `id`로 조인.
- **제출 파일 = `id, action`** 2컬럼. sample_submission의 **id 순서/컬럼 그대로** 채워야 함(baseline이 그렇게 함).

### 샘플 스키마 (train.jsonl 한 줄)
```jsonc
{
  "id": "sess_sim_20260522_024730-step_08",
  "session_meta": {
    "user_tier": "enterprise",          // free | pro | enterprise
    "language_pref": "en",              // ko(최다) | en | mixed
    "workspace": {
      "language_mix": {"py":0.82,"yaml":0.1,...},  // 코드 언어 비율(dict)
      "loc": 15326,                     // 코드 라인 수 (227 ~ 90000)
      "git_dirty": true,                // bool
      "open_files": ["src/...py", ...], // 열린 파일 목록(list)
      "last_ci_status": "passed"        // passed | failed | none
    },
    "budget_tokens_remaining": 131818,  // 잔여 토큰 예산 (131 ~ 199,741)
    "turn_index": 8,                    // 현재 턴 번호 (1 ~ 18)
    "elapsed_session_sec": 720          // 세션 경과초 (30 ~ 1482)
  },
  "history": [                          // user ↔ assistant_action 번갈아, 길이 0~12(평균 7)
    {"role":"user","content":"..."},
    {"role":"assistant_action","name":"read_file","args":{...},"result_summary":"ok; read ... (173L)"},
    ...
  ],
  "current_prompt": "schemas cmd should print ..."  // 현재 사용자 발화
}
```
- `history`의 `assistant_action.name`은 예측 대상 클래스와 **같은 어휘 집합**(13종 관측; `respond_only`는 history엔 거의 없음 → 종료성 액션). → **history의 마지막 action + action n-gram이 매우 강한 피처.**
- `language_pref`는 **ko가 최다**(샘플상 ~64%). 발화가 한국어/영어/혼용 → **다국어 인코더(XLM-R/mDeBERTa 등)나 한국어 인코더 고려**. 영어 전용 모델은 불리할 수 있음.

### 14개 클래스 분포 (train 70,000 기준, Macro-F1이라 소수 클래스 중요)
| action | 비율 | | action | 비율 |
|---|---|---|---|---|
| edit_file | 15.96% | | list_directory | 6.18% |
| grep_search | 14.16% | | ask_user | 3.86% |
| read_file | 13.22% | | plan_task | 3.83% |
| glob_pattern | 7.55% | | lint_or_typecheck | 3.26% |
| respond_only | 7.40% | | write_file | 2.12% |
| run_bash | 7.24% | | **web_search** | **1.82% (최소)** |
| apply_patch | 6.89% | | | |
| run_tests | 6.52% | | | |

- **불균형비 약 8.8배**(edit_file vs web_search). 극단적이진 않지만 Macro-F1이라 소수 클래스(web_search, write_file, lint_or_typecheck)의 F1 확보가 순위를 가른다.

---

## 3. 평가 (Evaluation)

- **지표: Macro-F1 Score** (14개 클래스 F1의 단순 평균)
  - → **클래스 불균형이 점수를 지배**한다. 소수 클래스 1개를 놓치면 전체 평균이 크게 하락.
  - → accuracy 최적화가 아니라 **소수 클래스 recall/F1 확보**가 최우선 목표.
- **Public Score**: 전체 테스트 데이터 100% 기준
- **Private Score**: 예선 평가 종료 시점(7.17 10:00)의 Public Score
- **예선 → 본선**: 상위 12팀만 코드 검증 후 본선. 
- **본선 점수 = Private(50%) + 추론 속도(10%) + 전문가 심사(40%)**
  - → 속도가 점수화되고, 설계·발표 논리가 40%. **가볍고 빠르고 설명 가능한 모델이 유리.**

---

## 4. ⛔ 하드 제약 (Hard Constraints) — 위반 시 제출/평가 불가

### 4.1 평가 서버 사양
| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA T4 (VRAM 16GB) |
| CPU | 3 vCPU |
| RAM | 12GB |
| Python | 3.11.15 |
| CUDA | 12.8 |
| 인터넷 | ❌ 비활성화 (패키지 설치 외 외부 연결/다운로드 불가) |

### 4.2 시간·용량 제한
- 패키지 설치 시간 **≤ 10분**
- 추론 코드(script.py) 실행 시간 **≤ 10분**
- 제출 zip 용량 **≤ 1GB**

### 4.3 오프라인 함의 (설계 시 반드시 반영)
- **모델 가중치는 zip에 동봉**해야 함. 런타임에 HF Hub 등에서 다운로드하는 코드는 전부 실패한다.
- `from_pretrained(...)`는 **로컬 경로**(`./model/...`)를 가리켜야 한다. HF repo id 사용 금지.
- 토크나이저 파일도 로컬 동봉.
- 1GB·10분·16GB VRAM 안에서 돌아야 하므로 **대형 생성형 LLM(7B+)은 사실상 부적합**. encoder 계열 fine-tuning 또는 임베딩+경량 분류기 권장.

---

## 5. 제출 형식 (Submission)

### 5.1 참가자가 구성하는 구조
```
submit.zip
├── model/              # 모델 가중치 (예: model.pt, config.json, tokenizer 파일 등)
├── script.py           # 추론 실행 코드 (평가 서버가 자동 실행)
└── requirements.txt    # 추가 설치 패키지 (버전 명시, pip install -r 로 설치 가능해야 함)
```
- 디렉토리명·파일명 **정확히 일치**해야 함 (불일치 시 설치 오류).

### 5.2 평가 서버가 자동 추가하는 항목
```
submit.zip
├── model/                    # 참가자 구성
├── script.py                 # 참가자 구성
├── requirements.txt          # 참가자 구성
├── data/                     # 평가용 테스트 데이터 (자동 생성, 읽기 전용)
└── output/submission.csv     # 추론 결과 저장 경로 (자동 생성)
```

### 5.3 script.py 계약 (contract)
- 테스트 데이터를 로드하여 추론하고, 결과를 **`output/submission.csv`** 로 저장해야 한다.
- 상대 경로 사용.
- **입력 데이터 경로 = `data/`** (공식 코드제출 가이드 확정). 가이드 예시가 `os.path.join('data', 'test.csv')`를 사용한다. 규칙 유의사항 문단의 "`open/` 디렉토리" 표현은 다른 대회에서 복붙된 오타로 판단되며, **정본은 `data/`**. (안전을 위해 `data/`를 기본으로 하되, 방어적으로 `open/`도 fallback 체크하는 것은 무해.)
- 표준 흐름: `load_data(data/) → load_model(model/) → predict → save output/submission.csv`.
- 예외 처리 코드 포함 권장(부분 실패로 전체 제출 오류 나는 것 방지).

### 5.5 샘플(더미) 데이터
- 참가자에게 **소량의 더미 평가 샘플**이 제공되며, 폴더 구조·파일 형식이 실제 평가 데이터와 **동일**하다.
- 용도: 로컬에서 script.py 파이프라인을 이 더미로 끝까지 돌려보고(추론→submission.csv 생성) 구조·경로·시간을 검증한 뒤 제출.
- 실제 평가 X(입력)는 **비공개**. 참가자는 평가 데이터를 볼 수 없다(치팅 방지). → test 셋에 대한 과적합 튜닝 불가, CV 설계가 그만큼 중요.

### 5.4 오류 구분 (제출 횟수 관리)
- **설치 오류** (구조 불일치 / 패키지 설치 실패): 일일 제출 횟수에 **반영 안 됨**.
- **제출 오류** (script.py 실행 후 발생하는 모든 오류): 일일 제출 횟수에 **반영됨**.
- **1일 최대 제출 10회.** → 실행 오류로 횟수 날리지 않도록 로컬에서 T4 유사 환경 검증 후 제출.

---

## 6. 사전 설치 패키지 (requirements.txt에 넣지 말 것)

아래는 서버에 **기본 설치**되어 있고, 다른 버전 지정 시 설치 에러 위험. 가급적 이 버전을 그대로 사용하고 requirements.txt에서 제외한다.

```
torch==2.7.1+cu128
pandas==2.0.3
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.8.0
joblib==1.5.3
threadpoolctl==3.6.0
narwhals==2.21.2
transformers==4.46.3
accelerate==1.9.0
sentencepiece==0.1.99
regex==2023.12.25
tqdm==4.66.4
loguru==0.7.2
pyyaml==6.0.1
rich==13.7.1
```
시스템 패키지: git, build-essential, python3.11(-dev/-venv), python3-pip, cmake, ninja-build, pkg-config, libgl1, libglib2.0-0, default-jre-headless, p7zip-full, unzip 등 설치됨.

> 함의: **transformers 4.46.3 / torch 2.7.1** 기준으로 코드 작성. LightGBM/XGBoost 등을 쓰려면 requirements.txt에 추가 필요(설치 10분·오프라인 wheel 여부 고려).

---

## 6-bis. 공식 베이스라인 분석 (baseline_submit.zip)

- **구성**: `model/tfidf_logreg.pkl` + `script.py` + `requirements.txt`(scikit-learn, joblib만).
- **방법**: **TF-IDF + LogisticRegression**, 입력은 **`current_prompt` 문자열 단독**.
- **script.py 흐름(그대로 재사용 가능한 계약)**:
  1. `./data/test.jsonl` 로드(한 줄=한 샘플), `./data/sample_submission.csv` 로드
  2. 각 샘플에서 `current_prompt`만 추출 → `model.predict(texts)`
  3. sample_submission의 **id 순서에 맞춰** `action` 채우고 `./output/submission.csv` 저장
  4. 필수 키 검증·빈 텍스트 방어 처리 포함
- **베이스라인의 결정적 약점 = 개선 포인트**:
  1. **`history`를 완전히 무시** — 특히 직전 `assistant_action.name`(다음 행동의 최강 예측자)을 안 씀.
  2. **`session_meta`를 완전히 무시** — user_tier / budget_tokens_remaining / last_ci_status / git_dirty / turn_index 등 행동 분기 신호를 안 씀.
  3. **단어 표면형 TF-IDF** — 한국어/영어 혼용을 제대로 못 다룸.
- → **개선의 정석**: (a) `current_prompt` + `history` 텍스트를 인코더로 임베딩, (b) 직전 action·action n-gram·session_meta 구조화 피처를 concat, (c) 소수 클래스 가중. 이 셋만 얹어도 baseline 대비 큰 폭 상승 여지.

---

## 7. 규칙 요약

- 언어: **Python만**.
- 사전학습 모델·외부 데이터·API: 법적/라이선스 제한 없으면 사용 가능. **출처 명시 필수**.
- 유료 API/서비스 비용은 참가자 부담. (단, 최종 추론은 오프라인 서버에서 돌아야 하므로 추론 단계에서 외부 API 호출 불가.)
- 본선 진출 시 **Private Score 복원 가능한 학습 코드** + 자원 출처 + 발표 PDF(10분) 제출.
- 부정행위 금지. 팀명 변경 불가.

---

## 8. 주요 일정

| 날짜 | 이벤트 |
|---|---|
| 07.01 10:00 | 예선 시작 |
| 07.15 10:00 | 예선 종료 (모델+추론코드 zip 제출 마감) |
| 07.17 10:00 | 예선 평가 (Private 확정) |
| 07.20 10:00 | 본선 후보팀 자료(학습코드+발표PDF) 제출 마감 |
| 07.24 | 코드 검증 |
| 07.27 14:00 | 예선 결과 발표 |
| 07.30 10:00 | 포스터 세션 자료 마감 |
| 08.11 | 본선 발표평가 및 시상식 |

---

## 9. 전략 노트 (설계 방향)

### 9.1 최우선 목표: Macro-F1 = 소수 클래스 살리기
- 클래스 분포 EDA 먼저. 소수 클래스 식별.
- 대응: `class_weight` / focal loss / 소수 클래스 오버샘플링 / 클래스별 decision threshold 튜닝 / stratified K-fold.
- CV는 반드시 **Macro-F1로 검증**하고, 클래스별 F1 표를 매번 확인.

### 9.2 모델 후보 (제약 친화적)
1. **Encoder fine-tuning** (1순위): DeBERTa-v3 / RoBERTa / ELECTRA 계열(데이터 언어에 맞는 encoder). base~small 사이즈면 T4·1GB·10분에 여유.
2. **텍스트 임베딩 + 구조화 피처 → 부스팅/MLP**: encoder에서 뽑은 [CLS]/pooled 임베딩 + session_meta 피처를 concat 후 LightGBM/MLP. 속도·경량 강점(본선 속도 점수 유리).
3. 앙상블은 1GB·10분 예산 안에서만. 무거운 앙상블은 속도 점수에서 손해.

### 9.3 피처 엔지니어링 (session_meta·history) — 확정 필드 기준
- **history (최우선)**: 직전 `assistant_action.name`(=강한 예측자), 마지막 N개 action 시퀀스/n-gram, history 길이, user↔action 패턴, 마지막 result_summary의 신호(ok/fail, PASS/FAIL 등).
- **session_meta 구조화 피처**:
  - `user_tier` (free/pro/enterprise, 범주형), `language_pref` (ko/en/mixed)
  - `budget_tokens_remaining` (수치·구간화; 낮으면 respond_only/ask_user 경향 가설)
  - `turn_index`, `elapsed_session_sec` (진행도)
  - `workspace.last_ci_status` (failed → run_tests/lint 가설), `workspace.git_dirty`, `workspace.loc`, `workspace.language_mix`(주 언어), `len(open_files)`
- **current_prompt**: 의도 키워드(read/edit/run/test/search/ask 등) + 인코더 임베딩. **한국어 포함**이므로 다국어/한국어 토크나이저 유리.

### 9.4 추론 속도 (본선 10%)
- 배치 추론, fp16(T4 지원), 짧은 max_length, 불필요한 재로딩 제거.
- 모델 하나로 빠르게 vs 앙상블로 느리게: 본선에서는 전자가 유리할 수 있음.

### 9.5 재현성 (본선 필수)
- seed 고정, 학습 코드로 Private Score 복원 가능해야 함.
- 학습/추론 코드 분리, 상대 경로, UTF-8, 라이브러리 버전 기록.

---

## 10. 개발 체크리스트

- [x] 데이터 스키마·14개 클래스·분포 확정 (EDA 완료 → 2-bis 참고)
- [x] 클래스 분포 확인 (불균형 8.8배, 소수: web_search/write_file/lint_or_typecheck)
- [x] baseline(TF-IDF+LogReg) 재현: `baseline/` 압축해제 완료, `sklearn 1.8.0` env에서 로드·5샘플 추론 검증 통과
- [x] 입력 데이터 경로 = `data/` (베이스라인 script.py로 확정), 더미 5샘플 end-to-end 확인
- [x] **CV 세팅**: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`. `notebooks/05_cv_setup.md` 참조. 상혁 v01 val과 100% 정렬
- [ ] baseline encoder fine-tuning + CV 평가 (첫 벤치마크 F1 확보)
- [ ] session_meta 피처 엔지니어링 추가 실험
- [ ] 소수 클래스 대응(weight/focal/threshold) 적용 및 클래스별 F1 모니터링
- [ ] script.py: 로컬 model/ 로드, output/submission.csv 저장, 오프라인 동작 확인
- [ ] requirements.txt: 사전설치 패키지 제외, 추가분만 (설치 10분·오프라인 검증)
- [ ] T4 유사 환경에서 추론 10분 이내·zip 1GB 이내 검증 후 제출 (10회/일 아껴 쓰기)
- [ ] (본선 대비) 학습 코드 재현성 + 발표 PDF 스토리라인 정리

---

## 10-bis. 추가 발견 사항 (팩트만)

### 데이터 leak 존재 확인
- 한 세션 안에서 스텝 K의 라벨 = 그 세션의 다른 스텝 M(M>K)의 history 안에 저장된 스텝 K의 행동 name
- 전체 70,000 샘플 × 231,664 (스텝K, 스텝M) 쌍 비교, **100% 일치, 불일치 0**
- 로컬 test.jsonl 5샘플 = train_labels.csv에도 동일 id로 존재 (완전 동일 샘플)
- 로컬 5샘플 정답 5/5 sample_submission.csv 값과 일치
- 시뮬 (train 세션에서 스텝 하나 제외 후 lookup): 972건 중 793건(81.6%) 성공, 성공한 것 100% 정답
- 상세: `notebooks/04_data_leak_finding.md`

### id prefix 두 종류 존재
- `sess_sim_YYYYMMDD_XXXXXX-step_NN`: 64,975건 (92.8%), 8,330 세션
- `sess_au_XXXXXX_XXX-step_NN`: 5,025건 (7.2%), 1,099 세션
- 두 prefix의 클래스 분포 상당한 차이 (`sess_au`에서 read_file 25.69% vs `sess_sim` 12.26% 등)
- 대회 공식 docs·팀 카톡·상혁 코드 어디에도 prefix 정의 없음
- 상세: `notebooks/03_data_noise_audit.md`

### history truncation 규칙 (실측)
- 스텝 N 샘플의 history 길이 = `min(2*(N-1), 12)`
- 즉 스텝 1~7 샘플: 이전 스텝 모두 담김 (0, 2, 4, 6, 8, 10, 12개)
- 스텝 8+ 샘플: **12개(=6쌍)만 유지, 오래된 스텝은 잘림**
- 스텝 N의 history 내용 = 스텝 max(1, N-6) ~ N-1 의 (사용자 발화, 행동) 쌍

### 세션 스텝 갭 존재
- 9,429 세션 중 1,679 세션(17.81%)에 라벨 없는 중간 스텝 있음
- 갭 스텝 총 3,205개 (`train.jsonl`에도 `train_labels.csv`에도 없음)
- 갭 스텝의 원본 정보는 다른 스텝의 history에 여전히 담김 (회수 가능: 3,181/3,205 = 99.3%)
- prefix별 갭 비율: sess_sim 18.70%, sess_au 11.01%

### 위 발견들의 해석·활용 후보 (미검증 가설)
- 위 팩트를 서버 30k test 예측에 활용할 수 있는지는 **실측 필요** (1회 제출로 확인 가능)
- prefix 의미 (`sim`/`au`가 무엇을 뜻하는지)는 미확인
- 갭 스텝이 test로 갔는지 여부도 미확인

---

## 11. 자주 실패하는 지점 (Gotchas)

- HF Hub에서 런타임 다운로드 → 오프라인이라 **무조건 실패**. 로컬 경로로 고정.
- requirements.txt에 사전설치 패키지의 다른 버전 명시 → 설치 에러.
- 제출 zip 디렉토리/파일명 불일치 → 설치 오류.
- output/submission.csv 파일명·경로 틀림 → 채점 불가.
- 무거운 모델/앙상블로 10분·1GB·16GB VRAM 초과.
- accuracy만 보고 튜닝 → Macro-F1에서 소수 클래스 때문에 순위 하락.
