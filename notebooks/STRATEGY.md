# STRATEGY.md — 실험 설계/전략 (Dacon 236694, 코딩 에이전트 다음 행동 예측)

> Claude Code용 전략 컨텍스트. 규칙·제약은 `COMPETITION_CONTEXT.md`, 데이터 분석은 `EDA_REPORT.md` 참조.
> 이 문서는 "무엇을 왜 어떤 순서로" 실험하는지의 계획서다. 코드 작성 시 이 순서와 규약을 따른다.

---

## 0. TL;DR — 큰 그림

1. **CV(GroupKFold by session)를 먼저 고정** = 모든 실험의 계측기. 이게 없으면 아무것도 비교 불가.
2. **범용(general-purpose) encoder search**(경로 A)로 1GB 이하 student 상한을 찾는다. 특화 모델 X, 범용 O.
3. **불균형(Macro-F1)** 처리 = class weight/focal + 클래스별 threshold. 소수 3클래스(web_search·write_file·lint)가 승부처.
4. **컬럼 활용**(history + session_meta)을 backbone 위에 얹어 추가 이득.
5. 필요하면 **H200으로 대형 teacher 앙상블 → 1GB student 증류**로 1위 격차 공략.
6. 제출물은 항상 **≤1GB·T4 10분·오프라인** (H200는 학습용일 뿐, 추론은 T4).

핵심 원칙: **큰 레버(backbone) 먼저, 그 위에 작은 레버(컬럼·기법). 모든 판단은 CV 위에서. 되돌릴 일 없는 순서로.**

---

## 1. 현재 위치와 목표

- 현재: **4위 ~0.724** (Macro-F1). 본선 컷(상위 12) ~0.695 → **본선은 사실상 확보**.
- baseline(TF-IDF+LogReg 제출) ≈ 0.436. 1위 JunCTion ≈ 0.759 (2위와 0.032 격차 = 구조적).
- 2~5위는 0.0035 안 노이즈 밴드 → 마진 튜닝으로 순위 출렁, **1위 격차는 backbone/증류로만 메움**.
- 진짜 목표: 예선에서 1위 추격(0.74+) + **본선 대비**(Private 50% + 속도 10% + 전문가심사 40%).

---

## 2. CV 설계 (⛔ 타협 불가, 첫 단계)

**GroupKFold(session) 5-fold, seed 고정.** group key = `id.split("-step_")[0]`.

**왜 session 단위여야만 하는가 (데이터로 검증됨):**
- 각 세션은 여러 step이 개별 샘플로 존재(세션당 평균 7.4 step, 97.7%가 2+ step).
- **이전 step의 정답(label)이 이후 step의 `history`에 그대로 누적**되어 있음 (history = 그 세션의 행동 로그).
- 랜덤 split이면 같은 세션의 step들이 train/val로 쪼개져 → 모델이 세션 궤적을 암기해 val 답을 조회 → **CV가 뻥튀기, 리더보드 전이 안 됨.**
- **테스트는 세션당 1 step**(dummy로 확인). GroupKFold는 이 조건(새 세션 통째)을 그대로 재현.

**검증 절차:** CV 돌린 뒤
- `overlap sessions == 0` 확인 (누수 없음)
- OOF Macro-F1 + **클래스별 F1 표** 확보
- OOF Macro-F1 ≈ 리더보드(±0.01~0.02)인지 대조 → 계측기 신뢰 확인
- 신뢰 확인 후에야 제출 없이 GPU에서 무한 실험 가능.

**주의:** history를 피처로 쓰는 것은 정당(테스트에도 있음). leak은 "피처"가 아니라 "분할"의 문제. → history 활용(O) + session 분할(GroupKFold)은 독립.

---

## 2.5 모델 후보 판별 규칙 (3-게이트 필터) — 새 모델 볼 때마다 이걸로 거른다

어떤 모델(BFCL/tool-use 리더보드 상위, 최신 SOTA LLM 등)을 발견하든, 아래 3개 게이트를 순서대로 통과하는지만 확인한다. 하나라도 막히면 그 용도로는 못 쓴다.

**게이트 1 — 제출 모델로?** 제출 zip ≤1GB (T4). fp16 기준 상한 **~0.5B**. 거대 LLM은 4bit로도 초과 → **전부 탈락**. (예: GLM-4.6 355B는 4bit ~180GB = 1GB의 180배. 양자화가 존재하는 것 ≠ 1GB에 담기는 것.) → 제출은 encoder(≤0.5B)만.

**게이트 2 — teacher로? (단일 H200 141GB)** 4bit로 ~141GB 안에 들어와야 파인튜닝 가능 = **~100B급이 상한**.
- 통과: GLM-4.5-Air(106B), EXAONE-32B, Qwen-14/32B, Kanana-8B, GLM-4-9B.
- 탈락: GLM-4.6(355B), GLM-5/5.1/5.2(744B+) → 단일 H200 초과.

**게이트 3 — 스킬이 맞나?** function-calling(도구호출 **생성**) ≠ 우리 태스크(다음 action **분류**). tool-use 특화 모델은 brittle(WildToolBench). → BFCL 순위는 teacher tiebreaker 정도의 약한 신호일 뿐, 선택 기준 아님. 최종 판단은 CV.

**요지:** 3게이트 다 통과하는 거대/특화 모델은 사실상 없다. 제출=encoder, teacher=≤100B 범용 LLM. (GLM 버전 참고: 4.5→4.6→5→5.1→**5.2(최신)**. "GLM-6" 없음. GLM 계열 teacher는 4.5-Air/4-9B가 상한.)

---

## 3. 경로 A: 범용 encoder search (메인 레버)

**전제:** 분류/NLU에선 범용 encoder가 곧 SOTA (mDeBERTa 276M이 XNLI에서 70B LLM을 이김; KLUE-BERT가 한국어에서 8B/12.8B 이김; tool-use 특화 모델은 brittle). → 범용 encoder 파인튜닝이 정답.

**후보 (축별 대표만 스윕, 전수조사 금지):**

| 축 | 모델 | 파라미터 | transformers | 비고 |
|---|---|---|---|---|
| 모던 다국어 | `jhu-clsp/mmBERT-base` | 307M | ≥4.48 | 코드+다국어, XLM-R 상회, 빠름 (1순위) |
| 모던 다국어(속도) | `jhu-clsp/mmBERT-small` | ~140M | ≥4.48 | 더 빠름 → 본선 속도 유리 |
| 검증 다국어 | `microsoft/mdeberta-v3-base` | 276M | 4.46.3 OK | 현재 기준선, 안전 |
| 한국어 특화 | `klue/roberta-large` | ~337M | 4.46.3 OK | ko 64%라 필수 실험 |
| 임베딩 계열 | `BAAI/bge-m3` 또는 `intfloat/multilingual-e5-large` | ~560M | 4.46.3 OK | pooling 분류, 앙상블 다양성 |
| (코드+다국어) | `ibm-granite/granite-embedding-...multilingual` | 311M | ≥4.48 | mmBERT 레시피 |
| (실험 1회) | `Qwen/Qwen2.5-0.5B` SeqCls+LoRA | 0.5B | ≥4.46 | "소형 LLM이 encoder 이기나" 확인 |

**추천 스윕 순서:** ① mdeberta(기준선 재현) → ② mmBERT-base → ③ KLUE-roberta-large → ④ bge-m3 → (여유) mmBERT-small, Qwen-0.5B. 동일 CV·동일 입력으로 `--model`만 교체, OOF Macro-F1 표로 줄 세우고 상위 2~3개 심화.

**공통 세팅:** max_length 128~256(입력 짧음, median 56자), fp16, LR 1e-5~5e-5, weight_decay 0.01, pooling은 CLS/mean 둘 다 실험.

**1GB 예산:** fp16 = ~0.5B 상한 / int8 = ~1B / int4 = ~2B. encoder는 다 여유, **소형 2~3개 앙상블도 1GB 안**.

---

## 4. 불균형 처리 (Macro-F1 직결)

- 분포: edit_file 16% → web_search 1.8% (불균형 8.8배). **소수: web_search·write_file·lint_or_typecheck**.
- 대응: `weighted_ce`(inverse-freq) 또는 `focal`(γ=2) + **클래스별 threshold/logit 보정**(Macro-F1 기준 좌표하강). class-balanced sampler / logit-adjustment도 실험.
- **혼동 위험쌍 모니터링**(confusion matrix): glob↔grep(파일명 vs 내용), edit↔write↔apply_patch(수정 vs 신규 vs diff), run_tests↔lint(실행 vs 정적), respond_only↔ask_user.
- 검증은 항상 Macro-F1 + 클래스별 F1로. accuracy만 보지 말 것.

---

## 5. 컬럼 활용 (history + session_meta) — backbone 위에 얹는 레버

**history (신호 제일 셈):** 직전 `assistant_action.name`(전이 top-1 23%), 최근 action n-gram, `result_summary`의 PASS/FAIL, history 길이.
**session_meta (텍스트로 못 잡는 부분):** `turn_index`(국면 신호, 단독 예측력 최고), `budget_tokens_remaining`(구간화), `last_ci_status`, `git_dirty`, `user_tier`, `language_pref`, `loc`, `len(open_files)`.
**current_prompt:** 의도 키워드 + 인코더 임베딩(한/영 혼용 → 다국어 모델).

**활용 방식 2갈래:** (a) 텍스트로 이어붙여 인코더에 넣기(간단), (b) 수치/범주는 별도 벡터로 뽑아 fusion 헤드/LightGBM에 결합. 어느 쪽이 나은지는 CV로 결정.
**주의:** 이 실험들은 **확정된 backbone 위에서** 검증 (backbone 바뀌면 재검증). caution: 데이터가 합성(`sess_sim_`)이라 일부 상관은 생성기 아티팩트일 수 있음 — 인과로 과신 금지, CV로만 판단.

---

## 6. 증류 경로 (조건부, 1위 격차 공략용)

**언제:** 경로 A 단일 모델 상한을 확인한 뒤, "대형 teacher가 CV에서 확실히 더 높은데 크기 때문에 1GB에 못 담을 때"만 발동. 무조건 아님.

**H200(141GB) 확보 → teacher 상한 해제:**
- teacher(학습용, 크기 무제한): 범용 instruct LLM들을 QLoRA SeqCls로 파인튜닝 → OOF soft logit.
  후보: **EXAONE-3.5-32B**(작년 우승팀), **GLM-4.5-Air(106B, 4bit)**, **Kanana-1.5-8B**, Qwen-14/32B, **GLM-4-9B-Chat**. (풀 GLM-4.5 355B는 한 장엔 무리 → Air까지.)
- **중형 앙상블 teacher가 단일 대형보다 대체로 우수**(soft label 캘리브레이션·다양성 → 소수클래스 유리).
- student(제출, ≤1GB): mmBERT/mdeberta 단일 모델에 KD(`--teacher_logits`, 상혁 코드에 구현됨).

**핵심 구분:** H200은 **teacher만 키움**. 제출 student는 여전히 ≤1GB·T4. teacher가 강해질수록 → 증류로 넘길 지식↑ → 1GB student 성능↑. 이 경로로만 이득이 옴.
**검증:** 32B teacher가 8B teacher보다 CV에서 유의미하게 높은지 먼저 확인("키울 수 있다"≠"이득이다").
**주의:** teacher는 반드시 **우리 70k로 파인튜닝**. 범용/pre-distilled 모델은 14 라벨을 모름 → 태스크 파인튜닝 생략 불가.

---

## 7. 경로 A·증류 외 다른 방법

- **소형 앙상블(증류 없이, 1GB 안)**: mdeberta+KLUE+mmBERT를 각 fp16으로 담아 logit 평균. teacher가 작으면 증류 없이 앙상블 이득 그대로.
- **양자화된 중형 모델(경로 C)**: 1~2B LLM을 int8/int4로 1GB에 담기. T4 속도·bitsandbytes 검증 필요.
- **LightGBM (피처 기반)**: encoder 임베딩 + 구조화 컬럼 → 초경량·초고속, 앙상블 멤버(6번 컬럼 작업과 시너지).
- **학습 기법(경로 A 내 레버)**: InfoNCE/SupCon 대조학습(작년 우승 사용), coarse 4그룹 보조헤드(hierarchical), FGM/AWP 적대학습, train 텍스트 MLM 도메인 적응.
- **RL: 스킵.** 고정 라벨 지도학습(BC)이라 부적합. soft-F1 loss 정도가 유일한 접점이나 threshold 보정이 지배.

---

## 8. 하드웨어

- **학습(teacher/student 훈련): H200 141GB** — 32B급 teacher 앙상블까지 여유. GPU 병목 없음.
- **대회 추론(제출): Dacon T4 16GB, 오프라인, 10분** — 이건 GPU 늘어도 불변. student는 항상 ≤1GB·T4 마진 확보.

---

## 9. 팀 역할 & 공통 규약

- **2번 (상혁 + 나): CV 계측기 → 범용 encoder search → teacher 앙상블 → student 증류.** 메인 레버.
- **3번 (경모 + 용욱): 같은 CV 위에서 컬럼 활용** — history 명시피처(직전 action·n-gram·result PASS/FAIL) + session_meta fusion(turn·budget·ci·git). backbone 위에 얹는 레버.
- **공통 규약 (반드시 공유):**
  - fold 분할 = **group(session), seed 42** 공통. 두 팀이 같은 fold를 써야 최종 병합·비교 가능.
  - 공용 지표 = OOF Macro-F1 + 클래스별 F1.
  - 초반에 2번이 **기준 backbone 하나(mdeberta) 빨리 고정** → 3번이 그 위에서 병렬로 컬럼 실험.
  - 1주차 끝 **머지 포인트**: 각자 best 합쳐 CV 확인.
  - 제출 전 항상 **오프라인 dry-run**(dummy test.jsonl로 end-to-end).

---

## 10. 제출 위생 (상세는 COMPETITION_CONTEXT.md)

- 제출 zip: `model/` + `script.py` + `requirements.txt`, ≤1GB. 입력 `data/`, 출력 `output/submission.csv`(컬럼 `id,action`, 대소문자 정확).
- 오프라인: 모델 로컬 경로 로드(HF Hub 다운로드 금지). mmBERT/granite 쓰면 requirements.txt에 `transformers>=4.48` + 설치 10분·T4 로드 검증.
- 제출 10회/일. 실행오류는 카운트됨 → dry-run 필수. 설치오류는 카운트 안 됨.

---

## 11. 타임라인 (예선 마감 7/15)

- **1주차(~7/8):** CV 고정 → 범용 encoder search → 불균형 처리로 단단한 단일모델(현재 위로). 3번은 병렬로 컬럼 실험.
- **2주차(~7/15):** H200 teacher 앙상블 → 1GB student 증류로 1위 격차 공략, 최종 제출 확정.
- **예선 후(~8/11):** 속도 최적화 + 재현 학습코드 + 발표(40%). 서사: "작년 우승 인사이트(LLM 앙상블)를 1GB·T4에 증류로 압축해 앙상블급 성능을 경량·고속으로 재현."

---

## 12. 지금 당장의 다음 액션

**CV 계측기 고정 1회.** 현재 제출과 동일 설정으로 `--split group` 실행 → `overlap 0` / 클래스별 F1 / OOF Macro-F1 확보 → 리더보드와 대조. 이 숫자가 나와야 encoder search·teacher 규모 결정이 전부 그 위에서 이뤄진다.
(주의: train.jsonl의 중첩 `history`/`session_meta` 파싱이 하네스에서 되는지 첫 실행에서 확인.)

---

### ✅ 완료 (2026-07-02)

**CV 계측기 세팅 완료** — `notebooks/05_cv_setup.md` 참조.

- 방식: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` — 상혁 v01 코드와 완전 정렬
- Group key: `id.split("-step_")[0]` (세션 id)
- 기본 val = `fold_0` (7,000건, 10%)
- 산출: `src/cv.py`, `data/processed/cv_splits/fold_assignments.csv`
- 검증 통과: overlap 0, 클래스 stratify 표준편차 <0.01%p, rare class fold별 편차 ±1건 이하
- 상혁 val 재현본과 100% 일치 → val_probs.npz 그대로 앙상블 정렬 가능
- Sanity check로 추가 확인 필요: OOF Macro-F1과 리더보드(0.77585) 대조 — **다음 액션**

**다음 미결 항목** (이후 처리):
- OOF Macro-F1 뽑아 리더보드 대조 (계측기 신뢰 확인)
- encoder search 스윕 시작 (§3 경로 A)
