# STATUS — oh-my-sinsa 공유 대시보드

> **여러 세션이 공유하는 단일 진행상황 보드.** 작업 시작/완료할 때마다 여기부터 갱신한다.
> 상세 근거는 `notebooks/`, 팀 맥락은 `team_chat/SUMMARY.md`, 팀원 코드는 `team_ref/` 참조.

**최종 갱신**: 2026-07-03 (세션: exp001 klue-large 시작→3090 느려 yj-2 이관)
**마감**: 2026-07-15 10:00 예선 종료 · **D-12**
**팀 리더보드 (07-03)**: 🥉 **14위 0.77585** — 상위 13개 팀 튀어옴. 1위 고대 야호 0.79186 (**gap 0.01601, 4배 확대**). 우리 점수 그대로. 팀별 12~30회 제출 활발. 제작자 상혁(mDeBERTa+KD+AWP)
**사용자(김건우) 방향**: 상혁 라인과 **diverse한 축** — rare 3클래스(`web_search`·`write_file`·`lint_or_typecheck`) F1 확보가 시상권 lever

---

## 📋 갱신 규약 (병렬 세션 필수)

1. 무엇이든 손대기 전에 이 파일부터 읽는다. 끝나면 **"최종 갱신" 날짜/세션명**을 바꾼다.
2. TODO는 **한 항목 = 한 세션이 소유**. 집는 순간 `🔴 진행 중`으로 옮기고 `(담당·세션)` 태그를 단다. 중복 착수 방지.
3. 실험은 반드시 **exp_id 한 줄을 트래커에 먼저 추가**(상태 `📋 계획`)한 뒤 시작한다. 끝나면 점수·경로 채우고 상태 갱신.
4. 산출물 경로는 항상 한 줄로 남긴다 (`→ saved: ...`). 실험 결과는 `experiments/{exp_id}/`에 config+model+log+metrics 세트로.
5. 충돌 나면 덮어쓰지 말고 두 항목 다 남긴 뒤 조율. 삭제 대신 `✅ 완료`/`⛔ 폐기`로 이동.

---

## 🗂 TODO 보드

담당 태그: `건우`(사용자) · `상혁` · `용욱` · `경모` · `규중` · `claude`(세션 작업)

### 🔴 진행 중 (In Progress)

- [ ] `건우` **exp001: klue/roberta-large + 상혁 serialize 학습 → yj-2 A100로 이동** — 3090에서 ETA 9~10시간 확인 후 kill (07-03). `experiments/exp001_klue_large/`. **사용자가 yj2-A100-kinamkim SSH 접속 후 batch_size=32로 재시작 필요**. 예상 3~4시간

### 🟡 다음 (Next — 착수 준비됨)

- [ ] `건우/claude` **exp001: Qwen2.5-0.5B + LoRA 분류 head 파이프라인 구축** — 모델 리서치 Tier 1 최우선. 상혁 `serialize()` 재사용, tokenizer 교체. 아키텍처 완전 diverse(causal LM). → `src/` 스크립트 초안부터
- [ ] `건우/claude` **exp002: multilingual-e5-base + head** — 임베딩 objective가 mDeBERTa와 다름, 앙상블 재료. exp001과 병렬
- [ ] `건우/claude` **exp003: au sample 취급 ablation** — 상혁 코드에 `sample_weight` 추가해 (i)그대로 / (ii)downweight 0.5 / (iii)제외 3조건 val Macro-F1 비교. 노이즈 감사 축C 후속
- [ ] `건우/claude` **`last_action` categorical feature 실험** — Naive Markov top1 22.94%(majority +6.98%p). LightGBM/CatBoost 재료로 상혁 encoder와 diverse (EDA #01 §9)
- [ ] `건우/claude` **rare class booster rule 검증** — `list_directory→write_file`(lift 6.88×), `apply_patch→lint`(4.73×) 등 전이 rule로 rare 커버율 측정 (EDA #01 §9.5)
- [ ] `상혁` **CUE 정규식 확장 ablation** — au 어휘 `typecheck/요약/됐다/다시`를 `_explore_cues`에 추가 시 rare F1 변화 (노이즈 감사 축C 시사)
- [ ] `상혁/경모` **DACON 게시판 prefix 의미 문의** — `sess_sim_` vs `sess_au_`가 뭔지, test 30k의 origin이 sim/au/혼재인지 공식 확인
- [ ] `용욱` **유사 tool-calling 벤치 데이터 조사** — pretraining/증강 재료 (상혁 "알아오세요")

### ✅ 완료 (Done)

- [x] `건우/claude` EDA #01 — 클래스분포·prefix(sim/au)·step편향·turn_index·history 전이매트릭스 → `notebooks/01_eda_findings.md`
- [x] `건우/claude` 모델 리서치 — 다국어 encoder/causal LM 후보 Tier 분류, 제약(tf 4.46.3, ≤1GB, ≥50 smp/s) → `notebooks/02_model_research.md`
- [x] `건우/claude` 데이터 노이즈 감사 — 축A(토큰) / 축B(step gap) / 축C(sim vs au) → `notebooks/03_data_noise_audit.md`, `data/processed/noise_candidates.csv`
- [x] `건우/claude` **데이터 leak 검증** — 세션 안 스텝K 라벨 = 스텝M(M>K) history 안 값 (231,664쌍 100% 일치). 로컬 test 5샘플 100% lookup 성공 → `notebooks/04_data_leak_finding.md`
- [x] `건우/claude` **CV 세팅** — `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`. 상혁 val과 100% 정렬 → `notebooks/05_cv_setup.md`, `src/cv.py`, `data/processed/cv_splits/fold_assignments.csv`
- [x] `건우/claude` **baseline OOF (TF-IDF+LogReg 10-fold)** — 전체 Macro-F1 0.4312 ± 0.0069 (fold별 0.42~0.45). respond_only 0.998, write_file 0.976 극단 편향, glob/list 0.19 최저 → `notebooks/06_baseline_oof_result.md`, `data/processed/baseline_oof.{csv,_probs.npz}`
- [x] `건우/claude` 팀 카톡 요약 → `team_chat/SUMMARY.md`
- [x] `건우/claude` 상혁 v01 코드 아카이브 → `team_ref/sanghyuk_v01_lb0.77585/`
- [x] `건우/claude` 공유 대시보드 신설 → `STATUS.md` (이 파일)

### ⛔ 블록 / 대기 (Blocked)

- [ ] `건우` **상혁 추론 `script.py` + model weight(.safetensors) + teacher logits npz 수급** — 앙상블·재현에 필요, 아직 미수급 (팀 카톡 07-02). 확보 전까지 상혁 코드 완전 재현 불가
- [ ] **prefix 의미 확정** — 게시판 답변 대기 중이면 au 취급 결정(exp003) 방향은 잠정으로 진행

---

## 🧪 실험 트래커

> 새 실험은 착수 **전에** `📋 계획` 행부터 추가. `experiments/{exp_id}/`에 config+log+metrics 저장.
> rare-F1은 `web_search / write_file / lint_or_typecheck` 순. val split은 상혁 규약(동일 val_id 순서 `val_probs.npz`) 준수 → 앙상블 재료화.

**상태 범례**: `📋 계획` · `🔄 학습중` · `✅ 완료` · `❌ 실패/폐기` · `⭐ 앙상블 채택`

| exp_id | 모델 / 방법 | 조건 | 학습데이터 | val Macro-F1 | 서버 LB | rare-F1 (web/write/lint) | 상태 | 산출물 경로 | 비고 |
|---|---|---|---|---|---|---|---|---|---|
| baseline_oof | TF-IDF + LogReg | 공식 baseline pkl 파라미터 재현, 10-fold OOF | full 70k | **0.4312 ± 0.0069** | — | 0.2483 / 0.9759 / 0.2603 | ✅ 완료 | `data/processed/baseline_oof.{csv,_probs.npz}` | 계측기 신뢰 확인용. respond_only 0.998, write_file 0.976 극단. glob/list 0.19 최저 |
| team_mdeberta_ko_cls_3ep | mDeBERTa-v3-base CLS pool | 3 epoch, KD/AWP 없음 | full 70k | — | **0.6954392568** | — | ✅ 완료(팀 실측) | 팀원 공유 (경로 미확인) | 07-02 팀에서 공유. 서버 실측 점수 — 우리 OOF 0.4312 → mDeBERTa 서버 0.6954 (+26%p) |
| ref_sanghyuk_v01 | mDeBERTa-v3-base + KD + AWP | full pipeline | full 70k | — | **0.77585** | — | ✅ 완료(팀) | `team_ref/sanghyuk_v01_lb0.77585/` | 팀 최고 4위. 재현 검증 pending(weight 미수급) |
| exp001 | **klue/roberta-large + 상혁 serialize** | **4 epoch, CLS pool, weighted CE, thresh tune, group split, GPU 5** | full 70k | 🔄 학습중 | — | — | 🔄 학습중 | `experiments/exp001_klue_large/` | 한국어 SOTA(337M). 상혁 mDeBERTa와 diverse. 07-03 시작, background id bozqyss1r |
| exp001_qwen | Qwen2.5-0.5B + LoRA head | 4 epoch, 3090 | full 70k | — | — | — | 📋 계획 | `experiments/exp001_qwen05_lora/` | Tier1 대안. causal LM 완전 diverse. exp001 결과 본 뒤 |
| exp002 | multilingual-e5-base + head | — | full 70k | — | — | — | 📋 계획 | `experiments/exp002_e5base/` | 임베딩 objective diverse, 앙상블 |
| exp003a | mDeBERTa (상혁 코드) | au **그대로** | full 70k | — | — | — | 📋 계획 | `experiments/exp003_au_ablation/a/` | au 취급 3조건 비교 baseline |
| exp003b | mDeBERTa (상혁 코드) | au **downweight 0.5** | 70k, w=0.5 | — | — | — | 📋 계획 | `experiments/exp003_au_ablation/b/` | noise_candidates.csv 활용 |
| exp003c | mDeBERTa (상혁 코드) | au **제외** | sim 64,975 | — | — | — | 📋 계획 | `experiments/exp003_au_ablation/c/` | test가 sim이면 최적 가설 |
| exp_leak_lookup | Lookup-only (모델 없음) | 세션내 lookup + 실패시 majority | full 70k | — | — | — | 📋 계획 (팀 승인 대기) | `submissions/vXXX_lookup.zip` | data leak 서버 검증용. 1회 제출로 서버 30k leak률 역산 |

*표는 실험이 늘면 계속 추가. 리더보드 제출본은 `submissions/v###_*.zip`으로 버전 매핑.*

---

## 🔗 빠른 참조

| 항목 | 값 |
|---|---|
| 개발 서버 | 3090 (Ampere, bf16, 24GB) — conda env `aichallenge` |
| 제출 제약 | submit.zip ≤ 1GB · 추론 ≤ 10분 · 오프라인 · T4 16GB · **tf==4.46.3 고정** |
| 처리량 요구 | 30,000건 / 10분 = **≥ 50 samples/sec** (배치 필수) |
| 평가지표 | Macro-F1 (14 클래스, imbalance 8.8:1) |
| 데이터 | train 70k (`data/raw/train.jsonl` + `train_labels.csv`), prefix `sess_sim_` 93% / `sess_au_` 7% |
| 노이즈 후보 | `data/processed/noise_candidates.csv` (7,073 rows / 6,784 uid) |
| 대회 링크 | https://dacon.io/competitions/official/236694/overview/description |
