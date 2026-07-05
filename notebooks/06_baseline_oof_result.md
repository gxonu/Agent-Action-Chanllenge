# Baseline OOF 결과 (팩트만)

**작성**: 2026-07-02
**목적**: DACON 공식 baseline(TF-IDF + LogReg)을 우리 CV에 맞춰 10-fold OOF로 학습·평가. 계측기(CV) 신뢰성 확인 + 팀 리더보드와의 gap 계량.

**재현**:
```bash
cd AI-Action-Chanllenge
python -m src.baseline_oof
```

**산출**:
- `data/processed/baseline_oof.csv` — 70,000행 (id, fold, action_true, action_pred)
- `data/processed/baseline_oof_probs.npz` — 14 클래스 확률 매트릭스

---

## 1. 설정

### 파이프라인 (`baseline/model/tfidf_logreg.pkl` 재현)
- `TfidfVectorizer`: analyzer=word, ngram=(1,2), min_df=2, max_features=80,000, sublinear_tf=True, use_idf=True, norm=l2
- `LogisticRegression`: C=2.0, class_weight='balanced', solver=lbfgs, max_iter=500

### 입력 필드
- `current_prompt` 문자열 단독 (baseline과 동일. history/session_meta 미사용)

### CV
- `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` — 세션 단위 group split
- 10-fold 각각 train 63,000 (9-fold) → val 7,000 (1-fold) 예측
- 총 학습 시간: ~13분 (fold당 68~97초)

---

## 2. Fold별 결과

| fold | val Macro-F1 | val 크기 | 학습 시간 |
|---:|---:|---:|---:|
| 0 | 0.4286 | 7,000 | 68.7s |
| 1 | 0.4324 | 7,000 | 86.5s |
| 2 | 0.4263 | 7,000 | 84.2s |
| 3 | 0.4327 | 7,000 | 88.9s |
| 4 | 0.4366 | 7,000 | 96.9s |
| 5 | 0.4344 | 7,001 | 87.7s |
| 6 | 0.4289 | 7,000 | 76.8s |
| 7 | 0.4186 | 7,000 | 75.0s |
| 8 | 0.4262 | 6,999 | 90.3s |
| 9 | 0.4458 | 7,000 | 47.8s |

**평균 ± 표준편차: 0.4310 ± 0.0069**

---

## 3. 전체 OOF 지표 (n=70,000)

| 지표 | 값 |
|---|---:|
| **Macro-F1** | **0.4312** |
| Micro-F1 (=accuracy) | 0.4194 |
| Weighted-F1 | 0.4265 |

---

## 4. 클래스별 F1 (support 기준 정렬)

| 클래스 | F1 | precision | recall | support |
|---|---:|---:|---:|---:|
| edit_file | 0.6334 | 0.6826 | 0.5907 | 11,171 |
| grep_search | 0.2904 | 0.3395 | 0.2536 | 9,912 |
| read_file | 0.3149 | 0.3455 | 0.2893 | 9,257 |
| glob_pattern | 0.1978 | 0.1784 | 0.2218 | 5,284 |
| **respond_only** | **0.9977** | 0.9994 | 0.9959 | 5,178 |
| run_bash | 0.4127 | 0.4401 | 0.3885 | 5,068 |
| apply_patch | 0.3385 | 0.2938 | 0.3993 | 4,823 |
| run_tests | 0.3824 | 0.3933 | 0.3721 | 4,561 |
| list_directory | 0.1869 | 0.1549 | 0.2356 | 4,329 |
| ask_user | 0.4008 | 0.4192 | 0.3839 | 2,701 |
| plan_task | 0.3963 | 0.4164 | 0.3781 | 2,679 |
| lint_or_typecheck | 0.2603 | 0.2227 | 0.3132 | 2,283 |
| **write_file** | **0.9759** | 0.9565 | 0.9959 | 1,481 |
| web_search | 0.2483 | 0.2091 | 0.3056 | 1,273 |

### rare 3개 (Macro-F1 비중 3/14 = 21.4%)

| 클래스 | F1 | support |
|---|---:|---:|
| web_search | 0.2483 | 1,273 |
| write_file | 0.9759 | 1,481 |
| lint_or_typecheck | 0.2603 | 2,283 |

- rare 3 평균 F1: **0.4948**
- common 11 평균 F1: **0.4138**

---

## 5. 주요 혼동 쌍 (top 15)

| 실제 | → 예측 | 건수 | 실제 대비 |
|---|---|---:|---:|
| edit_file | apply_patch | 4,453 | 39.86% |
| apply_patch | edit_file | 2,845 | 58.99% |
| grep_search | read_file | 2,515 | 25.37% |
| grep_search | glob_pattern | 2,319 | 23.40% |
| grep_search | list_directory | 2,309 | 23.29% |
| read_file | grep_search | 2,231 | 24.10% |
| read_file | glob_pattern | 2,062 | 22.28% |
| read_file | list_directory | 2,037 | 22.00% |
| run_bash | run_tests | 1,744 | 34.41% |
| run_tests | run_bash | 1,631 | 35.76% |
| glob_pattern | grep_search | 1,441 | 27.27% |
| glob_pattern | read_file | 1,358 | 25.70% |
| run_bash | lint_or_typecheck | 1,264 | 24.94% |
| glob_pattern | list_directory | 1,172 | 22.18% |
| run_tests | lint_or_typecheck | 1,151 | 25.24% |

---

## 6. 리더보드 대조 (참고 팩트)

| 항목 | 값 |
|---|---:|
| baseline OOF Macro-F1 (실측) | 0.4312 |
| 우리 팀 (oh-my-sinsa, 상혁 mDeBERTa+KD+AWP) | 0.77585 |
| 1위 (SSU퍼루키) | 0.78525 |
| baseline → 우리 팀 gap | +34.47%p |

**미확인**:
- 대회 공식 baseline이 서버에서 어떤 점수 나오는지 (직접 제출 안 함)
- 우리 OOF 값과 서버 baseline 값의 근접 여부 (계측기 신뢰 확인 근거)

---

## 7. 파일 위치

- 학습 코드: `src/baseline_oof.py`
- OOF 예측 (팀 공유): `data/processed/baseline_oof.csv`
- OOF 확률 (앙상블 재료): `data/processed/baseline_oof_probs.npz`
- CV 세팅: `notebooks/05_cv_setup.md`
