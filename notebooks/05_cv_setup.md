# CV(교차 검증) 세팅

**작성**: 2026-07-02
**목적**: 팀원 전원이 **같은 val fold**를 써서 실험 결과를 비교 가능하게 함. 앙상블 재료로도 정렬된 val_probs 축적.

---

## 결정한 방식

**상혁 v01(LB 0.77585) 코드와 완전 정렬**:
- `sklearn.model_selection.StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`
- `groups` = 세션 id (id에서 `-step_NN` 잘라낸 것)
- `y` = 14 클래스 라벨 (train_labels.csv의 action)
- **fold_0을 val로 기본 사용**. 나머지 9 fold(fold_1~9)를 train

이렇게 하면 상혁이 만든 val_probs.npz와 정확히 같은 val 샘플을 씀 → 앙상블에서 로짓·확률 그대로 정렬됨.

## 왜 이 방식인가

1. **세션 단위 그룹핑**: 같은 세션의 여러 스텝이 train과 val에 갈리면 누수 발생 (스텝 3의 history에 스텝 1, 2 정보가 있음). Group split이 이걸 막아줌.
2. **Stratify**: 14 클래스 분포를 각 fold에 균등하게. 특히 rare 3개(`web_search`, `write_file`, `lint_or_typecheck`)가 fold마다 균등해야 rare F1을 신뢰성 있게 측정 가능.
3. **seed=42**: 상혁 코드 유지. 이 seed와 다르면 val 집합이 완전히 달라져 비교 불가능.
4. **10-fold의 첫 fold만 val 사용**: val_size=10% 확보. 필요하면 다른 fold도 val로 로테이션해서 실험 반복 가능.

## 검증 결과 (요약)

- ✅ **각 fold 클래스 분포 균등**: rare class도 fold별 편차 ±1건 이하 (예: web_search 평균 127.3 ± 0.5)
- ✅ **세션 leak 없음**: 9,429개 세션 중 여러 fold에 걸친 세션 0개
- ✅ **fold 크기 균등**: 각 fold 7,000건 안팎 (최대 편차 2건)
- ✅ **fold_0 val size 정확히 10%**: 7,000건, 946 세션

### 각 fold의 rare class 분포

| 클래스 | fold별 개수 | 평균 ± 표준편차 |
|---|---|---|
| web_search | [128, 127, 128, 127, 127, 127, 128, 127, 127, 127] | 127.3 ± 0.5 |
| write_file | [148, 148, 148, 148, 148, 148, 148, 149, 148, 148] | 148.1 ± 0.3 |
| lint_or_typecheck | [228, 228, 228, 229, 229, 229, 228, 228, 228, 228] | 228.3 ± 0.5 |

**해석**: rare class도 각 fold에 골고루 분산돼서 fold별 Macro-F1이 신뢰 가능.

## 파일 위치·사용법

**코드**: `src/cv.py`
- `compute_fold_assignments()` — fold 할당 계산 (내부용)
- `load_fold_assignments()` — CSV 로드. 없으면 자동 계산
- `get_val_ids(fold=0)` — 지정 fold의 id 목록 (val 용)
- `get_train_ids(fold=0)` — 나머지 id (train 용)

**할당 CSV**: `data/processed/cv_splits/fold_assignments.csv`
- 컬럼: `id, session, action, fold`
- 팀원 누구든 이 파일 읽어 같은 val 집합 재현 가능

### 사용 예 (Python)

```python
from src.cv import get_val_ids, get_train_ids

val_ids = set(get_val_ids(fold=0))
train_ids = set(get_train_ids(fold=0))

# jsonl 로드 후 필터
train_samples = [obj for obj in samples if obj['id'] in train_ids]
val_samples = [obj for obj in samples if obj['id'] in val_ids]
```

### CSV 직접 로드 (다른 언어 or shell)

```python
import pandas as pd
df = pd.read_csv('data/processed/cv_splits/fold_assignments.csv')
val_ids = df[df.fold == 0]['id'].tolist()
```

## 상혁 val 정렬 검증 완료 ✅

상혁 코드 그대로 재현해서 얻은 val 집합과 우리 `fold_0` val 집합을 정확히 비교:
- 상혁 val 7,000건 vs 우리 fold_0 val 7,000건
- **완전 일치 (다른 것 0건)**
- 즉 나중에 상혁이 만든 `val_probs.npz`(같은 val 순서)와 우리 결과를 그대로 stacking 가능

## Prefix (sim/au) 편향 검증 ✅ 없음

각 fold의 sim/au 비율이 원본과 거의 같음:
- sim 평균 92.82%, 표준편차 0.75%p (원본 92.82%)
- au 평균 7.18%, 표준편차 0.75%p (원본 7.18%)
- StratifiedGroupKFold의 stratify가 클래스 기준이지만, 결과적으로 prefix도 균등하게 분포됨

## 두 가지 leak 개념 구분 (혼동 주의)

**Type 1 — 전통적 CV leak (train → val)**
- 같은 세션이 train/val에 걸치면 발생
- 우리 CV는 세션 단위 group split으로 완벽 방지 ✅
- 이게 CV 설계 목적이었음

**Type 2 — Cross-sample lookup (batch inference trick)**
- 같은 세션의 여러 스텝이 한 fold에 함께 있으면, 그 sample들끼리 답 lookup 가능
- 예: val fold_0의 스텝 6과 스텝 7이 같이 있으면 스텝 7의 history에 스텝 6 답 있음
- 이건 CV design 문제가 아니라 **데이터·태스크의 성질**
- Val에서 시뮬해보니 val 7,000건 중 6,054건 (86.5%) 이 방식으로 답 확정 가능

### 이게 CV·평가에 뭘 의미하나

**모델 혼자 예측 (전통적 val Macro-F1)** — CV의 원래 목적
```python
for sample in val_samples:
    pred = model(sample)  # sample 하나씩 독립적으로
```
- 각 sample의 자기 history만 봄. 다른 val sample은 못 봄
- **이게 CV가 재는 순수 모델 실력. 우리 개선 목표.**
- 여기에 86.49%라는 숫자는 안 나옴

**서버 script.py에서 예측 (batch trick 가능)**
```python
for target in test:
    if lookup(target, all_test_samples):  # 다른 test 다 봄
        pred = lookup_answer
    else:
        pred = model(target)
```
- 다른 test sample들의 history를 볼 수 있음 (batch level)
- 서버 30k에서도 leak 통한다면 hybrid가 유효
- **다만 이게 서버에서 실제 통할지는 별개 문제** (지금까지 미확인)

### 결론

- CV는 **모델 val Macro-F1**을 재는 걸 목적으로 함. 이게 순수 모델 실력.
- Hybrid val Macro-F1은 script.py 만들어 시뮬할 때 별도로 계산
- 두 지표 다 유용하지만 서로 다른 걸 측정

## 재현 방법

```bash
cd AI-Action-Chanllenge
/home/nas4_user/kinamkim/anaconda3/envs/aichallenge/bin/python -m src.cv
```

결과가 `data/processed/cv_splits/fold_assignments.csv`로 저장. seed 고정이라 몇 번 돌려도 같은 결과.

## 상혁 코드와의 상호 운용

상혁 `train.py`는 자체적으로 split을 계산하지만, 파라미터가 동일하니 **결과는 정확히 같음**. 검증:

```python
# 상혁 코드
from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
tr_idx, va_idx = next(sgkf.split(np.arange(len(texts)), y, groups))

# 우리 코드
fold_of[val_idx] = fold_idx  # for fold_idx, (_, val_idx) in enumerate(...)
# fold_of == 0 을 val로 → 상혁의 va_idx와 동일
```

즉 상혁이 만든 `val_probs.npz`를 우리 실험 결과와 앙상블할 때 id 순서가 이미 정렬돼 있음.

---

## 팀 카톡 공유용 요약

```
[CV 세팅 완료]

방식
- StratifiedGroupKFold(n_splits=10, shuffle=True, seed=42)
- 세션 id로 그룹 (누수 방지)
- 클래스로 stratify (14 클래스 균등)
- fold_0 = val 기본값 (10% = 7,000건)
- 상혁이형 train.py 코드와 완전 동일 파라미터

산출
- src/cv.py (스크립트)
- data/processed/cv_splits/fold_assignments.csv (id, session, action, fold)
  → 팀원 누구든 이 파일 읽어 같은 val 재현

검증 통과
- 세션 leak 없음 (같은 세션이 여러 fold에 안 갈림)
- 클래스 분포 fold별 균등 (rare class도 편차 ±1건 이하)
- fold_0 크기 = 7,000건 = 10%

사용법
  from src.cv import get_val_ids, get_train_ids
  val_ids = set(get_val_ids(fold=0))

이제 앙상블·비교 실험 시 모두 이 fold_0 val 씀. 상혁이형 val_probs.npz와도 id 정렬됨.

추가 발견
- 상혁 코드 그대로 재현해서 얻은 val 7,000건 vs 우리 fold_0 = 완벽 일치. 앙상블 정렬 확실
- prefix(sim/au) 편향 없음. fold별 sim 비율 표준편차 0.75%p

주의점 (혼동 방지)
- CV는 세션 단위 group split로 train→val leak 완벽 방지 (설계 목적 그대로)
- 다른 이슈로 cross-sample lookup(batch trick)이 가능함이 별개로 발견됨
  - Val 7,000건 중 6,054건(86.5%)이 batch 방식으로 답 확정 가능
  - 이건 CV 문제 아니라 데이터·태스크의 성질
  - 서버 script.py에서 hybrid로 활용 가능 (통할지는 미확인)
- 즉 CV로 재는 모델 val Macro-F1 = 순수 모델 실력. 이걸 개선 목표로 삼음
```
