# 경모 validation 분석 v01 — mmbert_open 검증 리포트

## 메타
- **받은 날짜**: 2026-07-11
- **저자**: 경모
- **원본 위치**: 카톡/노션 공유 → `team_ref/validation_report.md` (2026-07-12 이 폴더로 정리)
- **받은 파일**: `validation_report.md` (동봉 CSV들 — validation_all/correct/wrong.csv, class_metrics.csv, confusion_matrix.csv 등 — 은 미수급, 리포트 본문만)
- **리더보드 관련성**: 직접 없음 (검증 분석). 단 **CV-LB 격차 원인 규명에 핵심**

## 접근 요약
- 모델: `mmbert_open` (mmbert + open_files serialize, = 우리 exp006/상혁 계열)
- **Split: `random`, seed 42, val 0.1 (7000)** ← ⚠️ 이 부분이 핵심 이슈 (아래)
- 산출: 전체/클래스별 성능, 오분류 조합, 확신도별 오답/정답 예시 테이블

## 🔴 핵심 검증 결과 (yj-3-S2, 2026-07-11)

### ⚠️ 헤드라인 Macro-F1 0.807은 **누수로 부풀려진 값** — 진짜 아님

- **원인 = Split이 `random`**. 경모 설정(random, seed42, val0.1)을 재현해보니:
  - **val 7000개 중 99.5%가 train에도 존재하는 세션** (같은 세션의 다른 step이 train/val에 갈림)
  - 우리 데이터는 history가 누적구조라, 같은 세션이 train에 있으면 **컨텍스트 누수 = 모델이 "본 세션" 컨닝** → 점수 뻥튀기
- **진짜 점수 = group split 기준 우리 exp006 = 0.767** (상혁 LB 0.77585와 정합)
- **0.807(random CV) − 0.775(LB) = 0.032 = 통째로 누수 = CV-LB 격차의 정체**
- 참고: group split이면 val 세션의 train 존재율 = 0.0% (정상)

### ➡️ 팀 액션 (중요)
- **"0.807 넘었다" 착시 금지.** 모델·코드는 그대로 두고 **split만 `group`(세션 단위)으로 재실행**하면 진짜 CV(~0.767) 나옴.
- **더 큰 함의**: 용욱 "eval 0.8인데 LB 0.75"(07-05 카톡), 규중 "eval +4%"도 **같은 random-split 누수일 가능성** → 팀의 모든 "개선"을 **group split로 재평가**해야 진짜/가짜 구분됨. (남은 기간 삽질 방지)

## ✅ 분석의 "구조/통찰"은 전부 정확 (우리와 일치)
절대 숫자만 누수로 떴을 뿐, 문제 진단은 정확:
- **최하위 4개 = read/grep/list/glob (inspect)** — 우리 4중검증(t-SNE/수치/규칙탐색/실제예시)과 동일 결론
- **오분류 삼각형**: grep→read 246, read→list 197, grep→list 148 — 우리 exp006 confusion과 동일 구조
- **list_directory 과예측** (precision 0.41, "쓰레기통") — 동일
- **bias 적용 +0.000** — 우리 threshold held-out 무효와 정합
- 최상위 respond 1.0 / write 0.99 / edit 0.98 — 동일

## 예시 테이블(리포트 §5~7)의 가치
- **§6 경계 오답**: 전부 inspect, prediction_margin ≈ 0 (예: grep 0.31 vs read 0.30) → **inspect 동전던지기** 실증
- **§7 확신 낮게 맞춤**: 확률 0.28~0.32 → inspect에서 사실상 찍어서 우연히 맞음
- **§5 확신 높은 오답**: edit_file→apply_patch 지배(0.99 확신) → **edit↔apply는 "확신하며 틀림"**(둘이 거의 동의어 라벨). inspect(낮은확신 혼동)와 다른 종류.

## 개선 여지 / 다음
1. **[필수] group split로 재실행** → 진짜 CV 확보 (경모 or 우리가 같은 fold로)
2. 팀 공유 group-fold 파일 하나로 통일 → 모든 실험 누수 착시 차단
3. 이 리포트의 오답 예시들은 "inspect 손댈 수 없음"의 좋은 정성 증거 → 발표자료 활용 가능
