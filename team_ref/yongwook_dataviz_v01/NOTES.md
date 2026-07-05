# yongwook v01 — train.jsonl HTML 뷰어

## 메타
- **받은 날짜**: 2026-07-02
- **저자**: 용욱
- **원본 위치**: `team_ref/_incoming/` (SCP 업로드)
- **리더보드 관련성**: 없음 (분석·시각화 유틸)
- **받은 파일**:
  - `train_viewer_small.html` (2.1 MB, **층화 샘플 1,400건** = 14 클래스 × 100)
  - `train_viewer_large.html` (97 MB, **전체 70,000건** embed)

## 뷰어 기능 요약

**단일 HTML 파일** — 데이터를 embed된 `DATA` JS 배열로 담고, 브라우저에서만 열면 즉시 렌더. 서버 불필요.

**UI 구성**
- **좌측 사이드바** (320px 고정)
  - 검색 input: prompt / history / id 통합 텍스트 검색
  - 클래스 드롭다운: 14 클래스로 필터
  - 리스트: 매칭된 샘플 스크롤. 각 항목에 id + 클래스 태그
- **우측 메인 패널**
  - 현재 선택된 샘플의 상세
  - `session_meta` KV 그리드 + `language_mix` 스택 바 차트
  - `open_files` chip 리스트
  - `history` 타임라인 (user 발화 왼쪽·파랑, assistant_action 오른쪽·노랑 스타일링)
  - `current_prompt` 강조 박스
  - `_target` (정답 라벨) — **train.jsonl에는 없는 필드**로 별도 조인해 embed
- **키보드 네비**: `j` (다음 샘플) / `k` (이전 샘플) — vim 스타일

**데이터 스키마 (embed된 `DATA` 배열의 각 원소)**
- 원본 train.jsonl 필드 (`id`, `session_meta`, `history`, `current_prompt`) 모두 유지
- **추가 필드 2개**:
  - `_target`: 정답 액션 클래스 (train_labels.csv에서 조인)
  - `_session`: id에서 `-step_NN` 잘라낸 세션 id (예: `sess_sim_20260522_023347`)

## 재현 방법 (뷰어 열기)

### A. 로컬 다운로드 후 브라우저
```bash
# 로컬에서 (SCP는 로컬 machine에서 실행)
scp <server>:/home/nas4_user/kinamkim/Repos/geonwoo/AI-Action-Chanllenge/team_ref/yongwook_dataviz_v01/train_viewer_small.html .
open train_viewer_small.html   # 또는 브라우저에 드래그
```
- **small (2.1MB)**: 즉시 렌더링, 다운로드 순식간
- **large (97MB)**: 브라우저 로드에 몇 초~10초 걸림. 검색은 클라이언트 JS라 반응 느릴 수 있음

### B. 서버에서 SSH tunnel + http.server (다운로드 안 하고 보기)
```bash
# 서버에서
cd /home/nas4_user/kinamkim/Repos/geonwoo/AI-Action-Chanllenge/team_ref/yongwook_dataviz_v01
python -m http.server 8080

# 로컬 machine에서 (다른 터미널)
ssh -L 8080:localhost:8080 <server-alias>
# 브라우저에서 http://localhost:8080/train_viewer_small.html 열기
```

## 파악한 핵심 아이디어

1. **`_target` embed로 라벨 실시간 확인** — jsonl+CSV 조인 필요 없이 뷰어 열자마자 정답 보임. 라벨별 대표 케이스 훑기 편함
2. **`_session` embed** — 같은 세션의 여러 step이 흩어져 있는 걸 시각화하기 좋음. 세션 그룹 분리 필요성을 직관적으로 이해 가능
3. **층화 샘플 1,400 (small)**: rare class(`web_search`, `write_file`)도 100건씩 확보 → **rare class 실패 패턴 훑기에 최적**. 대회 데이터 원본은 imbalance라 rare 몇 개밖에 안 보임
4. **history 타임라인 UI**: user↔action alternation을 시각적으로 재구성. `role`, `name`, `args`, `result_summary` 렌더링 규약이 상혁이형 `_stringify_history` 파싱 로직과 동일 가정에 서 있음 → 뷰어로 눈으로 확인한 것을 그대로 학습 입력으로 신뢰 가능

## 활용 방안 (사용자 EDA에)

1. **rare class 오답 케이스 진단**: baseline 예측을 `_target`과 비교해 오답 id를 뷰어에서 검색 → 어떤 문맥에서 실패하는지 눈으로 파악
2. **`sess_au_` vs `sess_sim_` 차이 확인**: id 검색으로 `sess_au_` 필터 후 clsss dropdown 훑기 → 시뮬레이션 데이터와 다른 특성 있는지 시각적 스캔
3. **경계 클래스 학습**: `read_file`/`grep_search`/`list_directory`/`glob_pattern` 4개를 번갈아 필터 → 판단 근거가 되는 prompt 패턴 눈에 익히기 (상혁 CUE 정규식 개선안 발굴)
4. **history 활용도 감 잡기**: 짧은 prompt인데 정답이 잘 나오는 케이스 = history에 힌트 많음. 반대는 prompt-only 신호가 강한 케이스. 어느 케이스가 얼마나 되는지 정량화 가치

## 개선 여지 hypothesis (뷰어 자체)

- baseline/상혁 모델 예측값을 별도 필드(`_pred_baseline`, `_pred_sanghyuk`)로 넣으면 **정답 vs 예측 diff** 뷰어로 즉시 오답 분석 가능
- confusion matrix에서 특정 (true, pred) 셀 클릭 → 해당 오답 case로 자동 필터되면 이상적. 지금은 텍스트 검색으로만 접근 가능
- 클래스 분포 요약 상단 배너 (전체 vs 필터링 후) 있으면 좋음

## 미수급 자료

- 뷰어 생성 스크립트 (Python?) — 뷰어에 예측 필드 추가하려면 원본 생성 코드 필요. 용욱이한테 요청 가치 있음
