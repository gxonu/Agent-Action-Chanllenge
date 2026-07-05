# AI-Action-Chanllenge — CLAUDE.md
> **2026 AI·SW중심대학 디지털 경진대회: AI부문** 참가 작업 공간
> 참가 팀: **인하대 5인 팀** (팀명: `oh-my-sinsa` 추정)
> 사용자 역할: 김건우 — **모델 담당** (팀원 중 1명)
> 대회 링크: https://dacon.io/competitions/official/236694/overview/description

---

## 🏆 팀 컨텍스트

**리더보드 (2026-07-02 기준)**: **2위 0.77585** (1위 HallaAI 0.77975, gap 0.0039)

**팀원 역할 분담**
| 팀원 | 담당 |
|---|---|
| 용욱 | 학습 전략 (데이터 split, 학습 방법) |
| 상혁 | **모델 distillation (GLM-2.5 / Claude / qwen2.5) — 현 2위 점수 제작자** |
| 경모 | 데이터 증강 |
| **김건우 (사용자)** | **모델쪽 쓸만한거 test** (현재 데이터 분석부터 진입) |
| 규중 | 데이터 분석 |

**사용자 방향성**
- 상혁이 이미 distillation으로 top-tier 확보 → 사용자는 **diverse한 방향**으로 접근
  - 상혁 코드 완벽 이해 → 개선 여지 진단
  - EDA로 rare class(`web_search` 1.8%, `write_file` 2.1%) 실패 패턴 파악
  - Distillation 외 별개 pretrained (xlm-roberta, bge-m3, ModernBERT 등) or LightGBM+rich feature 등으로 앙상블 재료 or 단독 개선안 제공

**팀원 코드 관리**: `team_ref/` 참조. read-only 스냅샷 원칙, 버전 태그 규칙, `NOTES.md` 템플릿은 `team_ref/README.md` 참조.

**팀 대화 로그**: `team_chat/YYYY-MM-DD.txt` 형식으로 카톡 대화 export 저장. 사용자가 주기적으로 업로드. **민감정보 포함 가능 — `.gitignore`에 이미 등록, 절대 커밋 X**. Claude는 세션 컨텍스트 파악 시 참조 (예: "팀에서 X 얘기 나왔나?" 시 grep).

---

## 1. 대회 한줄 요약

**AI Agent 세션 상태 → 다음 행동 14클래스 예측** (Macro-F1, T4 16GB, 1GB/10분/오프라인, 코드+모델 submit.zip 제출)

상세 문서는 [`docs/`](./docs/)에 정리됨:
- [docs/README.md](./docs/README.md) — 대회 한눈에 보기
- [docs/overview.md](./docs/overview.md) — 개요·목적·문제 설명
- [docs/rules.md](./docs/rules.md) — 규칙
- [docs/data.md](./docs/data.md) — 데이터 구조, 14클래스
- [docs/schedule.md](./docs/schedule.md) — 일정
- [docs/submission.md](./docs/submission.md) — 제출 형식
- [docs/board.md](./docs/board.md) — 게시판·베이스라인 링크

---

## 2. 핵심 일정 (2026)

- **07.01 10:00** — 예선 시작 ✅
- **07.15 10:00** — 예선 종료 ⏳ **← 마감**
- **07.17** — 예선 평가
- **07.20 10:00** — 본선 진출 후보팀 자료 제출 (해당 시)
- **07.27 14:00** — 예선 결과 발표
- **08.11** — 본선 발표평가 및 시상식

**작업 시작일: 2026-07-02 (D-13)**

---

## 3. 확정 결정사항

- **개발 서버**: 3090 (Ampere, bf16 지원, 24GB 여유)
  - 2080Ti는 target T4(16GB)보다 좁아 실험 제약 큼 → 이전 결정
  - 최종 submit 전 2080Ti에서 실행 검증 (11GB 통과 시 T4 16GB 확정 통과)
  - H200/A100은 이 대회에서 낭비 (1GB 모델 상한 때문에 큰 GPU 필요 없음)
- **Conda env**: `aichallenge` (fliption/video 등과 독립)
- **평가지표**: Macro-F1 (14 클래스)
- **제출 제약**: `submit.zip ≤ 1GB` / 추론 ≤ 10분 / 오프라인 / T4 16GB
- **평가 데이터**: 30,000건 → 10분 안에 추론 = **50 samples/sec 이상 필요** → 배치 처리 필수
- **학습 데이터**: 70,000건 (JSONL + labels CSV)

---

## 4. 폴더 구조 컨벤션

```
AI-Action-Chanllenge/
├── CLAUDE.md               # 이 파일 (불변 규칙·컨텍스트)
├── STATUS.md               # ★ 공유 진행상황 대시보드 (세션 간 TODO·실험 트래커). 작업 전후로 갱신
├── docs/                   # 대회 공식 문서 정리
├── data/
│   ├── raw/                # open.zip 원본 (read-only)
│   ├── processed/          # 전처리 결과, split, cache
│   └── external/           # 외부 데이터 (사용 시)
├── baseline/               # DACON 공식 baseline (TF-IDF+LogReg)
├── notebooks/              # EDA, 분석
├── src/
│   ├── data.py             # 로더, 전처리
│   ├── features.py         # 피처 엔지니어링
│   ├── models/             # 모델 클래스
│   ├── train.py            # 학습 진입점
│   └── inference.py        # 추론 (submit용 script.py 원형)
├── experiments/            # exp_name별 config/model/log/metrics
│   └── exp001_baseline/
├── submissions/            # 실제 제출한 zip 버전 관리
│   └── v001_baseline.zip
└── scripts/                # utility (zip 패키징, offline 검증)
```

**원칙:**
- `data/raw/`는 read-only. 절대 수정 금지
- 실험은 `experiments/{exp_name}/`에 config + 모델 + 로그 + metrics를 세트로 저장 (재현성)
- 제출 zip은 `submissions/`에 버전별 보관해 리더보드 점수 매핑

---

## 5. 로드맵

| Phase | 기간 | 상태 | 산출물 |
|-------|------|------|--------|
| **P1. 셋업** | Day 1 (07.02) | 🟡 진행중 | 3090 이전, `aichallenge` env, 데이터 배치 |
| **P2. 베이스라인 재현** | Day 1–2 | ⏳ | DACON TF-IDF+LogReg 제출, 첫 리더보드 점수 |
| **P3. EDA** | Day 2–3 | ⏳ | 14클래스 분포, 필드별 신호, class-wise F1 진단 |
| **P4. 후보 병렬 실험** | Day 3–6 | ⏳ | LightGBM+피처 / DistilBERT / 소형 임베딩+MLP / 앙상블 |
| **P5. 승자 튜닝** | Day 6–10 | ⏳ | HP 서치, class imbalance 대응 |
| **P6. 앙상블·최종 검증** | Day 10–13 | ⏳ | T4 시뮬(2080Ti), 1GB/10분 검증, 최종 submit |

---

## 6. 다음 액션 (P1 진행 중)

1. **3090 서버 이전** — 사용자가 SSH 이동 (NAS 공유이므로 파일은 그대로)
2. **`aichallenge` conda env 생성** — Python, PyTorch(3090이므로 CUDA 11.8+/bf16 활용), transformers, sklearn, lightgbm 등
3. **폴더 스켈레톤 생성** — `data/`, `baseline/`, `notebooks/`, `src/`, `experiments/`, `submissions/`, `scripts/`
4. **데이터 배치** — 사용자가 다운받은 `open.zip`을 `data/raw/`에 압축 해제
5. **baseline 압축 해제 + 실행** — `baseline_submit.zip`을 `baseline/`에 풀고 학습→추론 pipeline 이해
6. **첫 제출** — 리더보드 baseline 점수 확보

---

## 7. 접근 후보 (P4용, 잠정)

**모델 후보 (1GB 제한 안):**
- **(a) LightGBM + 확장 피처** — session_meta, workspace, history 통계 피처 + TF-IDF. 오프라인 100% 안전, 초고속 추론
- **(b) DistilBERT-multilingual 파인튜닝** — current_prompt + history 요약 → 14클래스. 260MB, 배치 추론 빠름
- **(c) 소형 임베딩 + MLP head** — 프리트레인 임베딩(고정) → MLP. 대안 견제용
- **(d) 앙상블** — (a) + (b) 로짓 가중 평균

**Macro-F1 대응:**
- Class imbalance는 EDA 후 결정 (class weight / focal loss / 오버샘플링)
- 희귀 클래스별 F1 개선 여부를 실험 판단 기준으로 삼음

---

## 8. 상위 CLAUDE.md 상속

이 대회 작업은 `/home/nas4_user/kinamkim/Repos/geonwoo/CLAUDE.md` (Fliption 프로젝트 규칙)의 **6번 코딩 규칙을 그대로 상속**한다:
- conda `base` 환경 절대 건드리지 않기
- 새 파일 만들기 전 확인
- 학습 스크립트 실행 전 확인
- 삭제·rm·mv 실행 전 확인
- GPU 사용 시 gpustat 확인 후 지정
- Python 실행은 반드시 `aichallenge` env activate 상태에서
- 실험 output은 구조적으로 구분
- **작업 완료 시 저장 경로 한 줄 보고**

단, Fliption 관련 서버(H200/A100/A6000, yj-N 시리즈)와 데이터셋(NAVI, MVImgNet) 지시는 **이 대회에는 적용 안 함**. 이 대회는 별도 3090 서버 + DACON 데이터로 진행.

---

## 9. 세션 이어가기 팁

이 파일이 있으면 어느 서버에서든 `AI-Action-Chanllenge/`에서 `claude` 실행 시 컨텍스트가 자동 로드됨. 대화 히스토리(`~/.claude/`)는 로컬 디스크에 있어 서버 간 공유 안 되지만, **불변 규칙·컨텍스트**는 이 파일, **살아있는 진행상황(TODO·실험 상태)** 은 [`STATUS.md`](./STATUS.md)에 있음. CLAUDE.md는 잘 안 바뀌는 것, STATUS.md는 매 세션 갱신되는 것으로 역할 분리.

**여러 세션이 병렬로 도는 중** — 무엇이든 손대기 전에 `STATUS.md`부터 읽어 이미 누가 집었는지 확인하고, 착수하면 해당 항목을 `🔴 진행 중`으로 옮긴 뒤 시작한다. 끝나면 상태·산출물 경로를 갱신한다.

새 세션 시작 시 첫 프롬프트 예: *"STATUS.md 확인하고 다음 항목 이어가자"*
