# 규중 v01 — RL 입력-편집 에이전트 (LB 미제출, 로컬 eval +4%)

## 메타
- **받은 날짜**: 2026-07-05
- **저자**: 규중
- **원본 위치**: 카톡 공유 (07-04~05 대화). 노션 "규중:데이터 분석" 하위에도 코드 있음
- **리더보드 점수**: 없음. 카톡상 **base 대비 로컬 eval +4%** (07-05 "입력 바꾸는거 base보다 4퍼 더 높게"). **서버 LB 미검증**
- **받은 파일**: `rl_input_edit.ipynb` (Kaggle 노트북, 27셀). 안에 `config.py`/`common.py`/`train.py`/`exemplar_bank.py`/`rl_edit.py`를 `%%writefile`로 생성

## 접근 요약

**핵심 아이디어**: 메인 분류기는 **고정(freeze)**, 헷갈리는 입력을 **추론 시점에 편집**해서 프로즌 분류기가 더 잘 맞추게 만드는 별도 정책망(PolicyNet)을 RL로 학습. 상혁 "다 concat" 방식과 달리 **입력 자체를 손본다**.

- **메인 분류기**: `google/t5gemma-2-270m-270m` 인코더 + LoRA + attention pooling.
  - ⚠️ **`transformers>=4.50.0` 필요** (T5Gemma2 arch). 서버 기본 4.46.3 불가 → 업그레이드 전제 (건우 mmbert 제출로 서버 tf 업그레이드 실증됨 → 제출 가능성은 있음).
  - 안정화: bf16(fp16 NaN 이력), eager attention 강제(sdpa 오류 회피), vision_tower/multi_modal_projector 제거(텍스트 인코더만), LoRA dropout=0.0.
  - `MAX_LENGTH=256` (메모리 절약).
- **RL 입력-편집 (탐색 4클래스 전용)**: `read_file/grep_search/list_directory/glob_pattern` — 경계가 모호해 오답 많은 4개.
  1. **exemplar bank** (`exemplar_bank.py`): 탐색 4클래스 중 **kNN purity 높은(임베딩 공간에서 잘 분리되는)** 샘플만 골라 `(클래스, history 길이 L)` 버킷별 저장. ⚠️ t-SNE 눈대중 금지, **원본 고차원 pooled 임베딩 kNN purity로 정량 판정**.
  2. **PolicyNet** (`rl_edit.py`): 헷갈리는 샘플의 history를 **마지막 step부터 역순으로** 훑으며, 그 위치에 **어떤 (도너 클래스, exemplar) pair를 끼워넣을지**를 정책이 선택. 위치는 고정(정책이 안 고름), 내용만 선택.
  3. **학습**: 지도학습(정답 swap 미리 정함) 아님. 최종 분류 정오답 + margin 개선분을 보상으로 **REINFORCE(+baseline)**. 메인 분류기 파라미터는 전혀 안 건드림.
  4. **효율**: 샘플 1개씩 forward 안 함. 매 라운드 "아직 안 끝난 에피소드 전체"를 한 배치로 프로즌 인코더에 통과 → forward 호출이 N×12가 아니라 (배치수)×12 수준.
- **학습 데이터**: full 70k에서 탐색 4클래스만 필터, val split은 메인 모델 학습 때 것 재사용.
- **평가**: 탐색 4클래스만 편집, 나머지 10클래스는 원본 그대로 → 전체 14클래스 val acc before/after 비교.

## 노트북 구성 (셀)
- [2] `pip install "transformers>=4.50.0" ... peft torchao`
- [4] config.py — `MODEL_NAME=google/t5gemma-2-270m-270m`, `MAX_LENGTH=256`, LoRA 설정
- [6] common.py — 상혁 serialize 계열 + **`get_history_pairs`(시간순 pair 리스트 반환)**, `serialize_from_pairs`(편집된 pair로 재직렬화). serialize 결과는 상혁 것과 100% 동일 유지
- [8][9] train.py — `T5GemmaEncoderClassifier`, `_force_eager_attention` 등 (노트북에선 실행 안 함, 클래스 재사용용)
- [17] exemplar_bank.py — `extract_pooled_embeddings`, `knn_purity`, `build_exemplar_bank`
- [19] rl_edit.py — `train_policy_vectorized`, `evaluate_policy_vectorized`, PolicyNet
- [23] 재로드 버그 진단 (LoRA merge 방식 4가지 A/B/C/D 비교)
- [25][26] 통합 실행 + before/after 비교

## 🔴 보안 경고 (조치 필요)
- **`rl_input_edit.ipynb` cell 21에 실 HuggingFace 토큰 하드코딩** (`hf_...`).
- 이 파일은 현재 **git 추적 대상**. commit/push 시 private repo에 토큰 유출.
- **권고**: (1) 규중이 해당 토큰 **rotate(폐기·재발급)**, (2) 커밋 전 노트북에서 토큰 **스크럽**하거나 `team_ref/**/*.ipynb`를 `.gitignore`에. (아직 미커밋이라 지금 조치하면 안전)

## 재현 방법
- Kaggle 환경 전제 (`/kaggle/input/...`, `/kaggle/working/...` 경로 하드코딩). 로컬 재현하려면 경로 수정 필요.
- **선행 필요물**: 학습 완료된 `model/` 체크포인트(T5Gemma2 + LoRA + heads.pt + val_probs.npz + backbone_meta.json) — 노트북은 이걸 **가정만** 하고 안 만듦. 즉 재현하려면 먼저 T5Gemma2 분류기를 학습해야 함.
- 재현 검증: **미수행** (체크포인트 미수급). "재현 OK" 미기록.

## 파악한 핵심 아이디어
1. **입력을 손대는 축** — 팀 다수가 "모델/loss/데이터"를 건드릴 때, 규중은 **추론시점 입력 편집**이라는 직교 축. 메인 모델 재학습 불필요 → 어떤 분류기 위에도 얹을 수 있는 후처리 성격.
2. **kNN purity로 exemplar 선별** — "잘 분리되는 샘플"을 정량 기준으로 고르는 게 t-SNE 눈대중보다 신뢰성 높음. 이 유틸(`knn_purity`)은 우리 EDA/오답분석에도 재사용 가치.
3. **탐색 4클래스 타겟팅** — Macro-F1에서 이 4개가 최대 혼동원(상혁·용욱도 지목). 여기만 개선해도 lever.
4. **REINFORCE로 "정답 swap 미정의"** — 어떤 편집이 옳은지 사람이 안 정하고 보상으로 학습. 노이즈 심한 라벨에 강건할 수 있음.

## 개선 여지 / 우리(건우) 통합 hypothesis
- **mmbert_base(내 SOTA) 위에 이식 가능한가?** — 규중 정책망은 프로즌 분류기 + attn-pool 전제. 내 mmbert는 CLS pool → attn-pool 헤드로 바꾸거나 mean-pool로 pooled 벡터 뽑는 경로 추가 필요. **비자명한 작업량**. 별도 실험(exp)으로 스코프.
- **효과 미검증 리스크**: 카톡 +4%는 로컬 eval, LB 미확인. 용욱 언어별 LoRA도 eval↑였으나 LB 0.75(과적합)였음 → **LB 검증 없이 신뢰 금지**. 통합 전 규중에게 "LB 제출해봤나" 확인.
- **의존성**: T5Gemma2는 tf≥4.50. mmbert(ModernBERT)는 tf≥4.48. 통합 시 tf 버전 정합 확인.
