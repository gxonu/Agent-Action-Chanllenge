# 상혁 v02 — mmbert_base LB 0.77585 (제출 zip)

## 메타
- **받은 날짜**: 2026-07-06
- **저자**: 상혁
- **원본**: `submit.zip` (572MB, 건우가 카톡으로 수급)
- **리더보드**: 0.77585 (팀 최종 선택 제출, 07-02)
- **받은 것**: `extracted/` = 제출 zip 통째 (`script.py`, `requirements.txt`, `model/`)

## 실제 제출 아키텍처 (script.py 기준 = 진짜 동작)
- **plain `AutoModelForSequenceClassification`** (mmbert-base + linear CLS head). **fusion 아님.**
- 입력: `serialize()` → 텍스트 → mmbert CLS → softmax. `model/common.py::serialize`가 source of truth.
- **serialize_config**: `add_open_files=true`, 나머지(symbol_cues/prev_cue/langmix) **전부 off**.
  → 텍스트 = `[PROMPT] q/short [HISTORY] [META(open_files)]`. **우리 `exp006`/`sanghyuk_v01` serialize와 바이트 동일 (3000/3000 검증)**.
- **ensemble.json**: `method=mean`, `tags=[mmbert_open]`(모델 1개), **`bias` 전부 0 (이 제출엔 threshold 튜닝 없음)**, `max_length=512`.
- **requirements.txt**: `transformers==4.48.3` (주석: "서버 기본 4.46.3은 modernbert 미지원 → 업글"). **tf 업그레이드 실증 재확인**.
- 추론 최적화: 전체 토큰화 후 **길이순 정렬 버킷팅**(batch-max 패딩)으로 속도 확보. BATCH_SIZE=256.
- weight: `model/mmbert_open/model.safetensors` **615MB (fp16)** < 1GB.

## dead code (제출에 미사용, 실험 잔재)
- `common.py::_prev_cue(history)` — 직전 action **결과상태** discrete cue(cold/grep_empty/glob_empty/test_fail/bash_fail…). `add_prev_cue` flag로 켜지만 **이 제출은 off**. 반응문법 아이디어라 향후 실험 가치.
- `common.py::extract_features()` — 24-dim 구조 feature(nopen, ci, last_act one-hot, n_inspect…) fusion 헤드용. **이 제출은 미사용**(script.py가 안 부름).

## 재현/활용
- **CV baseline 재현**: 우리 `exp006`(plain mmbert + 이 serialize, held-out our fold)이 곧 이것. **⚠️ 상혁 weight 자체는 CV baseline 불가** — 전체 70k 학습이라 우리 val fold 오염(평가 시 뻥튀기). 깨끗한 delta엔 exp006 held-out run 필요.
- **최종 제출 앙상블 멤버로는 유효** — 전체 70k 학습이라 실제 30k test엔 오염 없음. 우리 diverse 모델과 stacking 재료.
- **제출 패키징 레퍼런스**: script.py 구조(ensemble.json + 길이버킷 추론 + offline)를 우리 제출에 재사용.

## 미수급
- [ ] **held-out(seed42 group split) `val_probs.npz`** — 있으면 exp006 재학습 없이 serialize_v2 델타 즉시 확정. 상혁에게 별도 요청 가치.
- [ ] 0.77585가 정확히 이 zip인지, bias=0 버전이 그 점수인지 확인 (제출 이력상 bias 버전도 있었음).
