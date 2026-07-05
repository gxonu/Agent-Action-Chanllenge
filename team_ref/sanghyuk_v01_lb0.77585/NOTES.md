# sanghyuk v01 — LB 0.77585 (학습 코드)

## 메타
- **받은 날짜**: 2026-07-02
- **저자**: 상혁
- **원본 위치**: `team_ref/_incoming/` (SCP 업로드, 이후 정리)
- **리더보드 점수**: 0.77585 (2위, 2026-07-02 기준. 1위 HallaAI 0.77975와 gap 0.0039)
- **받은 파일**: `config.py`, `common.py`, `train.py` (학습 코드만. **inference `script.py`와 모델 weight는 미포함 → 별도 수급 필요**)

## 접근 요약

### 모델
- **기본**: `microsoft/mdeberta-v3-base` (다국어, encoder)
- **대안**: `FacebookAI/xlm-roberta-base`
- **업그레이드**: `Qwen/Qwen2.5-1.5B` (LoRA 권장)
- **핵심 제약**: **대회 평가서버 `transformers==4.46.3` → ModernBERT/Qwen3 NOT AVAILABLE**. 4.46.3에서 로드 가능한 arch만 후보.
- **입력 언어**: EDA 결과 current_prompt **64%가 한국어(ko)** → 다국어/한국어 모델 필수 (영어 전용 BERT는 배제)

### 입력 직렬화 (`common.py::serialize`)
단일 문자열로 concat, 순서:
```
[PROMPT] {current_prompt}
[CUE] q={0/1} short={0/1} [path={0/1} sym={0/1} dir={0/1} glob={0/1}]  ← 옵션
[HISTORY] {(user: ... -> action_name(args): result_summary) 최근순 12쌍}
[META] tier= lang= turn= budget= ci= dirty= loc= toplang= nopen= hastest= hascfg= [open_files=...] [langmix=...]
```
- **history**: `(user 요청 → 다음 assistant_action)`을 pair로 묶어서 저장. 최신 pair를 앞에 (truncation 시 오래된 것부터 잘림)
- **META**: nested session_meta를 11개 flat feature로 (파일 경로는 과적합 → 이진 플래그 `has_test`, `has_cfg`로만 표현)
- **CUE**: prompt에 `?`끝? / 25자 미만? 이진 플래그. `add_symbol_cues` 옵션 시 path/sym/dir/glob lexical 패턴 매칭 (read↔grep↔list↔glob 경계용)
- MAX_LENGTH=512, max_history_steps=12

### 학습 (`train.py`)
- **Trainer**: HuggingFace `Trainer` 커스텀 (`ImbalancedTrainer`)
- **Loss 옵션 (동시 조합 가능)**:
  - `weighted_ce` (기본, class weight = inverse frequency) or `focal` (gamma=2.0) or `none`
  - `label_smoothing` (예: 0.1)
  - **KD (Knowledge Distillation)**: `--teacher_logits path.npz --kd_alpha 0.5 --kd_temp 2.0` → 팀 distillation 접근에 핵심
  - **SupCon (contrastive)**: `--supcon --supcon_weight 0.3 --supcon_temp 0.1` — 같은 클래스 임베딩 뭉치기
  - **Hierarchical coarse aux**: `--hierarchical --hier_lambda 0.3` — 14클래스를 4 coarse group (inspect/modify/execute/converse)으로 aux head 붙임
- **Adversarial 학습**:
  - `--fgm --fgm_eps 1.0` (embedding perturbation)
  - `--awp --awp_lr 1e-4 --awp_eps 1e-2 --awp_start_epoch 1` (weight perturbation, FGM보다 강함)
  - AWP + FGM 조합 시 FGM은 warmup, AWP는 이후 발동
- **Head 아키텍처 3종**:
  1. `AutoModelForSequenceClassification` (기본, [CLS])
  2. `AttnPoolClassifier` (`--pool attn`) — learnable attention weighted pooling
  3. `HierClassifier` (`--hierarchical`) — masked mean pool + fine(14) + coarse(4) 헤드
  4. `SupConClassifier` (`--supcon`) — masked mean pool + classifier + projection head
- **Split**: `--split group` (기본) — id=`sess_..._step_XX`에서 세션 id로 `StratifiedGroupKFold`. **누수 방지: 같은 세션이 train/val에 안 갈림** (매우 중요 — Macro-F1 스코어링이 세션 단위일 가능성)
- **하이퍼**: 4 epochs, batch 16, lr 2e-5, warmup 0.06, weight_decay 0.01, fp16 (또는 bf16)
- **Seed 관리**: `--seed`는 모델 weight init/학습만 변경, split seed는 42 고정 → **앙상블 정렬 보존** (여러 seed 실험 → 앙상블)

### 후처리 (Threshold tuning)
- `tune_thresholds()`: 각 클래스에 additive bias를 좌표하강으로 스윕 → val Macro-F1 최대화
- 3라운드, 클래스별 [-0.3, 0.3] 13-point sweep
- **rare class recall 끌어올리는 표준 트릭. Macro-F1 특화**

### 산출물 (`{output_dir}/`, 기본 `model/`)
| 파일 | 내용 |
|---|---|
| `config.json`, `model.safetensors`, tokenizer 파일들 | HF 표준 |
| `label_maps.json` | `{label2id, id2label}` |
| `serialize_config.json` | 직렬화 마커·옵션 잠금 (**추론과 동일 보장**) |
| `thresholds.json` | `{bias: [14], max_length}` |
| `val_probs.npz` | `{ids, probs, y, classes}` — **동일 holdout stacking 앙상블용** |
| `pool.json` (custom head 시) | `{pool: attn/hier/supcon, backbone: ...}` |

## 재현 방법

**필요한 것 (아직 미수급)**
1. **`script.py` (추론 스크립트)** — 상혁이 별도 관리. 서버에서 model/ 로드 → test.jsonl → submission.csv
2. **모델 weight (`model/` 산출물)** — 학습 결과. 없으면 재학습해야 함
3. **teacher logits npz** (KD 사용한 경우) — 팀 distillation flow에서 어느 단계 것인지 확인 필요

**재학습 시도 시 (검증용)**
```bash
cd team_ref/sanghyuk_v01_lb0.77585
# open.zip의 train.jsonl + train_labels.csv를 CSV로 조인한 파일 필요
# (train.py는 df에 current_prompt/history/session_meta/action 모두 있는 걸 기대)
python train.py --data ./train.csv \
    --model microsoft/mdeberta-v3-base \
    --pool cls \
    --split group \
    --fgm --awp \
    --epochs 4
```
CLI 조합이 실제 제출본과 다를 수 있음 (0.77585 재현 조합은 상혁에게 확인 필요).

## 파악한 핵심 아이디어

1. **세션 그룹 분리 (Group Split)** — id에서 `-step_\d+$` 잘라내면 세션 id. 같은 세션의 다른 step이 train/val에 갈리면 누수 → StratifiedGroupKFold로 방지. **일반 stratified split보다 훨씬 진짜 gen 성능 반영.**
2. **입력 순서: PROMPT → CUE → HISTORY → META**. 잘려도 prompt+cue는 보존. HISTORY는 최신순 정렬로 truncation 시 오래된 것 먼저 잘림. Truncation-robust 설계.
3. **META flat encoding**: nested dict를 11개 명시적 텍스트 feature로. 이진 플래그(has_test, has_cfg, git_dirty) 위주 — 파일 경로는 identifier라 과적합 위험, 텍스트 임베딩에 잘 안 태워짐.
4. **CUE 이진 플래그** (q/short/path/sym/dir/glob): tokenizer가 특수마커를 개별 토큰으로 취급 → 저비용 판별 신호. 특히 read↔grep↔list↔glob 경계에서 lexical 신호가 결정적.
5. **KD + AWP + Threshold tuning 3단 스택**:
   - KD: 큰 모델(GLM/Claude)의 softmax logits로 학생 모델 증류 → **팀 distillation flow의 백본**
   - AWP: 결정경계 강건화 → 저신뢰 rare class 개선
   - Threshold tune: post-hoc Macro-F1 최적화 → rare class recall 끌어올림
6. **앙상블 재료 준비**: `val_probs.npz`를 seed/config별로 저장. 동일 val id 순서로 정렬되어 있어 **stacking 즉시 가능**. 사용자가 만들 모델도 이 형식 맞추면 상혁 모델과 앙상블 가능.
7. **Hier coarse aux (선택)**: 14 → 4 coarse group (inspect/modify/execute/converse) aux loss. 클래스 구조가 잘 정의돼 있어 유용. inference에는 fine head만 사용.

## 개선 여지 hypothesis (사용자 담당 방향 후보)

1. **다른 backbone 시도** (상혁 이미 mdeberta/xlm-r 라인. 사용자는 다른 축):
   - `BAAI/bge-m3` (임베딩 fixed → MLP head) — 다국어 강함, 빠른 실험
   - `intfloat/multilingual-e5-large` — 마찬가지
   - `klue/roberta-large` — 한국어 특화 (전용) → prompt 64% 한국어이니 강할 수 있음. 단 다국어 시 영어 손해
   - `microsoft/deberta-v3-large` — mdeberta보다 큰 sibling (영어 편향은 있음)
2. **입력 표현 실험**:
   - CUE에 turn_index / budget bucket 추가 (수치 → 이진화)
   - HISTORY pair의 result_summary에 더 많은 정보 (예: `n_files`, `exit_code`) 파싱해 추가
3. **Rare class 특화** (Macro-F1 결정적):
   - `web_search`, `write_file`, `lint_or_typecheck` 실패 케이스 분석
   - 이 3개만 별도 lexical rule + model prob ensemble
4. **teacher-free contribution**:
   - 상혁 distillation은 큰 teacher 필요. 사용자는 non-teacher 방식으로 앙상블 다양성 확보
   - LightGBM + serialize 텍스트 tf-idf + meta flat 피처 → 완전히 다른 모델 family로 앙상블 재료
5. **Coarse-only 사전학습 → fine head fine-tune**: coarse(4)로 먼저 warmup → fine head 붙이기. 상혁 hier aux를 명시적 2단계로

## 미수급 자료

- [ ] `script.py` (추론 코드) — 상혁에게 요청 필요
- [ ] `model/` 산출물 (0.77585 재현본) — 상혁에게 요청 (그래야 별도 재학습 없이 앙상블 실험 가능)
- [ ] 0.77585를 만든 정확한 CLI 옵션 조합 (`--fgm --awp --teacher_logits ...`) — 상혁에게 확인
- [ ] teacher logits npz (KD 사용 시) — 어느 teacher model에서? Claude / GLM-2.5 / qwen2.5?

## 다음 액션

1. 상혁에게 위 미수급 자료 요청 (script.py, model/, CLI 옵션, teacher npz)
2. 사용자 EDA 진행하면서 위 hypothesis 검증 (특히 세션 그룹 분리 시 실제 val Macro-F1, rare class F1 분포)
3. bge-m3 or klue/roberta-large 사이드 실험 → val_probs.npz 형식 맞춰 저장 → 상혁 모델과 stacking 확인
