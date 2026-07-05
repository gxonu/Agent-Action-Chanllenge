# 모델 리서치 — 다국어 encoder 후보

**작성**: 2026-07-02
**목적**: 상혁이 부탁(카톡 2026-07-02 11:46) — 한국어+영어+코드 학습된 encoder / 학습 가능 데이터셋 후보 정리 → 노션 공유
**컨텍스트**: 팀 리더보드 4위 0.77585 (상혁 mDeBERTa-v3-base + KD + AWP), 1위 SSU퍼루키 0.78525. rare class 3개(`web_search`, `write_file`, `lint_or_typecheck`) 정복이 시상권 핵심 (Macro-F1 특성)

---

## 1. 제약조건 (필터)

후보가 되려면 **모두** 만족해야 함:

| 제약 | 값 | 근거 |
|---|---|---|
| transformers 버전 | **==4.46.3** | 대회 평가서버 고정 (`config.py` L14 명시) |
| submit.zip 총 용량 | **≤ 1 GB** | 대회 규정. 모델 자체는 fp16 기준 ≤ 750MB 안전 (script+tokenizer 여유) |
| 추론 속도 | **≥ 50 sample/sec** | 30,000건 / 10분 (T4 16GB). 배치 처리 필수 |
| 다국어 | ko+en+mixed | current_prompt 자동감지 **mixed 56%, en 27%, ko 16%**. session_meta.language_pref **ko 64%** (섹션 4 EDA #01) |
| classification 지원 | encoder or causal LM+head | 14-class softmax |

**파생 결론**:
- fp32 저장 시 params ≤ ~190M, fp16 저장 시 ≤ ~375M이 확실히 안전
- **300M 안팎이 sweet spot** (상혁 mdeberta도 278M)
- 큰 모델은 quantization (int8/4bit) 또는 LoRA로 압축 필요

---

## 2. Category A — 다국어 encoder (주류 후보)

| Model | Params | fp16 size | 다국어 | KR 강도 | 코드 | tf 4.46.3 | 비고 |
|---|---:|---:|---|---|---|---|---|
| `microsoft/mdeberta-v3-base` | 278M | ~560 MB | 100+ | 강 | 중 | ✅ | **상혁 현재 사용**. DeBERTa-v3 다국어 버전. 분류 SOTA 급 |
| `FacebookAI/xlm-roberta-base` | 278M | ~560 MB | 100 | 중강 | 중 | ✅ | 상혁 대안. mdeberta보다 살짝 약함 (분류) 안정성 강함 |
| `FacebookAI/xlm-roberta-large` | **560M** | ~1.1 GB | 100 | 강 | 중 | ✅ | **크기 초과 위험**. int8 양자화 시 ~560MB로 fit 가능 |
| `intfloat/multilingual-e5-small` | **118M** | ~236 MB | 100 | 중 | 약 | ✅ | **매우 경량**. 임베딩 특화라 분류 head 얹으면 신선한 재료 |
| `intfloat/multilingual-e5-base` | 278M | ~560 MB | 100 | 중강 | 중 | ✅ | 임베딩 SOTA. E5 시리즈 강함. 상혁 라인과 diverse |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 278M | ~560 MB | 50+ | 중 | 약 | ✅ | Sentence-BERT 기반 안정적. 분류엔 오버킬 |

**우선순위 (상혁 라인과 diverse한 관점)**:
1. **`intfloat/multilingual-e5-base`** — 임베딩 SOTA 특성, 상혁 mdeberta 라인과 아키텍처/훈련목표 다름 → 앙상블 재료로 diverse
2. `intfloat/multilingual-e5-small` — 극경량 118M, 앙상블 speed member로 재밌음
3. `xlm-roberta-large + int8` — 크기 리스크 있지만 가장 튼튼한 다국어

---

## 3. Category B — 한국어 특화 encoder

한국어 64% 데이터라 한국어 전용 모델의 signal 강할 수 있음. 단 영어/mixed prompt에서는 손해.

| Model | Params | fp16 size | 특기 | tf 4.46.3 | 비고 |
|---|---:|---:|---|---|---|
| `klue/roberta-base` | 110M | ~220 MB | KLUE 표준, YNAT topic classification 등 | ✅ | 경량. 앙상블 member로 |
| `klue/roberta-large` | **337M** | ~670 MB | KLUE 표준, larger | ✅ | 크기 borderline. **한국어에서는 mdeberta-base보다 강**할 것 |
| `klue/bert-base` | 110M | ~220 MB | KLUE BERT baseline | ✅ | 오래된 편, roberta-base가 상위 호환 |
| `kakaobank/kf-deberta-base` | ~110M | ~220 MB | 금융 특화 한국어 DeBERTa | ✅ | 도메인 미스매치 가능성 |
| `monologg/koelectra-base-v3-discriminator` | 110M | ~220 MB | ELECTRA 한국어 | ✅ | 상혁이 **KoELECTRA 5/7 카톡에서 배제**한 이력. 예전 모델 |
| `beomi/kcbert-large` | 340M | ~680 MB | 한국어 커뮤니티 특화 | ✅ | KLUE-RoBERTa-large 대비 잘 안 뽑힘 |

**우선순위**:
1. **`klue/roberta-large`** — 한국어 sample subset에서만 학습 or bilingual pipeline의 한국어 파트 담당하면 강함
2. `klue/roberta-base` — 경량 앙상블 member

**주의**: prompt 실제 텍스트는 **mixed 56%** (한/영 혼재). 순수 한국어 모델은 영어 기술용어 잘 못 다룰 가능성. 상혁 mdeberta가 더 균형있을 수 있음.

---

## 4. Category C — 코드 aware encoder

Prompt에 file path (`components/Button.tsx`), 함수명 (`useAuth`, `read_file`) 등 코드 용어 다수. 하지만 코드 자체는 아니라 code-aware 필요성 애매.

| Model | Params | fp16 size | 다국어 | 코드 강도 | 비고 |
|---|---:|---:|---|---|---|
| `microsoft/codebert-base` | 125M | ~250 MB | 영어 위주 | 강 | Korean 약함 |
| `microsoft/unixcoder-base` | 125M | ~250 MB | 영어 위주 | 강 | Korean 약함 |
| `microsoft/graphcodebert-base` | 125M | ~250 MB | 영어 위주 | 강 | Korean 약함 |
| `Salesforce/codet5-small` | 60M | ~120 MB | 영어 위주 | 중 | seq2seq, 분류 head 얹기 어색 |

**결론**: 한국어 약해서 **primary 후보 아님**. 단 앙상블에서 "코드 특화 view"로 넣으면 diverse 재료 가능성.

---

## 5. Category D — 소형 causal LM + LoRA (상혁 upgrade 옵션과 겹침)

Sequence classification head 얹거나, "다음 액션 이름"을 텍스트로 생성하는 방식. LoRA/QLoRA로 크기 압축.

| Model | Params | 원본 size | LoRA 압축 시 | tf 4.46.3 | 한국어 |
|---|---:|---:|---:|---|---|
| `Qwen/Qwen2.5-0.5B` | 500M | ~1 GB fp16 | ~500 MB + LoRA 수십MB | ✅ | 중강 |
| `Qwen/Qwen2.5-1.5B` | 1.5B | ~3 GB fp16 / 750MB int4 | int4 + LoRA로 ~900MB | ✅ | 강 (상혁 upgrade 옵션) |
| `Qwen/Qwen2.5-Coder-0.5B` | 500M | ~1 GB | ~500 MB int8 | ✅ | 중 | 코드 강, 한국어 중 |
| `Qwen/Qwen2.5-Coder-1.5B` | 1.5B | 크기 이슈 | int4 필요 | ✅ | 중 | 코드 강 |
| `meta-llama/Llama-3.2-1B` | 1B | ~2 GB / int4 ~500MB | int4 필요 | ✅ (4.45+) | 중약 |

**우선순위**:
1. **`Qwen/Qwen2.5-0.5B` + LoRA** — 안전한 크기, 한국어 지원, 상혁 encoder 라인과 완전히 diverse (causal LM + classification head). **가장 매력적**
2. `Qwen/Qwen2.5-Coder-0.5B` — 코드 이해 특화. Coder 시리즈가 일반 코드 심층 이해 나음
3. `Qwen/Qwen2.5-1.5B + int4 + LoRA` — 상혁 upgrade 옵션. 이미 상혁 검토했지만 사용자가 int4/QLoRA로 재시도 여지

**주의 — 상혁 카톡 관찰 (2026-07-02 14:23)**:
> "지금 어떻게 이게 Qwen 14b 파인튜닝 보다 잘나오지?"

상혁이 Qwen 14b 파인튜닝을 mDeBERTa base보다 못 뽑음 → **데이터가 얕아 큰 모델 효과 제한적**. Qwen small은 오히려 나을 수 있음 (오버피팅 억제).

---

## 6. Category E — 대형 embedding 모델 (encoder fixed + MLP head)

거대 encoder를 fixed feature extractor로 쓰고 뒤에 MLP만 학습. 크기 제약 있어서 **submit 시 encoder 통째로 담아야** 하므로 여전히 제약 적용됨.

| Model | Params | fp16 size | 다국어 | KR | 비고 |
|---|---:|---:|---|---|---|
| `BAAI/bge-m3` | 568M | ~1.14 GB | 100 | 강 | 크기 초과. **int8 양자화 시 ~570MB 가능** |
| `BAAI/bge-small-en-v1.5` | 33M | ~66 MB | 영어 only | 약 | 다국어 실패 |
| `intfloat/multilingual-e5-large` | 560M | ~1.12 GB | 100 | 강 | int8 필요 |
| `Alibaba-NLP/gte-multilingual-base` | 305M | ~610 MB | 100 | 중강 | fit. Long context 강함 |
| `jinaai/jina-embeddings-v3` | 570M | ~1.14 GB | 89 | 중강 | int8 필요, LoRA task modes 특이 |

**우선순위**:
1. **`Alibaba-NLP/gte-multilingual-base`** (305M, fit) — Long context (8K) 강함. history+meta 다 담아도 여유
2. `BAAI/bge-m3 + int8` — 다국어 SOTA embedding. 압축 필요

---

## 7. Category F — 2026 신규 후보 (검증 필요)

| Model | Params | 상태 | 리스크 |
|---|---:|---|---|
| `google/embeddinggemma-300m` | 300M | **transformers 4.45+ 필요, 4.49+ 권장** | 우리 **4.46.3에 marginal** — 로드 자체는 될 수도 있으나 최신 optimizations 미적용. 반드시 테스트 후 도입 |
| ModernBERT-base (`answerdotai/ModernBERT-base`) | 149M | transformers 4.48+ 필요 | **불가능** — 상혁 config에도 명시 |
| `Qwen/Qwen3-*` | - | transformers 4.51+ 필요 | **불가능** |
| Nomic AI 계열 | varies | 최근 모델 최신 tf 필요 | 대체로 불가능 |

**결론**: 최신 아키텍처는 서버 tf 4.46.3 제약으로 **원천 봉쇄**. EmbeddingGemma만 marginal 가능성. 다만 도입 시 상혁 시간 잡아먹으므로 **후순위**로 미룸.

---

## 8. 분류 Paradigm 선택 (아키텍처 옵션)

Task는 **14-class multi-class text classification**. 세 가지 paradigm 가능:

### A. Encoder + classification head (상혁 방식, 표준)
```
Input → Encoder → [CLS]/pooled → Linear(hidden→14) → softmax → argmax
```
- **장점**: 단순, 학습 안정, 빠른 추론 (T4에서 30k 10분 여유), 상혁 코드 재사용 가능
- **단점**: pretrained knowledge 활용 제한 (특히 라벨 이름의 의미)
- **head 변형** (상혁이 4개 실험 중): 기본 CLS / AttnPool / Hier(coarse aux) / SupCon(contrastive)
- **후보**: mDeBERTa, XLM-R, E5, KLUE-RoBERTa 등 Category A/B 대부분

### B. Causal LM + last-token head (LoRA)
```
Input + "다음 액션:" → Causal LM → last-token hidden → Linear(hidden→14) → softmax
```
- **장점**: pretrained 지식 (Qwen이 read_file/grep_search 등 tool 이름 사전학습에서 이해), LoRA로 크기 압축 가능
- **단점**: 추론 다소 느림 (KV cache 있어도 encoder보다 느림), 아키텍처 복잡
- **후보**: `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-Coder-0.5B`
- **본 대회 적합성**: T4에서 30k 10분 안에 가능한 크기 (0.5B). **1.5B는 int4 필수**

### C. Causal LM + generation (텍스트로 라벨 생성)
```
Input + few-shot → Causal LM → "read_file" 텍스트 생성 → 파싱
```
- **장점**: few-shot 가능, instruction-following 활용
- **단점**: **추론 매우 느림** (auto-regressive 생성). 30k × 짧아도 5-10 토큰 = **T4 10분 초과 위험**
- **판정**: **이 대회 시간 제약상 부적합**. 배제

**결론**: 사용자는 **A (encoder+head)** 우선, **B (causal LM+head, LoRA)** 병렬. **C 배제**.

---

## 9. Distillation 확장 방향 (상혁 KD와 안 겹치는 것)

### 팀 현재 상태
상혁이 이미 `--teacher_logits` KD 사용 중. Config 상 `kd_alpha=0.5, kd_temp=2.0`. 팀 카톡상 teacher 후보는 **GLM-2.5, Claude, Qwen2.5** 시리즈로 추정.

### 사용자 확장 방향 (중복 방지 = 상혁과 사전 정렬 필수)

1. **Ensemble teacher distillation** — 여러 teacher logits 평균 (Claude + GPT-5 + Gemini 2.5 Pro 조합). 노이즈 강건성 up
2. **Iterative self-distillation**: student → 다시 teacher → 새 student. 라벨 노이즈 완화 이론적 근거
3. **Feature-level KD**: logit뿐 아니라 중간 hidden states도 매칭 (더 rich supervision). ReviewKD, FitNets 스타일
4. **Progressive/cascade KD**: 큰 teacher → 중간 → 최종 student 단계별 축소
5. **Task-agnostic + task-specific 2단계**: 상혁 KD가 task-specific이라면, 사용자는 general text 도메인 KD 먼저 → task tune

**⚠️ 상혁과 사전 확인 필요**: 어떤 teacher 이미 시도 완료, 어떤 KD 옵션 이미 실험 함. **중복 실험 방지**.

**적용 시 아키텍처 매핑**:
- Encoder+head (Paradigm A) → **standard KL divergence KD** (상혁 방식 그대로 확장)
- Causal LM+head (Paradigm B) → **logit-KD + optional distillation of intermediate hiddens**

---

## 10. RL-adjacent / Long-tail loss (rare class 특화)

**Task 특성**: RL(REINFORCE, PPO)은 supervised 대비 sample-efficiency 나빠 **주력으로 부적합**. 하지만 "F1 직접 최적화" 정신을 이어받은 supervised loss는 유효.

### 상혁이 이미 반영한 것
- `weighted_ce` (inverse frequency class weight)
- `focal_loss` (γ=2.0)
- `label_smoothing`
- `threshold tuning` (post-hoc, 좌표하강 3라운드로 클래스별 bias)

### 상혁 미적용, 사용자 실험 가치 있음

| 방법 | 원리 | rare class 효과 |
|---|---|---|
| **Balanced-Softmax** ([Ren et al. 2020](https://arxiv.org/abs/2007.10740)) | 학습 시 logit에 log(class prior) 빼서 재조정 → test에서 uniform prior 가정 | 중 |
| **LDAM Loss** ([Cao et al. 2019](https://arxiv.org/abs/1906.07413)) | rare class에 큰 margin 부여 | 강 |
| **Class-balanced loss** ([Cui et al. 2019](https://arxiv.org/abs/1901.05555)) | 유효 sample 수 기반 가중치 (effective number) | 중 |
| **Soft-F1 loss** | F1 미분가능 surrogate로 직접 최적화 | rare에 강 |
| **Two-stage: base → rare class fine-tune** | Full data 학습 후 rare class만 재학습 | 강 |
| **CB Sampler / class-balanced sampler** | 배치에 rare class 비중 강제 유지 | 중 |
| **Mixup / CutMix (텍스트 버전)** | 데이터 aug로 rare boundary 강화 | 중 |
| **Pseudo-labeling on rare boundary** | confidence 낮은 rare 후보에 self-supervised 라벨 | 중강 |

### 순수 RL 접근 (참고, 비주력)

- **REINFORCE with F1 reward**: variance 커 실용성 낮음
- **Contrastive-RL**: (SupCon과 유사, 상혁 이미)
- **Preference optimization (DPO/GRPO)**: 분류엔 오버킬. 배제

### 우선순위 (rare class F1 개선 lever)

1. **Balanced-Softmax** or **LDAM** — 상혁 loss 옵션에 추가하는 실험. 코드 재사용 easy
2. **Two-stage rare class fine-tune** — 심플, 재현 쉬움, gain 크게 나올 수 있음
3. **Soft-F1 loss** — Macro-F1 직접 최적화. 다만 unstable할 수 있음

---

## 11. 종합 랭킹 — 사용자 담당(모델 test)에 추천

**상혁 mdeberta 라인과 다양성 확보** 관점에서 우선순위:

### Tier 1 (즉시 시도 가치)
1. **`Qwen/Qwen2.5-0.5B` + LoRA classification head** — 아키텍처 완전 diverse (causal LM). 크기 안전. 한국어 강. 상혁 관찰상 큰 Qwen 실패했으니 오히려 작은 게 나을 수도. **최우선 실험 추천**.
2. **`intfloat/multilingual-e5-base`** — 임베딩 특화 training objective가 mdeberta MLM과 다름. 크기 fit. **두 번째 실험**.
3. **`klue/roberta-large`** — 한국어 sample subset(64%) 특화. `session_meta.language_pref == 'ko'` 조건부 앙상블 member로 사용 가능. 세 번째.

### Tier 2 (재미있는 시도)
4. **`Alibaba-NLP/gte-multilingual-base`** — 8K long context. history 다 담을 수 있어 상혁 512 truncation 문제 우회 가능. 실험 가치.
5. **`BAAI/bge-m3 + int8 양자화`** — 다국어 embedding SOTA. 압축 성공 시 강력.
6. **`Qwen/Qwen2.5-Coder-0.5B`** — 코드 이해 특화 view.

### Tier 3 (백업)
7. **`FacebookAI/xlm-roberta-large + int8`** — 안전판. mdeberta의 튼튼한 sibling.
8. **`google/embeddinggemma-300m`** — 2026 SOTA (MTEB under 500M). 다만 tf 4.46.3 marginal. 로드 테스트부터.

---

## 12. Rare class 관점 코멘트

**Macro-F1은 rare 3개(`web_search`, `write_file`, `lint_or_typecheck`) F1이 총점 결정** (섹션 EDA #01의 4번). 각 모델 후보에 대해:

- **Encoder 계열 (mdeberta, xlm-r, e5)**: rare class는 학습 데이터 부족으로 배타적 특화 어려움. **loss 재구성 (class-balanced, focal, LDAM)** + **threshold tuning** (상혁 이미) 유효
- **Causal LM (Qwen-0.5B) + LoRA**: pretrained instruction-following 성향이 rare class 라벨을 더 자연스럽게 다룰 수 있음. **zero/few-shot classification** 시도 여지 (라벨 이름 자체가 tool 이름이라 사전학습 지식 활용 가능)
- **한국어 특화 (KLUE)**: `web_search`가 한국어 "검색" 키워드 잡을 때 강할 수 있음 (mdeberta보다 한국어 lexical 신호 감지 강)
- **Embedding + MLP** (bge-m3, e5): fixed embedding + head라 head가 라벨 분포에 빠르게 적응 → rare class에 class weight 걸기 용이

---

## 13. 다음 액션 (사용자 담당)

1. **Tier 1의 (1) Qwen2.5-0.5B + LoRA 파이프라인 구축**
   - HF model card 확인 → 로컬 다운로드 → `src/`에 학습 스크립트 초안 (상혁 `serialize()` 재사용, tokenizer만 교체)
   - Baseline 실험: 4 epoch, 3090에서 학습 → val Macro-F1 확인
2. **Tier 1의 (2) e5-base 실험 병렬**
3. **결과 정리**: `submissions/` 에 `v002_qwen25_lora.zip` 형태로 개인 시도본 관리
4. **상혁 val_probs.npz 규약 준수**: 앙상블 재료로 활용 가능하도록 동일 val_id 순서로 확률 저장

---

## 14. 소스

- [The Best Open-Source Embedding Models in 2026 (BentoML)](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)
- [EmbeddingGemma paper (Google DeepMind, 2026)](https://arxiv.org/html/2509.20354v3)
- [MTEB Leaderboard 2026 (CodeSOTA)](https://www.codesota.com/benchmarks/mteb)
- [Welcome EmbeddingGemma (HF Blog)](https://huggingface.co/blog/embeddinggemma)
- [google/embeddinggemma-300m (Hugging Face)](https://huggingface.co/google/embeddinggemma-300m)
- [KLUE Benchmark (arxiv)](https://arxiv.org/abs/2105.09680)
- [KLUE-benchmark/KLUE GitHub](https://github.com/KLUE-benchmark/KLUE)
- [Qwen2.5-1.5B-Instruct (Hugging Face)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Qwen2.5-0.5B (Hugging Face)](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [KoBigBird-large (arxiv 2023)](https://arxiv.org/pdf/2309.10339)
- [Transformer-based Korean PLMs Survey (arxiv)](https://arxiv.org/pdf/2112.03014)
