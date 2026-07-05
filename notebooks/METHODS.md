# METHODS.md — 모델/데이터 지형 + 다클래스 텍스트분류 SOTA 기법

> Claude Code용 기법 라이브러리. `STRATEGY.md`(계획)에서 참조하는 "무엇을 어떻게" 카탈로그.
> TODO 1(한/영/코드 인코더·데이터셋)과 TODO 2(Kaggle 다클래스 NLP SOTA)를 정리.

---

# Part 1 — 한국어+영어+코드 인코더 & 데이터셋 (TODO 1)

## 1-A. 인코더 (= encoder search 후보)

"한/영/코드를 함께 학습한" 조건에 얼마나 부합하는지 기준:

| 모델 | 다국어(ko) | 코드 | 파라미터 | transformers | 1번 부합도 |
|---|---|---|---|---|---|
| **jhu-clsp/mmBERT-base** | ✅ 1800+언어 | ✅ (Gemma2 tok, 코드 포함 3T) | 307M | ≥4.48 | **정확히 부합 (1순위)** |
| jhu-clsp/mmBERT-small | ✅ | ✅ | ~140M | ≥4.48 | 부합 + 빠름(본선 속도) |
| **ibm-granite/granite-embedding-*-multilingual-r2** | ✅ 200+언어 | ✅ 코드+텍스트 | 311M | ≥4.48 | **정확히 부합** |
| microsoft/mdeberta-v3-base | ✅ 100언어 | ✖ 코드 특화X | 276M | 4.46.3 OK | 부분(코드 약함), 안전 기준선 |
| FacebookAI/xlm-roberta-base/large | ✅ | ✖ | 270M/550M | 4.46.3 OK | 부분 |
| klue/roberta-large | ko 특화 | ✖ | ~337M | 4.46.3 OK | 한국어 강, 코드/영어 약 |
| BAAI/bge-m3 | ✅ | 일부 | ~560M | 4.46.3 OK | 임베딩 계열, 앙상블 다양성 |

**결론**: 1번을 문자 그대로 만족 = **mmBERT / Granite**. 단 입력이 코드 원문이 아니라 대부분 한국어 NL 프롬프트(+파일경로·심볼)라, 코드 특화가 결정적이진 않음 → mmBERT(코드+다국어+속도)가 최적 후보, 안전빵은 mdeberta, 한국어 검증용 KLUE. (상세·스윕순서는 STRATEGY.md §3.)

## 1-B. 데이터셋 (⚠️ 우선순위 낮음 — 판단 포함)

**핵심 판단: 인코더를 새로 학습(pretrain from scratch)하지 마라.** mmBERT가 이미 한/영/코드 3T토큰으로 학습돼 있어 재현일 뿐, ROI 없음. 의미 있는 외부-데이터 활용은 **도메인 적응(DAPT/TAPT)** 하나:

- **TAPT (task-adaptive)**: 우리 `train.jsonl` 텍스트(current_prompt+history)로 MLM 몇 epoch 추가학습 후 분류 파인튜닝. **외부 데이터 불필요, 가진 70k로 충분.** 이득 있으면 CV로 확인.
- **외부 코퍼스(참고용, from-scratch/대규모 DAPT 시에만)**: 코드 = The Stack v2 / StarCoder2 data / CommitPack(코드+NL 커밋). 다국어 = FineWeb-2 / CulturaX / mC4 / OSCAR. 한국어 = 국립국어원 모두의 말뭉치 / AI-Hub / KoWiki. → **우리 규모·일정엔 부적합, 하지 않는 걸 권장.**

정리: **TODO 1의 실질 답 = "인코더는 mmBERT/Granite, 데이터셋은 외부 말고 train.jsonl로 TAPT(선택)".**

---

# Part 2 — 다클래스 텍스트 분류 Kaggle SOTA 기법 (TODO 2)

Kaggle NLP 분류 상위 솔루션의 반복 레퍼토리를, **이 대회 제약(1GB·10분·오프라인·Macro-F1·테스트 비공개)** 에 매핑. ✅=적용 / △=조건부 / ✖=이 대회엔 부적용.

## 2-A. 검증 (Validation)
- ✅ **GroupKFold(session)** — 이 대회는 세션 누수 때문에 필수(EDA_REPORT §3). 일반 StratifiedKFold는 여기선 틀림.
- ✅ **OOF 예측 저장** — 앙상블/스태킹/threshold 튜닝의 재료.
- ✖ **Adversarial validation** — train/test 분포차 탐지. 테스트 비공개라 제한적.

## 2-B. 입력·풀링 헤드 (Pooling head)
- ✅ **Pooling 다양화**: CLS 대신 **mean-pooling / attention-pooling / weighted-layer-pooling / concat last-4 layers**. 보통 mean·attention이 CLS보다 나음.
- ✅ **Multi-sample dropout** — 헤드에서 여러 dropout 평균, 거의 공짜 정규화.
- ✅ **입력 설계** — prompt+history+meta 결합(STRATEGY §5). 이 태스크의 핵심 피처엔지니어링.

## 2-C. 학습 트릭 (점수 견인력 큰 것)
- ✅ **적대적 학습 FGM / AWP / PGD** — 임베딩 perturbation. 텍스트 분류 +0.3~1%p 흔함. **FGM은 거의 공짜, 최우선.**
- ✅ **EMA / SWA** — 가중치 지수평균/평균, 일반화·안정화.
- ✅ **Discriminative LR (layer-wise decay)** + warmup + cosine.
- ✅ **Label smoothing (0.05~0.1)**.
- △ **R-Drop / Mixout** — 정규화 추가 카드.
- △ **Re-init top-N layers / gradual unfreezing** — 전이 안정화.
- △ **Contrastive: SupCon / InfoNCE** — 작년 우승팀 사용. 표현 분리로 혼동쌍(glob↔grep 등)에 도움 가능.

## 2-D. 불균형 (Macro-F1 직결 — 이 대회 승부처)
- ✅ **weighted CE (inverse-freq)** 또는 **focal loss (γ=2)**.
- ✅ **클래스별 threshold / logit 보정** — Macro-F1 기준 좌표하강. 소수클래스 recall 확보.
- △ **class-balanced sampler / 오버샘플** — 소수클래스(web_search·write_file·lint).
- △ **LDAM / logit-adjustment** — long-tail 전용 손실.

## 2-E. 앙상블·경량화
- ✅ **다양성 앙상블**: backbone 다름(mmBERT+KLUE+mdeberta) + seed 다름. 가장 확실한 상승.
- ✅ **소형 앙상블 in 1GB**: 각 fp16으로 담아 logit 평균 (증류 없이 이득).
- ✅ **Knowledge Distillation**: 큰/앙상블 teacher → 1GB 단일 student. 본선 속도까지 확보 (H200 활용, STRATEGY §6).
- △ **Model soups** — 같은 구조 가중치 평균, 추론비용 0 증가.
- △ **Stacking/blending** — OOF로 메타모델(LogReg/LGBM). 단 제출 파이프라인 복잡도↑.

## 2-F. 이 대회엔 부적용 (❌ 하지 말 것)
- ✖ **Pseudo-labeling / self-training** — 테스트 비공개+코드제출이라 반복 라벨링 불가.
- ✖ **TTA(test-time augmentation)** — 텍스트라 이득 작고, 10분 추론예산 소모.
- ✖ **거대 LLM 직접 제출** — 1GB 초과.
- ✖ **RL** — 고정 라벨 지도학습(BC)이라 부적합(soft-F1 loss 정도만 접점).

## 2-G. 우선순위 (이 대회 기준)
1. GroupKFold + Macro-F1/클래스별 F1 계측 (전제)
2. backbone search (mmBERT/mdeberta/KLUE) + pooling
3. 불균형: focal/weighted + threshold 보정
4. FGM(+EMA) — 저비용 고효율
5. 컬럼 활용(history/meta) fusion
6. 앙상블 → (필요시) 증류
7. (여유) SupCon/InfoNCE, AWP, SWA

---

## 참고 출처(개념 근거)
- 작년 우승팀(AI-IDLE, 236473): LLM(Kanana-8B/EXAONE-32B/Qwen3/Gemma3) LoRA **SeqCls + InfoNCE**, 4모델×4fold stacking (당시 CSV제출·H100이라 크기 무제한).
- encoder>LLM(분류): mDeBERTa 276M이 XNLI에서 70B LLM 상회; 한국어 KLUE-BERT가 8B/12.8B 상회.
- mmBERT: ModernBERT 기반 다국어+코드, XLM-R 상회, 2~4x 빠름.
- WildToolBench(ICLR'26): tool-use **특화 모델이 범용보다 brittle** → 범용 backbone 지지.
