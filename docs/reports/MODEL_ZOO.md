# MODEL_ZOO.md — 모델 스카우트 리스트 (역할: 모델 서치 담당)

> 전제 물리: **student(제출)는 T4·10분·1GB → ≤2B가 천장** (fp16 0.5B / int8 1B / int4 2B; T4 컴퓨트로도 ~1-2B가 한계).
> **teacher(H200 141GB)는 ~100B(4bit)까지 자유.** transformers 버전은 requirements.txt로 업그레이드 가능(권장사항일 뿐).
> 모든 후보는 STRATEGY.md §2.5 3-게이트 필터 통과 기준으로 분류. 최종 판단은 항상 CV.

---

## A. STUDENT 후보 (제출용, ≤2B) — 3개 계열

### A-1. 범용 encoder (기본 노선, 분류 SOTA)
| 모델 | 크기 | 비고 |
|---|---|---|
| jhu-clsp/**mmBERT-base** | 307M | 다국어+코드, XLM-R 상회, 빠름. 1순위 |
| jhu-clsp/mmBERT-small | ~140M | 속도 특화 → 본선 속도 10% |
| microsoft/**mdeberta-v3-base** | 276M | 현재 기준선, 4.46.3 안전 |
| klue/**roberta-large** | 337M | 한국어 특화 (ko 64%) 필수 실험 |
| BAAI/bge-m3 · intfloat/multilingual-e5-large | ~560M | 임베딩 계열, 앙상블 다양성 |
| Alibaba-NLP/gte-multilingual-base | 305M | 임베딩 계열 대안 |
| ibm-granite/granite-embedding-*-multilingual | 311M | 코드+다국어 |

### A-2. 범용 소형 LLM (SeqCls 헤드로 분류기化)
| 모델 | 크기 | 비고 |
|---|---|---|
| Qwen/**Qwen2.5-0.5B/1.5B-Instruct** | 0.5/1.5B | 다국어·코드 균형. 1.5B는 int8/int4 |
| Qwen/**Qwen3-0.6B/1.7B** | 0.6/1.7B | 최신 세대, ko 강화 |
| google/gemma-3-1b-it | 1B | int8 |
| meta-llama/Llama-3.2-1B-Instruct | 1B | ko 약함, 우선순위 낮음 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.7B | 영어 위주 |
| **naver-hyperclovax/HyperCLOVA-X-SEED-1.5B** (0.5B도) | 0.5/1.5B | **한국어 최강급 소형**. 꼭 실험 |
| kakaocorp/**kanana-nano-2.1b**-instruct | 2.1B | 한국어, int4로 아슬 (마지막 카드) |
| LiquidAI/LFM2-1.2B | 1.2B | 온디바이스 특화, 빠름 |

### A-3. tool-calling 특화 소형 (비공식 OK — action prior 보유)
| 모델 | 크기 | 근거 |
|---|---|---|
| Salesforce/**xLAM-2-1b-fc-r** | ~1B | xLAM-2 계열 BFCL 상위권. Large Action Model |
| Salesforce/xLAM-1b-fc-r ("Tiny Giant") | 1.3B | BFCL 78.94%로 2B 미만 유일 상위권, GPT-3.5 상회 |
| MadeAgents/**Hammer2.1-0.5b / -1.5b** | 0.5/1.5B | 사이즈별 BFCL best-in-class 주장, function masking으로 강건성 |
| katanemo/Arch-Function-1.5B | 1.5B | 함수 라우팅 특화 |
⚠️ caution: WildToolBench에서 tool 특화 = brittle 경고. 단 우리는 70k로 재파인튜닝하므로 prior만 취함 → **범용 대응짝과 반드시 A/B** (예: Hammer1.5b vs Qwen2.5-1.5B).

---

## B. TEACHER 후보 (H200, 학습 전용, ≤~100B 4bit)

### B-1. 한국어 강한 범용 (ko 64%라 우선)
| 모델 | 크기 | 비고 |
|---|---|---|
| kakaocorp/**kanana-1.5-8b**-instruct | 8B | 작년 우승팀 사용, 한국어 |
| LGAI-EXAONE/**EXAONE-3.5-32B** (또는 EXAONE-4.0) | 32B | 작년 우승팀 사용 |
| K-intelligence/Midm-2.0-Base | 11.5B | KT 한국어 |
| skt/A.X-4.0-Light (풀 A.X-4.0은 72B, 4bit ~40GB 가능) | 7B/72B | SKT 한국어 |
| upstage/solar-pro-2 계열 | ~31B | 한국어 강 |

### B-2. 글로벌 범용
| 모델 | 크기 | 비고 |
|---|---|---|
| Qwen/**Qwen3-14B/32B** | 14/32B | 다국어·코드 균형, 최우선 |
| Qwen/Qwen2.5-72B-Instruct | 72B | 4bit ~40GB, H200 여유 |
| zai-org/**GLM-4.5-Air** | 106B | 4bit ~55GB. GLM teacher 상한 (4.6/5.x는 초과) |
| THUDM/glm-4-9b-chat | 9B | 가벼운 GLM |
| google/gemma-3-27b-it | 27B | 다국어 |
| mistralai/Mistral-Small-3.2-24B | 24B | |
| microsoft/phi-4 | 14B | 추론 강 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | 32B | 추론 증류판 |
| meta-llama/Llama-3.3-70B-Instruct | 70B | 4bit ~40GB |

### B-3. tool/agent 특화 (비공식, 실험 1-2개만)
| 모델 | 크기 | 근거 |
|---|---|---|
| Salesforce/**xLAM-2-8b/32b-fc-r** | 8/32B | BFCL #1 계열(GPT-4o 상회 주장), τ-bench 강 |
| Team-ACE/ToolACE-2-8B | 8B | BFCL 상위 |
| MadeAgents/Hammer2.1-7b | 7B | 강건성 설계 |
| watt-ai/watt-tool-8B | 8B | BFCL 상위 |
⚠️ 동일 caution. teacher로서 "agentic prior가 증류 신호를 좋게 하나?"는 열린 질문 → **범용 teacher와 CV 대조 1회**로 판정. (발표 40%용 연구 서사로도 좋음.)

---

## C. 다양한 접근 아이디어 (모델 외)

1. **Decoder→분류기化 2방식 비교**: (a) SeqCls 헤드(1 forward, 생성 없음 → 10분 예산 OK) vs (b) **label-token 방식** — 14개 action을 단일 토큰으로 매핑해 "다음 행동은?" 프롬프트의 next-token 확률만 읽기(생성 루프 없이 1 forward). (b)는 instruct 모델의 사전지식을 더 직접 활용. 소형 LLM에서 A/B.
2. **API teacher 증류**: 규칙이 개발 단계 API 사용을 명시 허용(비용 자부담, 출처 명시). GPT/Claude/Gemini few-shot으로 70k 일부에 soft label 생성 → teacher 앙상블에 이질적 신호 추가. 특히 소수 클래스(web_search 1.8%)에 유용 가능.
3. **특화 vs 범용 A/B** (A-3/B-3): tool-prior가 파인튜닝 후에도 남는지 검증. 연구 질문이자 발표 소재.
4. **TAPT**: train.jsonl 텍스트로 MLM 도메인 적응 후 파인튜닝 (외부 데이터 불필요).
5. **하이브리드**: encoder 임베딩 + 구조화 컬럼 → LightGBM (경모·용욱 작업과 시너지, 본선 속도 유리).
6. **1GB 내 이종 앙상블**: encoder(0.3GB) + int4 0.5B LLM(0.3GB) + KLUE(0.7GB는 초과 주의) 조합 설계. 서로 다른 계열이라 앙상블 이득 큼.
7. **속도 플레이**: mmBERT-small 증류 student → 본선 속도 10% 극대화.

---

## D. 스윕 우선순위 (CV 고정 후)

**Student**: mdeberta(기준선) → mmBERT-base → KLUE-RoBERTa-large → HyperCLOVA-SEED-1.5B → Qwen2.5-1.5B(SeqCls) → xLAM-2-1b vs Qwen 대조 → bge-m3.
**Teacher**(student 확정 후): Qwen3-32B → EXAONE-32B → Kanana-8B → (대조) xLAM-2-32b → GLM-4.5-Air.
각 실험 = 동일 GroupKFold(seed 42)·동일 입력, OOF Macro-F1 + 클래스별 F1 기록.
