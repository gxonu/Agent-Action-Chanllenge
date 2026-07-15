# STATUS — oh-my-sinsa 공유 대시보드

> **여러 세션이 공유하는 단일 진행상황 보드.** 작업 시작/완료할 때마다 여기부터 갱신한다.
> 상세 근거는 `notebooks/`, 팀 맥락은 `team_chat/SUMMARY.md`, 팀원 코드는 `team_ref/` 참조.

**최종 갱신**: 2026-07-06 (세션: `yj-3-S2` — 리포트 전수검증 + 심층 EDA v3 신규발견 + serialize A/B **학습완료(GPU7)**: cue −0.0020 / reorder +0.0003 = **⛔negative, 두 축 다 무이득** → serialize 재설계 축 접음, exp007 실질이득 ~0 시사) + **라벨 100%일관 실측(오라벨 0%)** + **07-06 내부미팅 정리→카톡반영+TODO 분담 갱신** + 언어진단 3종 TODO 추가) · **07-11 `yj-3-S2`** — turn_index 신뢰도 검증(버리지마라) · inspect 4클래스 **분리불가 4중 검증완료**(t-SNE/수치/규칙탐색/실제예시) · 앙상블 **diversity 분석** · **LightGBM 멤버 실패→ML 제외** · 리더보드 확인(pack 0.79, Private shuffle 예상) · **결론: 데이터 축 소진, 남은 길=강한 인코더 2개 양자화 앙상블(S1)** · **07-13 `yj-3-S2`** — 앙상블 조합매트릭스 실측(**3-way xlmr+mmbert+bge=0.7745 최고**, v007 2-way 0.7712 대비 +0.003) · **vocab pruning 성공(3-way 827MB<1GB, 무손실, S1 int8벽 해소 enabler)** · Roka jeong브랜치(0.79026) 확인 → 이종백본이 그들 0.79 미탐색 top가설=우리 직격 · **3-way tri_cond 패키징 스크립트 ✅완료·검증통과**(T4 ~9:00<10분, 822MB<1GB, tri_cond=full 3-way 0.7749) → S1 신모델 갈아끼우면 즉시 제출 · int4도 무손실 확인(공간 병목아님, 시간이 병목) · `yj-3-S1` — exp007(serialize_v2)·exp001(klue-large) yj-3 학습 launch, 🔴진행중 TODO 실상 갱신, S1↔S2 serialize 중복 조율안 기록) · **07-14 `yj-3-S1`** — Roka(jsj) 0.79026 재현·해부(jeong브랜치 th85=xv6/mde/xv4 w0.6/0.15/0.25+gen_rescue) → **v016 정확재현 LB 0.78326**(재현gap δ=−0.0070, 체크포인트부재로 숫자재현 불가 실증). **v017 mht12(history 8→12=입력축) 앙상블 = LB 0.78700 신SOTA**(+0.00315 vs v012 0.78385, 535s<600). 입력축 고전이 확증→재현보다 자체강화가 정답. mde@384 프로브 진행중(GPU4). ⚠️정정: exp001(klue) S1 kill·exp007 미실행 = **8 GPU 전부 idle**(07-14 06:3x) · **07-15 `yj-3-S2`** — 디코더/tool모델(qwen 0.7013·hammer 0.7090) 단독약함+앙상블기여0(decorrelated 하지만 약해 가중치0) **축사망** · **번역증강 A/B = −0.0081 해로움**(구조보존 NLLB도 음수) · **⛔au제거는 S1 v019 LB 0.7178(−0.069)로 반증**(test에 au있어 sim-only OOD붕괴, sim-val 착시) → **데이터축 완전소진**(증강·au제거·디코더 전부 사망) · 블렌드 grid-search 하네스 준비완료 · **📦S1 인수인계: 코드특화 인코더 3종(codebert/graphcodebert/unixcoder) = decorrelated 강멤버 유일 미탐색후보, fold호환 검증완료, S1이 train.py로 실행**(하단 S2→S1 블록)
**마감**: 2026-07-15 10:00 예선 종료 · **D-10**
**팀 리더보드 (07-05, DACON 확인)**: **20위 0.77585** — 상위권 더 밀집, 우리 점수 그대로라 순위 하락(07-03 14위 → 07-05 20위). 1위 고대 야호 **0.79362**, gap **0.018**. **팀 관찰: "다들 0.79 못 넘음" = 0.79가 현재 벽**. 제작자 상혁(mDeBERTa+KD+AWP)
**팀 SOTA (상혁)**: **mmbert_base + CLS + 5에폭 + bias(threshold) + open_files = 0.77585** (07-02, 팀 최종 선택 제출). ⚠️ **이건 상혁(팀원) 것 — 건우 개인 제출 아님**(DACON은 팀 제출 통합 표시). **07-02 이후 정체** — 07-05 시도들(0.7755, Language별LoRA 0.7544) 미돌파. mmbert_base=ModernBERT arch(tf≥4.48), native 8192 ctx.
**방향(07-05 갱신)**: gte 신모델 대신 **내 SOTA mmbert_base를 직접 개선**. ⚠️ **max_length bump 재평가**: 실측 결과 512 truncation은 **3.2%뿐**(41.5% 오측정 정정 — raw JSON이 아니라 상혁 compact serialize+mmBERT 토큰 기준, max 682tok, 768이면 100%커버). → 1024 bump는 **저기대 곁다리**로 강등. **진짜 lever 후보(보고서 기반)**: F4 open_files 범주형(0/1/2/3+), F6 OOF threshold 튜닝(CV-LB 0.025 격차), KLUE-large 등 diverse 앙상블. rare-F1/노이즈제거/leak은 팀 실측 소진.

---

## 🔴 S1 현황 (07-14, `건우/yj-3-S1`) — S2 공유용

**팀 신SOTA = v017 LB 0.78700** (mmbert-mht12 앙상블, 상혁 0.77585 → +0.011). 계보:
```
v012(mmb512+bge+mde) 0.78385 → v016(Roka th85 정확재현) 0.78326 → v017(mmbert-mht12) 0.78700
```

**핵심 발견 (실측 근거)**:
1. **mmbert(ModernBERT)가 유일한 "입력축" 백본** — max_len·history 늘릴수록 단조개선 (@320 0.7565 → @512 0.7601 → @768+mht12 0.7664). **mht12(history 8→12)=+0.010 단일, LB +0.00315**. mmbert만 이게 됨(xlmr @384는 오히려 하락=길이축 폐쇄, S2 발견과 정합). **mht12=history 천장**(데이터 ~12턴 슬라이딩).
2. **Roka 0.79026 재현 종결** — jeong브랜치 th85=xv6/mde/xv4(w0.6/0.15/0.25)+gen_rescue 정확재현했으나 **LB 0.78326**. 재현gap δ=−0.0070(그들 체크포인트 미공개, 코드만; Roka 자체도 팀원모델 −0.0089 못따라잡음). **단일은 우리≈그들(0.763 vs 0.7645)** — 재현 잘됨, 마지막 1마일이 벽.
3. **gen_rescue = 클린 LB +0.0019 검증** (코드 그들과 byte동일). th·가중=저전이(Roka 실측 0.04~0.26)라 스킵.
4. **우리 4멤버(bge/xlmr/mde/mht) 전부 ~90% 일치 = 앙상블 천장 ~0.78** — xlmr은 bge와 중복(+0.0004). **decorrelated 강멤버가 유일한 돌파구.**

**진행중**: v018(mmbert 전행 해방, CV 0.7797, ~0.789 예상) 빌드완료·제출대기 / **decorrelated 강멤버 프로브 4개 병렬**(GPU4-7): mde@384·klue-large@512·koelectra@512·**mmbert-v4@768**(Roka식 same-backbone 직렬화변형). 완료시 강+diverse 멤버만 편입해 4-way 신조합 제출.

**⚠️ S2에게 — 전략 공유**: **모델축 현실 천장 ~0.790~0.791로 수렴**(Roka도 0.790 정체, honest상한~0.796). **0.793(컷) 이상은 model축으로 거의 불가 → 데이터축 우위 필요.** 죽은축(재확인 불필요): 직렬화 재배치(S2 cue/reorder 실측 무이득), 극단가중, th/bias, 일반증강(팀 부정적), 클래스 specialist(inspect 분리불가). **질문: 데이터축에 아직 살아있는 레버 있나?**(07-06 언어균형증강 외 / S2 07-11 "데이터축 소진" 결론 재검토 필요한지)

**🆕 S1 착수 예약 (`건우/yj-3-S1`, 07-14 — 중복방지)**: 데이터축 2건 **직접 재검증** 맡음. ①**경모 증강 재검증** — 기존 "1000샘플 무효"가 in-sample/검증부실 의심 → **honest 반분-CV A/B**(base vs base+aug, fold0, out-of-sample 델타)로 재판정. 경모 증강데이터(방식·수량·언어) 수급 필요. ②**au ablation 재검증** — au/sim은 이미 v6 헤더 `[GEN] au/sim` 마커로 들어감(gen_rescue가 이 헤더 보호). 팀의 "au제거=부정적"은 마커 유무·검증 불명확 → **마커 있는 상태에서 au-downweight/제외 클린 A/B**. test가 sim-only면 이득 가능. → 4프로브 종료 후 GPU 여유 생기면 착수. **S2·경모와 겹치면 조율 요망.**
> ✅✅ **[07-14 조율완료 — S2가 두 건 다 실행함, S1 재실행 불필요]** honest 반분-CV A/B(동일 sim-only val 6537, mmbert 4ep) 완료 → 아래 S2 블록(`experiments/exp_data_ab/`) 참조. **결론: ①au 제거 = +0.0083 sim-test 이득(→최종모델 sim-only 학습 권고) ②경모 증강 = −0.0119 해로움(증강본 폐기).** ⚠️ 1seed·"test=sim"가정 → S1은 **최종모델 학습 시 au 제외 반영 + 제출로 LB 확정**만 하면 됨(A/B 재실행 X). 미확인분: 다른 seed 재현(S2가 할지 조율).

### 🌙 07-15 새벽 `건우/yj-3-S1` — 마감 스퍼트 (13 GPU: yj-3 8 + yj-4 5)

**🆕 발견 A — 길이축은 mmbert 전용 아님(mde·bge도 먹음)**: fold0 프로브 실측
- `mde@320 0.7472 → @384 0.7584 → @512 0.7648` (**+0.0176**!) / `bge@320 0.7589 → @512mht12 0.7651` (+0.0062)
- **업글 앙상블(mmbert@768+bge@512+mde@512) CV ~0.7820** = 기존(0.7797) **+0.0046**, 입력축이라 LB전이 기대(v018 coverage와 달리)
- 폐기: mmbert-v4(0.7645, 같은백본 중복 +0.000) / klue@512(0.7314)·koel@512(0.7294) 약함
- ⚠️ 배포시간: 전부-base @512/@768 = ~736s DQ. **tri_cond(mmbert@768 base + bge@512/mde@512 cond50%) = ~508s 안전**

**🆕 발견 B — au제거를 최종모델에 반영중(S2 exp_data_ab +0.0083 → LB 확정용)**
- train_full_cli에 `AD_EXCLUDE_AU` 추가(70000→64975). mmbert@768·bge sim-only FULL 완료, bge@512·mde@512 sim-only FULL 학습중
- ⚠️ test au비율 미확정 → **au-포함 길이최적 FULL도 병행(yj-4)** = 헤지. 둘 다 제출해 LB로 판정

**🆕 진행 — 팀 input 아이디어 honest fold0 검증(용욱·경모)**: 프리앰블(입력앞 액션설명) 4종 프로브 중
- full-14(152tok)·hints-4(4개만41tok)·용욱원안(full+SELECTION TASK 184tok)·압축용욱 — **baseline 0.7664 넘나 판정중(~2:40 완료)**
- truncation 실측: @768 잘림0%, **@512는 full-14 32%잘림→hints-4(51tok)만 안전**. teacher_cli/train_full_cli에 `AD_ACTIONS_PREAMBLE=1/2/3/4` 추가
- ⚠️우리 실험사(cue/PACE 죽음)상 죽을확률↑ but 다른축(레이블 grounding)이라 검증. 통하면 캐스케이드/하이브리드 후속

**학습 현황(07-15 01:3x)**: yj-3 8GPU(sim-only 길이 3 + 프리앰블프로브 4 + mmbert-hints4-sim) / yj-4 5GPU(`experiments/yj4_train/HANDOFF_yj4.md` = au포함 길이최적 bge@512-full·mde@512-full + 프리앰블-sim 3) = **모든 레버조합(길이×au×프리앰블) 커버**
**타임라인**: 2:40 프리앰블판정 → 3:30 1차 au-sim 앙상블 제출 → 5:00 au+길이 결합 제출 → 6:00 LB보고 타겟 최종wave(~3h, 마지노선 6:30발진) → 9:xx 최종제출
**은행 SOTA 유지: v017 0.78700 / v018 0.78663(flat, coverage=저전이 확인)**

> 🚨 **[07-15 04:xx 결정적 실측 — au 제거는 재앙, sim-only 전면 폐기]** v019(sim-only 앙상블, v017과 동일구조·멤버만 sim-only) 제출 **LB=0.71780 = v017(0.787) 대비 −0.0692 대폭락**. **원인**: test에 au 상당수 → sim-only 모델이 au 샘플서 완전 OOD(학습때 [GEN] au 한번도 못봄)로 무너짐. **⚠️ 위 S2 exp_data_ab의 "au제거 +0.0083 → sim-only 학습 권고"는 LB로 반증됨** — sim-only val엔 au가 없어 생긴 착시. **결론: au 제거 절대 금지, au-포함이 정답(v005~v018 전부 au-포함이 옳았음).** 조치: yj-3 sim-only 학습 전부 kill, **새 본命 = au-포함 길이앙상블(mmbert@768-full + bge@512-full + mde@512-full, 전부 au-포함)** = v017 + 길이업글(+0.0046 au-포함 fold0 측정) → 예상 ~0.790. yj-4 G0/G1(bge@512-full·mde@512-full au-포함)이 본命 멤버(~7:30 완료). au-포함 mmbert seed soup(원본+seed2024+seed777) 병행. **v019 카드 1장으로 "au빼면 폭망" 확정 = 최종 sim-only 제출 참사 방지(값어치 있었음).**

#### 🖥️ `건우/yj-4-S1` 발진현황 (07-15 01:37 발진 → 02:5x 프리앰블 kill → **03:xx au-포함 본命 확정**)
> 🚨 **v019(sim-only 앙상블) LB 0.7178 = v017(au-포함 0.787) 대비 −0.069 대폭락.** test에 au 상당수 → sim-only는 au샘플서 OOD로 붕괴. exp_data_ab "+0.0083"은 sim-only val 착시. **→ au 제거=재앙, au-포함이 정답.** yj-4는 애초에 au-포함(G0/G1)+au-포함 seed만 돌던 중이라 **kill할 sim-only 없음**(도는 3종 전부 `EXCLUDE_AU=none` 검증완료). au-포함 3종이 이제 **본命**.
> ⚠️ env: PATH가 `nas5/miniconda3/bin`을 앞세워 `python`이 base로 튐 → 규칙대로 **env 절대경로 `envs/intern_aichallenge/bin/python`**로 전부 실행. tmux 세션명 `yj4`. 로그·member_는 NAS 공유(yj-1-S4 패키징이 읽음).

| TAG | GPU | 상태 | 용도 |
|---|---|---|---|
| `bge_512mht12_full` (G0) | **0** | ✅100% 학습중 | **au-포함 길이헤지 (유지, +0.0046 proven)** |
| `mde_512_full` (G1) | **1** | ✅100% 학습중 | **au-포함 길이헤지 (유지)** |
| `mmbert_768_full_seed2024` | **7** | ✅100% 학습중 | **au-포함 mmbert seed soup용** (kill한 G4 자리 재활용) |
| ~~bge_512_hints4_simonly (G2)~~ | ~~2~~ | ⛔kill | 프리앰블 폐기(fold0 hints-4 −0.0015) |
| ~~mde_512_hints4_simonly (G3)~~ | ~~6~~ | ⛔kill | 프리앰블 폐기 |
| ~~mmbert_768_full14_simonly (G4)~~ | ~~7~~ | ⛔kill→seed2024로 교체 | 프리앰블 폐기(full-14 −0.0119) |

- 현재 **3종 활성**(G0/G1 au-헤지 + seed2024 au-mmbert), GPU **2·6 유휴**(seed 2~3개면 충분 → 억지로 안 채움, 필요시 bge au-seed 추가 가능)
- kill 3종 = `KeyboardInterrupt` 클린종료, GPU 즉시 해제 확인. seed2024 `[full]` au-포함 마커 확인(exclude_au·preamble 없음)
- 로그: `logs/{bge512_full,mde512_full,mmb768_full_seed2024}.log` / 결과예정: `experiments/yj4_train/member_{bge_512mht12_full,mde_512_full,mmbert_768_full_seed2024}/`

---

## 📨 팀 카톡 반영 (~2026-07-05) — 전략에 직결되는 실측 신호

> 출처 `team_chat/2026-07-0{1..5}.txt`. **여러 축이 이미 팀 실측으로 소진됨** — 중복 삽질 방지용으로 여기 고정.

- **노이즈 제거·증강 = 팀 실측 부정적.** 상혁이 "이상한 라벨" 반복 제거 실험 → **뺄수록 점수 하락**. 경모 증강 1000개 → 효과 없음/하락. 데이터는 명백 LLM생성(sim), test도 같은 분포 추정. 팀 결론: **"잘못 라벨된 데이터에 오히려 피팅하는 게 맞다"**. → 내 `exp003(au ablation)`·rare booster·노이즈 제거 축 **강등**.
- **대형 LLM / 앙상블 = 이번엔 안 통함.** Qwen3.5-122B zero-shot 0.30, Claude 0.21 vs 우리 mDeBERTa 0.77. 작년 솔루션(카나나·엑사원·젬마·큐원 디코더+헤드 스태킹) 다 해봤으나 **현 모델보다 낮음**. **ablation 할수록 하락 = 모든 feature 다 넣는 게 최선.**
- **명시적 feature concat = 용욱이 이미 시도, 미미.** step/git_dirty/open_files → mlp+mDeBERTa fusion. eval +0.01, list_directory precision 개선 거의 없음. open_files만 미세. → 내 `last_action` feature 기대 낮춤.
- **data leak 제출 = 상혁 반대.** 건우 07-03 질문에 상혁 "아마 규정위반·불공정". → `exp_leak_lookup` **보류**.
- **상혁 코드·백본 공유됨(blocked 해소 방향).** 0.775 input형식+config+common → 노션 업로드(07-02). **0.775 백본 weight → 용욱에게 메일 전달(07-05)**. 노션/용욱 통해 수급 가능.
- **팀 진행 중 축**: 용욱 = 언어별(ko/en/mixed) **LoRA adapter + 언어 라우팅**(eval↑, LB 0.75 과적합) + 번역 증강(경모 착수 예정). 규중 = history 방해 step **모델이 스스로 마스킹/입력수정**(base 대비 **eval +4%**, LB 미검증, 상혁이 참고해 돌려볼 예정).

**→ 내 실효 lever 재정의**: ① 상혁 mDeBERTa-base와 **diverse한 강한 인코더**(klue-large 등) 단독 성능 + 앙상블, ② 팀 미탐색 축 결합. 노이즈/leak/feature-concat은 팀이 이미 소진.

### 🆕 07-06 내부미팅(56분 STT, `team_chat/2026.7.6...미칭.txt`) — 새 신호·분담

- **⭐언어 혼재(mixed)가 성능 갉아먹음**: 용욱 실측 mDeBERTa base 70% → ko/en LoRA 74% but **mixed 69%**. mixed 데이터도 ~10%로 적음. 팀 가설: public test에도 mixed 있고 **주최측이 증강 의도**. → **팀 방향이 "언어 균형 증강 + 도메인 SFT"로 선회**(증강=부정적이던 07-05 기조에서 언어균형 목적으로 재개).
- **turn_index 신뢰 의심(신규)**: turn=1인데 elapsed 381~495초·토큰 2.5만 소진 샘플 존재 → **세션 중간부터 잘라온 것**, 진짜 첫턴 아님. history는 최근 6쌍(12턴) 슬라이딩(=S2 sliding-window 발견 일치). 중간 스텝 결측 ~9200. "turn_index 빼면 오히려 노이즈일 수도" 의견.
- **au=sim 언어변환본? 가설(미검증)**: sim↔au 매칭 필요. prefix 의미 여전히 미궁.
- **분담 확정**: 경모=언어증강(mixed 우선, ko/en/mixed 4:4:4, 구독LLM, 코딩용어 보존, 배치) · 상혁=AI생성체/한국어 도메인 SFT("우리만의 KO ModernBert") · 건우=규중 코드(RL/다른형식) ModernBert 백본 이식 테스트 + tool-calling decoder · 규중=코드 노션(백본 ModernBert 아님, val↑).
- **후처리 KNN 가점 아이디어**(미할당). 모델 탐색 소진(ModernBert 유지). **멘토링 금 19시**.
- **→ S2(데이터셋 축) 기여점**: 미팅서 나온 데이터 의문들이 곧 내 EDA 과제 — (a)turn_index 신뢰도 실측 (b)au=sim 매칭 검증 (c)언어별 val F1 분해로 mixed 갉아먹기 재확인. [아래 TODO]

---

## 📋 갱신 규약 (병렬 세션 필수)

1. 무엇이든 손대기 전에 이 파일부터 읽는다. 끝나면 **"최종 갱신" 날짜/세션명**을 바꾼다.
2. TODO는 **한 항목 = 한 세션이 소유**. 집는 순간 `🔴 진행 중`으로 옮기고 `(담당·세션)` 태그를 단다. 중복 착수 방지.
3. 실험은 반드시 **exp_id 한 줄을 트래커에 먼저 추가**(상태 `📋 계획`)한 뒤 시작한다. 끝나면 점수·경로 채우고 상태 갱신.
4. 산출물 경로는 항상 한 줄로 남긴다 (`→ saved: ...`). 실험 결과는 `experiments/{exp_id}/`에 config+model+log+metrics 세트로.
5. 충돌 나면 덮어쓰지 말고 두 항목 다 남긴 뒤 조율. 삭제 대신 `✅ 완료`/`⛔ 폐기`로 이동.

---

## 🗂 TODO 보드

담당 태그: `건우`(사용자) · `상혁` · `용욱` · `경모` · `규중` · `claude`(세션 작업)

### 🏆 제출 로그 (건우/yj-3-S1, DACON public LB)

| 버전 | 구성 | 우리 CV | **public LB** | 추론 | 비고 |
|---|---|---|---|---|---|
| v005 | mmbert 10ep 오버핏 단독 | 0.7366 | 0.74621 | 2:57 | ⛔ train 오버핏은 LB 무익 (SOTA −0.030). CV→LB +0.010 |
| **v007** | **mmbert+bge 수동int8 앙상블(bnb無)** | 0.7754 | **0.782125** 🏆 | 7:49 | ✅ **새 팀 SOTA (상혁 0.77585 대비 +0.0064)**. 0.81GB·설치리스크0·CV→LB +0.007 |

**핵심 실증**: ①**diverse 앙상블은 우리 케이스에서 이득**(mmbert·bge 동급강도→상호보완). 경쟁팀 Roka는 강한 large를 약멤버로 희석해 앙상블 폐기했지만 우리와 상황 다름. ②**수동 per-row int8**(임베딩+선형층 직접 int8 저장→script.py fp16복원)으로 2모델을 1GB·오프라인·fp16속도로 패키징 성공 (bnb 4bit는 임베딩 미양자화로 1.07GB 초과+업로드 절단됨). ③CV→public 캘리브레이션 **+0.007~0.010**(우리 group-split val이 보수적).

**🔄 진행중(건우/yj-3-S1) — Roka 기법 × 이종백본 조합 매트릭스** [07-13]:
경쟁팀 Roka-jsj 레포(github, `jeong`브랜치 최신 **LB 0.79026**) 분석완료 → 그들 레시피(**FGM+v6+8ep+LLRD**)를 우리 4-백본에 적용 후 앙상블.
- **학습중(yj-3 8 GPU)**: 각 fold0(CV)+full(배포), Roka레시피. **fold0 CV**: mmbert **0.7565**✅ / xlmr ~0.757 / bge ~0.756 / mdeberta 0.729(8ep 약함→12ep 재실측중). *xlmr sim 0.7463 = Roka 0.7456 재현확인.*
- **핵심 우위**: 우리 **mmbert(ModernBERT)+mdeberta(DeBERTa)+bge/xlmr(XLM-R) = 3개 다른 arch**. Roka는 XLM-R+한국어모델 편중 → **그들이 "0.79갭=이종백본"으로 미탐색한 축을 우리가 직격**(mmbert는 그들 안 씀).
- **gen_rescue 포팅완료**(`scratchpad/gen_rescue.py`): v6 좌측절단서 [GEN] 잃는 ~11.8%행 헤더복원. **추론시점 적용=Roka 확정 +0.00314(전이0.82)**, 재학습 불필요. 우리 fold0 실측중(mmbert).
- **Roka 통일이론**(메모리 `action-challenge-roka-strategy-theory`): 죽은축 M1~M6(train선택편향/FULL-holdout/분포매칭/th선택기/약멤버희석/public승자의저주), 작동축 S1~S5(bias+0.0058/강-강조건부/[GEN]수호/gen_rescue). **정직상한 ~0.7837, 0.79대는 public착시 위험**.
- **다음**: fold0 4종 완료 → 조합 CV(mmbert⊕bge, +mdeberta 3-way) + gen_rescue → best 제출.

**✅ S2 완료 — vocab pruning (3-way 1GB enabler)** [07-13 `yj-3-S2`]:
> 3-way int8만으론 ~1.16GB>1GB (v007 2모델도 0.81GB). **어휘 pruning으로 임베딩(모델의 절반~2/3) 축소 → 3-way 827MB 달성, 무손실 검증.**
- **원리**: 각 모델 사전학습 vocab(mmbert 256k / bge·xlmr 250k)의 **3.8%(~9.7k)만 실제 사용**(12-repo 한+영+코드). 안 쓰는 24만 토큰 임베딩 행 slice + id remap. **학습 무관 = 학습 끝난 모델에 후처리** (S1은 원본대로 학습만 하면 됨).
- **실측(exp_encoders 구버전 기준, `src/prune_apply.py` + `src/vocab_prune.py`)**: keep=등장9.7k+버퍼→34k. fp32 크기 mmbert 1230→551MB / bge 2271→1394MB / xlmr 2240→1363MB. **int8 추정 138+349+341=827MB ✅ 1GB 안.**
- **무손실**: pruned 재추론 Macro-F1 mmbert 0.7679→0.7671 / bge 0.7644→0.7641 / xlmr 0.7590→0.7581, argmax 99.6~99.9% 일치. **pruned 3-way(xlmr+mmbert+bge, w 0.15/0.45/0.4) = 0.7752** (v007 2-way 0.7712 대비 **+0.004**, LB추정 ~0.786).
- **⚠️ 조율**: 위 실측은 옛 exp_encoders 모델. **S1의 새 Roka-레시피 4-백본 완성되면 `src/prune_apply.py`로 재-pruning**해서 최종 패키징. 도구는 어느 모델이든 적용됨(`--model`/emb_key만 지정). 산출: `experiments/exp_prune/{mmbert,bge,xlmr}/`(pruned model+remap.npy+val_probs).
- **버퍼 주의**: keep은 train+dummy 기준. 실제 test 희귀토큰은 UNK 처리(12-repo 닫힌세계라 극소). 안전 위해 버퍼 30k 유지.

**✅ S2 완료 — 1GB 압축 아이데이션·검증 → "공간은 병목 아님, 시간이 병목"** [07-13 `yj-3-S2`]:
- **압축 맵(3-way 실측)**: fp16 2870MB / int8 1435 / **int8+prune 822 ✅** / **int4(NF4) 718 ✅** / **int4+prune 411 ✅**. int4도 무손실(xlmr fp16 0.7590→int4 0.7588, −0.0002). → 공간은 여러 방법으로 해결, 심지어 4~5모델 여유.
- **⚠️ 근데 앙상블 점수 포화**: 3-way 0.7745 → 4-way(+kd) 0.7751 → **5-way 0.7754 (겨우 +0.0009)**. 약멤버(e5/klue/gte) 희석. **강한 3~4개가 상한**(Roka M5 일치).
- **⚠️ 진짜 병목 = 시간(10분)**: v007 2모델 7:49. 3모델은 **tri_cond(저마진행만 3번째 재추론) 필수**. 4~5모델은 공간 돼도 시간 불가. **→ 공간 더 짜낼 필요 없음, 현재안(3-way int8+prune 822MB)이 최적.**
- **여지 1개**: mdeberta(DeBERTa=진짜 다른 arch, S1 학습중)는 kd/gte보다 diverse할 수 있음 → val_probs 나오면 4-way(mmbert+bge+xlmr+mdeberta) +0.001 검증 가치(단 시간 관건).
- 용량분해: 어휘임베딩이 mmbert 64%/bge·xlmr 45% = pruning이 mmbert에 특히 큼.

**🔄 S2 진행 — S1질문 "데이터축 살아있는 레버?" 답 + 예약2건 실행** [07-14 `yj-3-S2`]:
- **🟢 유일 살아있는 레버 = au/sim** (S1 gen_rescue [GEN]와 정합): **au(7%)는 훨씬 쉬움**(mmbert au-only 0.867 vs sim-only 0.758) + **read-heavy 다른 분포**(read 25% vs sim 12%). **test dummy 5개 전부 sim** → 진짜 지표=**sim-only CV**. au가 전체 CV를 **+0.010 부풀림**(bge 특히 Δ−0.0123→sim-only선 bge·xlmr 동급). ⚠️ sim-only 재가중 easy win은 **+0.0004뿐**(marginal) → 진짜 카드=**au-제거 학습 A/B**.
- **🔴 세션레벨 규칙 = 죽음**: 세션 오답률이 repo/언어/prefix로 안 갈림(균일). 못하는 세션 공통패턴 없음.
- **🟡 스텝별**: step1-2가 최악(acc 62/68%, inspect 48-49%=동전던지기, history없어 스크램블 최악). inspect라 못 고침. 손실은 step1-2 inspect에 집중되나 4중검증 "불가" 영역.
- **✅ honest 반분-CV A/B 완료**(`experiments/exp_data_ab/`, 동일 sim-only val 6537, mmbert 4ep):
  | train | sim-val Macro-F1 |
  |---|---|
  | **sim만(au제거)** | **0.7545** 🥇 **+0.0083** |
  | full(sim+au, 현행) | 0.7462 |
  | augNoau(sim+aug,au제거) | 0.7455 |
  | aug(sim+au+경모5299) | 0.7343 **−0.0119** |
  | **trans(sim+구조보존번역증강3826)** | **0.7464 −0.0081** [07-15 신규] |
  - **🟢 au 제거 = +0.0083 (진짜 데이터축 레버! = S1 질문 답).** au는 read-heavy off-dist라 sim-test 방해. **→ 최종 4-백본 sim-only 학습 권고(전모델 +0.008 기대).**
  - **🔴 [07-15 S2] 번역증강 = −0.0081 해로움 확정.** NLLB 구조보존(코드용어 보존, val누수 방지, history/meta/label 원본유지)으로 경모 3대문제 우회했는데도 음수. **모든 "추가형" 증강(경모/번역)이 sim-val 오염.**
  - **⛔ [07-15 정정 — au제거는 LB로 반증됨]** 위 "au제거 +0.0083 → sim-only 학습 권고"는 **S1 v019 제출 LB 0.71780(−0.069 폭락)으로 반증**됨. **test에 au 상당수 → sim-only는 au에서 OOD 붕괴.** sim-only val엔 au가 없어 생긴 착시. **→ au-포함이 정답, au 제거 절대금지.** **데이터축 결론: 추가형증강·au제거·디코더 전부 사망 = 데이터축 완전 소진(살아있는 레버 없음).**
  - **🔴 경모 증강 = −0.0119 확정 해로움**(honest out-of-sample). augNoau도 sim보다 낮음. **증강 축 완전 종료.**
  - **팀 옛 "au제거 부정적"(상혁 07-05) 반증**: 그건 au를 val에서도 뺐거나 random-split 누수 탓. au는 **train만 제거+sim-only 평가**하니 정반대. S1 "재검증 필요" 의심 정확.
  - ⚠️ **1 fold·1 seed·"test=sim" 가정** → 다른 seed 재확인 + 제출로 LB 확정 필요(M1/M2 CV 뒤집힘 위험). 근거(au off-dist+test sim) 탄탄해 방향 신뢰.

**📦 [07-15 S2→S1 인수인계] 코드특화 인코더 = "decorrelated 강멤버" 유일 미탐색 후보 (S1이 학습 실행)**:
- **동기**: 지금까지 멤버 전부 다국어 **텍스트** 인코더(mmbert/bge/xlmr/mde/klue/koelectra) → 전부 ~90% 일치, 앙상블 천장 0.78(S1 발견4). 근데 태스크는 **코드 액션 예측** → **코드 사전학습 인코더는 다른 신호 = decorrelated 가능성.** 디코더(qwen/hammer)는 decorrelated이나 약해서 기여0이었음 → 코드인코더는 인코더라 더 강할 여지.
- **모델 3종**: `microsoft/codebert-base` · `microsoft/graphcodebert-base` · `microsoft/unixcoder-base` (전부 RoBERTa arch = AutoModel OK, 인터넷 다운로드 확인됨)
- **실행**(기존 하네스 그대로, 새 파일 X. bge와 동일 호출):
  ```
  cd experiments/exp_encoders
  python train.py --data ../../data/processed/train_merged.jsonl --model microsoft/codebert-base --tag codebert --split group --max_length 512
  # graphcodebert-base → --tag graphcodebert / unixcoder-base → --tag unixcoder (GPU 5·6·7 idle, 각 ~40분)
  ```
- **fold 호환 검증완료**: `train_merged.jsonl` group-split seed42 = val ids0 `sess_sim_20260522_024730-step_08` **✅기존 멤버와 동일 fold** → 나온 `runs/<tag>/val_probs.npz`는 S1 앙상블·S2 블렌드하네스에 즉시 편입 가능.
- **평가**: 단일 CV + **앙상블 기여**(mmbert+bge+xlmr에 추가시 Δ) — S2 blend_grid 하네스로 측정. **decorrelated+강이면 편입, 약하거나 redundant면 폐기**(디코더처럼).
- ⚠️ **리스크**: 코드인코더는 영어-코드 BPE라 **한국어 프롬프트(64%)서 약할 수 있음**(UNK/분절). 단독은 낮아도 decorrelated이면 기여 가능(klue 0.73도 기여했듯). test=au포함이므로 학습은 **au-포함(EXCLUDE_AU 미설정)** 유지.

**🟦 S2 병렬 assignment (건우 제안, GPU 안 겹치는 코딩축)**:
> ⚠️ **th실험·직렬화 repositioning은 Roka가 이미 실측 기각**(th=선택기상한 honest음수 종결 / v8 [GEN]재배치 −0.0023 M1 / v7[PACE] −0.0082). 재탐색 비추.
살아있는 축으로:
1. **[✅ 완료 07-13 S2] 3-way tri_cond 패키징 스크립트** → `experiments/exp_3way_pkg/script.py` (v007 확장: N-모델 base+conditional + vocab-prune remap + 수동int8 + per-class bias). **ensemble.json v2**(base/conditional/weights/cond_fraction/bias). **더미 30k 검증 통과**: ①tri_cond 저마진 27%(또는 15%)만 xlmr 재추론=**full 3-way와 동일 0.7749**(고마진행은 3번째가 안 바꿈) ②시간 A6000 2-way 2:41 vs 3-way 3:05=**1.15×** → **T4 추정 ~9:00<10분** ③용량 int8+prune **822MB<1GB** ④출력·remap·int8 dequant 정상. **안전레버: cond_fraction 0.15로 낮추면 T4 ~8:30.** → **S1 신버전 모델 나오면 tag/weight만 교체→즉시 제출.** gen_rescue는 [GEN]학습 모델일때만 훅 추가(현 모델 미해당).
2. th는 "탐색"이 아니라 **조건부 앙상블의 CV-최적 th 1개 고정**(public 스윕=M6 금지)으로만.
3. (여력시) 문샷=합성데이터 증강 — 단 경모 담당축과 조율.

### 🔴 진행 중 (In Progress)

- [x] `건우/yj-3-S1` ✅ **exp007: mmbert@512 + serialize_v2** — 완료(07-05 19:53). **CV Macro-F1 0.7540→0.7606(tuned)**. 클래스별: **write_file 0.99★**(rare인데!, write cue 골격35× 효과 추정)·apply_patch 0.96·edit 0.98·respond 1.00·web_search 0.72·lint 0.65★. ⚠️ 병목=inspect(list 0.49/read 0.58/grep 0.59/glob 0.64, EDA F1 천장~43%). `experiments/exp007_serialize_v2/model/` (+val_probs.npz). ⚠️ **델타 미확정** — 같은 fold baseline(상혁 or 자체) 필요. **model.safetensors 1.23GB(fp32) → 제출 시 fp16(0.62GB) 변환 필요**. S2 exp_serialize_ab와 축 겹침(both 커버)
- [x] `건우/yj-3-S1` ✅ **exp001: klue/roberta-large (diverse, 원본 serialize)** — 완료(07-05 21:05). **CV Macro-F1 0.7379→0.7422(tuned)**. exp007(mmbert+v2 0.7606)보다 낮음 = 단독 약함, **앙상블/증류 재료**. `experiments/exp001_klue_large/model/`(+val_probs.npz). **⚠️ 관찰: klue는 cue 없는 원본 serialize인데도 write_file 1.00** → write_file은 cue 없이도 쉬운 클래스, exp007 write_file 0.99가 write cue 덕분이라 단정 불가. **model 1.35GB(fp32)→fp16 필요**
- [x] `건우/yj-3-S1` ✅ **앙상블 stacking (exp007 + klue-large, 같은 fold)** — 재학습0, val_probs blend. **최적 w=0.6(mmbert):0.4(klue) → CV 0.7647→0.7697(tuned), 단일최고 대비 +0.0091**. diverse 앙상블 첫 실증(klue 단독 약해도 diverse해서 이득). ⚠️ 제출 제약: mmbert+klue fp16=1.29GB>1GB → **증류/양자화/소형멤버** 필요(직접제출 불가, 상한 증거)
- [x] `건우/yj-3-S1` ✅ **상혁 mmbert 제출본 수급·분석** → `team_ref/sanghyuk_v02_mmbert_lb0.77585/`(+NOTES). **제출=plain CLS mmbert + base serialize(open_files), bias=0, tf4.48.3**. common.py의 _prev_cue/extract_features는 미사용 dead code. **상혁 serialize = 우리 exp006 serialize 바이트동일(3000/3000)** → exp006이 그의 baseline 충실 재현. ⚠️ 그의 weight는 전체70k학습이라 CV baseline 불가(우리 fold 오염), val_probs 미포함
- [x] `건우/yj-3-S1` ✅ **exp006: mmbert baseline (상혁 방법 재현)** — 완료(07-06). **CV Macro-F1 0.7679→0.7708(tuned) = 우리 best single** (상혁 LB 0.77585와 정합). 클래스별: write_file 0.99·apply 0.97·web_search 0.75·lint 0.67·list 0.51. `experiments/exp006_mmbert_1024/model/`(+val_probs).
  **⛔ serialize_v2 판정 확정**: exp006(0.7708) vs exp007(0.7606) = **−0.0102, serialize_v2가 오히려 깎음**(web_search 0.75→0.72, write_file 무효). S2 ablation(cue−0.002)과 일치. **serialize 재설계 축 폐기.**
  **🎯 앙상블(같은 fold OOF)**: exp006단독 0.7708 / exp006+klue 0.7744(+0.0036) / **3-way exp006+klue+exp007(0.4/0.3/0.3) 0.7758(+0.0050)**. exp007도 diverse 멤버로 소폭 기여. ⚠️ 3모델 fp16 ~1.9GB>1GB → 증류/양자화/소형멤버로만 제출 가능
- [x] `건우/yj-3-S1` ⛔ **exp010: 규중 RL 입력편집 → mmbert 이식 (07-06 미팅 할당과제)** — 규중 코드(exemplar bank + REINFORCE 정책)를 exp006 mmbert(frozen) backbone으로 이식 완료(`experiments/exp010_rl_edit_mmbert/`, serialize 규중=exp006 바이트동일, pooled 추출만 ModernBert mean-pool로 교체). **결과: gold-리크 평가 +0.0199(0.7674→0.7873)로 혹했으나, gold-free(테스트-유효) 재평가 = −0.0009 ≈ 0.** 리크 원인: confusing 판정에 `pred!=gold` + inspect선택에 gold action 사용(테스트엔 gold 없음). margin만으론 "어느 게 틀렸나" 못 가림 → 편집이 맞던 것도 깨서 상쇄. **⛔ 규중 방식 실전 전이 안 됨 (규중 +4% local도 동일 리크 의심). 팀 공유: 이 축 베팅 금지.** 남은 과제: tool-calling decoder(Hammer, full-FT OOM→LoRA 필요)
- [ ] `yj-3-S1↔S2 조율` **serialize 실험 중복 정리** — S1 exp007=cue+reorder+last_action **묶음(=both)** 학습중. S2 exp_serialize_ab=4변형(base/cue/reorder/both) 분리 하네스. → **중복 회피**: exp007이 both를 커버하므로, S2 하네스는 **base/cue/reorder 3개만** 돌려 개별 축 기여 분리하면 상보적(both 재학습 불필요). exp007 결과 나오면 확정

### ⚠️ 팀 경고 (즉시 공유)

- [ ] `건우/yj-3-S2 검증` 🔴 **경모 validation_report.md의 Macro-F1 0.807은 누수 부풀림 — 진짜 아님** (`team_ref/validation_report.md`). **원인: Split=`random`** → val 7000개 중 **99.5%가 train에도 있는 세션**(같은 세션 다른 step이 train/val에 갈림, history 누적구조상 컨텍스트 누수). **진짜 점수 = group split 우리 exp006 = 0.767** (상혁 LB 0.775와 정합). **0.807 CV − 0.775 LB = 0.032 = 통째로 누수 = CV-LB 격차의 정체.** → **팀에 "0.807 착시 금지, group split로 재실행" 즉시 공유 필요.** ✅단 경모 분석의 *구조/통찰*(inspect 4개 최하위·grep↔read↔list 삼각형·list 과예측 0.41·bias무효)은 전부 우리와 일치=정확. 예시테이블도 inspect 동전던지기(margin~0)·edit↔apply 고확신혼동 확인.

- [ ] `건우/yj-3-S2 검증` **경모 mixed 증강(`augmented_..._5299`) = 도움 안 됨, 증강 축 종료 권장** (`team_ref/kyungmo_validation_v01/AUG_NOTES.md`). 5,299개 검증: ①분포 인위적 균등화(write 7.4%/web 6.9% vs real 2.1/1.8 → test 불균형과 시프트) ②67% 단일턴(history 없음, real 12.9%뿐) ③언어 어긋남(영어 43%/한국어 3개, "mixed 위주" 실패) ④**inspect 라벨 스크램블 그대로 상속**("열어줘"인데 glob). → 경모 실측(하락)+내 V4-c(언어↔클래스 왜곡)+이 검증 전부 일치. **번역·생성 증강 축 종료.** 쓰려면 real 불균형분포+멀티턴+group split CV 필수.

### 🟡 다음 (Next — 착수 준비됨)

- [x] `건우/yj-3-S2` ✅ **exp_serialize_ab: CUE확장·재직렬화 2축 분리 A/B — 완료(07-06, GPU7)** — 결과 **base 0.7562 / cue 0.7542(−0.0020) / reorder 0.7565(+0.0003)**. **⛔ negative: 두 축 다 Macro-F1 못 올림.** cue는 web_search만 깎음(큐 정밀도 낮음), reorder는 노이즈 수준. → EDA v3 예측대로(state/rare어휘 정보는 이미 serialize 텍스트에 존재 + inspect ~43% 바닥이라 재강조 무이득). **함의: serialize 재설계 축은 접어도 됨.** exp007(=both)도 자체 baseline 대비 실질이득 ~0 시사(→ S1 exp006으로 최종 확정). `experiments/exp_serialize_ab/results_all.json`
- [x] `건우/yj-3-S2` ✅ **[07-11] 데이터 축 총정리 — 전 축 검증 완료, 결론 확정** — 남은 5일 삽질 방지용 요약:
  - **inspect 4클래스 분리 = 4중 검증으로 "불가능" 확정**: ①t-SNE(read/grep/list 3개 삼각형 엉킴+glob만 분리) ②수치천장 43~55% ③**규칙 전수탐색: 65%+ 규칙 0개, 최고 51%**(prompt 키워드 무력) ④실제예시("열어줘"인데 정답 read/grep 갈림, 모델확률 0.3 도토리키재기). 혼동구조=read↔grep 쌍둥이(대칭 377), list=과예측 쓰레기통(비대칭), glob=OK. → **손댈 수 없는 영역 확정, 팀도 그만 파야 함**
  - **threshold 재조정 = held-out에서 −0.003(과적합)**. 모든 범위 일관. **팀 "tuned" CV는 과적합 부풀림 = CV-LB 격차 원인.** → 모델 비교는 **raw CV로**. 제대로 하려면 전체 OOF 필요(S1)
  - **turn_index = 신뢰됨(버리지마라)**: hlen/elapsed/budget 세션내 100% 단조. "turn=1 이상"은 elapsed=세션나이아님+budget=tier knob 때문이지 버그 아님
  - **라벨 100% 일관**(다음step 기록 vs 라벨 58,326쌍 일치) = 오라벨 0%, 노이즈청소 축 사망
  - **앙상블 diversity 분석**: exp007(같은mmbert)=이득0 **드롭**. klue(다른백본)=+0.0023. **mmbert+klue=1.2GB>1GB → 양자화 필수**. 도움되는 다양성=다른 강한 텍스트인코더뿐
  - **⛔ LightGBM(ML) 멤버 실패 → 제외**: 단독 0.41(텍스트 못읽음), 불일치49%지만 앙상블 +0.0002. **비-텍스트 모델은 diverse해도 너무 약해 무용**. `experiments/exp_lgbm/` 폐기
  - **→ 0.79 향 유일 경로 = 강한 텍스트 인코더 2개(mmbert+klue/decoder) 양자화(int8)로 1GB 맞춰 앙상블 = S1 축.** + Public 과적합 금지(Private shuffle 대비 로버스트 유지)
- [ ] `건우/yj-3-S2` 🆕**[07-06미팅, 데이터셋축]** **언어/구조 진단 3종 (순수 EDA)** — 미팅 데이터 의문 실측.
  - [x] ✅**(a) turn_index 신뢰도 — 완료**: **turn_index는 신뢰됨(버리지 마라).** hlen==min(2*(turn-1),12) **100% 일치**, 세션내 elapsed 단조증가·budget 단조감소·turn~elapsed 상관 **+1.000**, turn=1 history빈 **100%**. ⚠️단 turn=1이 "생판 처음"은 아님(elapsed>300s 47.6%): 이유는 **elapsed=세션나이 아님 + budget=tier knob**(free40k/pro118k/ent160k)이지 turn 버그 아님. "turn=1 read_file"=inspect 스크램블(생성기 설계, 라벨 100%일관). **미팅 "turn빼자" 직관 반증.**
  - [ ] (b) **au=sim 언어변환본 가설**: sim↔au 내용/구조 매칭 시도.
  - [ ] (c) **언어별(ko/en/mixed) val F1 분해**: exp006 val_probs로 mixed가 정말 갉아먹는지 우리 CV 재확인 → 경모 증강·용욱 LoRA 근거.
- [ ] `건우/yj-3-S2` 🆕**[07-06, 데이터 사실 확정]** **라벨 100% 일관 = 오라벨 0% (실측완료, 문서화 대기)** — 다음step 기록 action과 라벨 58,326쌍 **100% 일치** → CSV 오라벨 없음, "노이즈"는 prompt↔라벨 설계지 라벨오류 아님. **노이즈 청소 축 완전 사망 확정.** → EDA_REPORT §7.6 **V6**로 추가 예정(팀이 아직 "노이즈 제거" 논의 中이라 공유 가치)
- [ ] `경모` 🆕**[07-06 확정]** **언어 균형 증강** — mixed 우선, ko/en/mixed **4:4:4 비율 맞춤**, Claude/GPT **구독**으로(API 비용X), **코딩용어·함수명 보존**, 배치로 분할(할루시네이션). ⚠️ S2 경고(EDA v3 V4-c): 언어↔클래스 커플링 왜곡 위험 → 증강 후 언어별 클래스 prior 보존 확인 권장
- [ ] `상혁` 🆕**[07-06 확정]** **AI생성체/한국어 도메인 SFT** — "우리만의 KO ModernBert" 도메인 적응(MLM/TAPT), GPU 있는 상혁 코드. mmBERT가 한국어 혼재/AI체 정렬 약하다는 가설 대응
- [ ] `건우(S1?)` 🆕**[07-06]** **규중 코드(RL/다른 입력형식) ModernBert 백본 이식·테스트** — 규중 val↑인데 백본이 ModernBert 아님. 노션 코드. ⚠️ **모델 축 = S1 담당 확인 필요**(S2는 데이터셋 축)
- [ ] `팀(미할당)` 🆕**[07-06 아이디어]** **후처리 KNN 가점** — baseline 출력 + test 임베딩 위치 기반 근접 비율 가점. ⚠️ "train이 진짜냐" 함정 언급됨
- [ ] `건우/claude` **exp001: Qwen2.5-0.5B + LoRA 분류 head 파이프라인 구축** — 모델 리서치 Tier 1 최우선. 상혁 `serialize()` 재사용, tokenizer 교체. 아키텍처 완전 diverse(causal LM). → `src/` 스크립트 초안부터
- [ ] `건우/claude` **exp002: multilingual-e5-base + head** — 임베딩 objective가 mDeBERTa와 다름, 앙상블 재료. exp001과 병렬
- [ ] `건우/claude` ⏬**[강등]** **exp003: au sample 취급 ablation** — 상혁 코드에 `sample_weight` 추가해 (i)그대로 / (ii)downweight 0.5 / (iii)제외 3조건 val Macro-F1 비교. ⚠️ **팀 실측상 노이즈 제거는 하락** — 우리 CV로 재확인 정도의 낮은 우선순위
- [ ] `건우/claude` ⏬**[강등]** **`last_action` categorical feature 실험** — Naive Markov top1 22.94%(majority +6.98%p). ⚠️ **용욱이 step/open_files fusion 이미 시도 → eval +0.01 미미**. LightGBM 별도 축이면 시도 가치 있으나 기대 낮춤
- [ ] `건우/claude` ⏬**[강등]** **rare class booster rule 검증** — `list_directory→write_file`(lift 6.88×) 등 전이 rule. ⚠️ 상혁 "헷갈리는 4클래스만/super-class 2단계도 안 됨" 실측
- [ ] `상혁` **CUE 정규식 확장 ablation** — au 어휘 `typecheck/요약/됐다/다시`를 `_explore_cues`에 추가 시 rare F1 변화 (노이즈 감사 축C 시사)
- [ ] `상혁/경모` **DACON 게시판 prefix 의미 문의** — `sess_sim_` vs `sess_au_`가 뭔지, test 30k의 origin이 sim/au/혼재인지 공식 확인
- [ ] `용욱` **유사 tool-calling 벤치 데이터 조사** — pretraining/증강 재료 (상혁 "알아오세요")
- [x] `건우` ✅**[해결됨]** **transformers 업그레이드 서버 통과 = 실증** — 건우 mmbert_base 제출이 07-02 서버 채점 완료(ModernBERT arch, tf≥4.48). probe 불필요. → ModernBERT/mmBERT/T5Gemma2(규중) 계열 서버 사용 가능 확정

### ✅ 완료 (Done)

- [x] `건우/claude` EDA #01 — 클래스분포·prefix(sim/au)·step편향·turn_index·history 전이매트릭스 → `notebooks/01_eda_findings.md`
- [x] `건우/claude` 모델 리서치 — 다국어 encoder/causal LM 후보 Tier 분류, 제약(tf 4.46.3, ≤1GB, ≥50 smp/s) → `notebooks/02_model_research.md`
- [x] `건우/claude` 데이터 노이즈 감사 — 축A(토큰) / 축B(step gap) / 축C(sim vs au) → `notebooks/03_data_noise_audit.md`, `data/processed/noise_candidates.csv`
- [x] `건우/claude` **데이터 leak 검증** — 세션 안 스텝K 라벨 = 스텝M(M>K) history 안 값 (231,664쌍 100% 일치). 로컬 test 5샘플 100% lookup 성공 → `notebooks/04_data_leak_finding.md`
- [x] `건우/claude` **CV 세팅** — `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`. 상혁 val과 100% 정렬 → `notebooks/05_cv_setup.md`, `src/cv.py`, `data/processed/cv_splits/fold_assignments.csv`
- [x] `건우/claude` **baseline OOF (TF-IDF+LogReg 10-fold)** — 전체 Macro-F1 0.4312 ± 0.0069 (fold별 0.42~0.45). respond_only 0.998, write_file 0.976 극단 편향, glob/list 0.19 최저 → `notebooks/06_baseline_oof_result.md`, `data/processed/baseline_oof.{csv,_probs.npz}`
- [x] `건우/claude` 팀 카톡 요약 → `team_chat/SUMMARY.md`
- [x] `건우/claude` 상혁 v01 코드 아카이브 → `team_ref/sanghyuk_v01_lb0.77585/`
- [x] `건우/claude` 공유 대시보드 신설 → `STATUS.md` (이 파일)
- [x] `건우/yj-3-S2` **EDA v3 — 리포트 전수 재검증 + 심층 신규발견** → `docs/reports/EDA_REPORT.md §7.6`. ①기존 리포트 ~60개 수치 전부 실측 일치 확인. ②V1: 중복prompt 충돌은 노이즈 아니라 상태-조건부 규칙(prompt만 74.9%→+last_action/turn/n_open 89.7%). ③V1-b: 용욱 contradiction 207개=inspect 100%, state로도 16%만 설명=inspect 바닥(~43% 천장), 발견1과 상보적. ④V3: 512 truncation 3.4%(41.5% 재현실패) + META 맨끝 순서결함. ⑤V4: 실제 prompt언어 mixed57.7/en27.4/ko_pure14.8(선언 ko64.3과 상이), ko_pure→실행형 극단편향→번역증강 경고. ⑤V5: rare 판별어휘(타입체크39:0, 골격53:1)
- [x] `건우/yj-3-S2` **serialize A/B 하네스 구축·dry-run 검증** → `src/serialize_variants.py` + `experiments/exp_serialize_ab/run_ab.py`. control=상혁 baseline 바이트동일(selftest 3000/3000). dry-run 실측 rare cue 정밀도: **newf→write_file 76.4%(36×lift), ws→web_search 18.7%(10×), tc→lint 25.1%(7.7×)**. GPU 학습만 남음(→ 다음 칸 exp_serialize_ab)

- [x] `건우/yj-3-S1` **Roka(jsj) 0.79026 완전 해부 + 재현** — jeong브랜치 DEBATE.md 1차출처 확인: th85=base xv6(v6,w0.6)+cond[mde(v6,w0.15),xv4(v4,w0.25)]+th0.85+gen_rescue. 0.79 실동력=**gen_rescue(+0.0031, 입력축 고전이)**, 가중/th=저전이. gen_rescue 코드 fetch·byte동일 검증. **v016 정확재현 LB 0.78326** → 재현gap **δ=−0.0070**(그들 체크포인트 미공개, 코드만; Roka 자체 m1t3도 −0.0089 동류). 단일모델은 우리≈그들(CV 0.7631 vs 0.7645). → `submissions/v015,v016_roka_exact_th85_w6.zip`
- [x] `건우/yj-3-S1` **v017 mht12 앙상블 = 팀 신SOTA 0.78700** (535s) — base=bge(.4)+mde(.2)@320, cond=mmbert-**mht12**@768/mht12(.4) 저마진20%, gen_rescue, per-member 직렬화/mht. mht12=history 8→12(입력축)라 LB 전이 큼: **+0.00315 vs v012**. cf0.2로 시간눌렀는데도(535s, 65s여유) → **cf↑로 mht12 커버리지 더 주면 상승 여지**. Roka와 gap 0.00326으로 축소 → `submissions/v017_mht12_ens_bge_mde.zip`
- [🔄] `건우/yj-3-S1` **mde@384 fold0 프로브** (GPU4, tmux adrepro:mde384, ETA ~1.5h) — mde12@320(0.7472)와 max_len만 다른 클린비교. >0.7472면 FULL승격, ≤면 폐기(xlmr 길이축처럼)

### ⛔ 블록 / 대기 (Blocked)

- [ ] `건우` **상혁 0.775 백본 weight + config/common/input형식 수급** — 07-02 input형식·config·common **노션 업로드됨**, 07-05 **0.775 백본 weight 용욱에게 메일 전달됨**. → 노션/용욱에게서 받아오면 blocked 해제. teacher logits npz는 아직 확인 필요
- [ ] **prefix(sim/au) 의미 확정** — 경모가 DACON 문의 예정(07-03), 답변 아직 없음. 단 au 취급(exp003)은 **노이즈 제거 축이 팀 실측 부정적이라 우선순위 하락**

---

## 🧪 실험 트래커

> 새 실험은 착수 **전에** `📋 계획` 행부터 추가. `experiments/{exp_id}/`에 config+log+metrics 저장.
> rare-F1은 `web_search / write_file / lint_or_typecheck` 순. val split은 상혁 규약(동일 val_id 순서 `val_probs.npz`) 준수 → 앙상블 재료화.

**상태 범례**: `📋 계획` · `🔄 학습중` · `✅ 완료` · `❌ 실패/폐기` · `⭐ 앙상블 채택`

| exp_id | 모델 / 방법 | 조건 | 학습데이터 | val Macro-F1 | 서버 LB | rare-F1 (web/write/lint) | 상태 | 산출물 경로 | 비고 |
|---|---|---|---|---|---|---|---|---|---|
| baseline_oof | TF-IDF + LogReg | 공식 baseline pkl 파라미터 재현, 10-fold OOF | full 70k | **0.4312 ± 0.0069** | — | 0.2483 / 0.9759 / 0.2603 | ✅ 완료 | `data/processed/baseline_oof.{csv,_probs.npz}` | 계측기 신뢰 확인용. respond_only 0.998, write_file 0.976 극단. glob/list 0.19 최저 |
| team_mdeberta_ko_cls_3ep | mDeBERTa-v3-base CLS pool | 3 epoch, KD/AWP 없음 | full 70k | — | **0.6954392568** | — | ✅ 완료(팀 실측) | 팀원 공유 (경로 미확인) | 07-02 팀에서 공유. 서버 실측 점수 — 우리 OOF 0.4312 → mDeBERTa 서버 0.6954 (+26%p) |
| ref_sanghyuk_v01 | mDeBERTa-v3-base + KD + AWP | full pipeline | full 70k | — | **0.77585** | — | ✅ 완료(팀) | `team_ref/sanghyuk_v01_lb0.77585/` | 팀 최고 4위. 재현 검증 pending(weight 미수급) |
| exp001 | **klue/roberta-large + 원본 serialize** | 5 epoch, CLS, weighted CE, thresh tune, group split, add_openfiles, batch32, fp16 | full 70k | 🔄 학습중 | — | — | 🔄 학습중(yj-3 GPU5, bg bqho121fd) | `experiments/exp001_klue_large/` | 한국어 native(337M, ≤1GB single 제출가능). 상혁 mmbert와 diverse. **07-05 yj-3에서 재시작**(옛 3090 run 중단분 대체). env intern_aichallenge |
| exp001_qwen | Qwen2.5-0.5B + LoRA head | 4 epoch, 3090 | full 70k | — | — | — | 📋 계획 | `experiments/exp001_qwen05_lora/` | Tier1 대안. causal LM 완전 diverse. exp001 결과 본 뒤 |
| exp002 | multilingual-e5-base + head | — | full 70k | — | — | — | 📋 계획 | `experiments/exp002_e5base/` | 임베딩 objective diverse, 앙상블 |
| **exp004** | **gte-multilingual-base @ max_len=1024** | 상혁 serialize 재사용, mean/attn pool, no-KD 우선 | full 70k | 📋 계획 | — | — | ⏬ 강등(앙상블 멤버) | `experiments/exp004_gte_1024/` | ~~long-context 메인~~ → **강등**: mmbert_base(SOTA)가 이미 modern+long-capable이라 gte의 long-context 가치 소멸. gte는 diverse 앙상블 멤버로만. ✅ 게이트 통과 기록 유지: tf4.46.3+trust_remote_code 로드OK, fp16 0.61GB, T4 ~93smp/s |
| **exp006a** | **mmbert_base @ 512 (baseline)** | CLS+5ep+bias+open_files | full 70k | — | — | — | ⏸ 상혁 수급 예정 | (상혁 `model/`+`val_probs.npz`) | baseline은 재학습 대신 **상혁한테 받기로**(07-05). 파이프라인은 스모크로 검증됨. ⚠️ 앙상블/A-B 비교 시 상혁 val fold가 우리 exp007과 같은지 id 대조 필요 |
| **exp007** | **mmbert_base @ 512 + serialize_v2** | CLS+5ep+bias+open_files, **serialize_v2**(last_action 표면화 + lint/write cue + META재배치) | full 70k | 🔄 학습중 | — | — | 🔄 학습중(GPU6, bg btt4snlo9) | `experiments/exp007_serialize_v2/` | **현재 메인 실험**(EDA v3 §7.6 A). 데이터검증: lint 17.5×·write 35× lift, last_action 충돌55→30%. 목표: write_file/lint rare-F1↑. baseline(상혁 or exp006a) 대비 델타로 판정 |
| **exp006b** | mmbert_base @ 1024 (곁다리) | exp006a + `--max_length 1024` | full 70k | 📋 계획 | — | — | ⏬ 저기대(곁다리) | `experiments/exp006_mmbert_1024/` | ⚠️ **재평가**: 512 truncation 3.2%뿐(41.5% 오측정 정정). 기대이득 미미. 코드0수정이라 exp006a와 페어로 확인만 |
| **exp005** | **mDeBERTa-v3-base @ max_len=1024 (대조군)** | 상혁 train.py에서 MAX_LENGTH만 512→1024, no-KD | full 70k | 📋 계획 | — | — | 📋 계획 | `experiments/exp005_mdeberta_1024/` | context 효과 격리 측정. mdeberta가 >512 처리 가능한지도 검증. no-KD baseline은 팀 실측 0.695 기준 |
| **exp_serialize_ab** | **mmbert_base @ 512 + serialize 2축 분리** | **base/cue/reorder 3변형**(both은 exp007 커버→제외, S1 조율), GroupKFold(seed42, n5) OOF eval_folds1(val 14,001), weighted CE, 3ep, batch32, raw-argmax | full 70k | **base 0.7562 / cue 0.7542 / reorder 0.7565** | — | base(0.72/0.99/0.63) cue(0.69/0.99/0.63) reorder(0.72/0.99/0.63) | ✅ 완료(07-06 03:2x, GPU7) | `experiments/exp_serialize_ab/` (results_all.json + {v}/val_probs.npz) | **⛔ negative result**: cue **−0.0020**(ws 0.72→0.69, 큐 정밀도 낮아 노이즈), reorder **+0.0003**(노이즈 수준). **두 축 다 Macro-F1 못 올림** → EDA v3 예측 일치(정보 이미 텍스트에 있음+inspect ~43% 바닥). ⚠️ **exp007 델타 해석**: cue·reorder 개별무이득 → exp007 serialize_v2(=both) 실질이득 ~0 시사(자체 baseline은 S1 exp006으로 확정 예정). ⚠️ val fold가 S1(7,000)과 달라(14,001) 절대치 직접비교 불가, 하네스 **내부** 델타만 유효 |
| **exp_lgbm** | 구조화 피처 + **LightGBM(트리 ML, 비-딥러닝)** | 40+ 피처(turn/last_action/n_open/ci/dirty/prompt신호), exp006 val fold 미러링, class_weight balanced, n_jobs=8 | full 70k | **0.4113** | — | — | **⛔ 폐기(ML 제외)** | `experiments/exp_lgbm/` | 앙상블 diverse 멤버 시도. **텍스트 못 읽어 단독 0.41**(TF-IDF baseline급). mmbert와 불일치 49%지만 **너무 약해 앙상블 +0.0002**. 비-텍스트 모델은 다양해도 무용 → **ML 축 제외 확정**. ⚠️ n_jobs=-1로 56코어 점유해 서버 load 72(타인 민폐)—8코어로 재실행함 |
| exp003a | mDeBERTa (상혁 코드) | au **그대로** | full 70k | — | — | — | 📋 계획 | `experiments/exp003_au_ablation/a/` | au 취급 3조건 비교 baseline |
| exp003b | mDeBERTa (상혁 코드) | au **downweight 0.5** | 70k, w=0.5 | — | — | — | 📋 계획 | `experiments/exp003_au_ablation/b/` | noise_candidates.csv 활용 |
| exp003c | mDeBERTa (상혁 코드) | au **제외** | sim 64,975 | — | — | — | 📋 계획 | `experiments/exp003_au_ablation/c/` | test가 sim이면 최적 가설 |
| exp_leak_lookup | Lookup-only (모델 없음) | 세션내 lookup + 실패시 majority | full 70k | — | — | — | ⛔ 보류 (상혁 반대) | `submissions/vXXX_lookup.zip` | data leak 서버 검증용. 상혁 07-03 "규정위반·불공정 소지" → 제출 보류 |

| v016 | **Roka th85 정확재현** (xv6+mde+xv4 tri_cond) | w0.6/0.15/0.25, th0.85, gen_rescue, per-member v6/v4 | full 70k | CV 0.7704 | **0.78326** (9:22) | — | ✅ 완료(S1) | `submissions/v016_roka_exact_th85_w6.zip` | 재현gap δ=−0.0070(체크포인트부재). Roka 0.79026과 −0.0070 |
| **v017** | **mht12 앙상블** (bge+mde base, mmbert-mht12@768 cond) | w0.4/0.2/0.4, cf0.2, gen_rescue, mht12=history12 | full 70k | CV 0.7777 | **⭐0.78700** (8:55) | — | ⭐앙상블 채택(S1 신SOTA) | `submissions/v017_mht12_ens_bge_mde.zip` | **팀 신기록**. mht12 입력축 +0.00315 vs v012. cf↑ 여지(535s<600) |
| mde_384_probe | mDeBERTa-v3-base @384 (mht=8) | mde12@320와 max_len만 상이, 12ep b64 FGM | full 70k | 🔄 fold0 학습중 | — | — | 🔄 학습중(GPU4) | `experiments/.../mdeberta_384_12ep_f0` | >0.7472(@320)면 FULL승격. ETA~40m |
| **v018** | **mmbert 해방** (mht12@768 전행 base + bge/mde cond) | w mht.4/bge.4/mde.2, cf0.5, gen_rescue | full 70k | CV 0.7797 | 🔜제출대기 (~445-525s) | — | 🔜 대기(S1) | `submissions/v018_mhtbase_full_bge_mde_cond.zip` | v017 상위호환(+0.002, mmbert 20%→전행). 예상 ~0.789 |
| klue_512_probe | klue/roberta-large @512 mht8 | v6 FGM LLRD 8ep b32 fold0 | full 70k | 🔄 학습중 | — | — | 🔄 학습중(GPU5) | `.../klue_large_512_f0` | 한국어native=decorrelate 베팅. 강+diverse면 편입 |
| koel_512_probe | koelectra-base-v3 @512 mht8 | v6 FGM LLRD 8ep b48 fold0 | full 70k | 🔄 학습중 | — | — | 🔄 학습중(GPU6) | `.../koel_512_f0` | ELECTRA decorrelated 확인됨(0.71). 약점 극복 시도 |
| mmbertv4_probe | mmBERT-base @768 mht12 **v4직렬화** | v6판과 직렬화만 상이(b32 tok-FGM 동일) | full 70k | 🔄 학습중 | — | — | 🔄 학습중(GPU7) | `.../mmbert_mht12_v4_f0` | **Roka식 same-backbone 변형**. 강함보장(~0.76), redundancy 리스크 |

*표는 실험이 늘면 계속 추가. 리더보드 제출본은 `submissions/v###_*.zip`으로 버전 매핑.*

---

## 🔗 빠른 참조

| 항목 | 값 |
|---|---|
| 개발 서버 | **yj-3 (A6000 48GB, GPU5-7 idle) — conda env `intern_aichallenge` 신규 생성 필요** (기존 3090 env는 대전 소재라 여기선 접근 불가) |
| **평가서버 환경 (DACON rules 확정)** | Ubuntu 22.04.5 · T4 16GB · 3vCPU · 12GB RAM · **Python 3.11.15 · CUDA 12.8 · torch==2.7.1+cu128 · 기본 transformers==4.46.3 · numpy==1.26.4 · pandas==2.0.3** · sklearn/scipy/tqdm |
| **✅ transformers 업그레이드 = 실측 성공** | 건우 **mmbert_base(ModernBERT arch, tf≥4.48)** 제출이 **서버 채점 완료(07-02, 추론 2분51초)**. `requirements.txt` 상위 tf 업그레이드가 서버 설치·실행 통과함이 **실증됨**. → **ModernBERT/mmBERT 계열 사용 가능**. 규칙의 "고정버전 변경 시 설치에러 가능"은 경고일 뿐, tf 업그레이드는 실제로 통과 |
| 제출 제약 | submit.zip ≤ 1GB · 추론 ≤ 10분 · 설치 ≤ 10분 · 오프라인 · T4 16GB · 1일 10회 |
| 처리량 요구 | 30,000건 / 10분 = **≥ 50 samples/sec** (배치 필수) |
| 평가지표 | Macro-F1 (14 클래스, imbalance 8.8:1) |
| 데이터 | train 70k (`data/raw/train.jsonl` + `train_labels.csv`), prefix `sess_sim_` 93% / `sess_au_` 7% |
| 노이즈 후보 | `data/processed/noise_candidates.csv` (7,073 rows / 6,784 uid) |
| 대회 링크 | https://dacon.io/competitions/official/236694/overview/description |
