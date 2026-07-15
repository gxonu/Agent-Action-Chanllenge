# 경모 mixed 증강 검증 — `augmented_only_clean_after_wrapper_removal_5299`

## 메타
- **받은 날짜**: 2026-07-12
- **저자**: 경모
- **파일**: `team_ref/augmented_only_clean_after_wrapper_removal_5299.jsonl` (+ `_labels.csv`), 5,299개
- **목적**: mixed 언어 클래스 성능 보강 (미팅서 mixed val이 ko/en보다 낮다 → 번역증강으로 mixed 늘리기)
- **방법(추정)**: ko/en 실제 샘플을 mixed로 LLM 번역, 라벨은 원본 상속. id = `aug_mixed_llm_run0001_...`

## 🔴 검증 결론 (yj-3-S2, 2026-07-12): 도움 안 될 가능성 높음 — 증강 축 종료 권장

경모 미팅 실측("점수 오히려 약간 떨어짐")과 정합. 이유 4가지:

### ① 클래스 분포가 real과 완전 다름 (인위적 균등화)
- 증강은 거의 **균등(~7%씩)**: write_file 7.4%(real 2.1), web_search 6.9%(real 1.8), plan_task 7.6%(real 3.8), ask_user 7.2%(real 3.9). edit_file은 7.0%(real 16.0)로 축소.
- rare를 3~4배 부풀림 → **test(real처럼 불균형)와 분포 시프트** → 학습 시 오히려 예측 왜곡.

### ② 구조 mismatch: 67%가 history 없는 단일턴 (real은 12.9%만 빈값)
- step_1이 3,536/5,299. real은 멀티턴 누적 구조인데 증강은 대부분 맥락 없는 단발.

### ③ 언어가 의도와 어긋남
- 실제 prompt 언어: mixed 3,027 + **영어 2,269(43%)** + 한국어 **3개**. "mixed 위주" 의도였으나 번역이 덜 돼 43%가 영어로 남고 ko는 사실상 0.

### ④ 치명적 — inspect 라벨이 스크램블 그대로 상속됨
- 원본의 뒤엉킨 inspect 라벨을 그대로 물려받음. 예(전부 glob_pattern 라벨):
  - "parser.go then here **열어줘**" (→read인데 glob)
  - "service.yaml **열어서** 보여줘" (→read인데 glob)
  - "그 field **grep으로**" (grep이라면서 glob)
  - "workflow **살펴봐줘**" (→read인데 glob)
- inspect는 라벨 자체가 랜덤이라 **증강으로 못 고침**. 오히려 "깨끗한 예시"를 만들면 scramble된 test와 더 어긋남.

## 함의 / 권장
- **증강 축(번역·생성) 종료 권장.** 팀 실측(경모 1000개·이 5299개 다 하락/무효) + 내 EDA(V4-c 언어↔클래스 왜곡 경고) + 이 검증이 전부 같은 방향.
- 굳이 쓰려면: **real과 같은 불균형 분포 + 멀티턴 + 깨끗한 rare-separable 클래스(write/web)만** 조심스럽게. 단 이미 F1 0.99/0.77이라 여지 적음.
- **반드시 group split CV로 진짜 효과 확인** (random split이면 또 누수 착시). [[kyungmo_validation_v01/NOTES]]
