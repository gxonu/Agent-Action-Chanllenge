# Validation 분석 보고서

## 1. 실행 설정

- Split 방식: `random`
- Seed: `42`
- Validation 비율: `0.1`
- 사용 모델: `mmbert_open`
- Max length: `512`
- Batch size: `64`

## 2. 전체 성능

- 전체 데이터 수: **70000**
- Train 데이터 수: **63000**
- Validation 데이터 수: **7000**
- 맞춘 데이터 수: **5549**
- 틀린 데이터 수: **1451**
- Accuracy: **0.792714**
- Macro-F1: **0.807233**
- Weighted-F1: **0.795519**
- Micro-F1: **0.792714**
- Bias 적용 전 Macro-F1: **0.807233**
- Bias 적용에 따른 변화: **+0.000000**

## 3. 클래스별 성능

| class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| read_file | 0.5988023952095808 | 0.6479481641468683 | 0.6224066390041494 | 926.0 |
| grep_search | 0.7991071428571429 | 0.5418768920282543 | 0.6458208057727 | 991.0 |
| list_directory | 0.4132231404958678 | 0.6928406466512702 | 0.5176876617773943 | 433.0 |
| glob_pattern | 0.7319148936170212 | 0.6502835538752363 | 0.6886886886886887 | 529.0 |
| edit_file | 0.9918181818181818 | 0.9767233661593554 | 0.984212900315742 | 1117.0 |
| write_file | 0.9932885906040269 | 1.0 | 0.9966329966329966 | 148.0 |
| apply_patch | 0.948 | 0.983402489626556 | 0.9653767820773932 | 482.0 |
| run_bash | 0.8637274549098196 | 0.8500986193293886 | 0.856858846918489 | 507.0 |
| run_tests | 0.8466386554621849 | 0.8837719298245614 | 0.8648068669527897 | 456.0 |
| lint_or_typecheck | 0.7926267281105991 | 0.7543859649122807 | 0.7730337078651686 | 228.0 |
| ask_user | 0.8360655737704918 | 0.7555555555555555 | 0.7937743190661478 | 270.0 |
| plan_task | 0.7673611111111112 | 0.8246268656716418 | 0.7949640287769785 | 268.0 |
| web_search | 0.762589928057554 | 0.8346456692913385 | 0.7969924812030075 | 127.0 |
| respond_only | 1.0 | 1.0 | 1.0 | 518.0 |
| accuracy | 0.7927142857142857 | 0.7927142857142857 | 0.7927142857142857 | 0.7927142857142857 |
| macro avg | 0.8103688425731129 | 0.8140114083623077 | 0.8072326232179748 | 7000.0 |
| weighted avg | 0.8107505370988988 | 0.7927142857142857 | 0.7955185236149234 | 7000.0 |

## 4. 주요 오분류 조합

| true_action | pred_action | count |
| --- | --- | --- |
| grep_search | read_file | 246 |
| read_file | list_directory | 197 |
| grep_search | list_directory | 148 |
| list_directory | read_file | 83 |
| glob_pattern | list_directory | 81 |
| read_file | grep_search | 78 |
| glob_pattern | read_file | 73 |
| grep_search | glob_pattern | 56 |
| ask_user | plan_task | 52 |
| read_file | glob_pattern | 48 |
| run_bash | run_tests | 45 |
| run_tests | run_bash | 38 |
| plan_task | ask_user | 32 |
| run_bash | lint_or_typecheck | 30 |
| glob_pattern | grep_search | 29 |
| lint_or_typecheck | run_bash | 29 |
| list_directory | grep_search | 27 |
| lint_or_typecheck | run_tests | 27 |
| edit_file | apply_patch | 26 |
| list_directory | glob_pattern | 22 |
| plan_task | web_search | 15 |
| run_tests | lint_or_typecheck | 15 |
| ask_user | web_search | 14 |
| web_search | plan_task | 13 |
| apply_patch | edit_file | 8 |
| web_search | ask_user | 8 |
| grep_search | web_search | 3 |
| grep_search | run_bash | 1 |
| glob_pattern | web_search | 1 |
| glob_pattern | plan_task | 1 |

## 5. 확신도가 높은 오답

| id | true_action | pred_action | pred_probability | true_probability | prediction_margin | top2_action | current_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sess_sim_20260522_007828-step_01 | read_file | write_file | 0.9999316 | 3.6285417e-06 | 0.99989665 | list_directory | 이벤트 적재 검증용 dag를 하나 새로 만들려고요. 기존 dbt_project.yml 보면서 패턴 좀 익히고 싶어요 가볍게 |
| sess_sim_20260522_021882-step_08 | edit_file | apply_patch | 0.9995161 | 0.00048217535 | 0.9990339 | edit_file | 근데 session.py에 그 단계 로직 넣어줘 |
| sess_sim_20260522_035072-step_09 | edit_file | apply_patch | 0.998654 | 0.0013406131 | 0.9973134 | edit_file | 역시 poetry인데 Dockerfile은 pip install requirements.txt 하고 있어서 깨지는 거였네. Dockerfile 그 부분 poetry install로 바꿔줘 한번 더 |
| sess_sim_20260522_031147-step_05 | lint_or_typecheck | run_bash | 0.99843866 | 0.0015608886 | 0.9968778 | lint_or_typecheck | now try a clean build to confirm the symbol resolves |
| sess_sim_20260522_041995-step_03 | plan_task | ask_user | 0.996979 | 0.0028667164 | 0.99411225 | plan_task | 없는 거 확인. 그럼 설계부터 잡고 가자, 단계 좀 쪼개줘 |
| sess_sim_20260522_013159-step_09 | edit_file | apply_patch | 0.99431276 | 0.005685899 | 0.98862684 | edit_file | 잠깐 yeah there's a Commands section. drop the make lint line in there too! |
| sess_sim_20260522_012046-step_06 | edit_file | apply_patch | 0.99379224 | 0.006192898 | 0.9875993 | edit_file | _revoke_jti에서 타입 힌트 빠진거랑 안쓰는 import 때문이네. preprocess 쪽 정리해줘 |
| sess_sim_20260522_041438-step_06 | run_bash | run_tests | 0.9923415 | 0.007062187 | 0.9852793 | run_bash | fmt 한번 정리해주고 급해 |
| sess_sim_20260522_027895-step_08 | edit_file | apply_patch | 0.99025166 | 0.009708086 | 0.98054355 | edit_file | oh the limiter resets per test method so the counter never crosses. i need the 11 calls in one method against the same bucket. fix the test thanks |
| sess_sim_20260522_023073-step_03 | ask_user | web_search | 0.98984176 | 0.0042726425 | 0.98397887 | plan_task | 음 pg pool 직접 쓰는 구조네. 이 패턴 next.js에서 connection 새는 거 흔하다던데 맞아? 공식 문서 한번 찾아봐줘 |
| sess_au_816338_009-step_03 | edit_file | apply_patch | 0.9896718 | 0.010328153 | 0.97934365 | edit_file | query랑 withTransaction 둘 다 제네릭 시그니처 손봐야 해서 두 함수 같이 고쳐줘 |
| sess_sim_20260522_003903-step_04 | plan_task | ask_user | 0.9893946 | 0.00869474 | 0.98069984 | plan_task | 신규 입사자라 이 인프라 전체 구조를 파악해야 해. 어디부터 보면 좋을지 단계 좀 잡아줘 |
| sess_sim_20260522_017950-step_07 | ask_user | web_search | 0.98850703 | 0.004402327 | 0.98141694 | plan_task | we keep calling it UserCard everywhere but half the codebase still says Config. i want to standardize on UserCard. before we touch anything, lay out the steps for now |
| sess_sim_20260522_045301-step_06 | edit_file | apply_patch | 0.98839873 | 0.011597316 | 0.9768014 | edit_file | btw good, three mutation methods. add the audit-write calls into service + a new repository method — that's two files so patch both consistently |
| sess_sim_20260522_042563-step_09 | plan_task | web_search | 0.9861886 | 0.009047073 | 0.9771415 | plan_task | 이제 전처리에 토큰 길이 기준으로 자르는 로직 새로 넣고 싶은데, 일단 어디서부터 손대야 할지 plan부터 좀 짜줘요 |
| sess_sim_20260522_030116-step_07 | edit_file | apply_patch | 0.9848082 | 0.015189007 | 0.9696192 | edit_file | 생성된 거 app에서 새 아이콘 prop 쓸 수 있게 연결 |
| sess_sim_20260522_013159-step_10 | edit_file | apply_patch | 0.98307645 | 0.016914764 | 0.96616167 | edit_file | images.remotePatterns가 비어있네models/marts/dim_users.sql CDN 도메인 하나 추가해줘 |
| sess_sim_20260522_032563-step_04 | plan_task | web_search | 0.9826042 | 0.004731626 | 0.96994174 | ask_user | 하는 김에 User라는 게 여러 군데 걸리네. 이거 환경변수 기준으로 갈지 파일 기준으로 갈지 좀 헷갈리는데, 어느 쪽으로 통일할까? |
| sess_sim_20260522_046125-step_07 | edit_file | apply_patch | 0.98185927 | 0.018124426 | 0.96373487 | edit_file | 아 둘이 비교 규칙이 미묘하게 다르다. 문자열 비교 vs 튜플 비교. 양쪽 다 튜플 비교로 통일해서 같이 고쳐 |
| sess_sim_20260522_044817-step_03 | run_bash | run_tests | 0.9793003 | 0.020644797 | 0.95865554 | run_bash | 한 번만 적용까지 해줘 |
| sess_sim_20260522_018642-step_04 | ask_user | plan_task | 0.9788721 | 0.021125136 | 0.957747 | ask_user | i kinda want shell resources for the cli but honestly i don't know where to even start with cobra completions. help me think through the approach? |
| sess_sim_20260522_004101-step_07 | edit_file | apply_patch | 0.9780336 | 0.021947848 | 0.95608574 | edit_file | 오케 모델 사이즈만 키워놨네. 여기 training 블록 밑에 grad_checkpoint: true 한 줄 추가해줘 한번만 더 |
| sess_sim_20260522_025642-step_12 | ask_user | web_search | 0.9770553 | 0.013936945 | 0.9631184 | ask_user | ok 구조 봤고. dry-run이면 useFetch에서 실제 exec 안 타고 로그만 찍게 하면 될 듯. 단계 좀 잡아줘 한번만 더 |
| sess_sim_20260522_016899-step_04 | plan_task | web_search | 0.9768429 | 0.009803734 | 0.963495 | ask_user | everything's unpinned. find the current best practice for pinning python deps in 2026 thx |
| sess_sim_20260522_012929-step_04 | run_bash | run_tests | 0.9757557 | 0.024237469 | 0.95151824 | run_bash | fingers crossed, try building again |
| sess_sim_20260522_008656-step_07 | run_tests | lint_or_typecheck | 0.9757499 | 0.018152919 | 0.957597 | run_tests | 타입 안 깨졌나 한번 돌려보자 간단히 |
| sess_sim_20260522_014934-step_12 | plan_task | web_search | 0.97421616 | 0.013311981 | 0.9609042 | plan_task | 그래서 둘 다 안 쓰는구나. 롤백 시 카나리 비중 줄이는 단계 넣고 싶은데, 비중 기본값을 0으로 죽일까 아니면 50으로 단계적으로 뺄까? |
| sess_sim_20260522_020411-step_03 | web_search | ask_user | 0.97282153 | 0.013397181 | 0.9591601 | plan_task | 잠깐만 리프레시 토큰 회전(rotation) 기능을 components에 넣어야 해. 손대기 전에 단계부터 한번 잡아보자 한번만 더 |
| sess_sim_20260522_025087-step_07 | edit_file | apply_patch | 0.97261196 | 0.027376164 | 0.9452358 | edit_file | wait, djangorestframework isn't in there but i swear the serializers use it. add it pinned to 3.15.x |
| sess_sim_20260522_036704-step_05 | run_bash | lint_or_typecheck | 0.9715452 | 0.0145802675 | 0.95696497 | run_bash | 타입 쪽도 깨끗하네요. 그래도 실제 동작은 테스트로 봐야 안심되니까 auth 테스트 전체 한번 가주세요 먼저요 |
| sess_sim_20260522_045106-step_07 | run_bash | run_tests | 0.9698667 | 0.017833255 | 0.95203346 | run_bash | 지금 다시 빌드 가자 |
| sess_sim_20260522_023042-step_07 | run_bash | lint_or_typecheck | 0.9687178 | 0.02156962 | 0.9471482 | run_bash | 아무튼 타입스크립트 한번 빌드 통과되나 보자 |
| sess_sim_20260522_002685-step_03 | web_search | plan_task | 0.96522576 | 0.01097697 | 0.9414364 | ask_user | 보니까 mixed precision 학습 옵션을 새로 넣을 건데 작업 범위가 좀 커. config 키 추가하고 routes 루프에 autocast/scaler 넣고 검증까지 해야 하는데 단계 먼저 쪼개줘 |
| sess_sim_20260522_037570-step_02 | run_bash | run_tests | 0.9640394 | 0.035947863 | 0.9280915 | run_bash | 혹시나 해서 이제 다 통과하나 ㅎ |
| sess_sim_20260522_035555-step_03 | read_file | glob_pattern | 0.9636501 | 0.015759557 | 0.9478905 | read_file | intermittent garbled log lines under -j. smells like a data race in the cmd, no rush |
| sess_sim_20260522_008432-step_05 | edit_file | apply_patch | 0.96235853 | 0.037607353 | 0.92475116 | edit_file | 그래서 ok 거의 비어있네. 그럼 그 두 함수를 test_auth.py 로 옮기고, tests 랑 train 쪽 import 도 같이 바꿔야 하니까 한 번에 정리해줘 |
| sess_sim_20260522_012691-step_10 | run_bash | lint_or_typecheck | 0.9621853 | 0.0137248365 | 0.9380975 | run_tests | static check on style too while we're here |
| sess_sim_20260522_023142-step_11 | ask_user | web_search | 0.9596253 | 0.022218337 | 0.93740696 | ask_user | 그러면 흠 하나 깨지네. null 체크 쪽이 좀 의심스러운데, 내가 의도한 게 빈 문자열까지 잡는 건지 진짜 null만인지 헷갈려서... 어느 쪽으로 갈까? |
| sess_sim_20260522_015473-step_10 | run_bash | lint_or_typecheck | 0.9554105 | 0.016960567 | 0.9279881 | run_tests | 다시 README 정적분석 돌려서 깨끗해졌는지 봐요 |
| sess_sim_20260522_023955-step_07 | grep_search | read_file | 0.9545241 | 0.026142353 | 0.92838174 | grep_search | android api started blowing up after I bumped a dependency. can you open the gradle config so we can eyeball what's set when free |
| sess_sim_20260522_014934-step_13 | ask_user | web_search | 0.9533639 | 0.026212769 | 0.92715114 | ask_user | 막혀서 그런데 src가 지금 JSON만 받는데 YAML 입력도 받게 만들어달라는 요청이 들어왔어. 근데 입력을 정확히 어떤 포맷까지 지원해야 하는지가 좀 애매해. 우선순위가 뭔지 너가 정해주기 그러니까 나한테 한번 물어봐줄래? |
| sess_au_079217_001-step_04 | run_tests | run_bash | 0.94802773 | 0.051838532 | 0.8961892 | run_tests | 고쳤으니 돌려보자 |
| sess_sim_20260522_025238-step_04 | run_bash | run_tests | 0.948022 | 0.030119082 | 0.91790295 | run_bash | 고친 걸로 안드 빌드 다시 돌려봐 |
| sess_sim_20260522_003380-step_03 | ask_user | plan_task | 0.9475547 | 0.0408274 | 0.9067273 | ask_user | 이름 바꾸는 게 한 파일만 고친다고 끝나는 게 아니라서 좀 무섭네요.. 어디부터 어떻게 손대면 좋을지 단계 좀 잡아줄래요? |
| sess_sim_20260522_042100-step_04 | ask_user | web_search | 0.9473798 | 0.04589501 | 0.90148485 | ask_user | 음 잠시만 이거 동시 요청 들어올 때 카운터가 안전한지 확인하고 싶은데, 슬로우API 같은 라이브러리 락 처리 어떻게 하는지 좀 검색해봐 |
| sess_sim_20260522_017587-step_09 | run_tests | lint_or_typecheck | 0.9472231 | 0.02697582 | 0.9202473 | run_tests | 이제 안정적인지 다시 ㅠ |
| sess_sim_20260522_026812-step_07 | run_bash | run_tests | 0.9444309 | 0.02874302 | 0.91568786 | run_bash | DAG 한번 돌려보자 깨지나 안깨지나 |
| sess_sim_20260522_038188-step_07 | ask_user | web_search | 0.94393194 | 0.03438272 | 0.90954924 | ask_user | before i write the loop — should the backoff be capped at a fixed ceiling or grow unbounded? what's your call |
| sess_sim_20260522_035531-step_04 | run_bash | run_tests | 0.94285995 | 0.056623142 | 0.8862368 | run_bash | minor — build it and see |
| sess_sim_20260522_032685-step_05 | lint_or_typecheck | run_bash | 0.9425002 | 0.05749288 | 0.8850073 | lint_or_typecheck | first off, and the metro.config suite for me please |

## 6. 경계에 가까운 오답

| id | true_action | pred_action | pred_probability | true_probability | prediction_margin | top2_action | current_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sess_sim_20260522_038715-step_09 | glob_pattern | read_file | 0.3391313 | 0.1391792 | 0.0 | list_directory | 그나저나 딥링크 핸들링 기능 추가하려는데 라우팅 관련 코드가 src 안 어디에 흩어져있는지 모르겠어 |
| sess_sim_20260522_046696-step_08 | grep_search | read_file | 0.27929235 | 0.24551372 | 0.0 | list_directory | 좋아. 우선 라우트랑 쿼리 어떻게 도는지부터 보자 한번 더 |
| sess_sim_20260522_015278-step_16 | apply_patch | edit_file | 0.38749522 | 0.10187698 | 0.0 | run_tests | 아 있네 굿. 액션 메시지에 몇 건 처리됐는지 message_user로 띄워주게 다듬어줘 |
| sess_sim_20260522_041595-step_04 | glob_pattern | read_file | 0.29591084 | 0.14996031 | 0.0 | grep_search | 아 max_seq_len 같은 게 config에 박혀 있을 텐데 Cargo 설정 쪽 값 한번 확인하자 가볍게 |
| sess_sim_20260522_024556-step_05 | glob_pattern | list_directory | 0.265365 | 0.23694429 | 0.001034528 | read_file | 한번만 계획대로 하자. build_model이 hidden dim 어디서 받는지부터 확인해야 하니까 package.json도 같이 보자 |
| sess_sim_20260522_042570-step_07 | list_directory | grep_search | 0.29464215 | 0.29349345 | 0.0011487007 | list_directory | the theme is read from a cookie at render — classic server/client drift. is etl_events.py doing the same — sorry to bug you |
| sess_sim_20260522_042523-step_03 | list_directory | read_file | 0.3076667 | 0.30646724 | 0.0011994541 | list_directory | 근데 token_expiry one is the suspect. let me see the actual route logic! |
| sess_sim_20260522_046120-step_03 | read_file | grep_search | 0.30832693 | 0.30712488 | 0.0012020469 | read_file | env 처리 라이브러리 안 깔려있는 듯. 일단 metro 설정에서 정의해 쓰는 방법도 있던데 scripts 보여줘 가능하면 |
| sess_sim_20260522_013031-step_04 | read_file | grep_search | 0.3235428 | 0.32228145 | 0.0012613535 | read_file | mod.rs 안에 load_dataset 함수가 좀 비대해진 거 같은데 일단 지금 어떻게 생겼는지 한번 보자 빨리 |
| sess_sim_20260522_024121-step_04 | read_file | glob_pattern | 0.3255979 | 0.32432854 | 0.0012693703 | read_file | 이 프로젝트 처음 보는데 app 폴더 안에 뭐뭐 들어있는지 궁금해요. 한번 풀어서 보여줄래요? |
| sess_sim_20260522_037930-step_03 | read_file | grep_search | 0.2736998 | 0.27156982 | 0.002129972 | read_file | stdin 없이 실행하면 패닉 메시지가 raw하게 그대로 노출돼. 이거 좀 점잖게 처리하고 싶은데 어디서 잡아야 하나 lib 엔트리부터 보자 먼저 |
| sess_sim_20260522_046669-step_03 | grep_search | glob_pattern | 0.29056102 | 0.14724916 | 0.0022611618 | read_file | FeatureGrid 위에 Hero 컴포넌트 넣을 거야. 비슷한 스타일 토큰 쓰는 css가 있나 한번 찾아봐 이 부분만 |
| sess_sim_20260522_029703-step_10 | glob_pattern | grep_search | 0.29608253 | 0.22089101 | 0.0023041368 | read_file | 프론트 카드 컴포넌트 스타일 다듬자. 일단 Dockerfile 지금 상태 좀 봐줘 |
| sess_sim_20260522_043889-step_03 | grep_search | read_file | 0.3231178 | 0.32060325 | 0.0025145411 | grep_search | 혹시 헬스체크 엔드포인트 하나 새로 붙이려고. 일단 components 어떻게 잡혀있는지 보자 이 부분만 |
| sess_sim_20260522_038105-step_07 | grep_search | read_file | 0.32575268 | 0.32321766 | 0.0025350153 | grep_search | 88번 줄 근처가 뭔지 보게 dim_users.sql 열어줘 ㅠ |
| sess_sim_20260522_009859-step_02 | read_file | list_directory | 0.28286842 | 0.2795729 | 0.003295511 | read_file | uh there's a _revoke but refresh never calls it. where's the token store backed? |
| sess_sim_20260522_018580-step_05 | grep_search | read_file | 0.30612165 | 0.23471162 | 0.0035664141 | list_directory | show me the dbt project config... |
| sess_sim_20260522_033759-step_06 | ask_user | web_search | 0.47377017 | 0.47008327 | 0.003686905 | ask_user | so we're adding mixed precision training. i keep forgetting whether torch's autocast wants the scaler inside or outside the accum loop. can you check the current recommended pattern |
| sess_sim_20260522_040575-step_05 | grep_search | read_file | 0.31964418 | 0.31592023 | 0.0037239492 | grep_search | where do we parse the config flags? can't find it |
| sess_sim_20260522_011224-step_01 | grep_search | list_directory | 0.34582824 | 0.34179923 | 0.0040290058 | grep_search | real quick q — all in src? read it so i can triage |
| sess_sim_20260522_041878-step_08 | list_directory | read_file | 0.28741062 | 0.28295475 | 0.0044558644 | list_directory | 이번엔 tsx 컴포넌트 파일들 전체 목록이 궁금한데, 패턴으로 한번 쭉 뽑아줄 수 있어? |
| sess_sim_20260522_004883-step_05 | grep_search | read_file | 0.2980836 | 0.2559623 | 0.0046213567 | list_directory | 이 디렉토리에 뭐뭐 들어있는지부터 한번 보여줄래 먼저 |
| sess_au_601527_004-step_02 | glob_pattern | list_directory | 0.3981997 | 0.39356056 | 0.0046391487 | glob_pattern | 헬스체크 응답에 버전 박으려는데 버전 문자열 어디 박혀있나 .py 전부 뒤져 |
| sess_sim_20260522_044795-step_06 | glob_pattern | list_directory | 0.27305102 | 0.22814268 | 0.0052812696 | grep_search | main.tf가 너무 길어서 일단 어떤 리소스들 정의돼 있는지 resource 블록만 검색해줘 |
| sess_sim_20260522_006088-step_07 | grep_search | read_file | 0.46923488 | 0.46376815 | 0.0054667294 | grep_search | k8s/service.yaml 쪽 열어봐 |
| sess_sim_20260522_037597-step_11 | run_tests | run_bash | 0.4996453 | 0.49382427 | 0.0058210194 | run_tests | 노드로 한번 띄워볼 수 있나? tests/test_dags.py 그냥 실행해보자 |
| sess_sim_20260522_045090-step_02 | grep_search | read_file | 0.33823234 | 0.2573136 | 0.006541997 | list_directory | 뭐가 걸렸대? |
| sess_sim_20260522_005823-step_01 | read_file | list_directory | 0.3458788 | 0.33918893 | 0.0066898763 | read_file | yeah figured. pull up README pls |
| sess_sim_20260522_020610-step_07 | list_directory | read_file | 0.28879663 | 0.20801243 | 0.0066899657 | grep_search | 어 유저 프로필 수정 PATCH 엔드포인트 새로 하나 추가하고 싶은데 기존 update가 이미 있는지부터 확인해줘 가볍게 |
| sess_sim_20260522_010991-step_02 | glob_pattern | read_file | 0.29542714 | 0.2014638 | 0.006843567 | grep_search | did i leave any unused imports after that? scan layout.tsx |
| sess_sim_20260522_003415-step_12 | edit_file | apply_patch | 0.5036108 | 0.49580303 | 0.0078077614 | edit_file | real quick, add limit/offset query params to validate, appreciate it |
| sess_sim_20260522_025638-step_10 | plan_task | ask_user | 0.50386983 | 0.49605805 | 0.0078117847 | plan_task | im supposed to add a feature but the task ticket just says 'improve the list view'. that's super vague, what do they actually want me to change? |
| sess_sim_20260522_003879-step_07 | list_directory | glob_pattern | 0.43511048 | 0.098615155 | 0.008415788 | grep_search | btw 음 세 개밖에 안 되네. 그럼 그중에 제일 큰 k8s/deployment.yaml부터 보자, 어떻게 생겼나 thx |
| sess_sim_20260522_016947-step_09 | grep_search | list_directory | 0.2987833 | 0.21774232 | 0.009192586 | read_file | 그러면 다크모드 토글 만들거임. theme 관련 코드 어디 흩어져 있는지부터 찾아줘 여기부터 |
| sess_sim_20260522_019421-step_05 | grep_search | read_file | 0.4071101 | 0.39767942 | 0.009430677 | grep_search | 혹시 로그인 성공했는데도 자꾸 /login으로 다시 튕겨. 리다이렉트 로직 어디서 도는지부터 좀 찾아보자 가볍게 |
| sess_sim_20260522_022066-step_06 | glob_pattern | grep_search | 0.35119662 | 0.17453538 | 0.009472936 | read_file | im adding a dark mode toggle to the little web page. lets start by seeing the markup if you can |
| sess_sim_20260522_033150-step_01 | grep_search | list_directory | 0.35357007 | 0.34403312 | 0.009536952 | grep_search | by the way, the data loading is slow as hell. read models.py |
| sess_sim_20260522_036177-step_06 | grep_search | list_directory | 0.28452405 | 0.27469507 | 0.009828985 | grep_search | 하는 김에 stg_users.sql 폴더 들어가서 뭐있나 보자 한번만 더 |
| sess_sim_20260522_047060-step_06 | grep_search | read_file | 0.45392537 | 0.4434102 | 0.010515183 | grep_search | build이랑 collect_results 이름이 너무 비슷해서 헷갈려. build을 read_source로 바꾸고 싶은데 참조부터 보자 꼼꼼히 |
| sess_sim_20260522_031185-step_04 | grep_search | read_file | 0.41063163 | 0.39955553 | 0.011076093 | grep_search | alright starting a typing pass on the model. show me ci.yml, any time |
| sess_sim_20260522_008927-step_06 | glob_pattern | grep_search | 0.323937 | 0.2602907 | 0.011190534 | read_file | service is one of the callers, pull that screen up |
| sess_sim_20260522_010787-step_09 | grep_search | glob_pattern | 0.29261336 | 0.27921364 | 0.011209846 | read_file | 오케이 프로덕션에서 workflows 스크립트 돌렸는데 직전 버전이 아니라 완전 옛날 리비전으로 돌아가버림. 이거 심각함. ci.yml 먼저 통째로 보자 |
| sess_sim_20260522_019523-step_03 | list_directory | read_file | 0.32630613 | 0.21651931 | 0.011272371 | grep_search | integration.rs에서 db를 어떻게 import 하고 있는지도 보자 |
| sess_sim_20260522_012882-step_03 | glob_pattern | read_file | 0.29932103 | 0.28785422 | 0.011466801 | glob_pattern | init에서 AddCommand로 붙이는 구조구나. 그럼 같은 패턴으로 README.md가 어떻게 생겼는지도 참고로 보고. 이 부분만 |
| sess_sim_20260522_001979-step_04 | grep_search | read_file | 0.33533743 | 0.32375306 | 0.011584371 | grep_search | 혹시나 해서 dags 돌리면 f1이 0.9 넘게 나오는데 리더보드 채점이랑 0.2씩 차이가 남. 뭔가 평가 코드가 잘못된 거 같은데 한번 같이 봐줄래? |
| sess_sim_20260522_012046-step_07 | grep_search | glob_pattern | 0.28362292 | 0.27169412 | 0.011928797 | grep_search | 막혀서 그런데 existsByEmail 같은 거 리포에 이미 있나? |
| sess_sim_20260522_035058-step_06 | grep_search | web_search | 0.44508538 | 0.43307996 | 0.012005419 | grep_search | the eval pipeline naming is inconsistent, half of it says fetchUser the other half says fetchUser, appreciate it |
| sess_sim_20260522_036951-step_10 | list_directory | read_file | 0.3266994 | 0.31418374 | 0.012515664 | list_directory | 혹시나 해서 추론 서버 파드에 readiness/liveness probe 붙이려고. 지금 main 어떻게 생겼는지부터 좀 보자 |
| sess_sim_20260522_013957-step_14 | read_file | grep_search | 0.27579963 | 0.2268666 | 0.012629777 | list_directory | 오케이 parser만 디렉토리로 빠져있네. 다른 것도 커지면 이렇게 갈지 고민중인데 일단 .rs 파일들 전체 한번 훑자 |
| sess_sim_20260522_033094-step_05 | list_directory | grep_search | 0.30216724 | 0.2167944 | 0.012708753 | read_file | alright tracking a leak in the data layer. where does DataLoader get instantiated |

## 7. 확신도가 낮지만 맞춘 데이터

| id | true_action | pred_action | pred_probability | true_probability | prediction_margin | top2_action | current_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sess_sim_20260522_029815-step_05 | grep_search | grep_search | 0.2578313 | 0.2578313 | 0.0020064414 | read_file | 자 커스텀 오퍼레이터 하나 있다던데 어디? |
| sess_sim_20260522_001353-step_13 | run_tests | run_tests | 0.26684272 | 0.26684272 | 0.014694065 | lint_or_typecheck | build's red. what's blowing up? |
| sess_sim_20260522_016874-step_03 | glob_pattern | glob_pattern | 0.2733611 | 0.2733611 | 0.011497229 | grep_search | not urgent but module not found for lib/auth?? but the file is right there. maybe the path alias is busted. lemme look at the workflows |
| sess_sim_20260522_033652-step_07 | grep_search | grep_search | 0.280421 | 0.280421 | 0.004347503 | read_file | 하는 김에 runner이랑 Header만 있구나internal/runner/runner.go 루트 레이아웃부터 보고 어디에 Provider 끼울지 정하자 |
| sess_sim_20260522_010098-step_06 | read_file | read_file | 0.2837152 | 0.2837152 | 0.038180172 | grep_search | i wanna audit all our components for inline styles. how many tsx files are there even for me |
| sess_sim_20260522_043530-step_03 | read_file | read_file | 0.28589237 | 0.28589237 | 0.02148515 | grep_search | terraform 라우트 패턴 보고 비슷하게 갈게. 한번 보자 |
| sess_sim_20260522_014529-step_07 | grep_search | grep_search | 0.28607994 | 0.28607994 | 0.017332703 | list_directory | src/main/resources/application.yml에 헬스체크 엔드포인트 하나 붙이려는데 지금 라우팅 구조부터 보자 |
| sess_sim_20260522_013162-step_02 | read_file | read_file | 0.28705633 | 0.28705633 | 0.028733581 | grep_search | app 디렉토리 안쪽도 |
| sess_sim_20260522_008819-step_14 | grep_search | grep_search | 0.28712875 | 0.28712875 | 0.002234459 | read_file | 딥링크 기능 새로 넣을 거야. 우선 라우팅 관련 코드가 어디 있는지부터 훑자 한번만 더 |
| sess_sim_20260522_043254-step_03 | grep_search | grep_search | 0.28859156 | 0.28859156 | 0.018541813 | glob_pattern | 자 스크립트 폴더에 뭐있나 보자 천천히 |
| sess_sim_20260522_038418-step_05 | list_directory | list_directory | 0.2929387 | 0.2929387 | 0.01667118 | grep_search | 어 잠깐 어제부터 Makefile 워크플로가 변수 못 찾는다고 깨짐. 변수 어디 정의돼 있나 전체 검색 |
| sess_sim_20260522_015899-step_10 | read_file | read_file | 0.2934129 | 0.2934129 | 0.045367435 | glob_pattern | index.vue 좀 봐줘. 메타데이터 어떻게 들어가 있나 간단히 |
| sess_sim_20260522_010887-step_07 | read_file | read_file | 0.29482964 | 0.29482964 | 0.014598012 | grep_search | 음... models/marts/dim_users.sql에서 fetch 한방에 실패하면 그냥 죽어버려. 재시도 좀 넣고싶은데 일단 그 fetch 부분 보여줘 |
| sess_sim_20260522_028000-step_04 | read_file | read_file | 0.29578283 | 0.29578283 | 0.017920583 | grep_search | repository 통째로 한번 보자 |
| sess_sim_20260522_034964-step_04 | read_file | read_file | 0.29963395 | 0.29963395 | 0.0114788115 | list_directory | 가능하면 실제로 @/ alias를 어디서 쓰고 있는지 한번 찾아봐 간단히 |
| sess_sim_20260522_005720-step_03 | read_file | read_file | 0.29998294 | 0.29998294 | 0.033172578 | grep_search | 음 잠시만 src src에 로그아웃 엔드포인트가 없네. 지금 lib.rs 안에 뭐가 들어있는지 열어봐 이번 것만 |
| sess_sim_20260522_038884-step_04 | read_file | read_file | 0.30145177 | 0.30145177 | 0.033335 | grep_search | 프론트가 internal/runner/runner.go에서 백엔드 어떤 엔드포인트 때리는지 궁금해. fetch 호출 같은거 어디 있는지 찾아줘 간단히 |
| sess_sim_20260522_032287-step_02 | list_directory | list_directory | 0.30363303 | 0.30363303 | 0.026089966 | read_file | 헐 해싱 함수 자체가 없음? workflows 라우트 직접 보자 |
| sess_sim_20260522_022266-step_04 | grep_search | grep_search | 0.30482495 | 0.30482495 | 0.05408278 | read_file | lets start with the obvious one, open up the UserControllerTest.java mod and walk me through whats defined there thx |
| sess_sim_20260522_037196-step_04 | read_file | read_file | 0.3056548 | 0.3056548 | 0.01739484 | grep_search | i'm tacking an incremental load mode onto terraform. show me the current mart if you can |
| sess_sim_20260522_009774-step_03 | read_file | read_file | 0.30776078 | 0.30776078 | 0.012944013 | glob_pattern | pages도 테스트가 필요할 것 같은데, 우선 pages 구현부터 다시 확인하자 한 번만 |
| sess_sim_20260522_043735-step_12 | list_directory | list_directory | 0.30933607 | 0.30933607 | 0.038472325 | read_file | when you can, the metro.config side provisions that secret too right? where's it created |
| sess_sim_20260522_034667-step_09 | read_file | read_file | 0.30959582 | 0.30959582 | 0.0036068857 | grep_search | Airflow 설정을 좀 손봐야 할 것 같은데, 관련 설정 파일이랑 plugins 쪽이 어떻게 구성돼 있는지 프로젝트 루트부터 한번 훑고 싶어요 좀 |
| sess_sim_20260522_005019-step_08 | read_file | read_file | 0.31120098 | 0.31120098 | 0.02895394 | grep_search | fetchRecent가 클라에서 useEffect로 도는 구조네. 서버에서 미리 못 받아오나? db 쪽 쿼리 함수가 서버 컴포넌트에서 바로 쓸 수 있는 모양인지 좀 봐줘 |
| sess_sim_20260522_003608-step_03 | read_file | read_file | 0.31217965 | 0.31217965 | 0.0366821 | list_directory | i'm auditing how soft-deletes work across the codebase. where does deactivate get referenced? |
| sess_sim_20260522_017521-step_05 | read_file | read_file | 0.3127764 | 0.3127764 | 0.036752194 | glob_pattern | 좋아 일단 lib.rs가 지금 뭘 노출하고 있는지부터 같이 보자 |
| sess_sim_20260522_026693-step_02 | read_file | read_file | 0.31584314 | 0.31584314 | 0.056037635 | glob_pattern | wait, yeah it says run `python server.py` but i don't see a server.py anywhere. is that file actually gone? |
| sess_sim_20260522_017845-step_03 | grep_search | grep_search | 0.31694865 | 0.31694865 | 0.035048664 | read_file | 프론트 카드 컴포넌트 스타일 다듬자. 일단 style.css 지금 상태 좀 봐줘 좀 |
| sess_sim_20260522_009524-step_09 | read_file | read_file | 0.31721982 | 0.31721982 | 0.058312535 | list_directory | 어 model-paths에 Podfile 디렉토리가 빠졌네. ios/Podfile 하위에 진짜 파일들 있는지 한번 확인하자 이 부분만 |
| sess_sim_20260522_024745-step_03 | list_directory | list_directory | 0.31752872 | 0.31752872 | 0.030665427 | read_file | ok which ones broke, open the controller test |
| sess_sim_20260522_010904-step_02 | read_file | read_file | 0.31753176 | 0.31753176 | 0.037310988 | grep_search | 페이지 모바일에서 보면 사이드바가 본문 위로 막 겹쳐버려요ㅠ 일단 마크업 구조가 어떤지 dbt_project.yml부터 좀 봐줄래요? |
| sess_sim_20260522_012501-step_03 | list_directory | list_directory | 0.31775862 | 0.31775862 | 0.019252032 | read_file | terraform 출력 json 옵션 하나 붙이려고. 일단 어디 정의돼 있는지부터 가볍게 |
| sess_sim_20260522_005347-step_04 | grep_search | grep_search | 0.31784803 | 0.31784803 | 0.035148144 | list_directory | 이번엔 stg_users 회귀 잡으려면 dag 테스트에 고정 픽스처가 필요한데 지금은 없네. 일단 기존 테스트 구조부터 보자 |
| sess_sim_20260522_029318-step_02 | read_file | read_file | 0.31793803 | 0.31793803 | 0.036260545 | list_directory | 확인차 output이 status[0].load_balancer 참조하는데 LB가 아직 프로비저닝 중이면 빈값 나오는거 같아. 혹시 타임아웃 관련 변수가 이미 어딘가 정의돼있나? workflows 좀 봐줘 |
| sess_sim_20260522_033815-step_04 | grep_search | grep_search | 0.3182218 | 0.3182218 | 0.038486898 | read_file | 참, 어 뭐가 깨졌지... 스냅샷 기댓값이 옛날 거라 그런가? 테스트 파일 어디 있는지 찾아줘 |
| sess_sim_20260522_012220-step_03 | read_file | read_file | 0.3186422 | 0.3186422 | 0.03853771 | list_directory | right, any rewrites in there pointing at the auth api |
| sess_sim_20260522_032143-step_04 | read_file | read_file | 0.3198882 | 0.3198882 | 0.040876746 | grep_search | Makefile 쪽 포트랑 매칭 안 맞는 거 아니야? 한번 열어봐 |
| sess_sim_20260522_000640-step_05 | grep_search | grep_search | 0.3199246 | 0.3199246 | 0.06084177 | read_file | 아무튼 app/admin.py 모듈 경로랑 실제 import 경로 안 맞는 듯. app/admin.py 열어봐 한번 더 |
| sess_sim_20260522_042727-step_14 | list_directory | list_directory | 0.32077238 | 0.32077238 | 0.067020535 | read_file | 어 잠깐 handleLogin이 createSession 호출하는 흐름이구나~ 그럼 세션은 결국 db에 쌓이는 거야? Cargo/db 좀 보자 |
| sess_sim_20260522_003100-step_04 | grep_search | grep_search | 0.3207903 | 0.3207903 | 0.051710933 | read_file | open the schema |
| sess_sim_20260522_012098-step_02 | list_directory | list_directory | 0.3232166 | 0.3232166 | 0.026616365 | read_file | 막혀서 그런데 헤더에 다크모드 토글 버튼 하나 넣자. 근데 workflows가 workflows 컴포넌트 쓰는지부터 확인 좀. 어디서 import 하는지 검색해봐 |
| sess_sim_20260522_003590-step_02 | read_file | read_file | 0.32372078 | 0.32372078 | 0.021979809 | grep_search | is that symbol called anywhere else? don't wanna leave a dangling reference |
| sess_sim_20260522_008321-step_03 | read_file | read_file | 0.32373434 | 0.32373434 | 0.09146631 | list_directory | 막혀서 그런데 UserController.java도 같이 보고 가자 거기 데이터 로딩이랑 묶을 수 있을지 빨리 |
| sess_sim_20260522_015952-step_03 | grep_search | grep_search | 0.32508722 | 0.32508722 | 0.04154107 | read_file | 프로젝트 안에 url 설정 파일이 두 갠지 헷갈리는데 terraform 들어간 파일 다 찾아봐 줄래요? |
| sess_sim_20260522_011272-step_04 | grep_search | grep_search | 0.32514805 | 0.32514805 | 0.05347669 | list_directory | ci 워크플로가 단계가 너무 중구난방이라 정리하고 싶어. 일단 dbt_project.yml 내용 보여줘 급해 |
| sess_sim_20260522_021554-step_11 | grep_search | grep_search | 0.32572484 | 0.32572484 | 0.058835477 | read_file | 아무튼 여기 python run.py 라고 써있는데 우리 진입점은 main.py 잖아... 실제로 그런 파일 있는지 패턴으로 찾아봐 |
| sess_sim_20260522_035954-step_11 | list_directory | list_directory | 0.3263077 | 0.3263077 | 0.013724059 | read_file | 어 잠깐 requirements랑 requirements 두 개뿐이네. requirements 먼저 열어서 구조 보자 짧게 |
| sess_sim_20260522_041933-step_03 | read_file | read_file | 0.32657528 | 0.32657528 | 0.039497197 | list_directory | useAuth에서 EntityManager 직접 열고 close 안 하는거 보이지? 이거 누수다 |
| sess_sim_20260522_004658-step_03 | read_file | read_file | 0.32664683 | 0.32664683 | 0.06520197 | list_directory | actually 유저 삭제를 하드딜리트 말고 소프트딜리트로 바꾸려고 해. 영향 받는 곳부터 훑자. main 어디서 쓰는지 찾아줘 |
| sess_sim_20260522_035696-step_03 | read_file | read_file | 0.32682544 | 0.32682544 | 0.09416443 | list_directory | db.ts 안에 Config라는 함수가 두 개나 있는 거 같은데 이름만 같고 하는 일이 다른 거 같아ㅠ 일단 파일 좀 읽어줘 한번만 더 |

## 8. 생성된 상세 파일

- `validation_all.csv`: 전체 validation 예측 결과
- `validation_correct.csv`: 맞춘 데이터
- `validation_wrong.csv`: 틀린 데이터
- `class_metrics.csv`: 클래스별 precision, recall, F1
- `confusion_matrix.csv`: confusion matrix
- `confusion_pairs.csv`: 오분류 조합 빈도
- `summary.json`: 전체 성능 요약
