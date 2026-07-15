# yj-4 학습 handoff (건우/yj-3-S1 → yj-4)

> 목적: yj-3서 못 돌리는 **au-포함 길이최적 + 프리앰블 FULL** 5종을 yj-4 5 GPU로 병렬.
> 코드·데이터 전부 이 NAS 폴더에 있음. 학습결과(member_*, teacher_*.npz)도 이 폴더에 저장됨 → yj-3서 바로 읽어 패키징 가능.
> 마감 07-15 10:00. 각 학습 ~3-4h(FULL). 지금(01:2x) 발진하면 05:xx 완료.

## 0. 세팅 (yj-4 접속 후)
```bash
ssh yj4-A6000-kinamkim   # 또는 본인 alias
source /home/nas_main/kinamkim/miniconda3/etc/profile.d/conda.sh
conda activate intern_aichallenge
cd /home/nas5/kinamkim/Repos/geonwoo/2026-Agent-Action-Chanllenge/experiments/yj4_train
gpustat   # 빈 GPU 5개 확인 → 아래 CUDA_VISIBLE_DEVICES 번호를 빈 것으로 교체
```

## 1. 학습 커맨드 (GPU 번호는 빈 것으로 교체, tmux 권장)
각 줄을 **별도 tmux 창**에서. `> logs/xxx.log 2>&1`로 로그 남김. `mkdir -p logs` 먼저.

```bash
mkdir -p logs

# [G0] bge@512 au-포함 FULL (길이헤지, +0.0046 proven)
CUDA_VISIBLE_DEVICES=0 AD_WORK=$PWD AD_MODEL=BAAI/bge-m3 AD_VERSION=v6 AD_MAXLEN=512 AD_EPOCHS=8 AD_LR=2e-5 AD_BATCH=32 AD_LLRD=1 AD_FGM=1 AD_MHT=12 AD_PRUNE=1 AD_TAG=bge_512mht12_full nice -n 10 python -u action_decision_maximum/src/train_full_cli.py > logs/bge512_full.log 2>&1 &

# [G1] mde@512 au-포함 FULL (길이헤지)
CUDA_VISIBLE_DEVICES=1 AD_WORK=$PWD AD_MODEL=microsoft/mdeberta-v3-base AD_VERSION=v6 AD_MAXLEN=512 AD_EPOCHS=12 AD_LR=2e-5 AD_BATCH=48 AD_LLRD=1 AD_FGM=1 AD_PRUNE=1 AD_TAG=mde_512_full nice -n 10 python -u action_decision_maximum/src/train_full_cli.py > logs/mde512_full.log 2>&1 &

# [G2] bge@512 hints4 프리앰블 sim-only FULL (프리앰블+au제거 조합, @512엔 hints4만 안전)
CUDA_VISIBLE_DEVICES=2 AD_WORK=$PWD AD_MODEL=BAAI/bge-m3 AD_VERSION=v6 AD_MAXLEN=512 AD_EPOCHS=8 AD_LR=2e-5 AD_BATCH=32 AD_LLRD=1 AD_FGM=1 AD_MHT=12 AD_PRUNE=1 AD_EXCLUDE_AU=1 AD_ACTIONS_PREAMBLE=2 AD_TAG=bge_512_hints4_simonly_full nice -n 10 python -u action_decision_maximum/src/train_full_cli.py > logs/bge512_h4sim.log 2>&1 &

# [G3] mde@512 hints4 프리앰블 sim-only FULL
CUDA_VISIBLE_DEVICES=3 AD_WORK=$PWD AD_MODEL=microsoft/mdeberta-v3-base AD_VERSION=v6 AD_MAXLEN=512 AD_EPOCHS=12 AD_LR=2e-5 AD_BATCH=48 AD_LLRD=1 AD_FGM=1 AD_PRUNE=1 AD_EXCLUDE_AU=1 AD_ACTIONS_PREAMBLE=2 AD_TAG=mde_512_hints4_simonly_full nice -n 10 python -u action_decision_maximum/src/train_full_cli.py > logs/mde512_h4sim.log 2>&1 &

# [G4] mmbert@768 full-14 프리앰블 sim-only FULL (mmbert 프리앰블 alt버전)
CUDA_VISIBLE_DEVICES=4 AD_WORK=$PWD AD_MODEL=jhu-clsp/mmBERT-base AD_VERSION=v6 AD_MAXLEN=768 AD_EPOCHS=8 AD_LR=2e-5 AD_BATCH=32 AD_LLRD=1 AD_FGM=1 AD_FGM_EMB=tok_embeddings AD_MHT=12 AD_EXCLUDE_AU=1 AD_ACTIONS_PREAMBLE=1 AD_TAG=mmbert_768_full14_simonly_full nice -n 10 python -u action_decision_maximum/src/train_full_cli.py > logs/mmb768_f14sim.log 2>&1 &
```

## 2. 확인 (발진 직후)
```bash
# 각 로그에 [full] 헤더 + (해당시) [exclude_au]/[preamble] 뜨는지
grep -hE "\[full\]|exclude_au|preamble" logs/*.log
nvidia-smi   # 5 GPU 다 100% util
```
- `[exclude_au] train 70000 -> 64975` = sim-only 정상
- `[preamble] mode=N 추가` = 프리앰블 정상
- OOM 나면 AD_BATCH 낮춰 재발진 (bge@512 32→24, mde@512 48→32)

## 3. 완료 후 (yj-3서 자동 인식)
학습 끝나면 `member_<TAG>/` (양자화전 모델)이 이 폴더에 생성됨. yj-3(건우 세션)가 이 경로 읽어 패키징함. **완료되면 카톡/여기 알려주면 됨.**

## 왜 이 5개인가
| 모델 | 용도 |
|---|---|
| bge@512-full, mde@512-full | **au-포함 길이최적 앙상블** (길이 +0.0046, au리스크 없는 safe버전) |
| bge@512-hints4-simonly, mde@512-hints4-simonly | **프리앰블+au제거+길이 결합** (@512라 hints4만 안전) |
| mmbert@768-full14-simonly | mmbert 프리앰블 full-14 버전 (hints4 vs full14 비교용) |

yj-3서 이미 도는 것(중복금지): bge@512-simonly, mde@512-simonly, mde@320-simonly, mmbert-hints4-simonly, 프리앰블 fold0 프로브 4종.
