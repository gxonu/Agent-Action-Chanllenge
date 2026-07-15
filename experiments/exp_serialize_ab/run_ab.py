"""
exp_serialize_ab — serialize 변형 A/B (CUE 확장 · state 재직렬화)

같은 모델 / 같은 GroupKFold / 같은 하이퍼에서 serialize만 바꿔 OOF Macro-F1 비교.
변형: base(상혁 baseline) · cue(축A) · reorder(축B) · both

사용:
  # 0) 학습 없이 직렬화·truncation·예시만 (CPU, 안전) — 먼저 이걸로 sanity check
  python experiments/exp_serialize_ab/run_ab.py --dry_run

  # 1) 빠른 A/B (fold 1개만, mmBERT-base) — both은 S1 exp007이 커버하므로 제외
  python experiments/exp_serialize_ab/run_ab.py --gpu 5 --variants base,cue,reorder \
      --model jhu-clsp/mmBERT-base --n_folds 5 --eval_folds 1 --epochs 3

  # 2) 본격 (fold 3개 평균)
  python experiments/exp_serialize_ab/run_ab.py --gpu 5 --eval_folds 3 --epochs 4

주의(상위 CLAUDE.md): GPU 번호 필수 지정. mdeberta tokenizer는 protobuf/sentencepiece 필요.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import serialize_variants as sv

CLASS_NAMES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
RARE = ["web_search", "write_file", "lint_or_typecheck"]
INSPECT = ["read_file", "grep_search", "list_directory", "glob_pattern"]
SER_CFG = dict(prompt_marker="[PROMPT]", history_marker="[HISTORY]", meta_marker="[META]",
               max_history_steps=12, add_symbol_cues=True, add_open_files=True,
               add_langmix=False, max_open_files=8)


def load_df():
    lab = pd.read_csv(os.path.join(ROOT, "data", "raw", "train_labels.csv"))
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "raw", "train.jsonl"),
                                        encoding="utf-8")]
    df = pd.DataFrame(rows).merge(lab, on="id", how="left")
    assert df["action"].notna().all()
    df["session"] = df["id"].str.replace(r"-step_\d+$", "", regex=True)
    return df


def serialize_all(df, variant):
    fl = sv.VARIANTS[variant]
    return [sv.serialize_variant(p, h, m, SER_CFG, **fl)
            for p, h, m in zip(df["current_prompt"], df["history"], df["session_meta"])]


# ---------------------------------------------------------------------------
# dry-run: 학습 없이 직렬화 길이/truncation/예시 (CPU)
# ---------------------------------------------------------------------------
def dry_run(args):
    from transformers import AutoTokenizer
    df = load_df()
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[dry_run] model={args.model}  n={len(df)}  max_length={args.max_length}\n")
    header = f"{'variant':<9} {'p50':>5} {'p95':>5} {'p99':>5} {'max':>5} " \
             f"{'>512%':>7} {'>maxlen%':>9}"
    print(header); print("-" * len(header))
    base_texts = None
    for v in args.variants.split(","):
        texts = serialize_all(df, v)
        if v == "base":
            base_texts = texts
        L = []
        for i in range(0, len(texts), 2000):
            L.extend(len(x) for x in tok(texts[i:i+2000], add_special_tokens=True)["input_ids"])
        L = np.array(L)
        print(f"{v:<9} {np.percentile(L,50):5.0f} {np.percentile(L,95):5.0f} "
              f"{np.percentile(L,99):5.0f} {L.max():5.0f} "
              f"{(L>512).mean()*100:6.1f}% {(L>args.max_length).mean()*100:8.1f}%")
    # 변형별 추가 토큰 예시 3개
    print("\n[예시] (base 대비 무엇이 붙는지)")
    idx = [10, 100, 1000]
    for i in idx:
        r = df.iloc[i]
        print(f"\n--- id={r['id']} label={r['action']} ---")
        for v in args.variants.split(","):
            s = sv.serialize_variant(r["current_prompt"], r["history"], r["session_meta"],
                                     SER_CFG, **sv.VARIANTS[v])
            cue = s.split("[HISTORY]")[0].split("[CUE]")[-1].strip() if "[CUE]" in s else ""
            print(f"  {v:<8} CUE=[{cue}]")


def rare_cue_coverage(df):
    """각 rare cue가 얼마나 켜지고, 켜졌을 때 해당 rare 라벨 비율(정밀도 proxy)."""
    p = df["current_prompt"].astype(str)
    for flag, rgx, lbl in [("tc", sv._TC_RE, "lint_or_typecheck"),
                           ("newf", sv._NEWF_RE, "write_file"),
                           ("ws", sv._WS_RE, "web_search")]:
        on = p.apply(lambda s: bool(rgx.search(s)))
        n_on = int(on.sum())
        prec = (df[on]["action"] == lbl).mean() * 100 if n_on else 0.0
        recall = ((df["action"] == lbl) & on).sum() / max((df["action"] == lbl).sum(), 1) * 100
        base = (df["action"] == lbl).mean() * 100
        print(f"  {flag:<5} 발화 {n_on:>5} ({n_on/len(df)*100:4.1f}%) | "
              f"cue on일때 {lbl} = {prec:4.1f}% (base {base:.1f}%, lift {prec/max(base,1e-9):.1f}x) | "
              f"{lbl} recall {recall:.0f}%")


# ---------------------------------------------------------------------------
# train + OOF (GPU)
# ---------------------------------------------------------------------------
def run_train(args):
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, DataCollatorWithPadding)

    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokzr, max_length):
            self.enc = tokzr(list(texts), truncation=True, max_length=max_length)
            self.labels = [int(x) for x in labels]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {"input_ids": self.enc["input_ids"][i],
                    "attention_mask": self.enc["attention_mask"][i],
                    "labels": self.labels[i]}

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    df = load_df()
    y = df["action"].map({c: i for i, c in enumerate(CLASS_NAMES)}).values
    groups = df["session"].values
    tok = AutoTokenizer.from_pretrained(args.model)
    bf16 = torch.cuda.is_bf16_supported()

    # class weights (inverse freq) — 상혁 weighted_ce
    counts = np.bincount(y, minlength=14).astype(float)
    cw = torch.tensor(counts.sum() / (14 * counts), dtype=torch.float)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                out.logits, labels, weight=cw.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    skf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    folds = list(skf.split(df, y, groups))

    results = {}
    for v in args.variants.split(","):
        texts = np.array(serialize_all(df, v), dtype=object)
        oof = np.zeros((len(df), 14), dtype=np.float32)
        used = np.zeros(len(df), dtype=bool)
        for fi, (tr, va) in enumerate(folds[:args.eval_folds]):
            ds_tr = DS(texts[tr], y[tr], tok, args.max_length)
            ds_va = DS(texts[va], y[va], tok, args.max_length)
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model, num_labels=14,
                id2label={i: c for i, c in enumerate(CLASS_NAMES)},
                label2id={c: i for i, c in enumerate(CLASS_NAMES)})
            targs = TrainingArguments(
                output_dir=os.path.join(HERE, f"_tmp/{v}_f{fi}"),
                num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=128, learning_rate=2e-5, warmup_ratio=0.06,
                weight_decay=0.01, bf16=bf16, fp16=not bf16, report_to=[],
                logging_steps=200, save_strategy="no", eval_strategy="no", seed=42)
            tr_obj = WTrainer(model=model, args=targs, train_dataset=ds_tr,
                              data_collator=DataCollatorWithPadding(tok))
            tr_obj.train()
            logits = tr_obj.predict(ds_va).predictions
            oof[va] = torch.softmax(torch.tensor(logits), -1).numpy()
            used[va] = True
            del model, tr_obj; torch.cuda.empty_cache()
        pred = oof[used].argmax(1)
        yt = y[used]
        mf1 = f1_score(yt, pred, labels=list(range(14)), average="macro", zero_division=0)
        perclass = f1_score(yt, pred, labels=list(range(14)), average=None, zero_division=0)
        pcf = {CLASS_NAMES[i]: round(float(perclass[i]), 4) for i in range(14)}
        results[v] = dict(macro_f1=round(float(mf1), 4), n_eval=int(used.sum()),
                          rare_f1={k: pcf[k] for k in RARE},
                          inspect_f1={k: pcf[k] for k in INSPECT}, per_class=pcf)
        outd = os.path.join(HERE, v); os.makedirs(outd, exist_ok=True)
        np.savez(os.path.join(outd, "val_probs.npz"),
                 ids=df["id"].values[used], probs=oof[used], y=yt,
                 classes=np.array(CLASS_NAMES))
        json.dump(results[v], open(os.path.join(outd, "metrics.json"), "w"),
                  ensure_ascii=False, indent=2)
        print(f"\n=== [{v}] Macro-F1 {results[v]['macro_f1']} "
              f"(eval {results[v]['n_eval']}) rare {results[v]['rare_f1']}")

    print("\n" + "=" * 60 + "\nSUMMARY (Macro-F1 / rare avg / inspect avg)\n" + "=" * 60)
    b = results.get("base", {}).get("macro_f1")
    for v, r in results.items():
        d = f"(Δ{r['macro_f1']-b:+.4f})" if b else ""
        ra = np.mean(list(r["rare_f1"].values())); ins = np.mean(list(r["inspect_f1"].values()))
        print(f"  {v:<9} {r['macro_f1']:.4f} {d:<12} rare {ra:.4f}  inspect {ins:.4f}")
    json.dump(results, open(os.path.join(HERE, "results_all.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"\n[saved] {os.path.join(HERE, 'results_all.json')} + {{variant}}/val_probs.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    # both은 S1 exp007(serialize_v2)이 커버 → S2는 개별 축 분리용 base/cue/reorder만 (S1 세션 조율 결정)
    ap.add_argument("--variants", default="base,cue,reorder")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--eval_folds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args)
        print("\n[rare cue coverage — 발화율/정밀도/재현율]")
        rare_cue_coverage(load_df())
    else:
        if args.gpu is None:
            sys.exit("ERROR: --gpu 필수. gpustat으로 빈 GPU 확인 후 지정하세요.")
        run_train(args)
