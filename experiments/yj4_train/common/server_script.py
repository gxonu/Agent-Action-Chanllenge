"""정본 서버 추론 스크립트 (오프라인). 두 트랙 공용 단일 소스.

train_cli.py 와 노트북이 이 파일을 읽어 제출 model/ 옆 script.py 로 배치.
model/ 의 ad_lib.py + run_meta.json(version,max_len) + postproc.json 사용.
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import csv
import json

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model")
sys.path.insert(0, MODEL)
import ad_lib


def load_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    os.makedirs("output", exist_ok=True)
    samples = load_jsonl(os.path.join("data", "test.jsonl"))
    meta = json.load(open(os.path.join(MODEL, "run_meta.json")))
    ppath = os.path.join(MODEL, "postproc.json")
    preds = ad_lib.predict(
        MODEL, samples,
        version=meta["version"], max_len=meta["max_len"],
        batch_size=meta.get("batch_size", 128),
        postproc_path=ppath if os.path.exists(ppath) else None,
        meta=meta,   # "ensemble" 키 존재 시 다중모델 평균 경로
    )
    rows = list(csv.DictReader(open(os.path.join("data", "sample_submission.csv"), encoding="utf-8")))
    miss = 0
    for r in rows:
        if r["id"] in preds:
            r["action"] = preds[r["id"]]
        else:
            miss += 1
    with open(os.path.join("output", "submission.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "action"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows (missing={miss})")


if __name__ == "__main__":
    main()
