"""Reconstruct top-line metrics from the dumped _pred.jsonl / _i2t.jsonl / _t2i.jsonl
and compare to the score JSON written by clip_benchmark.cli.

Run on login node after `sbatch runs/verify_predictions.sh` completes.
"""
import json
import os
import sys

RESULTS = "/lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/results/s1_verify"
PREDS = os.path.join(RESULTS, "preds")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def check_classification():
    score_path = os.path.join(RESULTS, "imagenet1k_s1_verify.json")
    pred_path = os.path.join(PREDS, "imagenet1k_s1_verify_pred.jsonl")
    score = json.load(open(score_path))
    rows = load_jsonl(pred_path)
    n = len(rows)
    acc1 = sum(1 for r in rows if r["topk"][0] == r["true"]) / n
    acc5 = sum(1 for r in rows if r["true"] in r["topk"][:5]) / n
    exp1 = score["metrics"]["acc1"]
    exp5 = score["metrics"]["acc5"]
    # Tolerance 1e-3 absorbs torch.topk tie-breaking nondeterminism (fp16 amp logits → ties).
    ok = abs(acc1 - exp1) < 1e-3 and abs(acc5 - exp5) < 1e-3
    print(f"[classification] n={n}  acc1: reconstructed={acc1:.5f} expected={exp1:.5f}  acc5: reconstructed={acc5:.5f} expected={exp5:.5f}  -> {'OK' if ok else 'MISMATCH'}")
    return ok


def check_retrieval():
    score_path = os.path.join(RESULTS, "xm3600_en_s1_verify.json")
    i2t_path = os.path.join(PREDS, "xm3600_en_s1_verify_i2t.jsonl")
    t2i_path = os.path.join(PREDS, "xm3600_en_s1_verify_t2i.jsonl")
    score = json.load(open(score_path))
    i2t = load_jsonl(i2t_path)
    t2i = load_jsonl(t2i_path)
    # text_retrieval_recall@k (per image): >=1 GT text in top-k
    def i2t_recall(k):
        hits = 0
        for r in i2t:
            gt = set(r["gt_text_ids"])
            if any(x in gt for x in r["ranked_text_ids"][:k]):
                hits += 1
        return hits / len(i2t)
    # image_retrieval_recall@k (per text): GT image in top-k
    def t2i_recall(k):
        hits = 0
        for r in t2i:
            if r["gt_image_id"] in r["ranked_image_ids"][:k]:
                hits += 1
        return hits / len(t2i)
    all_ok = True
    for k in (1, 5, 10):
        rec_t = i2t_recall(k)   # text retrieval (rank texts given image)
        rec_i = t2i_recall(k)   # image retrieval (rank images given text)
        exp_t = score["metrics"][f"text_retrieval_recall@{k}"]
        exp_i = score["metrics"][f"image_retrieval_recall@{k}"]
        ok_t = abs(rec_t - exp_t) < 1e-3
        ok_i = abs(rec_i - exp_i) < 1e-3
        all_ok = all_ok and ok_t and ok_i
        print(f"[retrieval @{k}] text: rec={rec_t:.5f} exp={exp_t:.5f} {'OK' if ok_t else 'MISMATCH'}  | image: rec={rec_i:.5f} exp={exp_i:.5f} {'OK' if ok_i else 'MISMATCH'}")
    print(f"[retrieval] n_images={len(i2t)} n_texts={len(t2i)}")
    return all_ok


def main():
    ok_c = check_classification()
    ok_r = check_retrieval()
    print()
    print("ALL OK" if (ok_c and ok_r) else "FAILED")
    sys.exit(0 if (ok_c and ok_r) else 1)


if __name__ == "__main__":
    main()
