"""Standalone CVQA evaluation adapted from metaclip/clipeval/cvqa/eval_cvqa.py.

CVQA = 4-way image-text multiple choice. For each row we encode:
  - 1 image
  - 4 (Question + Option_i) text candidates
Prediction = argmax cosine-sim. Two passes:
  - LOCAL: uses raw `Question` / `Options` (local language)
  - EN:    uses `Translated Question` / `Translated Options` (English)
"""
import argparse
import io
import json
import os
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from clip_benchmark.models import MODEL_TYPES, load_clip


def load_cvqa(cvqa_dir):
    import glob
    files = sorted(glob.glob(os.path.join(cvqa_dir, "data", "test-*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df


@torch.no_grad()
def eval_split(df, model, preprocess, tokenizer, key_prefix, device, batch_size, pred_writer=None, split_name=None):
    """key_prefix='' for LOCAL, 'Translated ' for EN."""
    q_col = f"{key_prefix}Question"
    o_col = f"{key_prefix}Options"

    correct = 0
    subset_correct = defaultdict(lambda: [0, 0])
    global_idx = 0

    for start in tqdm(range(0, len(df), batch_size), desc=f"split={key_prefix or 'LOCAL'}"):
        batch = df.iloc[start:start + batch_size]
        imgs, texts, labels, subsets = [], [], [], []
        for _, row in batch.iterrows():
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            imgs.append(preprocess(img))
            for opt in row[o_col]:
                texts.append(f"{row[q_col]} {opt}")
            labels.append(int(row["Label"]))
            subsets.append(tuple(row["Subset"]) if hasattr(row["Subset"], "__len__") else row["Subset"])

        images = torch.stack(imgs).to(device, non_blocking=True)
        tok = tokenizer(texts).to(device, non_blocking=True)

        img_emb = model.encode_image(images)
        txt_emb = model.encode_text(tok)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        N, D = img_emb.shape
        txt_emb = txt_emb.view(N, 4, D)
        sims = torch.bmm(img_emb.unsqueeze(1), txt_emb.transpose(1, 2)).squeeze(1)
        topk_vals, topk_idx = torch.topk(sims, k=4, dim=-1)
        pred = topk_idx[:, 0].cpu().numpy()
        if pred_writer is not None:
            sims_cpu = sims.cpu().tolist()
            topk_vals_cpu = topk_vals.cpu().tolist()
            topk_idx_cpu = topk_idx.cpu().tolist()
            for i, (lbl, sub) in enumerate(zip(labels, subsets)):
                row = {
                    "sample_idx": global_idx + i,
                    "split": split_name,
                    "subset": str(sub),
                    "true": int(lbl),
                    "topk": [int(x) for x in topk_idx_cpu[i]],
                    "topk_scores": [float(s) for s in topk_vals_cpu[i]],
                    "all_scores": [float(s) for s in sims_cpu[i]],
                }
                pred_writer.write(json.dumps(row) + "\n")
        for i, (p, lbl, sub) in enumerate(zip(pred, labels, subsets)):
            match = int(p == lbl)
            correct += match
            subset_correct[str(sub)][0] += match
            subset_correct[str(sub)][1] += 1
        global_idx += len(labels)

    acc = correct / len(df)
    subset_acc = {s: v[0] / v[1] for s, v in subset_correct.items()}
    return acc, subset_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model arch name (open_clip) or HF repo id (hf_transformers)")
    ap.add_argument("--pretrained", default="", help="open_clip pretrained tag or .pt path; '' for hf_transformers")
    ap.add_argument("--model_type", default="auto", choices=MODEL_TYPES + ["auto"], help="'auto' infers from --model: 'org/name' → hf_transformers, else open_clip.")
    ap.add_argument("--cvqa_dir", default="/project/lt200394-thllmV/kd_dataset/eval/cvqa")
    ap.add_argument("--cache_dir", default=os.environ.get("HF_HUB_CACHE"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--save_predictions", default=None, help="optional path to JSONL for per-sample predictions (both LOCAL and EN splits appended).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, model_type={args.model_type}, model={args.model}, pretrained={args.pretrained}")

    model, preprocess, tokenizer = load_clip(
        model_type=args.model_type,
        model_name=args.model,
        pretrained=args.pretrained,
        cache_dir=args.cache_dir,
        device=device,
    )
    model = model.eval()

    df = load_cvqa(args.cvqa_dir)
    print(f"CVQA rows: {len(df)}")

    results = {}
    pred_writer = None
    if args.save_predictions is not None:
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
        pred_writer = open(args.save_predictions, "w")
    try:
        for prefix, name in [("", "LOCAL"), ("Translated ", "EN")]:
            acc, sub = eval_split(df, model, preprocess, tokenizer, prefix, device, args.batch_size, pred_writer=pred_writer, split_name=name)
            results[name] = acc
            results[f"{name}_subset"] = sub
            print(f"{name}: acc={acc:.4f}  n_subsets={len(sub)}")
    finally:
        if pred_writer is not None:
            pred_writer.close()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
