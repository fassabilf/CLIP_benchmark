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

import open_clip


def load_cvqa(cvqa_dir):
    import glob
    files = sorted(glob.glob(os.path.join(cvqa_dir, "data", "test-*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df


@torch.no_grad()
def eval_split(df, model, preprocess, tokenizer, key_prefix, device, batch_size):
    """key_prefix='' for LOCAL, 'Translated ' for EN."""
    q_col = f"{key_prefix}Question"
    o_col = f"{key_prefix}Options"

    correct = 0
    subset_correct = defaultdict(lambda: [0, 0])

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
        pred = sims.argmax(dim=-1).cpu().numpy()
        for i, (p, lbl, sub) in enumerate(zip(pred, labels, subsets)):
            match = int(p == lbl)
            correct += match
            subset_correct[str(sub)][0] += match
            subset_correct[str(sub)][1] += 1

    acc = correct / len(df)
    subset_acc = {s: v[0] / v[1] for s, v in subset_correct.items()}
    return acc, subset_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="open_clip arch name")
    ap.add_argument("--pretrained", required=True, help="open_clip pretrained tag or .pt path")
    ap.add_argument("--cvqa_dir", default="/project/lt200394-thllmV/kd_dataset/eval/cvqa")
    ap.add_argument("--cache_dir", default=os.environ.get("HF_HUB_CACHE"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, model={args.model}, pretrained={args.pretrained}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, cache_dir=args.cache_dir,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    df = load_cvqa(args.cvqa_dir)
    print(f"CVQA rows: {len(df)}")

    results = {}
    for prefix, name in [("", "LOCAL"), ("Translated ", "EN")]:
        acc, sub = eval_split(df, model, preprocess, tokenizer, prefix, device, args.batch_size)
        results[name] = acc
        results[f"{name}_subset"] = sub
        print(f"{name}: acc={acc:.4f}  n_subsets={len(sub)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
