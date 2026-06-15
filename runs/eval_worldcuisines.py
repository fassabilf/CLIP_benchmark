"""WorldCuisines VQA evaluation for CLIP models.

WorldCuisines is a 5-way multiple-choice VQA benchmark for food identification
across 23+ languages. We encode the image and 5 "{question} {option}" texts,
then predict the option with the highest image-text cosine similarity.

Differs from eval_cvqa.py:
  - 5-way MC (not 4-way)
  - Images are pre-downloaded URLs (not embedded bytes in parquet)
  - Options are parsed from the multi_choice_prompt string
  - multi_choice_answer is 1-indexed (not 0-indexed)

Usage:
    python eval_worldcuisines.py \
        --model ViT-T-16 \
        --pretrained /path/to/checkpoint.pt \
        --wc_images_dir /project/.../eval/worldcuisines \
        --output /path/to/result.json \
        [--task task1] [--split test_large] [--batch_size 64]
"""
import argparse
import io
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from clip_benchmark.models import MODEL_TYPES, load_clip

# WorldCuisines uses granular register codes (not ISO-2): id_casual/formal, jv_krama/ngoko, su_loma, th, tl
SEA_LANGS = {"id_casual", "id_formal", "jv_krama", "jv_ngoko", "su_loma", "th", "tl"}


def parse_prompt(prompt):
    """Extract (question, [opt1, opt2, opt3, opt4, opt5]) from multi_choice_prompt.

    Handles two formats from WorldCuisines:
      test_small: "Question? 1. OptionA 2. OptionB ...  Print only..."
      test_large: "Question?\n1. OptionA\n2. OptionB\n\nPrint only..."
    """
    # strip the LLM-instruction tail (handles spaces or newlines before "Print only")
    stripped = re.sub(r"\s*Print only.*", "", prompt, flags=re.DOTALL).strip()
    # split on whitespace + "N. " markers; \s+ covers both \n and space separators
    parts = re.split(r"\s+(\d+)\. ", stripped, maxsplit=10)
    # parts is: [question, "1", opt1, "2", opt2, ...]
    question = parts[0].strip()
    options = []
    i = 1
    while i + 1 < len(parts):
        options.append(parts[i + 1].strip())
        i += 2
    return question, options


def load_dataset_offline(task, split, hf_cache):
    os.environ.setdefault("HF_DATASETS_CACHE", hf_cache)
    # allow online if cache not pre-downloaded
    from datasets import load_dataset
    return load_dataset("worldcuisines/vqa", task, split=split,
                        cache_dir=hf_cache)


@torch.no_grad()
def run_eval(ds, model, preprocess, tokenizer, images_dir, device, batch_size,
             pred_writer=None):
    images_dir = Path(images_dir)
    correct = 0
    total = 0
    skipped = 0
    lang_correct = defaultdict(lambda: [0, 0])  # lang → [correct, total]
    global_idx = 0

    rows = list(ds)

    for start in tqdm(range(0, len(rows), batch_size)):
        batch = rows[start:start + batch_size]
        imgs, text_batch, labels, langs, n_opts_batch = [], [], [], [], []

        for row in batch:
            img_path = images_dir / row["image_path"]
            if not img_path.exists():
                skipped += 1
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                skipped += 1
                continue

            question, options = parse_prompt(row["multi_choice_prompt"])
            if len(options) == 0:
                skipped += 1
                continue

            imgs.append(preprocess(img))
            # encode question + each option as candidate text
            for opt in options:
                text_batch.append(f"{question} {opt}")
            labels.append(int(row["multi_choice_answer"]) - 1)  # 1→0-indexed
            langs.append(row["lang"])
            n_opts_batch.append(len(options))

        if not imgs:
            global_idx += len(batch)
            continue

        images = torch.stack(imgs).to(device, non_blocking=True)
        tok = tokenizer(text_batch).to(device, non_blocking=True)

        img_emb = model.encode_image(images)
        txt_emb = model.encode_text(tok)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        # build per-row tensors (variable n_opts supported)
        offset = 0
        for i, (n_opts, lbl, lang) in enumerate(zip(n_opts_batch, labels, langs)):
            t = txt_emb[offset:offset + n_opts]  # (n_opts, D)
            offset += n_opts
            sims = (img_emb[i] @ t.T)  # (n_opts,)
            pred = int(sims.argmax().item())
            match = int(pred == lbl)
            correct += match
            total += 1
            lang_correct[lang][0] += match
            lang_correct[lang][1] += 1

            if pred_writer is not None:
                pred_writer.write(json.dumps({
                    "sample_idx": global_idx + i,
                    "lang": lang,
                    "true": lbl,
                    "pred": pred,
                    "scores": sims.cpu().tolist(),
                }) + "\n")

        global_idx += len(batch)

    acc = correct / total if total > 0 else 0.0

    per_lang = {lang: v[0] / v[1] for lang, v in lang_correct.items()}
    sea_langs = {k: v for k, v in per_lang.items() if k in SEA_LANGS}
    non_sea = {k: v for k, v in per_lang.items() if k not in SEA_LANGS}
    mean_sea = sum(sea_langs.values()) / len(sea_langs) if sea_langs else 0.0
    mean_non_sea = sum(non_sea.values()) / len(non_sea) if non_sea else 0.0

    print(f"  acc={acc:.4f}  n={total}  skipped={skipped}")
    print(f"  mean_SEA={mean_sea:.4f}  mean_non_SEA={mean_non_sea:.4f}")
    for lang in sorted(per_lang):
        cnt = lang_correct[lang][1]
        print(f"    {lang}: {per_lang[lang]:.4f} ({cnt})")

    return {
        "acc": acc,
        "n": total,
        "skipped": skipped,
        "mean_SEA": mean_sea,
        "mean_non_SEA": mean_non_sea,
        "per_lang": per_lang,
        "lang_counts": {k: v[1] for k, v in lang_correct.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pretrained", default="")
    ap.add_argument("--model_type", default="auto",
                    choices=MODEL_TYPES + ["auto"])
    ap.add_argument("--wc_images_dir",
                    default="/project/lt200394-thllmV/kd_dataset/eval/worldcuisines")
    ap.add_argument("--hf_cache",
                    default=os.environ.get("HF_DATASETS_CACHE",
                        "/project/lt200394-thllmV/benchmark/.cache/huggingface/datasets"))
    ap.add_argument("--task", default="task1", choices=["task1", "task2"])
    ap.add_argument("--split", default="test_large")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--save_predictions", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}  pretrained={args.pretrained}")
    print(f"task={args.task}  split={args.split}")

    # parse check on 3 samples
    from datasets import load_dataset as _ld
    ds_check = _ld("worldcuisines/vqa", args.task, split=args.split,
                   cache_dir=args.hf_cache)
    print(f"Dataset rows: {len(ds_check)}")
    for row in list(ds_check)[:3]:
        q, opts = parse_prompt(row["multi_choice_prompt"])
        ans_idx = int(row["multi_choice_answer"]) - 1
        print(f"  q={q[:60]!r}  n_opts={len(opts)}  label={ans_idx}  "
              f"correct_opt={opts[ans_idx][:30]!r}")

    model, preprocess, tokenizer = load_clip(
        model_type=args.model_type,
        model_name=args.model,
        pretrained=args.pretrained,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        device=device,
    )
    model = model.eval()

    pred_writer = None
    if args.save_predictions is not None:
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
        pred_writer = open(args.save_predictions, "w")

    try:
        results = run_eval(
            ds_check, model, preprocess, tokenizer,
            args.wc_images_dir, device, args.batch_size,
            pred_writer=pred_writer,
        )
    finally:
        if pred_writer is not None:
            pred_writer.close()

    results["model"] = args.model
    results["pretrained"] = args.pretrained
    results["task"] = args.task
    results["split"] = args.split

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
