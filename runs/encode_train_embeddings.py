#!/usr/bin/env python3
"""Phase A — encode image+text embeddings over the ACTUAL training shards.

PATCHED copy of kd_dataset/scripts/encode_train_embeddings.py — adds an
`--model-type hf_transformers` path so HF dual-encoders (e.g. MetaCLIP2-B16-worldwide,
`facebook/metaclip-2-worldwide-b16`) can be probed too, not just open_clip checkpoints.
The HF path defers to clip_benchmark's own loader
(`clip_benchmark.models.load_clip`, model_type="hf_transformers") — the SAME wrapper
the eval uses — so there is one source of truth for image/text feature extraction.
It is imported lazily inside the hf_transformers branch, so the open_clip path keeps
working in envs where clip_benchmark is not installed (e.g. mc2_eval_env).

Reads the webdataset shards that the KD dataloader consumed, runs the model over every
(image, caption) pair, and persists L2-normalized embeddings + the per-pair diagonal
cosine similarity. Retrieval / analysis happens later, offline, in
`retrieval_from_emb.py` (Phase B).

Envs:
  - open_clip students (ViT-T-16 CLIP-BPE):  conda env `mc2_eval_env`
  - hf_transformers teacher (MetaCLIP2-B16):  conda env `mteb_env2` (has transformers + MetaClip2)

Output per (dataset, lang):
  {out_dir}/{tag}/{dataset}-{lang}.npz   images_emb[N,D] fp16, texts_emb[N,D] fp16,
                                         diag_sim[N] fp32
  {out_dir}/{tag}/{dataset}-{lang}.parquet   key, caption, lang   (strings, sidecar)

Bloom is stored as a single `bloom-all.*` (sea7 langs interleaved; per-sample `lang`
parsed from the wds key) — Phase B groups it by lang.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WDS_ROOT = "/lustrefs/disk/project/lt200394-thllmV/kd_dataset/webdataset"

# Per-language shard dirs that actually exist for each pool (see TRAIN_MULTILINGUAL.md).
CG_LANGS = ["id", "jv", "ms", "su", "th", "vi"]          # CG-OE-Filt (no my)
WIT_LANGS = ["id", "jv", "ms", "my", "th", "vi"]         # WIT witbase (no su)


def load_model(args, device):
    """Return (model, preprocess, tokenizer) for either backend.

    Both backends expose the same interface used by encode_job:
      model.encode_image(pixel_values) / model.encode_text(tokens),
      preprocess(PIL) -> tensor, tokenizer(list[str]) -> tokens.
    """
    if args.model_type == "hf_transformers":
        # Defer to clip_benchmark's loader (same wrapper the eval uses). Imported
        # lazily so the open_clip path doesn't require clip_benchmark to be installed.
        from clip_benchmark.models import load_clip
        print(f"hf_transformers via clip_benchmark.load_clip: {args.model} "
              f"(cache_dir={args.model_cache_dir})", flush=True)
        model, preprocess, tokenizer = load_clip(
            model_type="hf_transformers",
            model_name=args.model,                 # e.g. facebook/metaclip-2-worldwide-b16
            pretrained=args.ckpt or "",
            cache_dir=args.model_cache_dir,
            device=device,
        )
        return model, preprocess, tokenizer

    # default: open_clip
    import open_clip
    print("open_clip from:", open_clip.__file__, flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.ckpt)
    tokenizer = open_clip.get_tokenizer(args.model)
    return model.to(device).eval(), preprocess, tokenizer


def shard_urls(dirpath):
    tars = sorted(str(p) for p in Path(dirpath).glob("*.tar"))
    if not tars:
        print(f"  WARN: no shards in {dirpath}", file=sys.stderr)
    return tars


def build_jobs(datasets, langs_filter):
    """Return list of (dataset, lang_or_None, urls)."""
    jobs = []
    for ds in datasets:
        if ds == "cgoe":
            for lang in CG_LANGS:
                if langs_filter and lang not in langs_filter:
                    continue
                jobs.append(("cgoe", lang, shard_urls(f"{WDS_ROOT}/cultural-ground-oe-filt-{lang}")))
        elif ds == "wit":
            for lang in WIT_LANGS:
                if langs_filter and lang not in langs_filter:
                    continue
                jobs.append(("wit", lang, shard_urls(f"{WDS_ROOT}/wit-hf-base-{lang}")))
        elif ds == "bloom":
            jobs.append(("bloom", None, shard_urls(f"{WDS_ROOT}/bloom-sea7")))
        else:
            raise ValueError(f"unknown dataset {ds!r}")
    return jobs


def make_loader(urls, preprocess, batch_size, num_workers, max_n):
    import webdataset as wds

    def _identity(x):
        return x

    pipeline = (
        # empty_check=False: with many workers and few shards, some workers legitimately get
        # 0 shards; without this wds 1.x raises "No samples found ... fewer shards than workers".
        wds.WebDataset(urls, shardshuffle=False, empty_check=False, handler=wds.warn_and_continue)
        .decode("pilrgb", handler=wds.warn_and_continue)
        .to_tuple("__key__", "jpg", "txt", handler=wds.warn_and_continue)
        .map_tuple(_identity, preprocess, _identity)
    )
    if max_n:
        pipeline = pipeline.slice(max_n)

    def collate(samples):
        keys = [s[0] for s in samples]
        imgs = torch.stack([s[1] for s in samples])
        txts = [s[2] for s in samples]
        return keys, imgs, txts

    return torch.utils.data.DataLoader(
        pipeline, batch_size=batch_size, num_workers=num_workers, collate_fn=collate,
    )


def bloom_lang_of(key):
    # keys look like "bloom-tha-00000000" -> lang token is field [1]
    parts = key.split("-")
    return parts[1] if len(parts) >= 3 else "unk"


@torch.no_grad()
def encode_job(model, tokenizer, preprocess, urls, device, amp, batch_size, num_workers, max_n):
    nw = min(num_workers, max(1, len(urls)))   # no idle workers when a pool has few shards
    loader = make_loader(urls, preprocess, batch_size, nw, max_n)
    ie_chunks, te_chunks, diag_chunks = [], [], []
    keys, caps = [], []
    n = 0
    autocast = torch.autocast(device_type="cuda", enabled=amp) if device == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    for bkeys, imgs, txts in loader:
        imgs = imgs.to(device, non_blocking=True)
        toks = tokenizer(list(txts)).to(device)
        with autocast:
            ie = F.normalize(model.encode_image(imgs), dim=-1).float()
            te = F.normalize(model.encode_text(toks), dim=-1).float()
        diag = (ie * te).sum(-1)
        ie_chunks.append(ie.half().cpu())
        te_chunks.append(te.half().cpu())
        diag_chunks.append(diag.cpu())
        keys.extend(bkeys)
        caps.extend(txts)
        n += len(bkeys)
        if n % (batch_size * 20) < batch_size:
            print(f"    ... {n} pairs", flush=True)
    if n == 0:
        return None
    return {
        "images_emb": torch.cat(ie_chunks).numpy(),
        "texts_emb": torch.cat(te_chunks).numpy(),
        "diag_sim": torch.cat(diag_chunks).numpy(),
        "key": keys,
        "caption": caps,
    }


def save_shard(out_stem, images_emb, texts_emb, diag_sim, keys, caps, langs):
    np.savez(out_stem + ".npz",
             images_emb=images_emb.astype(np.float16),
             texts_emb=texts_emb.astype(np.float16),
             diag_sim=diag_sim.astype(np.float32))
    df = {"key": keys, "caption": caps, "lang": langs}
    try:
        import pandas as pd
        pd.DataFrame(df).to_parquet(out_stem + ".parquet", index=False)
    except Exception as e:
        print(f"    parquet failed ({e}); writing csv", file=sys.stderr)
        import csv
        with open(out_stem + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "caption", "lang"])
            for k, c, l in zip(keys, caps, langs):
                w.writerow([k, c.replace("\n", " "), l])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="open_clip pretrained tag/path (required for --model-type open_clip; "
                         "ignored for hf_transformers)")
    ap.add_argument("--tag", required=True, help="e.g. mc2_e32 / metaclip2_b16")
    ap.add_argument("--out-dir", default="/lustrefs/disk/project/lt200394-thllmV/kd_dataset/eval/train_probe/emb")
    ap.add_argument("--model", default="ViT-T-16",
                    help="open_clip arch, OR HF repo id (e.g. facebook/metaclip-2-worldwide-b16)")
    ap.add_argument("--model-type", choices=["open_clip", "hf_transformers"], default="open_clip")
    ap.add_argument("--model-cache-dir", default=None, help="HF cache dir (hf_transformers only)")
    ap.add_argument("--datasets", nargs="+", default=["bloom", "cgoe", "wit"])
    ap.add_argument("--langs", nargs="+", default=None, help="filter cg/wit langs; None=all")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-per-lang", type=int, default=None, help="smoke cap per job")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    if args.model_type == "open_clip" and not args.ckpt:
        ap.error("--ckpt is required when --model-type open_clip")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = load_model(args, device)
    amp = (not args.no_amp) and device == "cuda"

    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    src = args.ckpt if args.model_type == "open_clip" else args.model
    print(f"tag={args.tag} model_type={args.model_type} src={src}\nout={out_dir}\ndevice={device} amp={amp}", flush=True)

    jobs = build_jobs(args.datasets, args.langs)
    for ds, lang, urls in jobs:
        if not urls:
            print(f"skip {ds}/{lang}: no shards", flush=True)
            continue
        name = f"{ds}-{lang}" if lang else f"{ds}-all"
        out_stem = str(out_dir / name)
        if os.path.exists(out_stem + ".npz"):
            print(f"skip {name}: exists", flush=True)
            continue
        t0 = time.time()
        print(f"[{name}] encoding {len(urls)} shards ...", flush=True)
        res = encode_job(model, tokenizer, preprocess, urls, device, amp,
                         args.batch_size, args.num_workers, args.max_per_lang)
        if res is None:
            print(f"  {name}: 0 pairs, skip", flush=True)
            continue
        langs = [bloom_lang_of(k) for k in res["key"]] if ds == "bloom" else [lang] * len(res["key"])
        save_shard(out_stem, res["images_emb"], res["texts_emb"], res["diag_sim"],
                   res["key"], res["caption"], langs)
        dt = time.time() - t0
        print(f"  {name}: {len(res['key'])} pairs, mean_diag_cos={res['diag_sim'].mean():.4f}, "
              f"{dt:.1f}s -> {out_stem}.npz", flush=True)


if __name__ == "__main__":
    main()
