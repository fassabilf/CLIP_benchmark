#!/usr/bin/env python3
"""Inspect a raw open_clip checkpoint to determine its tokenizer/config variant
without instantiating a model -- avoids the token_embedding size-mismatch trap
(see runs/env.sh CKDONLY_V1 comment: checkpoint turned out to be vocab=256000
SigLIP2, not the CLIP-BPE 49408 we first assumed for ckdonly_v1).

Reports token_embedding shape (vocab size = real config, not a placeholder --
read straight from the saved tensor), positional_embedding / context length,
weight stats (to confirm it's trained, not zero-init), and param counts.

Usage:
  python3 runs/inspect_checkpoint.py /path/to/epoch_32.pt
  python3 runs/inspect_checkpoint.py ckpt1.pt ckpt2.pt ckpt3.pt
"""
import argparse

import torch

KNOWN_VOCABS = {
    49408: "CLIP-BPE (ctx 77) -- mc2*/metaclip2_kd family, eval in mc2_eval_env",
    256000: "SigLIP2 HFTokenizer -- mammoth/selflearn/ckdonly family, eval in mteb_env2",
}


def load_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "state_dict" and k != "optimizer"}
    else:
        sd = ckpt
        meta = {}
    # strip DDP "module." prefix
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    return sd, meta


def find_key(sd, suffix):
    if suffix in sd:
        return suffix
    candidates = [k for k in sd if k.endswith(suffix)]
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+", help="one or more .pt checkpoint paths")
    args = ap.parse_args()

    for path in args.ckpts:
        print(f"\n=== {path} ===")
        sd, meta = load_state_dict(path)
        if meta:
            print(f"  checkpoint meta: {meta}")

        tok_key = find_key(sd, "token_embedding.weight")
        if tok_key is None:
            print("  WARNING: no token_embedding.weight found -- can't determine vocab/tokenizer")
            continue

        w = sd[tok_key]
        vocab_size, embed_dim = w.shape
        print(f"  {tok_key}: {tuple(w.shape)}  (vocab={vocab_size}, embed_dim={embed_dim})")
        print(f"    weight stats: mean={w.float().mean():.6f} std={w.float().std():.6f} "
              f"min={w.float().min():.4f} max={w.float().max():.4f}")
        if w.float().std() < 1e-6:
            print("    !! std ~= 0 -- looks like an untrained/placeholder tensor, not a real checkpoint")

        pos_key = find_key(sd, "positional_embedding")
        if pos_key is not None:
            ctx_len = sd[pos_key].shape[0]
            print(f"  {pos_key}: {tuple(sd[pos_key].shape)}  (context_length={ctx_len})")

        variant = KNOWN_VOCABS.get(
            vocab_size,
            "UNKNOWN vocab size -- not CLIP-BPE (49408) or SigLIP2 (256000), check manually",
        )
        print(f"  => variant: {variant}")

        total_params = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
        visual_params = sum(v.numel() for k, v in sd.items() if torch.is_tensor(v) and k.startswith("visual."))
        other_params = total_params - visual_params
        print(f"  total params: {total_params:,}  (visual: {visual_params:,}, text+other: {other_params:,})")


if __name__ == "__main__":
    main()
