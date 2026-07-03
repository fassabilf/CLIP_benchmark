#!/usr/bin/env python3
"""Upload per-sample prediction dumps (preds/) to the HuggingFace dataset repo.

Per-sample preds/ dumps are excluded from git (see runs/.gitignore) and live on
HF instead, so results stay reproducible without bloating the repo.

Usage:
  python runs/push_clipkd_mammoth_bpe_v1_preds_to_hf.py
"""
from huggingface_hub import HfApi

api = HfApi()
api.upload_large_folder(
    folder_path="runs/results",
    repo_id="fassabilf/sea-clip-eval-predictions",
    repo_type="dataset",
    allow_patterns=[
        "clipkd_mammoth_bpe_v1_e8/preds/*.jsonl",
        "clipkd_mammoth_bpe_v1_e16/preds/*.jsonl",
        "clipkd_mammoth_bpe_v1_e24/preds/*.jsonl",
        "clipkd_mammoth_bpe_v1_e32/preds/*.jsonl",
    ],
)

print("Done! View at: https://huggingface.co/datasets/fassabilf/sea-clip-eval-predictions")
