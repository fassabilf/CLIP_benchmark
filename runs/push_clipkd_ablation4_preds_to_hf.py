#!/usr/bin/env python3
"""Upload per-sample prediction dumps (preds/) to the HuggingFace dataset repo.

Per-sample preds/ dumps are excluded from git (see runs/.gitignore) and live on
HF instead, so results stay reproducible without bloating the repo.

Usage:
  python runs/push_clipkd_ablation4_preds_to_hf.py
"""
from types import SimpleNamespace

from huggingface_hub import HfApi

REPO_ID = "fassabilf/sea-clip-eval-predictions"

api = HfApi()
# huggingface_hub's upload_large_folder unconditionally calls create_repo(exist_ok=True)
# internally before uploading. Our fine-grained token is scoped to read/write on this
# existing repo only (no account-level repo-creation permission), so that call 401s even
# though exist_ok=True and the repo already exists. Skip it since we know the repo exists.
api.create_repo = lambda *a, **kw: SimpleNamespace(repo_id=kw.get("repo_id", REPO_ID))

MODELS = ["ckd_v1", "fdonly_v1", "icl_v1", "nokd_v1"]
EPOCHS = [8, 16, 24, 32]

api.upload_large_folder(
    folder_path="runs/results",
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=[
        f"clipkd_{m}_e{e}/preds/*.jsonl" for m in MODELS for e in EPOCHS
    ],
    # Default worker count (half of cpu cores, often 32-64 here) hammers HF's preupload
    # endpoint hard enough to get stuck in a permanent 429 retry loop. Single worker to be
    # as gentle as possible on the rate limit.
    num_workers=1,
)

print("Done! View at: https://huggingface.co/datasets/fassabilf/sea-clip-eval-predictions")
