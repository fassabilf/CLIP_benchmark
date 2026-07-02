#!/usr/bin/env python3
"""Upload ckdonly_v1 per-sample prediction dumps to the HuggingFace dataset repo.

Per-sample preds/ dumps are excluded from git (see runs/.gitignore) and live on
HF instead, so results stay reproducible without bloating the repo.

Usage:
  python runs/push_ckdonly_v1_preds_to_hf.py --token hf_xxxx
  # or: export HF_TOKEN=hf_xxxx && python runs/push_ckdonly_v1_preds_to_hf.py
  # or rely on the huggingface-cli cached login token (no --token / HF_TOKEN needed)
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "fassabilf/sea-clip-eval-predictions"
REPO_TYPE = "dataset"

RESULTS_DIR = Path(__file__).parent / "results"
TAGS = ["ckdonly_v1_e8", "ckdonly_v1_e16", "ckdonly_v1_e24", "ckdonly_v1_e32"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", None))
    args = parser.parse_args()

    api = HfApi(token=args.token)
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, token=args.token, exist_ok=True)
    print(f"Repository ready: https://huggingface.co/datasets/{REPO_ID}")

    for tag in TAGS:
        preds_dir = RESULTS_DIR / tag / "preds"
        if not preds_dir.exists():
            print(f"  SKIP (not found): {preds_dir}")
            continue
        print(f"  Uploading: {preds_dir} -> {tag}/")
        api.upload_folder(
            folder_path=str(preds_dir),
            path_in_repo=tag,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"  Done: {tag}")

    print(f"\nDone! View at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
