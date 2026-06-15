"""Download WorldCuisines VQA images from image_url to a local cache directory.

Run on login node (needs internet). Compute nodes have HF_DATASETS_OFFLINE=1 and
no internet access, so images must be pre-cached here before running eval.

Usage:
    python download_worldcuisines_images.py \
        --out_dir /project/lt200394-thllmV/kd_dataset/eval/worldcuisines \
        [--hf_cache /project/lt200394-thllmV/benchmark/.cache/huggingface/datasets]
"""
import argparse
import concurrent.futures
import os
import time
from pathlib import Path


import re

_HEADERS = {
    "User-Agent": (
        "WorldCuisinesResearch/1.0 "
        "(https://huggingface.co/datasets/worldcuisines/vqa; "
        "academic research) Python/3.11"
    ),
}


def normalize_url(image_url):
    """Strip ?download and convert thumb URLs to original file URLs.

    Wikimedia thumbnail rendering is blocked for many server IPs.
    The original (non-resized) file is available at the non-thumb URL:
      commons/thumb/A/AB/file.jpg/WIDTHpx-file.jpg → commons/A/AB/file.jpg
    """
    url = image_url.replace("?download", "")
    # commons/thumb/{a}/{ab}/{file}.jpg/{width}px-{file}.jpg → commons/{a}/{ab}/{file}.jpg
    url = re.sub(r"(/thumb)((/[^/]+){2}/[^/]+)\.\w+/\d+px-[^/]+$", r"\2", url)
    return url


def fetch_one(args):
    import requests
    image_path, image_url, out_dir, delay = args
    dest = out_dir / image_path
    if dest.exists():
        return "skip", image_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = normalize_url(image_url)
    if delay > 0:
        time.sleep(delay)
    for attempt in range(4):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=45)
            if resp.status_code == 429:
                wait = 10 * (2 ** attempt)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return "ok", image_path
        except Exception as e:
            if attempt == 3:
                return "fail", f"{image_path}: {e}"
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/project/lt200394-thllmV/kd_dataset/eval/worldcuisines")
    ap.add_argument("--hf_cache", default=os.environ.get("HF_DATASETS_CACHE",
        "/project/lt200394-thllmV/benchmark/.cache/huggingface/datasets"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.3,
                    help="per-request delay in seconds (spread load across workers)")
    ap.add_argument("--tasks", nargs="+", default=["task1", "task2"])
    ap.add_argument("--splits", nargs="+", default=["test_large"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_DATASETS_CACHE"] = args.hf_cache

    from datasets import load_dataset

    # Collect unique (image_path, image_url) pairs across all requested tasks/splits
    pairs = {}
    for task in args.tasks:
        for split in args.splits:
            print(f"Loading metadata: {task}/{split} ...")
            ds = load_dataset("worldcuisines/vqa", task, split=split,
                              cache_dir=args.hf_cache)
            for row in ds:
                ip = row["image_path"]  # e.g. "images/1462_xxx.jpg"
                iu = row["image_url"]
                if ip not in pairs:
                    pairs[ip] = iu
    print(f"Total unique images: {len(pairs)}")

    tasks = [(ip, iu, out_dir, args.delay) for ip, iu in pairs.items()]
    ok = skip = fail = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (status, msg) in enumerate(pool.map(fetch_one, tasks), 1):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"  FAIL: {msg}")
            if i % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {i}/{len(tasks)} done  ok={ok} skip={skip} fail={fail}  "
                      f"({elapsed:.0f}s elapsed)")

    print(f"\nDone. ok={ok}  skip={skip}  fail={fail}  total={len(pairs)}")
    if fail:
        print("WARNING: Some images failed to download. Eval will skip those rows.")


if __name__ == "__main__":
    main()
