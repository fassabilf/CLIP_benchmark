#!/usr/bin/env python3
"""Compute per-language mean metric for filling the tab:per_language LaTeX table.

For each language, averages metric across all benchmarks that support it:
  - id, th, vi : 4 benchmarks (babel_imagenet, crossmodal3600, flickr30k-200, xtd200)
  - ms, jv, su, my : 3 benchmarks (no XM3600 for these)

Metric used:
  - babel_imagenet → acc1
  - crossmodal3600, flickr30k-200, xtd200 → image_retrieval_recall@1

Usage:
  python3 runs/compute_per_lang_mean.py tinyclip mobileclip2_s0
  python3 runs/compute_per_lang_mean.py tinyclip          # single model
"""
import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Language → list of (dataset, lang_key) pairs to average
SEA_LANGS = {
    "th": [
        ("babel_imagenet", "th"),
        ("crossmodal3600", "th"),
        ("flickr30k-200",  "tha_Thai"),
        ("xtd200",         "tha_Thai"),
    ],
    "my": [
        ("babel_imagenet", "my"),
        ("flickr30k-200",  "mya_Mymr"),
        ("xtd200",         "mya_Mymr"),
    ],
    "ms": [
        ("babel_imagenet", "ms"),
        ("flickr30k-200",  "zsm_Latn"),
        ("xtd200",         "zsm_Latn"),
    ],
    "id": [
        ("babel_imagenet", "id"),
        ("crossmodal3600", "id"),
        ("flickr30k-200",  "ind_Latn"),
        ("xtd200",         "ind_Latn"),
    ],
    "jv": [
        ("babel_imagenet", "jv"),
        ("flickr30k-200",  "jav_Latn"),
        ("xtd200",         "jav_Latn"),
    ],
    "su": [
        ("babel_imagenet", "su"),
        ("flickr30k-200",  "sun_Latn"),
        ("xtd200",         "sun_Latn"),
    ],
    "vi": [
        ("babel_imagenet", "vi"),
        ("crossmodal3600", "vi"),
        ("flickr30k-200",  "vie_Latn"),
        ("xtd200",         "vie_Latn"),
    ],
}

TABLE_ORDER = ["th", "my", "ms", "id", "jv", "su", "vi"]
DISPLAY_NAMES = {
    "th": "Thai", "my": "Myanmar", "ms": "Malay",
    "id": "Indonesia", "jv": "Javanese", "su": "Sundanese", "vi": "Vietnamese",
}

METRIC_BY_DATASET = {
    "babel_imagenet":   "acc1",
    "crossmodal3600":   "image_retrieval_recall@1",
    "flickr30k-200":    "image_retrieval_recall@1",
    "xtd200":           "image_retrieval_recall@1",
}


def load_results(tag: str, retrieval_metric: str = "image_retrieval_recall@1") -> dict:
    """Load all JSON result files for a given tag into (dataset, lang) → metric."""
    tag_dir = RESULTS_DIR / tag
    if not tag_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {tag_dir}")
    results = {}
    for f in sorted(tag_dir.glob("*.json")):
        if "preds" in f.parts:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        ds = d.get("dataset", "")
        lang = d.get("language", "")
        task = d.get("task", "")
        m = d.get("metrics", {})
        if task == "zeroshot_classification":
            v = m.get("acc1")
        elif task == "zeroshot_retrieval":
            v = m.get(retrieval_metric)
        else:
            continue
        if v is not None:
            results[(ds, lang)] = v
    return results


def per_lang_mean(results: dict) -> dict:
    """Return per-language mean metric dict."""
    out = {}
    for sea_lang, benchmarks in SEA_LANGS.items():
        vals = []
        for ds, lang_key in benchmarks:
            v = results.get((ds, lang_key))
            if v is not None:
                vals.append(v)
            else:
                print(f"  MISSING: ({ds}, {lang_key}) for lang={sea_lang}")
        out[sea_lang] = (sum(vals) / len(vals)) if vals else None
    return out


def fmt(v):
    if v is None:
        return "---"
    return f"{v*100:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["r1", "r5", "r10"], default="r1",
                        help="Retrieval recall k to use (default: r1)")
    parser.add_argument("tags", nargs="*")
    args = parser.parse_args()

    k = args.metric[1:]
    retrieval_metric = f"image_retrieval_recall@{k}"
    tags = args.tags
    if not tags:
        # Auto-discover tags with result dirs
        tags = [d.name for d in sorted(RESULTS_DIR.iterdir())
                if d.is_dir() and d.name in ("tinyclip", "mobileclip2_s0")]
    if not tags:
        print("No result tags found. Specify: python3 compute_per_lang_mean.py tinyclip mobileclip2_s0")
        return

    print(f"[metric: {retrieval_metric}]")
    all_means = {}
    for tag in tags:
        try:
            results = load_results(tag, retrieval_metric)
            means = per_lang_mean(results)
            all_means[tag] = means
            print(f"\n=== {tag} ===")
            for lang in TABLE_ORDER:
                v = means.get(lang)
                print(f"  {DISPLAY_NAMES[lang]:<12} {fmt(v):>6}%")
            sea_vals = [v for v in means.values() if v is not None]
            avg = sum(sea_vals) / len(sea_vals) if sea_vals else None
            print(f"  {'Avg':<12} {fmt(avg):>6}%")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")

    # Print LaTeX row(s) for copy-paste
    if all_means:
        print("\n\n── LaTeX rows (paste into tab:per_language) ──")
        for tag, means in all_means.items():
            sea_vals = [v for v in means.values() if v is not None]
            avg = sum(sea_vals) / len(sea_vals) if sea_vals else None
            row_vals = " & ".join(
                fmt(means.get(lang)) for lang in ["th", "my", "ms", "id", "jv", "su", "vi"]
            )
            print(f"\\textbf{{{tag}}} & {row_vals} & {fmt(avg)} \\\\")


if __name__ == "__main__":
    main()
