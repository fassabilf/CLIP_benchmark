"""Compile WorldCuisines VQA results into a LaTeX table snippet.

Run after SLURM jobs complete to print the results for inserting into
sea_clip_report/tables/results_eval.tex.

Usage:
    python compile_worldcuisines.py [--results_dir runs/results]
"""
import argparse
import json
import sys
from pathlib import Path

# WorldCuisines uses granular register codes (not ISO-2)
SEA_LANGS = {"id_casual", "id_formal", "jv_krama", "jv_ngoko", "su_loma", "th", "tl"}

MODELS = [
    ("mc2_e0",       "e0 (init)"),
    ("mc2_e32",      "mc2-v1 e32"),
    ("mc2cc_e32",    "mc2-cc12m e32"),
    ("metaclip2_b16","B16 teacher"),
    ("mc2wit_e32",   "abl-WIT e32"),
    ("mc2cg_e32",    "abl-CG e32"),
]


def load_result(results_dir, tag, task, split="test_large"):
    path = results_dir / tag / f"worldcuisines_{task}_{split}_{tag}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def mean_acc(result, langs):
    if result is None:
        return None
    per_lang = result.get("per_lang", {})
    vals = [per_lang[l] for l in langs if l in per_lang]
    return sum(vals) / len(vals) if vals else None


def fmt(v):
    if v is None:
        return "—"
    return f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",
                    default="/lustrefs/disk/project/lt200394-thllmV/benchmark/CLIP_benchmark/runs/results")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    print("\n=== WorldCuisines VQA Results ===\n")
    print(f"{'Model':<20} {'Task':<8} {'Overall':>8} {'SEA':>8} {'non-SEA':>8} {'n':>6}")
    print("-" * 62)
    for tag, label in MODELS:
        for task in ["task1", "task2"]:
            r = load_result(results_dir, tag, task)
            if r is None:
                print(f"{label:<20} {task:<8} {'(missing)':>8}")
                continue
            per_lang = r.get("per_lang", {})
            sea_langs = [l for l in per_lang if l in SEA_LANGS]
            non_sea_langs = [l for l in per_lang if l not in SEA_LANGS]
            sea = mean_acc(r, sea_langs)
            non_sea = mean_acc(r, non_sea_langs)
            n = r.get("n", "?")
            print(f"{label:<20} {task:<8} {fmt(r.get('acc')):>8} {fmt(sea):>8} {fmt(non_sea):>8} {n:>6}")

    print("\n=== LaTeX snippet for results_eval.tex ===\n")
    print("% Add after CVQA rows:\n")
    for task in ["task1", "task2"]:
        # collect per-model values for SEA and non-SEA
        sea_vals = {}
        nonsea_vals = {}
        for tag, label in MODELS:
            r = load_result(results_dir, tag, task)
            if r is None:
                sea_vals[tag] = None
                nonsea_vals[tag] = None
            else:
                per_lang = r.get("per_lang", {})
                sea_langs = [l for l in per_lang if l in SEA_LANGS]
                non_sea_langs = [l for l in per_lang if l not in SEA_LANGS]
                sea_vals[tag] = mean_acc(r, sea_langs)
                nonsea_vals[tag] = mean_acc(r, non_sea_langs)

        col_tags = [t for t, _ in MODELS]

        def make_row(vals, bold=True):
            strs = [fmt(vals.get(t)) for t in col_tags]
            if bold:
                # bold the max
                nums = [(i, float(s)) for i, s in enumerate(strs) if s != "—"]
                if nums:
                    best_i = max(nums, key=lambda x: x[1])[0]
                    strs[best_i] = f"\\textbf{{{strs[best_i]}}}"
            return " & ".join(strs)

        non_sea_row = make_row(nonsea_vals)
        sea_row = make_row(sea_vals)
        label = f"WorldCuisines-{task[-1]}"
        print(f"{label:<22} & non-SEA & {non_sea_row} \\\\")
        print(f"\\searow {label:<15} & SEA     & {sea_row} \\\\")
    print()

    print("=== Per-language breakdown (SEA) ===\n")
    # collect all SEA lang codes actually present in results
    all_sea_langs = sorted(SEA_LANGS)
    tags = [t for t, _ in MODELS]
    labels = [l for _, l in MODELS]
    header = f"{'Lang':<20}" + "".join(f"{l[:12]:>12}" for l in labels)
    print(header)
    for task in ["task1", "task2"]:
        results_cache = {tag: load_result(results_dir, tag, task) for tag in tags}
        print(f"\n-- {task} --")
        for lang in all_sea_langs:
            row = f"{lang:<20}"
            for tag in tags:
                r = results_cache.get(tag)
                if r is None:
                    row += f"{'—':>12}"
                else:
                    v = r.get("per_lang", {}).get(lang)
                    row += f"{fmt(v):>12}"
            print(row)


if __name__ == "__main__":
    main()
