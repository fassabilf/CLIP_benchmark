#!/usr/bin/env python3
"""Summarize train-probe retrieval results from train_probe_curve_long_combined.csv.

Reads per-shard rows (tag × dataset × lang) and aggregates into a clean
comparison table showing mean t2i_r@1 / i2t_r@1 / mean_diag_cos per model.

Usage:
  python3 runs/analyze_train_probe.py                                  # default
  python3 runs/analyze_train_probe.py --csv /path/to/combined.csv
  python3 runs/analyze_train_probe.py --metric t2i_r@1                # default
  python3 runs/analyze_train_probe.py --metric i2t_r@1
  python3 runs/analyze_train_probe.py --metric mean_diag_cos
  python3 runs/analyze_train_probe.py --tags mc2_e32 selflearn_mammoth_v1_e32
  python3 runs/analyze_train_probe.py --latex
  python3 runs/analyze_train_probe.py --all-tags   # show every tag in the CSV
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CSV = (
    Path(__file__).parent / "results" / "train_probe" / "train_probe_curve_long_combined.csv"
)

# Langs excluded from means (too few samples / not main SEA langs)
SKIP_LANGS = {"ceb", "km", "fil", "dialect"}

# Datasets in the probe
DATASETS = ["bloom", "cgoe", "wit"]

# Default tag ordering (add new models here)
DEFAULT_TAGS = [
    # -- baselines --
    "mc2_e0",
    # -- CLIP-KD (mc2) epochs --
    "mc2_e8", "mc2_e16", "mc2_e24", "mc2_e32",
    # -- selflearn (no KD) epochs --
    "selflearn_mammoth_v1_e8", "selflearn_mammoth_v1_e16",
    "selflearn_mammoth_v1_e24", "selflearn_mammoth_v1_e32",
    # -- teacher (upper bound) --
    "metaclip2_b16",
    # -- ablation (placeholder, add when trained) --
    # "ablation_crd_e32", "ablation_icl_e32", "ablation_fd_e32",
]

# Display names for tags
TAG_LABELS = {
    "mc2_e0":                       "CLIP-KD init (e0)",
    "mc2_e8":                       "CLIP-KD (e8)",
    "mc2_e16":                      "CLIP-KD (e16)",
    "mc2_e24":                      "CLIP-KD (e24)",
    "mc2_e32":                      "CLIP-KD (e32)",
    "selflearn_mammoth_v1_e8":      "Selflearn (e8)",
    "selflearn_mammoth_v1_e16":     "Selflearn (e16)",
    "selflearn_mammoth_v1_e24":     "Selflearn (e24)",
    "selflearn_mammoth_v1_e32":     "Selflearn (e32)",
    "metaclip2_b16":                "Teacher (MC2-B16)",
    "metaclip2":                    "Teacher (MC2-H14)",
    "ablation_crd_e8":              "Abl-CRD (e8)",
    "ablation_crd_e16":             "Abl-CRD (e16)",
    "ablation_crd_e24":             "Abl-CRD (e24)",
    "ablation_crd_e32":             "Abl-CRD (e32)",
    "ablation_icl_e8":              "Abl-ICL (e8)",
    "ablation_icl_e16":             "Abl-ICL (e16)",
    "ablation_icl_e24":             "Abl-ICL (e24)",
    "ablation_icl_e32":             "Abl-ICL (e32)",
    "ablation_fd_e8":               "Abl-FD (e8)",
    "ablation_fd_e16":              "Abl-FD (e16)",
    "ablation_fd_e24":              "Abl-FD (e24)",
    "ablation_fd_e32":              "Abl-FD (e32)",
}


# ── Loading ────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict], metric: str, tags: list[str]) -> dict:
    """Return {tag: {dataset: mean_metric, 'all': mean_metric}}."""
    # accumulate per (tag, dataset)
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tag = r["tag"]
        if tags and tag not in tags:
            continue
        if r["lang"] in SKIP_LANGS:
            continue
        val = r.get(metric, "")
        if val == "":
            continue
        try:
            acc[tag][r["dataset"]].append(float(val))
        except ValueError:
            continue

    out = {}
    for tag in (tags if tags else sorted(acc)):
        if tag not in acc:
            continue
        ds_means = {}
        all_vals = []
        for ds in DATASETS:
            vals = acc[tag].get(ds, [])
            if vals:
                ds_means[ds] = sum(vals) / len(vals)
                all_vals.extend(vals)
        ds_means["all"] = sum(all_vals) / len(all_vals) if all_vals else None
        out[tag] = ds_means
    return out


# ── Printing ───────────────────────────────────────────────────────────────────

def fmt(v, decimals=3):
    return f"{v:.{decimals}f}" if v is not None else "—"


def print_table(data: dict, tags: list[str], metric: str):
    col_w = 10
    header_cols = DATASETS + ["all"]
    header_labels = ["Bloom", "CG-OE", "WIT", "Mean"]

    print(f"\n{'':=<72}")
    print(f"  Train-Probe — {metric}  (mean over main SEA langs, excl. ceb/km/fil)")
    print(f"{'':=<72}")
    print(f"  {'Model':<36}", end="")
    for h in header_labels:
        print(f"  {h:>{col_w}}", end="")
    print()
    print(f"  {'-'*36}", end="")
    for _ in header_labels:
        print(f"  {'-'*col_w}", end="")
    print()

    for tag in tags:
        if tag not in data:
            continue
        label = TAG_LABELS.get(tag, tag)
        row = data[tag]
        print(f"  {label:<36}", end="")
        for ds in header_cols:
            print(f"  {fmt(row.get(ds)):>{col_w}}", end="")
        print()

    print()


def print_latex(data: dict, tags: list[str], metric: str):
    print(f"\n% LaTeX table — {metric}")
    print(r"\begin{tabular}{l cccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Bloom} & \textbf{CG-OE} & \textbf{WIT} & \textbf{Mean} \\")
    print(r"\midrule")
    for tag in tags:
        if tag not in data:
            continue
        label = TAG_LABELS.get(tag, tag).replace("_", r"\_")
        row = data[tag]
        vals = " & ".join(fmt(row.get(ds)) for ds in DATASETS + ["all"])
        print(f"{label} & {vals} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_epoch_curve(data: dict, tags: list[str], metric: str):
    """Print selflearn vs mc2 side-by-side per epoch for quick comparison."""
    epochs = [8, 16, 24, 32]

    sl_tags  = [f"selflearn_mammoth_v1_e{e}" for e in epochs]
    mc2_tags = [f"mc2_e{e}" for e in epochs]

    if not any(t in data for t in sl_tags + mc2_tags):
        return

    print(f"\n{'':=<72}")
    print(f"  Epoch Curve — Selflearn vs CLIP-KD   ({metric}, Mean over all datasets)")
    print(f"{'':=<72}")
    print(f"  {'Epoch':<8}  {'Selflearn':>12}  {'CLIP-KD':>12}  {'Delta (KD-SL)':>15}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*15}")
    for e, sl_tag, mc2_tag in zip(epochs, sl_tags, mc2_tags):
        sl_val  = data.get(sl_tag,  {}).get("all")
        mc2_val = data.get(mc2_tag, {}).get("all")
        delta   = (mc2_val - sl_val) if (sl_val is not None and mc2_val is not None) else None
        print(f"  {f'e{e}':<8}  {fmt(sl_val):>12}  {fmt(mc2_val):>12}  {fmt(delta):>15}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="Path to train_probe_curve_long_combined.csv "
                         f"(default: {DEFAULT_CSV})")
    ap.add_argument("--metric", default="t2i_r@1",
                    choices=["t2i_r@1", "i2t_r@1", "mean_diag_cos",
                             "t2i_r@5", "t2i_r@10", "i2t_r@5", "i2t_r@10"],
                    help="Metric to aggregate (default: t2i_r@1)")
    ap.add_argument("--tags", nargs="+", default=None,
                    help="Tags to include (default: preset list in DEFAULT_TAGS)")
    ap.add_argument("--all-tags", action="store_true",
                    help="Show all tags found in the CSV")
    ap.add_argument("--latex", action="store_true",
                    help="Also print a LaTeX table")
    ap.add_argument("--no-curve", action="store_true",
                    help="Skip the selflearn vs CLIP-KD epoch curve")
    args = ap.parse_args()

    csv_path = args.csv or DEFAULT_CSV
    if not Path(csv_path).exists():
        sys.exit(f"CSV not found: {csv_path}\nPass --csv /path/to/combined.csv")

    rows = load_csv(csv_path)

    if args.all_tags:
        all_tags = sorted({r["tag"] for r in rows})
        tags = all_tags
    elif args.tags:
        tags = args.tags
    else:
        tags = DEFAULT_TAGS

    data = aggregate(rows, args.metric, tags)

    if not data:
        sys.exit("No matching rows found. Check --tags or --csv path.")

    print_table(data, tags, args.metric)

    if not args.no_curve:
        print_epoch_curve(data, tags, args.metric)

    if args.latex:
        print_latex(data, tags, args.metric)


if __name__ == "__main__":
    main()
