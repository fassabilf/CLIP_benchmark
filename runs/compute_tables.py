#!/usr/bin/env python3
"""Compute Table 2 (per-language) and Table 3 (per-task) values for SEA-CLIP paper.

Sources:
  - Retrieval R@k      : runs/results/<tag>/*.json  (image_retrieval_recall@k)
  - Babel-IN acc@k     : pred JSONL files (topk field), from HF or local preds/ dirs
  - CVQA               : runs/results/<tag>/cvqa_<tag>.json (LOCAL_subset)

Usage:
  python3 runs/compute_tables.py              # prints all tables
  python3 runs/compute_tables.py --latex      # also prints LaTeX rows
  python3 runs/compute_tables.py --cvqa sea7  # choose cvqa variant: all39 | sea7 | sea4

CVQA variants:
  all39  - mean over all 39 non-English subsets (current default)
  sea7   - mean over 7 SEA language subsets in CVQA
  sea4   - mean over 4 SEA langs also present in retrieval benchmarks
"""
import argparse
import ast
import json
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"

# Combined train-probe curve (per shard: tag × dataset × lang) used for the
# CG/WIT/Bloom R@1 columns of the dataset-ablation table (tab:training).
TRAIN_PROBE_CSV = RESULTS_DIR / "train_probe" / "train_probe_curve_long_combined.csv"

MODELS = [
    ("metaclip2_b16",  "Teacher",                   True),   # (tag, label, is_teacher)
    ("mc2_e0",         "CLIP-KD",                   False),
    ("tinyclip",       "TinyCLIP",                  False),
    ("mobileclip2_s0", "MobileCLIP2",               False),
    ("mc2cc_e32",      "SEA-CLIP-Tiny (CC12M)",     False),
    ("mv1_e32",        "SEA-CLIP-Tiny (SEA-only)",  False),
    ("mc2v3_e32",      "SEA-CLIP-Tiny (CC12M+SEA)", False),
]

# Languages for Table 2 (per-language) — in display order
TABLE2_LANGS = ["th", "my", "ms", "id", "jv", "su", "vi"]
LANG_DISPLAY = {
    "th": "Thai", "my": "Myanmar", "ms": "Malay",
    "id": "Indonesia", "jv": "Javanese", "su": "Sundanese", "vi": "Vietnamese",
}

# Benchmarks each language appears in (for per-language mean)
LANG_BENCHMARKS = {
    "th": [("babel_imagenet","th"),  ("crossmodal3600","th"),  ("flickr30k-200","tha_Thai"), ("xtd200","tha_Thai")],
    "my": [("babel_imagenet","my"),  ("flickr30k-200","mya_Mymr"), ("xtd200","mya_Mymr")],
    "ms": [("babel_imagenet","ms"),  ("flickr30k-200","zsm_Latn"), ("xtd200","zsm_Latn")],
    "id": [("babel_imagenet","id"),  ("crossmodal3600","id"),  ("flickr30k-200","ind_Latn"), ("xtd200","ind_Latn")],
    "jv": [("babel_imagenet","jv"),  ("flickr30k-200","jav_Latn"), ("xtd200","jav_Latn")],
    "su": [("babel_imagenet","su"),  ("flickr30k-200","sun_Latn"), ("xtd200","sun_Latn")],
    "vi": [("babel_imagenet","vi"),  ("crossmodal3600","vi"),  ("flickr30k-200","vie_Latn"), ("xtd200","vie_Latn")],
}

# CVQA SEA language sets
CVQA_SEA7 = {
    ("Indonesian","Indonesia"), ("Javanese","Indonesia"), ("Malay","Malaysia"),
    ("Sundanese","Indonesia"),  ("Minangkabau","Indonesia"),
    ("Filipino","Philippines"), ("Chinese","Singapore"),
}
CVQA_SEA4 = {
    ("Indonesian","Indonesia"), ("Javanese","Indonesia"),
    ("Malay","Malaysia"),       ("Sundanese","Indonesia"),
}

# Retrieval benchmarks for Table 3 (per-task) — SEA language keys per benchmark
TABLE3_BENCHMARKS = {
    "crossmodal3600": ["id", "th", "vi"],
    "flickr30k-200":  ["ind_Latn", "jav_Latn", "zsm_Latn", "mya_Mymr", "sun_Latn", "tha_Thai", "vie_Latn"],
    "xtd200":         ["ind_Latn", "jav_Latn", "zsm_Latn", "mya_Mymr", "sun_Latn", "tha_Thai", "vie_Latn"],
}
BENCH_DISPLAY = {
    "crossmodal3600": "XM3600", "flickr30k-200": "Flickr30k", "xtd200": "XTD-200",
}

# ─── Babel-IN pred file loading ───────────────────────────────────────────────

def _compute_acc_from_pred(pred_file: Path):
    """Compute acc@1/5/10 from a babel_imagenet pred JSONL file."""
    a1 = a5 = a10 = n = 0
    with open(pred_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            topk = d.get("topk", [])
            label = d.get("true")
            if topk and label is not None:
                a1  += int(label in topk[:1])
                a5  += int(label in topk[:5])
                a10 += int(label in topk[:10])
                n   += 1
    if n == 0:
        return None, None, None
    return 100 * a1 / n, 100 * a5 / n, 100 * a10 / n


def load_babel_acc(tag: str, langs=("id","jv","ms","my","su","th","vi","en")):
    """Return {lang: (acc1, acc5, acc10)} from pred files (local preds/ or HF cache)."""
    # Try local preds/ dir first
    preds_dir = RESULTS_DIR / tag / "preds"
    out = {}
    for lang in langs:
        pred_file = preds_dir / f"babel_imagenet_{lang}_{tag}_pred.jsonl"
        if pred_file.exists():
            a1, a5, a10 = _compute_acc_from_pred(pred_file)
            if a1 is not None:
                out[lang] = (a1, a5, a10)
                continue
        # Fallback: read acc1/acc5 from JSON result file (acc10 not stored for
        # classification, so it stays None unless per-sample preds are present).
        json_file = RESULTS_DIR / tag / f"babel_imagenet_{lang}_{tag}.json"
        if json_file.exists():
            m = json.loads(json_file.read_text()).get("metrics", {})
            acc1, acc5 = m.get("acc1"), m.get("acc5")
            if acc1 is not None:
                out[lang] = (acc1 * 100, acc5 * 100 if acc5 is not None else None, None)
    return out


# ─── Retrieval R@k loading ────────────────────────────────────────────────────

def load_retrieval(tag: str, k: int, direction: str = "t2i"):
    """Return {(dataset, lang_key): R@k} from JSON result files.

    direction: 't2i' (image_retrieval_recall) | 'i2t' (text_retrieval_recall) | 'mean'
    """
    metrics = {
        "t2i":  [f"image_retrieval_recall@{k}"],
        "i2t":  [f"text_retrieval_recall@{k}"],
        "mean": [f"image_retrieval_recall@{k}", f"text_retrieval_recall@{k}"],
    }[direction]
    out = {}
    tag_dir = RESULTS_DIR / tag
    if not tag_dir.exists():
        return out
    for f in sorted(tag_dir.glob("*.json")):
        if "preds" in f.parts or "cvqa" in f.name:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("task") != "zeroshot_retrieval":
            continue
        ds = d.get("dataset", "")
        lang = d.get("language", "")
        vals = [v for m in metrics for v in [d.get("metrics", {}).get(m)] if v is not None]
        if vals:
            out[(ds, lang)] = sum(vals) / len(vals) * 100
    return out


# ─── CVQA loading ─────────────────────────────────────────────────────────────

def _parse_cvqa_key(k):
    return ast.literal_eval("".join(ast.literal_eval(k)))


def load_cvqa(tag: str, variant: str = "all39"):
    """Return CVQA score for the given variant.

    variant: 'all39' | 'sea7' | 'sea4'
    """
    f = RESULTS_DIR / tag / f"cvqa_{tag}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if variant == "all39":
        return d["LOCAL"] * 100
    sea_set = CVQA_SEA7 if variant == "sea7" else CVQA_SEA4
    vals = [v for k, v in d["LOCAL_subset"].items()
            if _parse_cvqa_key(k) in sea_set]
    return sum(vals) / len(vals) * 100 if vals else None


# ─── Table 2: per-language ────────────────────────────────────────────────────

def compute_table2(k: int, direction: str = "t2i"):
    """Return {tag: {lang: mean_metric}} for Table 2."""
    results = {}
    for tag, label, _ in MODELS:
        babel = load_babel_acc(tag)
        ret   = load_retrieval(tag, k, direction)
        lang_means = {}
        for lang in TABLE2_LANGS:
            vals = []
            for ds, lang_key in LANG_BENCHMARKS[lang]:
                if ds == "babel_imagenet":
                    acc = babel.get(lang)
                    if acc and acc[0] is not None:
                        # pick the right k-index
                        v = acc[{1:0, 5:1, 10:2}[k]]
                        if v is not None:
                            vals.append(v)
                else:
                    v = ret.get((ds, lang_key))
                    if v is not None:
                        vals.append(v)
            lang_means[lang] = sum(vals) / len(vals) if vals else None
        sea_vals = [v for v in lang_means.values() if v is not None]
        lang_means["avg"] = sum(sea_vals) / len(sea_vals) if sea_vals else None
        results[tag] = lang_means
    return results


# ─── Table 3: per-task ────────────────────────────────────────────────────────

def compute_table3(cvqa_variant: str = "all39", direction: str = "t2i"):
    """Return per-task scores for all models and k=1/5/10."""
    results = {}
    for tag, label, _ in MODELS:
        babel = load_babel_acc(tag)
        cvqa  = load_cvqa(tag, cvqa_variant)

        per_k = {}
        for k in [1, 5, 10]:
            ki = {1: 0, 5: 1, 10: 2}[k]
            ret = load_retrieval(tag, k, direction)

            # Babel-IN SEA avg
            babel_sea = [babel[l][ki] for l in TABLE2_LANGS
                         if l in babel and babel[l][ki] is not None]
            babel_avg = sum(babel_sea) / len(babel_sea) if babel_sea else None

            # Per retrieval benchmark SEA avg
            bench_avgs = {}
            for ds, langs in TABLE3_BENCHMARKS.items():
                vals = [v for l in langs for v in [ret.get((ds, l))] if v is not None]
                bench_avgs[ds] = sum(vals) / len(vals) if vals else None

            # Retrieval Avg = mean(babel_avg, xm3600, flickr, xtd)
            ret_tasks = [v for v in [babel_avg] + [bench_avgs[ds] for ds in TABLE3_BENCHMARKS]
                         if v is not None]
            ret_avg = sum(ret_tasks) / len(ret_tasks) if ret_tasks else None

            # Total Avg = mean(babel, xm3600, flickr, xtd, cvqa) — equal weight per task
            all_tasks = [v for v in [babel_avg] + [bench_avgs[ds] for ds in TABLE3_BENCHMARKS]
                         if v is not None]
            if cvqa is not None:
                all_tasks.append(cvqa)
            total_avg = sum(all_tasks) / len(all_tasks) if all_tasks else None

            per_k[k] = {
                "babel":     babel_avg,
                "xm3600":    bench_avgs.get("crossmodal3600"),
                "flickr":    bench_avgs.get("flickr30k-200"),
                "xtd":       bench_avgs.get("xtd200"),
                "ret_avg":   ret_avg,
                "cvqa":      cvqa,
                "total_avg": total_avg,
            }
        results[tag] = per_k
    return results


# ─── Printing ─────────────────────────────────────────────────────────────────

def fmt(v, decimals=1):
    return f"{v:.{decimals}f}" if v is not None else "—"


def print_table2(k: int, direction: str = "t2i"):
    data = compute_table2(k, direction)
    langs = TABLE2_LANGS + ["avg"]
    header_langs = [LANG_DISPLAY.get(l, l.capitalize()) for l in TABLE2_LANGS] + ["Avg"]

    print(f"\n{'':=<90}")
    print(f"  TABLE 2 — Per-language mean metric  (k={k})")
    print(f"{'':=<90}")
    w = 10
    print(f"{'Model':<32}", end="")
    for h in header_langs:
        print(f"  {h:>{w}}", end="")
    print()
    print("-" * (32 + (w + 2) * len(langs)))
    for tag, label, is_teacher in MODELS:
        row = data.get(tag, {})
        marker = " *" if is_teacher else "  "
        print(f"{marker}{label:<30}", end="")
        for l in langs:
            print(f"  {fmt(row.get(l)):>{w}}", end="")
        print()


def print_table3(cvqa_variant: str = "all39", direction: str = "t2i"):
    data = compute_table3(cvqa_variant, direction)
    variant_label = {"all39": "39-lang global", "sea7": "SEA-7", "sea4": "SEA-4"}[cvqa_variant]
    dir_label = {"t2i": "T2I", "i2t": "I2T", "mean": "mean(I2T,T2I)"}[direction]

    print(f"\n{'':=<110}")
    print(f"  TABLE 3 — Per-task SEA performance  (CVQA = {variant_label}, retrieval = {dir_label})")
    print(f"{'':=<110}")

    cols = ["babel", "xm3600", "flickr", "xtd", "ret_avg", "cvqa", "total_avg"]
    headers = ["Babel-IN", "XM3600", "Flickr30k", "XTD-200", "Ret Avg", "CVQA", "Total Avg"]
    ks = [1, 5, 10]

    print(f"{'Model':<32}", end="")
    for h in headers:
        w = 18 if h in ("Ret Avg", "Total Avg") else 16
        print(f"  {h:^{w}}", end="")
    print()

    print(f"{'':32}", end="")
    for h in headers:
        w = 18 if h in ("Ret Avg", "Total Avg") else 16
        sub = "  ".join(f"@{k}" for k in ks) if h != "CVQA" else " acc@1"
        print(f"  {sub:^{w}}", end="")
    print()
    print("-" * 160)

    for tag, label, is_teacher in MODELS:
        row = data.get(tag, {})
        marker = " *" if is_teacher else "  "
        print(f"{marker}{label:<30}", end="")
        for col in cols:
            if col == "cvqa":
                v = row.get(1, {}).get("cvqa")
                print(f"  {fmt(v):^16}", end="")
            else:
                vals = "  ".join(fmt(row.get(k, {}).get(col)) for k in ks)
                w = 18 if col in ("ret_avg", "total_avg") else 16
                print(f"  {vals:^{w}}", end="")
        print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvqa", choices=["all39", "sea7", "sea4"], default="all39",
                        help="CVQA averaging variant (default: all39)")
    parser.add_argument("--direction", choices=["t2i", "i2t", "mean"], default="t2i",
                        help="Retrieval direction: t2i | i2t | mean (default: t2i)")
    parser.add_argument("--latex", action="store_true",
                        help="Also print raw numbers in a copy-paste format")
    parser.add_argument("--gen-pdfs", action="store_true",
                        help="Generate tables_t2i.pdf, tables_i2t.pdf, tables_mean.pdf, tables_training.pdf")
    parser.add_argument("--training", action="store_true",
                        help="Print the filled dataset-ablation table (tab:training) LaTeX")
    parser.add_argument("--selflearn-tables", action="store_true",
                        help="Generate a SEPARATE PDF of per-lang + per-task tables that include the selflearn (w/o KD) row")
    parser.add_argument("--mammoth-tables", action="store_true",
                        help="Generate a SEPARATE PDF of per-lang + per-task tables that include the clipkd_mammoth_v1 (KD + ML + Mammoth-VL-SEA) row")
    parser.add_argument("--ckdonly-tables", action="store_true",
                        help="Generate SEPARATE PDFs (t2i, i2t, mean) of per-lang + per-task tables that include the ckdonly_v1 (clipkd loss only) row")
    args = parser.parse_args()

    if args.training:
        print(generate_training_tex(args.direction, args.cvqa))
        return

    if args.selflearn_tables:
        print(f"Generating selflearn tables (direction={args.direction}, CVQA={args.cvqa}) ...")
        gen_selflearn_pdf(args.direction, args.cvqa)
        return

    if args.mammoth_tables:
        print(f"Generating mammoth tables (direction={args.direction}, CVQA={args.cvqa}) ...")
        gen_mammoth_pdf(args.direction, args.cvqa)
        return

    if args.ckdonly_tables:
        print(f"Generating ckdonly tables (t2i, i2t, mean; CVQA={args.cvqa}) ...")
        for direction in ["t2i", "i2t", "mean"]:
            gen_ckdonly_pdf(direction, args.cvqa)
        return

    if args.gen_pdfs:
        print(f"Generating PDFs (CVQA={args.cvqa}, direction={args.direction}) ...")
        gen_all_pdfs(args.cvqa)
        gen_training_pdf(args.direction, args.cvqa)
        return

    # Table 2 for each k
    for k in [1, 5, 10]:
        print_table2(k, args.direction)

    # Table 3 for the chosen CVQA variant (and all three if --latex)
    if args.latex:
        for variant in ["all39", "sea7", "sea4"]:
            print_table3(variant, args.direction)
    else:
        print_table3(args.cvqa, args.direction)

    if args.latex:
        print("\n\n" + "=" * 60)
        print("  RAW NUMBERS (Table 3, all CVQA variants)")
        print("=" * 60)
        for variant in ["all39", "sea7", "sea4"]:
            data = compute_table3(variant)
            vl = {"all39": "39-lang", "sea7": "SEA-7", "sea4": "SEA-4"}[variant]
            print(f"\n--- CVQA={vl} ---")
            print(f"{'Model':<32} {'CVQA':>6}  {'TA@1':>6} {'TA@5':>6} {'TA@10':>6}")
            for tag, label, _ in MODELS:
                row = data.get(tag, {})
                cvqa = fmt(row.get(1, {}).get("cvqa"))
                ta1  = fmt(row.get(1, {}).get("total_avg"))
                ta5  = fmt(row.get(5, {}).get("total_avg"))
                ta10 = fmt(row.get(10, {}).get("total_avg"))
                print(f"{label:<32} {cvqa:>6}  {ta1:>6} {ta5:>6} {ta10:>6}")


# ─── LaTeX generation ─────────────────────────────────────────────────────────

def _tex_val(v):
    return f"{v:.1f}" if v is not None else r"\text{---}"

def _tex_row2(label, data, langs, is_teacher, best_per_col):
    row = data.get(label, {})
    vals = [row.get(l) for l in langs + ["avg"]]
    triples = []
    for ki, k in enumerate([1, 5, 10]):
        col_vals = [row.get(l, {k: None}[k] if not isinstance(row.get(l), tuple) else row.get(l)[ki]) for l in langs + ["avg"]]
        triples.append(col_vals)
    # data is {lang: val} for a single k — need to call per k separately
    return None  # placeholder, handled inline


def generate_tex(direction: str, cvqa_variant: str = "sea7") -> str:
    dir_label = {"t2i": "T2I (text$\\to$image)", "i2t": "I2T (image$\\to$text)", "mean": "mean(I2T, T2I)"}[direction]
    cvqa_label = {"all39": "39-lang global", "sea7": "SEA-7", "sea4": "SEA-4"}[cvqa_variant]

    t2_k = {k: compute_table2(k, direction) for k in [1, 5, 10]}
    t3    = compute_table3(cvqa_variant, direction)

    LANGS_ORDER = ["th", "my", "ms", "id", "jv", "su", "vi"]
    LANG_NAMES  = ["Thai", "Myanmar", "Malay", "Indonesia", "Javanese", "Sundanese", "Vietnamese"]

    # ── find bests per column (Table 2, students only) ──
    student_tags = [tag for tag, _, is_t in MODELS if not is_t]
    t2_best = {}  # (k, lang_or_avg) -> best val
    for k in [1, 5, 10]:
        for col in LANGS_ORDER + ["avg"]:
            vals = [t2_k[k].get(tag, {}).get(col) for tag in student_tags]
            vals = [v for v in vals if v is not None]
            t2_best[(k, col)] = max(vals) if vals else None

    def t2_cell(k, col, v):
        if v is None:
            return r"\text{---}"
        s = f"{v:.1f}"
        if t2_best.get((k, col)) is not None and abs(v - t2_best[(k, col)]) < 0.05:
            return r"\textbf{" + s + "}"
        return s

    # ── find bests per column (Table 3, students only) ──
    t3_cols = ["babel", "xm3600", "flickr", "xtd", "ret_avg", "cvqa", "total_avg"]
    t3_best = {}  # (col, k_or_None) -> best val
    for col in t3_cols:
        if col == "cvqa":
            vals = [t3.get(tag, {}).get(1, {}).get("cvqa") for tag in student_tags]
            vals = [v for v in vals if v is not None]
            t3_best[(col, None)] = max(vals) if vals else None
        else:
            for k in [1, 5, 10]:
                vals = [t3.get(tag, {}).get(k, {}).get(col) for tag in student_tags]
                vals = [v for v in vals if v is not None]
                t3_best[(col, k)] = max(vals) if vals else None

    def t3_cell(col, k, v, is_teacher):
        if v is None:
            return r"\text{---}"
        s = f"{v:.1f}"
        if is_teacher:
            return r"\textit{" + s + "}"
        key = (col, None) if col == "cvqa" else (col, k)
        if t3_best.get(key) is not None and abs(v - t3_best[key]) < 0.05:
            return r"\textbf{" + s + "}"
        return s

    lines = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=0.5in]{geometry}")
    lines.append(r"\usepackage{booktabs,graphicx,multirow,colortbl,amsmath}")
    lines.append(r"\usepackage[table]{xcolor}")
    lines.append(r"\definecolor{bestrow}{RGB}{220,230,255}")
    lines.append(r"\begin{document}")
    lines.append(r"\small")

    # ── TABLE 2 ──
    lines.append(r"\begin{table}[h!]\centering\setlength{\tabcolsep}{3pt}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{l c " + " ".join(["ccc"] * 8) + "}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{\#Params}}")
    for ln in LANG_NAMES + ["Avg"]:
        lines.append(f"  & \\multicolumn{{3}}{{c}}{{\\textbf{{{ln}}}}}")
    lines[-1] += r" \\"
    cr = "".join(f"\\cmidrule(lr){{{3+i*3}-{5+i*3}}}" for i in range(8))
    lines.append(cr)
    lines.append("  & & " + " & ".join(["@1 & @5 & @10"] * 8) + r" \\")
    lines.append(r"\midrule")

    for tag, label, is_teacher in MODELS:
        if is_teacher:
            prefix = r"\textit{" ; suffix = "}"
            row_start = label
        else:
            prefix = "" ; suffix = ""
            row_start = label
        if tag == "mc2v3_e32":
            lines.append(r"\rowcolor{bestrow}")
            row_start = r"\textbf{" + label + "}"

        params = {
            "metaclip2_b16": "$85.80+499.50$",
            "mc2_e0":        "$5.62+40.49$",
            "tinyclip":      "$8.15+15.04$",
            "mobileclip2_s0":"$11.41+63.43$",
            "mc2cc_e32":     "$5.62+40.49$",
            "mv1_e32":       "$5.62+40.49$",
            "mc2v3_e32":     "$5.62+40.49$",
            "selflearn_mammoth_v1_e32": "$5.62+40.49$",
        }.get(tag, "$5.62+40.49$")

        cells = []
        for col in LANGS_ORDER + ["avg"]:
            triple = []
            for k in [1, 5, 10]:
                v = t2_k[k].get(tag, {}).get(col)
                triple.append(t2_cell(k, col, v) if not is_teacher else
                              (r"\textit{" + f"{v:.1f}" + "}" if v is not None else r"\text{---}"))
            cells.append(" & ".join(triple))

        line = f"{row_start} & {params}\n  & " + "\n  & ".join(cells) + r" \\"
        lines.append(line)
        if is_teacher:
            lines.append(r"\midrule")
        if tag == "mobileclip2_s0":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule\end{tabular}" + "\n}")
    lines.append(r"\caption{Per-language performance (mean across Babel-IN acc@$k$, XM3600, Flickr30k-200, XTD-200 " + dir_label + r" R@$k$), reported as \%. \textbf{Bold} = best among student models per column.}")
    lines.append(r"\label{tab:per_language_rk_" + direction + "}")
    lines.append(r"\end{table}")
    lines.append(r"\bigskip")

    # ── TABLE 3 ──
    lines.append(r"\begin{table}[h!]\centering\setlength{\tabcolsep}{3pt}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{l c ccc ccc ccc ccc ccc c ccc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{\#Params}}")
    for h in ["Babel-IN", "XM3600", "Flickr30k", "XTD-200", "Retrieval Avg"]:
        lines.append(f"  & \\multicolumn{{3}}{{c}}{{\\textbf{{{h}}}}}")
    lines.append(r"  & \multirow{2}{*}{\textbf{CVQA}}")
    lines.append(r"  & \multicolumn{3}{c}{\textbf{Total Avg}} \\")
    lines.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}\cmidrule(lr){15-17}\cmidrule(lr){19-21}")
    lines.append(r"  & & @1 & @5 & @10 & @1 & @5 & @10 & @1 & @5 & @10 & @1 & @5 & @10 & @1 & @5 & @10 & & @1 & @5 & @10 \\")
    lines.append(r"\midrule")

    col_order = ["babel", "xm3600", "flickr", "xtd", "ret_avg"]
    for tag, label, is_teacher in MODELS:
        if tag == "mc2v3_e32":
            lines.append(r"\rowcolor{bestrow}")
            label = r"\textbf{" + label + "}"

        params = {
            "metaclip2_b16": "$85.80+499.50$",
            "mc2_e0":        "$5.62+40.49$",
            "tinyclip":      "$8.15+15.04$",
            "mobileclip2_s0":"$11.41+63.43$",
            "mc2cc_e32":     "$5.62+40.49$",
            "mv1_e32":       "$5.62+40.49$",
            "mc2v3_e32":     "$5.62+40.49$",
            "selflearn_mammoth_v1_e32": "$5.62+40.49$",
        }.get(tag, "$5.62+40.49$")

        row_data = t3.get(tag, {})
        cells = []
        for col in col_order:
            for k in [1, 5, 10]:
                v = row_data.get(k, {}).get(col)
                cells.append(t3_cell(col, k, v, is_teacher))
        # CVQA (single value)
        v_cvqa = row_data.get(1, {}).get("cvqa")
        cells.append(t3_cell("cvqa", None, v_cvqa, is_teacher))
        # Total Avg
        for k in [1, 5, 10]:
            v = row_data.get(k, {}).get("total_avg")
            cells.append(t3_cell("total_avg", k, v, is_teacher))

        # insert & before CVQA (index 15 = after 5 cols × 3 k values)
        cell_str = " & ".join(cells[:15]) + " & " + cells[15] + " & " + " & ".join(cells[16:])
        line = f"{label} & {params}\n  & {cell_str} \\\\"
        lines.append(line)
        if is_teacher:
            lines.append(r"\midrule")
        if tag == "mobileclip2_s0":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule\end{tabular}" + "\n}")
    lines.append(r"\caption{Per-task SEA performance, reported as \%.")
    lines.append(r"  Babel-IN = acc@$k$ (SEA-lang avg); XM3600, Flickr30k, XTD-200 = " + dir_label + r" R@$k$;")
    lines.append(r"  CVQA = acc@1 (LOCAL, " + cvqa_label + r" subset).")
    lines.append(r"  \textbf{Retrieval Avg} = mean(Babel-IN, XM3600, Flickr30k, XTD-200).")
    lines.append(r"  \textbf{Total Avg} = equal-weight mean over all 5 tasks.")
    lines.append(r"  \textbf{Bold} = best student per column.}")
    lines.append(r"\label{tab:per_task_rk_" + direction + "}")
    lines.append(r"\end{table}")
    lines.append(r"\end{document}")

    return "\n".join(lines)


# ─── Table 4: dataset ablation (tab:training) ────────────────────────────────
#
# Columns:
#   CG / WIT / Bloom R@1 = train-probe i2t_r@1 × 100, mean over ALL langs, e32.
#   ImageNet             = imagenet1k_<tag>.json acc1 × 100.
#   R@1-Avg              = compute_table3 total_avg@1 (direction=mean, cvqa=sea7).
#
# (run_label, train_data_desc, probe_tag, eval_tag, style)
#   style: "choice"  → c1–c3, eligible for bold-best comparison
#          "searow"  → single-source rows (\rowcolor{searow})
#          "noKD"    → SEA-CLIP-Tiny w/o KD losses (selflearn)
#          "teacher" → reference row, values in \textit
TRAINING_ROWS = [
    ("SEA-CLIP-Tiny-c1 (SEA blend)",   r"CG-OE-filt + WIT-hf-base + Bloom \textit{(1.21M)}",    "mc2_e32",                  "mc2_e32",                  "choice"),
    ("SEA-CLIP-Tiny-c2 (CC12M)",       r"CC12M (10.97M)",                                       "mc2cc_e32",                "mc2cc_e32",                "choice"),
    ("SEA-CLIP-Tiny-c3 (CC12M + SEA)", r"CC12M + CG-OE-filt + WIT-hf + Bloom \textit{(12.18M)}", "mc2v3_e32",                "mc2v3_e32",                "choice"),
    ("SEA-CLIP-Tiny-WIT",              r"WIT-hf-base (487K)",                                   "mc2wit_e32",               "mc2wit_e32",               "searow"),
    ("SEA-CLIP-Tiny-Bloom",            r"Bloom-only (21K)",                                     "mc2bloom_e32",             "mc2bloom_e32",             "searow"),
    ("SEA-CLIP-Tiny-CG",               r"CG-only (703K)",                                       "mc2cg_e32",                "mc2cg_e32",                "searow"),
    ("SEA-CLIP-Tiny w/o KD losses",    r"CC12M + CG-OE-filt + WIT-hf + ML \textit{(12.33M)}", "selflearn_mammoth_v1_e32", "selflearn_mammoth_v1_e32", "noKD"),
    ("SEA-CLIP-Tiny + ML + Mammoth-VL-SEA", r"CC12M + CG-OE-filt + WIT-hf + ML + Mammoth-VL-SEA \textit{(12.87M)}", "clipkd_mammoth_v1_e32", "clipkd_mammoth_v1_e32", "mammothKD"),
    ("SEA-CLIP-Tiny (CKD only)",       r"CC12M + CG-OE-filt + WIT-hf + Bloom + Mammoth-VL-SEA \textit{(12.87M)}", "ckdonly_v1_e32", "ckdonly_v1_e32", "ckdKD"),
    ("Teacher",                        r"---",                                                  "metaclip2_b16",            "metaclip2_b16",            "teacher"),
]

# train-probe dataset key per ablation column
_PROBE_DS = {"cg": "cgoe", "wit": "wit", "bloom": "bloom"}


def load_train_probe(tag: str, dataset: str, metric: str = "i2t_r@1",
                     csv_path: Path = None):
    """Mean of `metric` over all languages for (tag, dataset), as a percentage.

    No language skipping — this matches the published ablation probe columns
    (e.g. mc2_e32 → CG 63.7 / WIT 54.6 / Bloom 24.9, mc2cg_e32 → CG 46.5).
    """
    path = Path(csv_path) if csv_path else TRAIN_PROBE_CSV
    if tag is None or not path.exists():
        return None
    vals = []
    with open(path, newline="") as f:
        import csv as _csv
        for r in _csv.DictReader(f):
            if r.get("tag") == tag and r.get("dataset") == dataset:
                v = r.get(metric, "")
                if v not in ("", None):
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
    return 100 * sum(vals) / len(vals) if vals else None


def _imagenet_acc1_pct(tag: str):
    f = RESULTS_DIR / tag / f"imagenet1k_{tag}.json"
    if not f.exists():
        return None
    acc1 = json.loads(f.read_text()).get("metrics", {}).get("acc1")
    return acc1 * 100 if acc1 is not None else None


def _total_avg1(tag: str, direction: str, cvqa: str):
    """Total-Average@1 (R@1-Avg) for a single tag, via compute_table3."""
    global MODELS
    saved = MODELS
    MODELS = [(tag, tag, False)]
    try:
        row = compute_table3(cvqa, direction).get(tag, {})
    finally:
        MODELS = saved
    return row.get(1, {}).get("total_avg")


def compute_training(direction: str = "mean", cvqa: str = "sea7"):
    """Return ordered list of (label, desc, style, {col: value}) for tab:training.

    cols: cg, wit, bloom, imagenet, r1avg
    """
    rows = []
    for label, desc, ptag, etag, style in TRAINING_ROWS:
        vals = {col: load_train_probe(ptag, ds) for col, ds in _PROBE_DS.items()}
        vals["imagenet"] = _imagenet_acc1_pct(etag)
        vals["r1avg"] = _total_avg1(etag, direction, cvqa)
        rows.append((label, desc, style, vals))
    return rows


def generate_training_tex(direction: str = "mean", cvqa: str = "sea7") -> str:
    """Filled LaTeX for tab:training (paste-ready, matches the paper layout)."""
    rows = compute_training(direction, cvqa)
    cols = ["cg", "wit", "bloom", "imagenet", "r1avg"]

    # best among the three ablation "choice" rows (c1–c3) per column → bold
    best = {}
    for c in cols:
        cand = [v[c] for (_l, _d, s, v) in rows if s == "choice" and v[c] is not None]
        best[c] = max(cand) if cand else None

    def cell(v, c, style):
        if v is None:
            return "---"
        s = f"{v:.1f}"
        if style == "teacher":
            return r"\textit{" + s + "}"
        if style == "choice" and best[c] is not None and abs(v - best[c]) < 0.05:
            return r"\textbf{" + s + "}"
        return s

    L = []
    L.append(r"\begin{table}[h!]")
    L.append(r"\centering")
    L.append(r"\resizebox{\linewidth}{!}{%")
    L.append(r"\begin{tabular}{l l c c c c c}")
    L.append(r"\toprule")
    L.append(r"\textbf{Run} & \textbf{Train data (\#sample)}")
    L.append(r"  & \textbf{CG R@1} & \textbf{WIT R@1} & \textbf{Bloom R@1}")
    L.append(r"  & \textbf{ImageNet} & \textbf{R@1-Avg} \\")
    L.append(r"\midrule")

    prev_style = None
    for label, desc, style, vals in rows:
        if (style == "noKD" and prev_style == "searow") or style == "teacher":
            L.append(r"\midrule")
        if style == "searow":
            L.append(r"\rowcolor{searow}")
        cells = " & ".join(cell(vals[c], c, style) for c in cols)
        L.append(f"{label}\n  & {desc}\n  & {cells} \\\\")
        prev_style = style

    cvqa_label = {"all39": "all 39 non-English", "sea7": "7 SEA", "sea4": "4 SEA"}[cvqa]
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}%")
    L.append("}")
    L.append(r"\caption{The dataset ablation study. All models are trained on the same "
             r"training pipeline and hyperparameters. We ran the ablation on SEA-CLIP-Tiny "
             r"using 3 choices: (c1) SEA blend, (c2) CC12M, and (c3) CC12M+SEA. We also "
             r"tested on a single source of the SEA dataset, i.e., only WIT, Bloom, or "
             r"CulturalGround (CG). We use \texttt{R@1-Avg} from Total Average@1 in "
             r"Table~\ref{tab:per_task}.}")
    L.append(r"\label{tab:training}")
    L.append(r"\vspace{-8mm}")
    L.append(r"\end{table}")
    return "\n".join(L)


def gen_training_pdf(direction: str = "mean", cvqa: str = "sea7"):
    """Write runs/results/tables_training.tex (fragment) + a standalone PDF."""
    import subprocess
    frag = generate_training_tex(direction, cvqa)
    (RESULTS_DIR / "tables_training.tex").write_text(frag)

    doc = "\n".join([
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.5in,landscape]{geometry}",
        r"\usepackage{booktabs,graphicx}",
        r"\usepackage[table]{xcolor}",
        r"\definecolor{searow}{RGB}{225,238,250}",
        r"\begin{document}\pagestyle{empty}\small",
        frag,
        r"\end{document}",
    ])
    doc_path = RESULTS_DIR / "tables_training_doc.tex"
    doc_path.write_text(doc)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", doc_path.name],
                   cwd=RESULTS_DIR, capture_output=True)
    print("  wrote tables_training.tex  +  compiled tables_training_doc.pdf")


def gen_selflearn_pdf(direction: str = "mean", cvqa: str = "sea7"):
    """Per-language + per-task tables INCLUDING the selflearn (w/o KD) row.

    Written to a SEPARATE file (tables_<dir>_selflearn.{tex,pdf}) so the main
    tables_<dir>.{tex,pdf} stay untouched.
    """
    import subprocess
    global MODELS
    saved = MODELS
    # append the self-learning student row after the SEA-CLIP-Tiny rows
    MODELS = saved + [("selflearn_mammoth_v1_e32", "SEA-CLIP-Tiny (w/o KD)", False)]
    try:
        tex = generate_tex(direction, cvqa)
    finally:
        MODELS = saved
    out = RESULTS_DIR / f"tables_{direction}_selflearn.tex"
    out.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", out.name],
                   cwd=RESULTS_DIR, capture_output=True)
    print(f"  wrote {out.name}  +  compiled tables_{direction}_selflearn.pdf")


def gen_mammoth_pdf(direction: str = "mean", cvqa: str = "sea7"):
    """Per-language + per-task tables INCLUDING the clipkd_mammoth_v1 (KD + ML +
    Mammoth-VL-SEA) row.

    Written to a SEPARATE file (tables_<dir>_mammoth.{tex,pdf}) so the main
    tables_<dir>.{tex,pdf} stay untouched.
    """
    import subprocess
    global MODELS
    saved = MODELS
    MODELS = saved + [("clipkd_mammoth_v1_e32", "SEA-CLIP-Tiny + ML + Mammoth-VL-SEA", False)]
    try:
        tex = generate_tex(direction, cvqa)
    finally:
        MODELS = saved
    out = RESULTS_DIR / f"tables_{direction}_mammoth.tex"
    out.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", out.name],
                   cwd=RESULTS_DIR, capture_output=True)
    print(f"  wrote {out.name}  +  compiled tables_{direction}_mammoth.pdf")


def gen_ckdonly_pdf(direction: str = "mean", cvqa: str = "sea7"):
    """Per-language + per-task tables INCLUDING the ckdonly_v1 (clipkd loss only,
    no FD/ICL) row.

    Written to a SEPARATE file (tables_<dir>_ckdonly.{tex,pdf}) so the main
    tables_<dir>.{tex,pdf} stay untouched.
    """
    import subprocess
    global MODELS
    saved = MODELS
    MODELS = saved + [("ckdonly_v1_e32", "SEA-CLIP-Tiny (CKD only)", False)]
    try:
        tex = generate_tex(direction, cvqa)
    finally:
        MODELS = saved
    out = RESULTS_DIR / f"tables_{direction}_ckdonly.tex"
    out.write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", out.name],
                   cwd=RESULTS_DIR, capture_output=True)
    print(f"  wrote {out.name}  +  compiled tables_{direction}_ckdonly.pdf")


def gen_all_pdfs(cvqa_variant: str = "sea7"):
    import subprocess, os
    out_dir = RESULTS_DIR
    for direction in ["t2i", "i2t", "mean"]:
        tex_path = out_dir / f"tables_{direction}.tex"
        tex_path.write_text(generate_tex(direction, cvqa_variant))
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=out_dir, capture_output=True
        )
        print(f"  compiled → tables_{direction}.pdf")


if __name__ == "__main__":
    main()
