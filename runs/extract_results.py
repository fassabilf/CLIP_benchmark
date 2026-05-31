#!/usr/bin/env python3
"""Compile per-(model, dataset, language) JSON results into two CSVs:

  - runs/results/benchmark_summary.csv   — full pivot (one row per dataset×lang)
  - runs/results/benchmark_aggregate.csv — ENG vs SEA-avg per benchmark

Run from CLIP_benchmark/ root or via `runs/extract.sh`.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Column rename: tag → full hp-annotated name shown in CSV header.
COL = {
    "metaclip2":     "metaclip2-worldwide-h14",
    "metaclip2_b16": "metaclip2-worldwide-b16",
    "s1":        "vit-t-16-clipkd-vit-b-32-lr2e-3-e32",
    "s1_e8":     "vit-t-16-clipkd-vit-b-32-lr2e-3-e8",
    "s2":        "vit-t-16-clipkd-vit-b-16-siglip2-lr2e-3-e32",
    "s3":        "vit-t-16-clipkd-vit-b-16-siglip2-lr4.47e-3-e100",
    "s3_e8":     "vit-t-16-clipkd-vit-b-16-siglip2-lr4.47e-3-e8",
    "wit_e8":    "vit-t-16-clipkd-vit-b-16-siglip2-wit-lr2e-3-e8",
    "wit_e32":   "vit-t-16-clipkd-vit-b-16-siglip2-wit-lr2e-3-e32",
    "mv1_e8":    "vit-t-16-clipkd-vit-b-16-siglip2-mv1-lr2e-3-e8",
    "mv1_e32":   "vit-t-16-clipkd-vit-b-16-siglip2-mv1-lr2e-3-e32",
    "clipkd_b16_laion": "clipkd-released-vit-b-16-teacher-laion400m",
    # Habibi metaclip2_kd run: ViT-T-16 (CLIP-BPE) <- MetaCLIP2-B16-worldwide, 3-blend SEA.
    "mc2_e0":    "vit-t-16-clipkd-metaclip2-b16-mc2-lr2e-3-e0init",
    "mc2_e8":    "vit-t-16-clipkd-metaclip2-b16-mc2-lr2e-3-e8",
    "mc2_e16":   "vit-t-16-clipkd-metaclip2-b16-mc2-lr2e-3-e16",
    "mc2_e24":   "vit-t-16-clipkd-metaclip2-b16-mc2-lr2e-3-e24",
    "mc2_e32":   "vit-t-16-clipkd-metaclip2-b16-mc2-lr2e-3-e32",
}
TAGS_ORDER = ["metaclip2", "metaclip2_b16", "s1", "s1_e8", "s2", "s3", "s3_e8",
              "wit_e8", "wit_e32", "mv1_e8", "mv1_e32", "clipkd_b16_laion",
              "mc2_e0", "mc2_e8", "mc2_e16", "mc2_e24", "mc2_e32"]

# Language groupings for the aggregate view.
SEA_LANGS = {
    "babel_imagenet":  ["id", "jv", "ms", "my", "su", "th", "vi"],
    "crossmodal3600":  ["id", "th", "vi"],
    "flickr30k-200":   ["ind_Latn", "jav_Latn", "zsm_Latn", "mya_Mymr",
                        "sun_Latn", "tha_Thai", "vie_Latn"],
    "xtd200":          ["ind_Latn", "jav_Latn", "zsm_Latn", "mya_Mymr",
                        "sun_Latn", "tha_Thai", "vie_Latn"],
}
ENG_LANG = {
    "imagenet1k-unverified": "en",
    "babel_imagenet":        "en",
    "crossmodal3600":        "en",
    "flickr30k-200":         "eng_Latn",
    "xtd200":                "eng_Latn",
}
BENCH_ORDER = ["imagenet1k-unverified", "babel_imagenet", "crossmodal3600",
               "flickr30k-200", "xtd200"]


def load_all(results_dir: Path):
    """Read every JSON under results_dir/<tag>/*.json into a (tag, ds, lang)→value map."""
    table = defaultdict(dict)  # (ds, lang) → {tag: value}
    for tagdir in sorted(results_dir.iterdir()):
        if not tagdir.is_dir() or tagdir.name not in COL:
            continue
        tag = tagdir.name
        for f in sorted(tagdir.glob("*.json")):
            d = json.loads(f.read_text())
            stem = f.stem
            if "cvqa" in stem:
                table[("cvqa", "LOCAL")][tag] = d.get("LOCAL", float("nan"))
                table[("cvqa", "EN")][tag]    = d.get("EN", float("nan"))
                continue
            ds = d.get("dataset", stem)
            lang = d.get("language", "")
            m = d.get("metrics", {})
            if d.get("task") == "zeroshot_classification":
                v = m.get("acc1", float("nan"))
            elif d.get("task") == "zeroshot_retrieval":
                v = m.get("image_retrieval_recall@1", float("nan"))
            else:
                continue
            table[(ds, lang)][tag] = v
    return table


def write_summary(table, out_path: Path):
    cols = [COL[t] for t in TAGS_ORDER]
    with out_path.open("w") as out:
        w = csv.writer(out)
        w.writerow(["dataset", "language"] + cols)
        for (ds, lang) in sorted(table.keys()):
            w.writerow([ds, lang] + [table[(ds, lang)].get(t, "") for t in TAGS_ORDER])


def _aggregate_rows(table):
    """Return [(label, group, [per-tag values-or-None])] in display order."""
    def avg(ds, langs, tag):
        vals = [table[(ds, l)][tag] for l in langs
                if (ds, l) in table and tag in table[(ds, l)]
                and table[(ds, l)][tag] == table[(ds, l)][tag]]  # not NaN
        return sum(vals) / len(vals) if vals else None

    rows = []
    for ds in BENCH_ORDER:
        label = ds.replace("-unverified", "")
        eng_lang = ENG_LANG.get(ds)
        eng_row = [table.get((ds, eng_lang), {}).get(t) or None for t in TAGS_ORDER]
        eng_row = [v if isinstance(v, (int, float)) else None for v in eng_row]
        rows.append((label, "ENG", eng_row))
        if ds in SEA_LANGS:
            rows.append((label, "SEA", [avg(ds, SEA_LANGS[ds], t) for t in TAGS_ORDER]))
    # CVQA: EN with ENG group, LOCAL with SEA group (local non-English languages).
    rows.append(("cvqa", "EN",    [table.get(("cvqa", "EN"),    {}).get(t) for t in TAGS_ORDER]))
    rows.append(("cvqa", "LOCAL", [table.get(("cvqa", "LOCAL"), {}).get(t) for t in TAGS_ORDER]))
    return rows


def _mean_rows(rows):
    """Compute ENG / SEA / ENG+SEA means across the aggregate rows.

    cvqa EN counts as ENG; cvqa LOCAL counts as SEA (local-language eval).
    """
    eng_groups = {"ENG", "EN"}
    sea_groups = {"SEA", "LOCAL"}

    def mean_over(groups):
        out = []
        for i in range(len(TAGS_ORDER)):
            vals = [r[2][i] for r in rows if r[1] in groups and isinstance(r[2][i], (int, float))]
            out.append(sum(vals) / len(vals) if vals else None)
        return out

    return [
        ("MEAN", "ENG",     mean_over(eng_groups)),
        ("MEAN", "SEA",     mean_over(sea_groups)),
        ("MEAN", "ENG+SEA", mean_over(eng_groups | sea_groups)),
    ]


# SEA languages whose script is NOT Latin (CLIP-BPE byte fallback hurts these).
NONLATIN = {"my", "th", "mya_Mymr", "tha_Thai"}


def _script_mean_rows(table):
    """MEAN over the per-language SEA benchmarks, split by script (Latin vs non-Latin).

    Mirrors how MEAN/SEA is built (per-benchmark lang-average, then mean across
    benchmarks), but restricts to the Latin- or non-Latin-script SEA languages.
    CVQA has no per-language SEA breakdown, so it is excluded here.
    """
    def script_mean(keep_latin):
        out = []
        for tag in TAGS_ORDER:
            per_bench = []
            for ds in BENCH_ORDER:
                if ds not in SEA_LANGS:
                    continue
                langs = [l for l in SEA_LANGS[ds] if (l in NONLATIN) != keep_latin]
                vals = [table[(ds, l)][tag] for l in langs
                        if (ds, l) in table and tag in table[(ds, l)]
                        and table[(ds, l)][tag] == table[(ds, l)][tag]]
                if vals:
                    per_bench.append(sum(vals) / len(vals))
            out.append(sum(per_bench) / len(per_bench) if per_bench else None)
        return out

    return [
        ("MEAN", "SEA-Latin",    script_mean(True)),
        ("MEAN", "SEA-nonLatin", script_mean(False)),
    ]


# Per-language MEAN rows: each language → its per-benchmark code (iso2 for babel/xm3600,
# NLLB code for flickr/xtd). Mean is taken across whichever benchmarks cover the language.
LANG_CODES = {
    "en": {"imagenet1k-unverified": "en", "babel_imagenet": "en", "crossmodal3600": "en",
           "flickr30k-200": "eng_Latn", "xtd200": "eng_Latn"},
    "id": {"babel_imagenet": "id", "crossmodal3600": "id", "flickr30k-200": "ind_Latn", "xtd200": "ind_Latn"},
    "jv": {"babel_imagenet": "jv", "flickr30k-200": "jav_Latn", "xtd200": "jav_Latn"},
    "ms": {"babel_imagenet": "ms", "flickr30k-200": "zsm_Latn", "xtd200": "zsm_Latn"},
    "my": {"babel_imagenet": "my", "flickr30k-200": "mya_Mymr", "xtd200": "mya_Mymr"},
    "su": {"babel_imagenet": "su", "flickr30k-200": "sun_Latn", "xtd200": "sun_Latn"},
    "th": {"babel_imagenet": "th", "crossmodal3600": "th", "flickr30k-200": "tha_Thai", "xtd200": "tha_Thai"},
    "vi": {"babel_imagenet": "vi", "crossmodal3600": "vi", "flickr30k-200": "vie_Latn", "xtd200": "vie_Latn"},
}
LANG_ORDER = ["en", "id", "jv", "ms", "my", "su", "th", "vi"]


def _lang_mean_rows(table):
    """One MEAN row per language: mean across the benchmarks that cover it."""
    rows = []
    for lang in LANG_ORDER:
        per_ds = LANG_CODES[lang]
        vals = []
        for tag in TAGS_ORDER:
            xs = [table[(ds, code)][tag] for ds, code in per_ds.items()
                  if (ds, code) in table and tag in table[(ds, code)]
                  and table[(ds, code)][tag] == table[(ds, code)][tag]]
            vals.append(sum(xs) / len(xs) if xs else None)
        rows.append(("MEAN-LANG", lang, vals))
    return rows


def write_aggregate(table, out_path: Path):
    cols = [COL[t] for t in TAGS_ORDER]
    rows = _aggregate_rows(table)
    means = _mean_rows(rows) + _script_mean_rows(table) + _lang_mean_rows(table)
    with out_path.open("w") as out:
        w = csv.writer(out)
        w.writerow(["benchmark", "column"] + cols)
        for label, group, vals in rows + means:
            w.writerow([label, group] +
                       [f"{v:.6f}" if isinstance(v, (int, float)) else "" for v in vals])


def fmt(v):
    if v in ("", None):
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "—"


def print_aggregate(table):
    cols = [COL[t][:32] for t in TAGS_ORDER]
    rows = _aggregate_rows(table)
    means = _mean_rows(rows) + _script_mean_rows(table) + _lang_mean_rows(table)
    print(f"\n{'Benchmark':22s} {'col':8s} " + " ".join(f"{c:>32s}" for c in cols))
    print("-" * 200)
    for label, group, vals in rows:
        print(f"{label:22s} {group:8s} " + " ".join(f"{fmt(v):>32s}" for v in vals))
    print("-" * 200)
    for label, group, vals in means:
        print(f"{label:22s} {group:8s} " + " ".join(f"{fmt(v):>32s}" for v in vals))


def main():
    root = Path(__file__).resolve().parent / "results"
    if not root.exists():
        print(f"ERR: {root} not found", file=sys.stderr); sys.exit(1)

    table = load_all(root)
    summary = root / "benchmark_summary.csv"
    aggregate = root / "benchmark_aggregate.csv"
    write_summary(table, summary)
    write_aggregate(table, aggregate)
    print(f"wrote {summary}")
    print(f"wrote {aggregate}")
    print_aggregate(table)


if __name__ == "__main__":
    main()
