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
    "metaclip2": "metaclip2-worldwide-h14",
    "s1":        "vit-t-16-clipkd-vit-b-32-lr2e-3-e32",
    "s1_e8":     "vit-t-16-clipkd-vit-b-32-lr2e-3-e8",
    "s2":        "vit-t-16-clipkd-vit-b-16-siglip2-lr2e-3-e32",
    "s3":        "vit-t-16-clipkd-vit-b-16-siglip2-lr4.47e-3-e100",
    "s3_e8":     "vit-t-16-clipkd-vit-b-16-siglip2-lr4.47e-3-e8",
}
TAGS_ORDER = ["metaclip2", "s1", "s1_e8", "s2", "s3", "s3_e8"]

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


def write_aggregate(table, out_path: Path):
    def avg(ds, langs, tag):
        vals = [table[(ds, l)][tag] for l in langs
                if (ds, l) in table and tag in table[(ds, l)]
                and table[(ds, l)][tag] == table[(ds, l)][tag]]  # not NaN
        return sum(vals) / len(vals) if vals else None

    cols = [COL[t] for t in TAGS_ORDER]
    with out_path.open("w") as out:
        w = csv.writer(out)
        w.writerow(["benchmark", "column"] + cols)
        for ds in BENCH_ORDER:
            label = ds.replace("-unverified", "")
            eng_lang = ENG_LANG.get(ds)
            eng_row = [table.get((ds, eng_lang), {}).get(t, "") for t in TAGS_ORDER]
            w.writerow([label, "ENG"] + eng_row)
            if ds in SEA_LANGS:
                sea_row = [avg(ds, SEA_LANGS[ds], t) for t in TAGS_ORDER]
                w.writerow([label, "SEA"] + [f"{v:.6f}" if v is not None else "" for v in sea_row])
        # CVQA: EN + LOCAL (LOCAL spans all countries, not SEA-only)
        for sub in ("EN", "LOCAL"):
            w.writerow(["cvqa", sub] + [table.get(("cvqa", sub), {}).get(t, "") for t in TAGS_ORDER])


def fmt(v):
    if v in ("", None):
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "—"


def print_aggregate(table):
    def avg(ds, langs, tag):
        vals = [table[(ds, l)][tag] for l in langs
                if (ds, l) in table and tag in table[(ds, l)]
                and table[(ds, l)][tag] == table[(ds, l)][tag]]
        return sum(vals) / len(vals) if vals else None

    cols = [COL[t][:32] for t in TAGS_ORDER]
    print(f"\n{'Benchmark':22s} {'col':6s} " + " ".join(f"{c:>32s}" for c in cols))
    print("-" * 180)
    for ds in BENCH_ORDER:
        label = ds.replace("-unverified", "")
        eng_lang = ENG_LANG.get(ds)
        eng = [table.get((ds, eng_lang), {}).get(t, "") for t in TAGS_ORDER]
        print(f"{label:22s} {'ENG':6s} " + " ".join(f"{fmt(v):>32s}" for v in eng))
        if ds in SEA_LANGS:
            sea = [avg(ds, SEA_LANGS[ds], t) for t in TAGS_ORDER]
            print(f"{label:22s} {'SEA':6s} " +
                  " ".join(f"{(f'{v:.3f}' if v is not None else '—'):>32s}" for v in sea))
    for sub in ("EN", "LOCAL"):
        vs = [table.get(("cvqa", sub), {}).get(t, "") for t in TAGS_ORDER]
        print(f"{'cvqa':22s} {sub:6s} " + " ".join(f"{fmt(v):>32s}" for v in vs))


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
