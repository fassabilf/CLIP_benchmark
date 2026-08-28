#!/usr/bin/env python3
"""Profile every model in the paper's comparison set and render the efficiency table.

Params, FLOPs and latency are all properties of the *architecture*, not of the
trained weights — a randomly-initialised model of the same config gives
identical FLOPs/params and statistically identical timings. So these rows are
keyed by architecture, not by run tag: `clipkd_*_e8` through `_e32` all share
one entry, and no checkpoint is required unless you pass one.

Usage
-----
  # measure everything, write runs/results/profile/*.json, print the table
  python3 runs/profile_models.py

  # params + FLOPs only (device-independent, runs anywhere)
  python3 runs/profile_models.py --no-latency

  # re-render the table from JSON already measured on the cluster
  python3 runs/profile_models.py --table-only

  # one model, e.g. while iterating
  python3 runs/profile_models.py --only sea_clip_tiny

Latency is device-bound: the GPU name is recorded in each JSON and printed in
the table footnote. Do not mix rows measured on different devices.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PROFILE_DIR = REPO_ROOT / "runs" / "results" / "profile"


def set_profile_dir(path):
    """Point reads and writes at another directory (e.g. a cpu/ sub-run).

    Latency is device-bound, so a CPU sweep must not overwrite the GPU JSONs it
    would otherwise share filenames with.
    """
    global PROFILE_DIR
    PROFILE_DIR = Path(path)

# (key, table label, model, pretrained, model_type)
#
# `pretrained=""` builds from the architecture config with random init, which is
# all that params/FLOPs/latency depend on. Point it at a checkpoint (and pass
# --check-vocab) only when you want the config↔checkpoint guard.
SPECS = [
    ("tinyclip",       "TinyCLIP",
     "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", "", "hf_transformers"),
    ("mobileclip2_s0", "MobileCLIP2",
     "MobileCLIP2-S0", "dfndr2b", "open_clip"),
    ("sea_clip_tiny",  "SEA-CLIP-Tiny",
     "ViT-T-16", "", "open_clip"),
    ("teacher_b16",    "MetaCLIP-2 B/16 (teacher)",
     "facebook/metaclip-2-worldwide-b16", "", "hf_transformers"),
]

# Rows to include in the headline table, in display order. The teacher is
# measured but kept out by default — it is a reference, not a comparison point.
DEFAULT_ROWS = ["tinyclip", "mobileclip2_s0", "sea_clip_tiny"]


def _rel(path):
    """Repo-relative when it can be, absolute otherwise (--out-dir may be elsewhere)."""
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def _slug(text):
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()


def specs_from_args(args):
    """Build the list of models to profile.

    Precedence: --models-file > --model > the built-in SPECS. Nothing about the
    profiler is tied to SPECS — it is a convenience list for the paper's
    comparison set, not a restriction.
    """
    if args.models_file:
        specs = []
        for raw in Path(args.models_file).read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [f.strip() for f in line.split(",")]
            model = parts[0]
            pretrained = parts[1] if len(parts) > 1 else ""
            model_type = parts[2] if len(parts) > 2 and parts[2] else "auto"
            label = parts[3] if len(parts) > 3 else model
            specs.append((_slug(f"{model}_{pretrained}"), label, model, pretrained, model_type))
        return specs
    if args.model:
        label = args.label or args.model
        return [(args.key or _slug(f"{args.model}_{args.pretrained}"),
                 label, args.model, args.pretrained, args.model_type)]
    return SPECS


def profile_one(spec, args):
    from clip_benchmark.metrics import profiling
    from clip_benchmark.models import load_clip
    import torch

    key, label, model_name, pretrained, model_type = spec
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== {label}  ({model_name}) ===")
    model, transform, tokenizer = load_clip(
        model_type=model_type,
        model_name=model_name,
        pretrained=pretrained,
        cache_dir=args.cache_dir,
        device=device,
    )
    model.eval()

    if args.check_vocab and Path(pretrained).is_file():
        profiling.assert_vocab_size(model, pretrained)

    result = profiling.profile(
        model, transform, tokenizer,
        device=device,
        amp=not args.fp32,
        latency_batch_sizes=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
        latency=not args.no_latency,
        memory=not args.no_memory,
        param_split=args.param_split,
        extra_image_hints=args.image_hint or (),
    )
    result.update({"key": key, "label": label, "model": model_name,
                   "pretrained": pretrained, "model_type": model_type})

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    out = PROFILE_DIR / f"{key}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  → {_rel(out)}")

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def load_results(keys):
    """Read whatever has been measured, in `keys` order. Missing = skipped."""
    results = []
    for key in keys:
        f = PROFILE_DIR / f"{key}.json"
        if f.exists():
            results.append(json.loads(f.read_text()))
        else:
            print(f"  (no profile for '{key}' — run without --table-only)", file=sys.stderr)
    return results


# ─── rendering ───────────────────────────────────────────────────────────────

def _m(n):
    return f"{n / 1e6:.2f}M"


def _g(n):
    return f"{n:.2f}"


def _tower_params(r, tower, exclude_projections=False):
    p = r["params"]
    n = p[tower]
    if exclude_projections:
        n -= p.get(f"{tower}_projection", 0)
    return n


def _batch_sizes(results):
    """(smallest, largest) batch size present in every result's latency block.

    Latency is read at the smallest (single-sample response time, what the
    rebuttal quotes as ms/image); throughput and peak memory at the largest
    (saturated the way a batched eval loop runs).
    """
    common = None
    for r in results:
        keys = set(r.get("latency", {}))
        common = keys if common is None else (common & keys)
    if not common:
        return None, None
    ordered = sorted(common, key=int)
    return ordered[0], ordered[-1]


def _has_memory(results, bs):
    return bs is not None and all(
        "peak_mem_alloc_mb" in r["latency"][bs]["image"] for r in results)


def render_markdown(results, with_latency=True, exclude_projections=False):
    has_lat = with_latency and all("latency" in r for r in results)
    bs = tbs = None
    if has_lat:
        bs, tbs = _batch_sizes(results)
        has_lat = bs is not None
    has_mem = has_lat and _has_memory(results, tbs)

    head = ["Model", "Image params", "Text params",
            "Image GFLOPs", "Text GFLOPs", "Total GFLOPs"]
    align = ["", "---:", "---:", "---:", "---:", "---:"]
    if has_lat:
        head += [f"Image ms (bs={bs})", f"Text ms (bs={bs})",
                 f"img/s (bs={tbs})", f"txt/s (bs={tbs})"]
        align += ["---:", "---:", "---:", "---:"]
    if has_mem:
        head += [f"Peak mem MB (bs={tbs})"]
        align += ["---:"]
    align[0] = "---"

    lines = ["| " + " | ".join(head) + " |",
             "| " + " | ".join(align) + " |"]
    for r in results:
        g = r["gflops"]
        cells = [r["label"],
                 _m(_tower_params(r, "image", exclude_projections)),
                 _m(_tower_params(r, "text", exclude_projections)),
                 _g(g["image"]), _g(g["text"]), _g(g["total"])]
        if has_lat:
            lat, tput = r["latency"][bs], r["latency"][tbs]
            cells += [f"{lat['image']['p50_ms']:.2f}", f"{lat['text']['p50_ms']:.2f}",
                      f"{tput['images_per_sec']:.0f}", f"{tput['texts_per_sec']:.0f}"]
        if has_mem:
            peak = max(r["latency"][tbs][t]["peak_mem_alloc_mb"] for t in ("image", "text"))
            cells += [f"{peak:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")

    notes = []
    r0 = results[0]
    notes.append(f"GFLOPs per sample, {r0['flops_convention']}.")
    shapes = {r["label"]: (tuple(r["image_input_shape"][1:]), r["text_input_shape"][1])
              for r in results}
    if len(set(shapes.values())) == 1:
        (img, ctx), = set(shapes.values())
        notes.append(f"Input {list(img)} image / {ctx}-token text for every model.")
    else:
        per = "; ".join(f"{label} {list(img)} / {ctx} tokens"
                        for label, (img, ctx) in shapes.items())
        notes.append("Each model at its own native input resolution and context "
                     f"length — {per}.")
    notes.append("Parameter counts "
                 + ("exclude" if exclude_projections else "include")
                 + " each tower's final embedding projection head "
                 + f"({', '.join(sorted(set(_m(r['params']['image_projection']) for r in results)))} image / "
                 + f"{', '.join(sorted(set(_m(r['params']['text_projection']) for r in results)))} text).")
    if has_lat:
        gpus = {r.get("gpu_name") or r["device"] for r in results}
        notes.append(f"Latency = median of {r0['latency'][bs]['image']['runs']} runs "
                     f"after {r0['latency'][bs]['image']['warmup']} warmup passes, "
                     f"{r0['precision']}, on {', '.join(sorted(gpus))}.")
        notes.append(f"Throughput = batch size / mean batch time at bs={tbs}, "
                     "same runs and precision.")
    if has_mem:
        notes.append(f"Peak memory = torch.cuda.max_memory_allocated over the measured "
                     f"bs={tbs} passes (weights resident + activations of the larger "
                     "tower), after warmup.")
    return "\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes)


def render_latex(results, exclude_projections=False):
    L = [r"\begin{table}[h!]", r"\centering",
         r"\begin{tabular}{l r r r r r}", r"\toprule",
         r"\textbf{Model} & \textbf{Image params} & \textbf{Text params}",
         r"  & \textbf{Image GFLOPs} & \textbf{Text GFLOPs} & \textbf{Total GFLOPs} \\",
         r"\midrule"]
    for r in results:
        g = r["gflops"]
        L.append(f"{r['label']} "
                 f"& {_m(_tower_params(r, 'image', exclude_projections))} "
                 f"& {_m(_tower_params(r, 'text', exclude_projections))} "
                 f"& {_g(g['image'])} & {_g(g['text'])} & {_g(g['total'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Efficiency comparison. GFLOPs are per sample for a single "
          r"forward pass of each tower, counted with \texttt{torch.utils.flop\_counter} "
          r"(1 MAC = 2 FLOPs), at each model's native input resolution and context length.}",
          r"\label{tab:efficiency}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="profile an arbitrary model instead of the built-in set, "
                         "using the same identifier `eval` takes.")
    ap.add_argument("--pretrained", default="",
                    help="checkpoint path or open_clip tag for --model. Optional: "
                         "params/FLOPs/latency depend on the architecture, not the weights.")
    ap.add_argument("--model-type", default="auto",
                    help="open_clip | hf_transformers | ja_clip | auto")
    ap.add_argument("--label", default=None, help="table label for --model.")
    ap.add_argument("--key", default=None, help="JSON filename stem for --model.")
    ap.add_argument("--models-file", default=None,
                    help="text file of models to profile, one per line: "
                         "`model,pretrained[,model_type[,label]]`.")
    ap.add_argument("--param-split", default="auto",
                    choices=["auto", "execution", "name"],
                    help="how to attribute parameters to towers. 'execution' traces "
                         "which parameters each forward pass touches (architecture-"
                         "agnostic); 'name' matches naming conventions.")
    ap.add_argument("--image-hint", nargs="+", default=None,
                    help="extra name components identifying the image tower, for "
                         "architectures the defaults do not cover.")
    ap.add_argument("--only", nargs="+", default=None,
                    help=f"subset of keys to profile. Available: {[s[0] for s in SPECS]}")
    ap.add_argument("--rows", nargs="+", default=DEFAULT_ROWS,
                    help="keys to include in the rendered table, in display order.")
    ap.add_argument("--table-only", action="store_true",
                    help="skip measurement, render from existing JSON.")
    ap.add_argument("--no-latency", action="store_true",
                    help="params + FLOPs only (device-independent).")
    ap.add_argument("--no-memory", action="store_true",
                    help="skip the peak-CUDA-memory measurement taken alongside latency.")
    ap.add_argument("--out-dir", default=None,
                    help="write/read the per-model JSON and the rendered table here "
                         "instead of runs/results/profile (use for a CPU sweep, whose "
                         "latency must not overwrite the GPU numbers).")
    ap.add_argument("--batch-size", type=int, nargs="+", default=[1, 64],
                    help="latency batch sizes. FLOPs are always per sample.")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--fp32", action="store_true",
                    help="measure latency in fp32 instead of the eval-time autocast.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--check-vocab", action="store_true",
                    help="assert config matches checkpoint when --pretrained is a file.")
    ap.add_argument("--exclude-projections", action="store_true",
                    help="report tower params without their final embedding projection "
                         "head. Reproduces the convention behind the commonly-quoted "
                         "TinyCLIP 8.15M/15.04M; the default includes the heads.")
    ap.add_argument("--latex", action="store_true", help="also print the LaTeX table.")
    args = ap.parse_args()

    if args.out_dir:
        set_profile_dir(args.out_dir)

    specs = specs_from_args(args)
    ad_hoc = args.model or args.models_file

    if not args.table_only:
        keys = args.only or [s[0] for s in specs]
        for spec in specs:
            if spec[0] in keys:
                try:
                    profile_one(spec, args)
                except Exception as e:  # one unavailable model must not sink the rest
                    print(f"  !! {spec[1]} failed: {type(e).__name__}: {e}", file=sys.stderr)

    # An explicit model/file defines its own rows; otherwise use the paper set.
    rows = [s[0] for s in specs] if ad_hoc else args.rows
    results = load_results(rows)
    if not results:
        print("nothing to render", file=sys.stderr)
        return 1

    table = render_markdown(results, with_latency=not args.no_latency,
                            exclude_projections=args.exclude_projections)
    print("\n" + table)
    md = PROFILE_DIR / "efficiency_table.md"
    md.write_text(table + "\n")
    print(f"\n→ {_rel(md)}")

    tex = render_latex(results, exclude_projections=args.exclude_projections)
    (PROFILE_DIR / "efficiency_table.tex").write_text(tex + "\n")
    if args.latex:
        print("\n" + tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
