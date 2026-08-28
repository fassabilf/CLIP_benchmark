#!/usr/bin/env python3
"""Median and spread of the repeated GPU profile (runs/results/profile/rep*/).

Batch-1 latency is kernel-launch-bound: precise within a run (std well under
1 ms) but reproducible to only ~10% across allocations. Quoting a single run
therefore quotes noise. This takes the median across repetitions and reports the
spread so the paper can state it.

Params and FLOPs are deterministic; they are checked for equality across
repetitions instead of averaged, which doubles as a guard that every repetition
really built the same model.

Usage:
  python3 runs/summarize_repeats.py                 # markdown table + spread
  python3 runs/summarize_repeats.py --write         # also write the median JSONs
                                                    # to runs/results/profile/
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = REPO_ROOT / "runs" / "results" / "profile"
KEYS = ["tinyclip", "mobileclip2_s0", "sea_clip_tiny", "teacher_b16"]
GLOB = "rep*"


def load_reps(key):
    reps = []
    for d in sorted(PROFILE_DIR.glob(GLOB)):
        f = d / f"{key}.json"
        if f.exists():
            reps.append(json.loads(f.read_text()))
    return reps


def median_result(reps):
    """One result whose latency is the per-field median across repetitions."""
    base = json.loads(json.dumps(reps[0]))
    for r in reps[1:]:
        if r["params"] != base["params"] or r["flops"] != base["flops"]:
            raise SystemExit(f"repetitions disagree on params/FLOPs for {base['label']} "
                             "— they did not build the same model")
    for bs in base.get("latency", {}):
        for tower in ("image", "text"):
            for field in base["latency"][bs][tower]:
                vals = [r["latency"][bs][tower][field] for r in reps]
                if all(isinstance(v, (int, float)) for v in vals):
                    base["latency"][bs][tower][field] = statistics.median(vals)
        for field in ("total_mean_ms", "images_per_sec", "texts_per_sec"):
            if field in base["latency"][bs]:
                base["latency"][bs][field] = statistics.median(
                    r["latency"][bs][field] for r in reps)
    base["repetitions"] = len(reps)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the median result over runs/results/profile/<key>.json")
    ap.add_argument("--glob", default="rep*",
                    help="which sub-run directories to pool (rep* = repetitions inside "
                         "one allocation; alloc* = separate allocations, which is where "
                         "the ~10%% of batch-1 variation actually lives).")
    args = ap.parse_args()
    global GLOB
    GLOB = args.glob

    print(f"| Model | bs=1 image ms (median) | spread | bs=1 text ms (median) | spread "
          f"| img/s (bs=1024) | spread |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in KEYS:
        reps = load_reps(key)
        if not reps:
            print(f"  (no repetitions for {key})", file=sys.stderr)
            continue
        med = median_result(reps)

        def spread(getter):
            vals = [getter(r) for r in reps]
            return f"{min(vals):.2f}–{max(vals):.2f}"

        img = med["latency"]["1"]["image"]["p50_ms"]
        txt = med["latency"]["1"]["text"]["p50_ms"]
        tput = med["latency"]["1024"]["images_per_sec"]
        print(f"| {med['label']} | {img:.2f} "
              f"| {spread(lambda r: r['latency']['1']['image']['p50_ms'])} "
              f"| {txt:.2f} "
              f"| {spread(lambda r: r['latency']['1']['text']['p50_ms'])} "
              f"| {tput:.0f} "
              f"| {spread(lambda r: r['latency']['1024']['images_per_sec']):s} |")

        if args.write:
            (PROFILE_DIR / f"{key}.json").write_text(json.dumps(med, indent=2))
    if args.write:
        print(f"\nwrote medians of {len(load_reps(KEYS[0]))} repetitions "
              f"to {PROFILE_DIR.relative_to(REPO_ROOT)}/<key>.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
