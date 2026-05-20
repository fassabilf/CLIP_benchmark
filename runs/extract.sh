#!/bin/bash
# One-button: regenerate runs/results/benchmark_{summary,aggregate}.csv
# from all per-(model,dataset,language) JSON files.
exec /home/ffirdaus/.conda/envs/mteb_env2/bin/python \
    "$(dirname "$0")/extract_results.py" "$@"
