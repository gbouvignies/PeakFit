#!/usr/bin/env bash
set -euo pipefail

mode="${1:-both}"

if [[ "$mode" != "local" && "$mode" != "basin" && "$mode" != "both" ]]; then
  echo "Usage: bash run.sh [local|basin|both]"
  exit 1
fi

if ! command -v peakfit >/dev/null 2>&1; then
  echo "Error: 'peakfit' command not found."
  exit 1
fi

for f in data/pseudo3d.ft2 data/pseudo3d.list data/b1_offsets.txt; do
  if [[ ! -f "$f" ]]; then
    echo "Missing input file: $f"
    exit 1
  fi
done

rm -rf Fits-local Fits-basin

run_fit() {
  optimizer="$1"
  out_root="$2"

  peakfit fit \
    data/pseudo3d.ft2 \
    data/pseudo3d.list \
    --z-values data/b1_offsets.txt \
    --optimizer "$optimizer" \
    --output "$out_root" >&2

  find "$out_root" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

local_run=""
basin_run=""

if [[ "$mode" == "local" || "$mode" == "both" ]]; then
  echo "Running local fit (varpro)..."
  local_run="$(run_fit varpro Fits-local)"
  echo "Local results: $local_run"
fi

if [[ "$mode" == "basin" || "$mode" == "both" ]]; then
  echo "Running global fit (basin_hopping)..."
  basin_run="$(run_fit basin_hopping Fits-basin)"
  echo "Basin-hopping results: $basin_run"
fi

if [[ "$mode" == "both" ]]; then
  python - "$local_run" "$basin_run" <<'PY'
import json
import sys

local_dir, basin_dir = sys.argv[1], sys.argv[2]


def read_chi2(run_dir: str) -> float:
    with open(f"{run_dir}/summary/fit.json", encoding="utf-8") as f:
        data = json.load(f)
    return float(data["global_statistics"]["chi_squared"])

local_chi2 = read_chi2(local_dir)
basin_chi2 = read_chi2(basin_dir)

print("\nChi-squared comparison")
print(f"  varpro        : {local_chi2:.6g}")
print(f"  basin_hopping : {basin_chi2:.6g}")
PY
fi
