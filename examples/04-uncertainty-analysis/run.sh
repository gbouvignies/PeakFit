#!/usr/bin/env bash
set -euo pipefail

echo "== PeakFit Example 4: MCMC Uncertainty Analysis =="

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

rm -rf Fits

peakfit fit \
  data/pseudo3d.ft2 \
  data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits

latest_run="$(find Fits -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"

peakfit mcmc "$latest_run" --peaks 2N-H --walkers 32 --steps 1000

echo
echo "MCMC complete."
echo "Results directory: $latest_run"
echo "Interactive quality check:"
echo "  peakfit plot spectrum --spectrum data/pseudo3d.ft2 --results $latest_run"
echo
echo "MCMC diagnostics PDF:"
echo "  peakfit plot mcmc $latest_run --output $latest_run/mcmc_diagnostics.pdf"
