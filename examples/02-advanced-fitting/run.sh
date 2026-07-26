#!/usr/bin/env bash
set -euo pipefail

echo "== PeakFit Example 2: Advanced Pseudo-ND CEST Workflow =="

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
mkdir -p "$latest_run/figures"

peakfit plot intensity "$latest_run" --output "$latest_run/figures/intensity_profiles.pdf"
peakfit plot cest "$latest_run" --output "$latest_run/figures/cest_profiles.pdf"

echo
echo "Run complete."
echo "Latest results: $latest_run"
echo "Intensity PDF: $latest_run/figures/intensity_profiles.pdf"
echo "CEST PDF: $latest_run/figures/cest_profiles.pdf"
echo "Interactive review command:"
echo "  peakfit plot spectrum --spectrum data/pseudo3d.ft2 --results $latest_run"
