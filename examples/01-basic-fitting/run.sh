#!/usr/bin/env bash
set -euo pipefail

echo "== PeakFit Example 1: Basic Fitting Template =="

if ! command -v peakfit >/dev/null 2>&1; then
  echo "Error: 'peakfit' command not found."
  echo "Install PeakFit first, then rerun this script."
  exit 1
fi

if [[ ! -f "data/spectrum.ft2" || ! -f "data/peaks.list" ]]; then
  echo "Missing input files in data/."
  echo "Required: data/spectrum.ft2 and data/peaks.list"
  echo "Optional: data/z_values.txt"
  exit 1
fi

rm -rf results

cmd=(peakfit fit data/spectrum.ft2 data/peaks.list --output results --output-verbosity standard)
if [[ -f "data/z_values.txt" ]]; then
  cmd+=(--z-values data/z_values.txt)
fi

"${cmd[@]}"

if [[ -d "results/summary" ]]; then
  latest_run="results"
else
  latest_run="$(find results -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

echo
echo "Run complete."
echo "Latest results: ${latest_run}"
echo "Inspect: ${latest_run}/summary/fit_summary.json"
