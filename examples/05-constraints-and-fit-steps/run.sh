#!/usr/bin/env bash
set -euo pipefail

echo "== PeakFit Example 5: Constraints and Fit Steps =="

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

run_scenario() {
  name="$1"
  config="$2"
  out_dir="$3"

  echo
  echo "Running $name..."
  peakfit fit \
    data/pseudo3d.ft2 \
    data/pseudo3d.list \
    --z-values data/b1_offsets.txt \
    --config "$config"

  python - "$out_dir" <<'PY'
import sys
from pathlib import Path
from peakfit.io.readers.results import ResultsLoader

out_dir = sys.argv[1]
chi2 = float(ResultsLoader(Path(out_dir)).load_summary().statistics.chi_squared)
print(f"  chi-squared: {chi2:.6g}")
PY
}

run_scenario "Scenario 1 (position windows)" "configs/position_windows.toml" "Fits/scenario1"
run_scenario "Scenario 2 (per-peak overrides)" "configs/per_peak.toml" "Fits/scenario2"
run_scenario "Scenario 3 (multi-step fit)" "configs/multi_step.toml" "Fits/scenario3"

echo
echo "All scenarios complete."
echo "Inspect outputs under Fits/scenario1, Fits/scenario2, Fits/scenario3"
