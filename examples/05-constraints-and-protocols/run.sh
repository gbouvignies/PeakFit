#!/bin/bash
# Example 5: Parameter Constraints and Multi-Step Protocols
# This script demonstrates advanced parameter control in PeakFit

set -e  # Exit on error

echo "=========================================="
echo "PeakFit Example 5: Constraints & Protocols"
echo "=========================================="
echo

# Check if PeakFit is installed
if ! command -v peakfit &> /dev/null; then
    echo "Error: peakfit command not found"
    echo "Please install PeakFit first:"
    echo "  uv sync --extra dev    # Recommended development install"
    echo "  OR from repository: pip install -e ."
    exit 1
fi

# Check data files exist (symlink to example 2 data)
if [ ! -f "data/pseudo3d.ft2" ]; then
    echo "Error: Data files not found"
    echo "Make sure you're running this script from the 05-constraints-and-protocols/ directory"
    exit 1
fi

# Clean previous results
if [ -d "Fits" ]; then
    echo "Cleaning previous results..."
    rm -rf Fits
fi

echo "This example runs 3 scenarios demonstrating:"
echo "  1. Position windows (global and per-dimension)"
echo "  2. Per-peak constraints"
echo "  3. Multi-step fitting protocol"
echo

# ============================================================
# Scenario 1: Position Windows
# ============================================================
echo "=========================================="
echo "Scenario 1: Position Windows"
echo "=========================================="
echo
echo "Using per-dimension position constraints:"
echo "  - F2 (15N): ±0.3 ppm"
echo "  - F3 (1H):  ±0.03 ppm"
echo

peakfit fit \
    data/pseudo3d.ft2 \
    data/pseudo3d.list \
    --z-values data/b1_offsets.txt \
    --config configs/position_windows.toml

echo
echo "✓ Scenario 1 complete: Fits/scenario1/"
echo

# ============================================================
# Scenario 2: Per-Peak Constraints
# ============================================================
echo "=========================================="
echo "Scenario 2: Per-Peak Constraints"
echo "=========================================="
echo
echo "Some peaks have special constraints:"
echo "  - Fix positions for well-resolved peaks"
echo "  - Wider windows for peaks that need to move"
echo

peakfit fit \
    data/pseudo3d.ft2 \
    data/pseudo3d.list \
    --z-values data/b1_offsets.txt \
    --config configs/per_peak.toml

echo
echo "✓ Scenario 2 complete: Fits/scenario2/"
echo

# ============================================================
# Scenario 3: Multi-Step Protocol
# ============================================================
echo "=========================================="
echo "Scenario 3: Multi-Step Protocol"
echo "=========================================="
echo
echo "Staged fitting approach:"
echo "  Step 1: Fix positions, optimize linewidths"
echo "  Step 2: Release positions with constraints"
echo "  Step 3: Full refinement"
echo

peakfit fit \
    data/pseudo3d.ft2 \
    data/pseudo3d.list \
    --z-values data/b1_offsets.txt \
    --config configs/multi_step.toml

echo
echo "✓ Scenario 3 complete: Fits/scenario3/"
echo

# ============================================================
# Summary
# ============================================================
echo
echo "=========================================="
echo "All scenarios complete!"
echo "=========================================="
echo
echo "Results saved to:"
echo
echo "  Scenario 1 (Position Windows):"
echo "    - Fits/scenario1/fit_results.json"
echo "    - Fits/scenario1/parameters.csv"
echo
echo "  Scenario 2 (Per-Peak Constraints):"
echo "    - Fits/scenario2/fit_results.json"
echo "    - Fits/scenario2/parameters.csv"
echo
echo "  Scenario 3 (Multi-Step Protocol):"
echo "    - Fits/scenario3/fit_results.json"
echo "    - Fits/scenario3/parameters.csv"
echo
echo "Compare results:"
echo "  • View parameters: cat Fits/scenario1/parameters.csv | head"
echo "  • Check logs: less Fits/scenario3/peakfit.log"
echo "  • Compare shifts: diff Fits/scenario1/shifts.csv Fits/scenario3/shifts.csv"
echo
echo "Configuration files:"
echo "  • configs/position_windows.toml"
echo "  • configs/per_peak.toml"
echo "  • configs/multi_step.toml"
echo
