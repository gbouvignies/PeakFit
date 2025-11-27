#!/bin/bash
# Example 1: Basic Fitting Template
# This is a template - you need to provide your own data files

set -e  # Exit on error

echo "=========================================="
echo "PeakFit Example 1: Basic Fitting"
echo "=========================================="
echo
echo "This is a template example."
echo "You need to add your own data files to run it."
echo

# Check if data files exist
if [ ! -f "data/spectrum.ft2" ] || [ ! -f "data/peaks.list" ]; then
    echo "❌ Data files not found!"
    echo
    echo "To use this example:"
    echo "  1. Copy your spectrum to:  data/spectrum.ft2"
    echo "  2. Copy your peak list to: data/peaks.list"
    echo "  3. (Optional) Z-values to: data/z_values.txt"
    echo
    echo "For a ready-to-run example with real data, see:"
    echo "  ../02-advanced-fitting/"
    echo
    exit 1
fi

echo "✓ Data files found"
echo

# Check if PeakFit is installed
if ! command -v peakfit &> /dev/null; then
    echo "Error: peakfit command not found"
    echo "Please install PeakFit first:"
    echo "  pip install peakfit"
    exit 1
fi

# Clean previous results
if [ -d "results" ]; then
    echo "Cleaning previous results..."
    rm -rf results
fi

echo "Step 1: Validating inputs..."
echo
peakfit validate data/spectrum.ft2 data/peaks.list

echo
echo "Step 2: Running fit..."
echo

# Check for Z-values file
if [ -f "data/z_values.txt" ]; then
    echo "  Detected Z-values file - running pseudo-3D fit"
    peakfit fit data/spectrum.ft2 data/peaks.list \
        --z-values data/z_values.txt \
        --output results/
else
    echo "  Running 2D fit (no Z-values file)"
    peakfit fit data/spectrum.ft2 data/peaks.list \
        --output results/
fi

echo
echo "=========================================="
echo "Fitting complete!"
echo "=========================================="
echo
echo "Results saved to: results/"
echo "  - results/*.out        : Individual peak profiles"
echo "  - results/shifts.list  : Fitted chemical shifts"
echo "  - results/peakfit.log  : Detailed log file"
echo
echo "Next steps:"
echo "  • View shifts: cat results/shifts.list"
echo "  • Check log: less results/peakfit.log"
echo
