#!/bin/bash
# Example 2: Advanced Fitting with CEST Analysis
# This script demonstrates fitting a pseudo-3D CEST NMR spectrum

set -e  # Exit on error

echo "=========================================="
echo "PeakFit Example 2: CEST Analysis"
echo "=========================================="
echo

# Check if PeakFit is installed
if ! command -v peakfit &> /dev/null; then
    echo "Error: peakfit command not found"
    echo "Please install PeakFit first:"
    echo "  pip install peakfit"
    echo "  OR from repository: pip install -e ."
    exit 1
fi

# Check data files exist
if [ ! -f "data/pseudo3d.ft2" ]; then
    echo "Error: Data files not found"
    echo "Make sure you're running this script from the 02-advanced-fitting/ directory"
    exit 1
fi

# Clean previous results
if [ -d "Fits" ]; then
    echo "Cleaning previous results..."
    rm -rf Fits
fi

echo "Step 1: Validating inputs..."
echo
peakfit validate data/pseudo3d.ft2 data/pseudo3d.list

echo
echo "Step 2: Running CEST fit..."
echo "  This will fit 166 peaks across 131 planes (~2-3 minutes)"
echo
peakfit fit \
    data/pseudo3d.ft2 \
    data/pseudo3d.list \
    --z-values data/b1_offsets.txt \
    --output Fits/

echo
echo "=========================================="
echo "Fitting complete!"
echo "=========================================="
echo
echo "Results saved to: Fits/"
echo "  - Fits/*.out        : Individual peak profiles"
echo "  - Fits/shifts.list  : Fitted chemical shifts"
echo "  - Fits/peakfit.log  : Detailed log file"
echo
echo "Next steps:"
echo "  • View log: less Fits/peakfit.log"
echo "  • Check shifts: cat Fits/shifts.list"
echo "  • Plot CEST profile: peakfit-plot cest -f Fits/*N-HN.out --ref 0"
echo "  • Open HTML report: open Fits/logs.html (if available)"
echo
