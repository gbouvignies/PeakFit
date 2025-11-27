#!/bin/bash
# Example 3: Global Optimization for Difficult Peaks
# Compares local vs global optimization with structured output comparison

set -e  # Exit on error

echo "=========================================="
echo "PeakFit Example 3: Global Optimization"
echo "=========================================="
echo
echo "This example compares local vs. global optimization methods"
echo "using the new structured JSON outputs for comparison."
echo
echo "⚠️  Warning: Global optimization is MUCH slower (10-30 minutes)"
echo

# Check if PeakFit is installed
if ! command -v peakfit &> /dev/null; then
    echo "Error: peakfit command not found"
    echo "Please install PeakFit first:"
    echo "  pip install peakfit"
    exit 1
fi

# Check data files exist
if [ ! -f "data/pseudo3d.ft2" ]; then
    echo "Error: Data files not found"
    echo "Make sure symbolic links are set up correctly"
    exit 1
fi

# Ask user which method to run
echo "Which optimization method would you like to run?"
echo "  1) Local optimization (fast, ~2-3 min) - baseline"
echo "  2) Basin-hopping (slow, ~10-20 min)"
echo "  3) Both and compare results"
echo
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo
        echo "=========================================="
        echo "Running LOCAL optimization..."
        echo "=========================================="
        echo
        rm -rf Fits-Local
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --output Fits-Local/

        echo
        echo "✓ Local optimization complete"
        echo
        echo "Results in: Fits-Local/"
        echo "  - fit_results.json  : Complete results"
        echo "  - parameters.csv    : Parameter table"
        echo "  - report.md         : Summary"
        ;;

    2)
        echo
        echo "=========================================="
        echo "Running BASIN-HOPPING optimization..."
        echo "  This will take 10-20 minutes..."
        echo "=========================================="
        echo
        rm -rf Fits-BH
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --optimizer basin_hopping \
            --output Fits-BH/

        echo
        echo "✓ Basin-hopping complete"
        echo
        echo "Results in: Fits-BH/"
        echo "  - fit_results.json  : Complete results"
        echo "  - parameters.csv    : Parameter table"
        echo "  - report.md         : Summary"
        ;;

    3)
        echo
        echo "=========================================="
        echo "Running BOTH optimizers for comparison"
        echo "=========================================="
        echo

        echo "Step 1/2: Local optimization (~2-3 min)..."
        rm -rf Fits-Local
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --output Fits-Local/
        echo "  ✓ Local complete"

        echo
        echo "Step 2/2: Basin-hopping (~10-20 min)..."
        rm -rf Fits-BH
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --optimizer basin_hopping \
            --output Fits-BH/
        echo "  ✓ Basin-hopping complete"

        echo
        echo "=========================================="
        echo "Comparison of Results"
        echo "=========================================="
        echo
        echo "Compare JSON results:"
        echo "  diff Fits-Local/fit_results.json Fits-BH/fit_results.json"
        echo
        echo "Compare parameter CSVs:"
        echo "  diff Fits-Local/parameters.csv Fits-BH/parameters.csv"
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo
echo "  • Compare results: diff Fits-Local/fit_results.json Fits-BH/fit_results.json"
echo "  • Compare CSVs: diff Fits-Local/parameters.csv Fits-BH/parameters.csv"
echo "  • Check logs: less Fits-*/peakfit.log"
echo "  • See README.md for more analysis examples"
echo
