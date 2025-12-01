#!/bin/bash
# Example 4: Uncertainty Analysis with MCMC
# Demonstrates MCMC-based uncertainty estimation with structured outputs

set -e  # Exit on error

echo "=========================================="
echo "PeakFit Example 4: MCMC Uncertainty"
echo "=========================================="
echo
echo "This example demonstrates MCMC-based uncertainty estimation"
echo "with comprehensive diagnostic outputs."
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

# Ask user which analysis to run
echo "Which analysis would you like to run?"
echo "  1) Quick fit with uncertainties (~2-3 min)"
echo "  2) Full MCMC analysis with chain storage (~10-15 min)"
echo
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo
        echo "=========================================="
        echo "Running fit with standard uncertainties"
        echo "=========================================="
        echo
        rm -rf Fits
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --output Fits/

        echo
        echo "=========================================="
        echo "Fit complete!"
        echo "=========================================="
        echo
        echo "Output files:"
        echo "  • Fits/fit_results.json  - Structured results with uncertainties"
        echo "  • Fits/parameters.csv    - All parameters"
        echo "  • Fits/shifts.csv        - Chemical shifts"
        echo "  • Fits/intensities.csv   - Fitted intensities"
        echo "  • Fits/report.md         - Human-readable report"
        echo
        echo "View uncertainties:"
        echo "  jq '.clusters[0].parameters' Fits/fit_results.json"
        echo
        ;;

    2)
        echo
        echo "=========================================="
        echo "Running MCMC analysis"
        echo "=========================================="
        echo
        echo "Note: MCMC fitting is run via 'peakfit analyze mcmc' after initial fitting."
        echo "First running standard fit..."
        echo
        rm -rf Fits
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --output Fits/

        echo
        echo "Now running MCMC analysis (this takes ~5-10 min)..."
        echo
        peakfit analyze mcmc Fits/ --chains 32 --samples 1000

        echo
        echo "=========================================="
        echo "MCMC analysis complete!"
        echo "=========================================="
        echo
        echo "Output files:"
        echo "  • Fits/*.out  - Peak profiles with uncertainties"
        echo
        ;;

    *)
        echo "Invalid choice. Please run again and select 1 or 2."
        exit 1
        ;;
esac

echo "=========================================="
echo "Next steps:"
echo "  • View results: ls -la Fits/"
echo "  • Check log: less Fits/peakfit.log"
echo "=========================================="
