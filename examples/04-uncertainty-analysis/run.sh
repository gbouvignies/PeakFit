#!/bin/bash
# Example 4: Uncertainty Analysis
# Demonstrates uncertainty estimation methods

set -e  # Exit on error

echo "=========================================="
echo "PeakFit Example 4: Uncertainty Analysis"
echo "=========================================="
echo
echo "This example demonstrates methods for estimating parameter uncertainties."
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
echo "Which uncertainty analysis would you like to run?"
echo "  1) Quick uncertainty check (fast, ~2-3 min)"
echo "  2) MCMC uncertainty analysis (slow, ~5-10 min)"
echo "  3) Profile likelihood for specific parameter (~2-3 min)"
echo "  4) Full analysis suite (all methods, ~10-15 min)"
echo
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo
        echo "=========================================="
        echo "Step 1: Running Fit"
        echo "=========================================="
        echo
        rm -rf Fits
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --output Fits/

        echo
        echo "=========================================="
        echo "Step 2: Displaying Uncertainties"
        echo "=========================================="
        echo
        echo "Covariance-based uncertainties from least-squares fit:"
        echo
        peakfit analyze uncertainty Fits/ --output Fits/uncertainty_summary.txt

        echo
        echo "✓ Quick uncertainty check complete"
        echo
        echo "Results:"
        echo "  • Fit results: Fits/"
        echo "  • Uncertainty summary: Fits/uncertainty_summary.txt"
        echo
        echo "Next steps:"
        echo "  • Check uncertainty_summary.txt for detailed results"
        echo "  • Run MCMC (option 2) for more accurate uncertainties"
        ;;

    2)
        echo
        echo "=========================================="
        echo "Step 1: Running Fit (if needed)"
        echo "=========================================="
        echo
        if [ ! -d "Fits" ]; then
            peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
                --z-values data/b1_offsets.txt \
                --output Fits/
        else
            echo "Using existing fit results in Fits/"
        fi

        echo
        echo "=========================================="
        echo "Step 2: MCMC Uncertainty Analysis"
        echo "=========================================="
        echo "  This will take 5-10 minutes..."
        echo "  Using 32 walkers, 1000 steps, 200 burn-in"
        echo
        peakfit analyze mcmc Fits/ \
            --chains 32 \
            --samples 1000 \
            --burn-in 200 \
            --output Fits/mcmc_results.txt

        echo
        echo "✓ MCMC analysis complete"
        echo
        echo "Results:"
        echo "  • MCMC results: Fits/mcmc_results.txt"
        echo "  • Includes correlation matrices for each cluster"
        echo
        echo "The MCMC results include:"
        echo "  - Median values and std errors"
        echo "  - 68% confidence intervals (1 sigma)"
        echo "  - 95% confidence intervals (2 sigma)"
        echo "  - Correlation matrices showing parameter dependencies"
        ;;

    3)
        echo
        echo "=========================================="
        echo "Step 1: Running Fit (if needed)"
        echo "=========================================="
        echo
        if [ ! -d "Fits" ]; then
            peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
                --z-values data/b1_offsets.txt \
                --output Fits/
        else
            echo "Using existing fit results in Fits/"
        fi

        echo
        echo "=========================================="
        echo "Step 2: Parameter Selection"
        echo "=========================================="
        echo
        echo "Available parameters in first cluster:"
        echo "  (Check Fits/*.out files for parameter names)"
        echo
        echo "Enter parameter name to profile (e.g., 2N-H_x0):"
        read -p "Parameter: " param_name

        if [ -z "$param_name" ]; then
            echo "No parameter specified, using example: 2N-H_x0"
            param_name="2N-H_x0"
        fi

        echo
        echo "=========================================="
        echo "Step 3: Profile Likelihood Analysis"
        echo "=========================================="
        echo "  Computing profile for: $param_name"
        echo "  This will take 2-3 minutes..."
        echo
        peakfit analyze profile Fits/ \
            --param "$param_name" \
            --points 20 \
            --confidence 0.95 \
            --output "Fits/profile_${param_name}.txt"

        echo
        echo "✓ Profile likelihood complete"
        echo
        echo "Results:"
        echo "  • Profile data: Fits/profile_${param_name}.txt"
        echo
        echo "Profile likelihood provides more accurate confidence intervals"
        echo "than the covariance matrix, especially for non-linear parameters."
        ;;

    4)
        echo
        echo "Running FULL uncertainty analysis suite..."
        echo "This will take 10-15 minutes total."
        echo

        # Step 1: Fit
        echo "=========================================="
        echo "Step 1/4: Fitting Data"
        echo "=========================================="
        echo
        rm -rf Fits
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --output Fits/
        echo "  ✓ Fit complete"

        # Step 2: Quick uncertainty check
        echo
        echo "=========================================="
        echo "Step 2/4: Covariance-based Uncertainties"
        echo "=========================================="
        echo
        peakfit analyze uncertainty Fits/ --output Fits/uncertainty_summary.txt
        echo "  ✓ Uncertainty summary saved"

        # Step 3: Correlation analysis
        echo
        echo "=========================================="
        echo "Step 3/4: Parameter Correlation Analysis"
        echo "=========================================="
        echo
        peakfit analyze correlation Fits/ --output Fits/correlation_summary.txt
        echo "  ✓ Correlation analysis complete"

        # Step 4: MCMC
        echo
        echo "=========================================="
        echo "Step 4/4: MCMC Analysis (this takes ~5-10 min)"
        echo "=========================================="
        echo
        peakfit analyze mcmc Fits/ \
            --chains 32 \
            --samples 1000 \
            --burn-in 200 \
            --output Fits/mcmc_results.txt
        echo "  ✓ MCMC complete"

        echo
        echo "=========================================="
        echo "All Analyses Complete!"
        echo "=========================================="
        echo
        echo "Results summary:"
        echo
        echo "1. Fit results:              Fits/"
        echo "2. Quick uncertainties:      Fits/uncertainty_summary.txt"
        echo "3. Parameter correlations:   Fits/correlation_summary.txt"
        echo "4. MCMC full analysis:       Fits/mcmc_results.txt"
        echo
        echo "Comparison of methods:"
        echo "  • Covariance: Fast, assumes Gaussian errors"
        echo "  • MCMC: Slow, full posterior with correlations"
        echo
        echo "To view specific results:"
        echo "  cat Fits/uncertainty_summary.txt"
        echo "  cat Fits/mcmc_results.txt"
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo
echo "Understanding uncertainties:"
echo "  • Small errors (<1%): Well-determined parameters"
echo "  • Large errors (>10%): Consider MCMC or more data"
echo "  • At boundary: May need to adjust parameter bounds"
echo
echo "Advanced analysis:"
echo "  • Profile likelihood: peakfit analyze profile Fits/ --param PARAM_NAME"
echo "  • Full MCMC: peakfit analyze mcmc Fits/ --chains 64 --samples 2000"
echo
echo "Documentation:"
echo "  • See README.md for detailed explanations"
echo "  • Check example outputs in Fits/ directory"
echo
