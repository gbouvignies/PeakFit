#!/bin/bash
# Example 3: Global Optimization for Difficult Peaks
# Demonstrates basin-hopping and differential evolution

set -e  # Exit on error

echo "=========================================="
echo "PeakFit Example 3: Global Optimization"
echo "=========================================="
echo
echo "This example compares local vs. global optimization methods."
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
echo "  3) Differential evolution (slowest, ~15-30 min)"
echo "  4) All three (for comparison, ~30-50 min total)"
echo
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo
        echo "Running LOCAL optimization..."
        echo
        rm -rf Fits-Local
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --output Fits-Local/

        echo
        echo "✓ Local optimization complete"
        echo "Results in: Fits-Local/"
        ;;

    2)
        echo
        echo "Running BASIN-HOPPING optimization..."
        echo "  This will take 10-20 minutes..."
        echo
        rm -rf Fits-BH
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --optimizer basin-hopping \
            --output Fits-BH/

        echo
        echo "✓ Basin-hopping complete"
        echo "Results in: Fits-BH/"
        ;;

    3)
        echo
        echo "Running DIFFERENTIAL EVOLUTION optimization..."
        echo "  This will take 15-30 minutes..."
        echo
        rm -rf Fits-DE
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --optimizer differential-evolution \
            --output Fits-DE/

        echo
        echo "✓ Differential evolution complete"
        echo "Results in: Fits-DE/"
        ;;

    4)
        echo
        echo "Running ALL optimizers (this will take 30-50 minutes)..."
        echo

        echo "Step 1/3: Local optimization (~2-3 min)..."
        rm -rf Fits-Local
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --output Fits-Local/
        echo "  ✓ Local complete"

        echo
        echo "Step 2/3: Basin-hopping (~10-20 min)..."
        rm -rf Fits-BH
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --optimizer basin-hopping \
            --output Fits-BH/
        echo "  ✓ Basin-hopping complete"

        echo
        echo "Step 3/3: Differential evolution (~15-30 min)..."
        rm -rf Fits-DE
        peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
            --z-values data/b1_offsets.txt \
            --optimizer differential-evolution \
            --output Fits-DE/
        echo "  ✓ Differential evolution complete"

        echo
        echo "=========================================="
        echo "All optimizers complete!"
        echo "=========================================="
        echo
        echo "Comparison:"
        echo
        grep "Successful" Fits-Local/peakfit.log | sed 's/^/  Local:        /'
        grep "Successful" Fits-BH/peakfit.log | sed 's/^/  Basin-hop:    /'
        grep "Successful" Fits-DE/peakfit.log | sed 's/^/  Diff-evol:    /'
        echo
        echo "Compare shifts:"
        echo "  diff Fits-Local/shifts.list Fits-BH/shifts.list"
        echo "  diff Fits-Local/shifts.list Fits-DE/shifts.list"
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
echo "  • View log: less Fits-*/peakfit.log"
echo "  • Compare results: diff Fits-Local/shifts.list Fits-BH/shifts.list"
echo "  • Check specific peak: cat Fits-*/10N-HN.out"
echo "  • See full README for analysis tips"
echo
