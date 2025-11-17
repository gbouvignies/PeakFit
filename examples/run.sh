#!/bin/bash

# PeakFit Example - CEST Analysis
# This script demonstrates fitting a pseudo-3D CEST NMR spectrum

# Clean up previous results
rm -rf Fits

# New CLI syntax (recommended)
peakfit fit \
    pseudo3d.ft2 \
    pseudo3d.list \
    --z-values b1_offsets.txt \
    --output Fits

# Plot CEST profiles
peakfit-plot cest -f Fits/*N-H.out --ref 0
