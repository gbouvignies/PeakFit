#!/bin/bash

rm -rf Fits

peakfit  \
    -s pseudo3d.ft2 \
    -l pseudo3d.list \
    -z b1_offsets.txt \
    -o Fits

peakfit-plot cest -f Fits/*N-H.out --ref 0
