# PeakFit Examples

This directory contains example files demonstrating PeakFit usage.

## CEST Analysis Example

This example shows how to fit a pseudo-3D CEST (Chemical Exchange Saturation Transfer) NMR spectrum.

### Files

- **pseudo3d.ft2** - Pseudo-3D NMRPipe spectrum (128 planes)
- **pseudo3d.list** - Peak list in Sparky format (name, F1 ppm, F2 ppm)
- **b1_offsets.txt** - Z-dimension values (B1 offsets in Hz)
- **run.sh** - Example script demonstrating the fitting workflow

### Quick Start

```bash
# Run the example
./run.sh

# Or manually:
peakfit fit pseudo3d.ft2 pseudo3d.list --z-values b1_offsets.txt --output Fits
```

### Using Configuration Files

You can also use a TOML configuration file for more control:

```bash
# Generate a default config
peakfit init peakfit.toml

# Edit the config as needed, then run:
peakfit fit pseudo3d.ft2 pseudo3d.list --z-values b1_offsets.txt --config peakfit.toml
```

Example configuration for CEST analysis:

```toml
[fitting]
lineshape = "auto"
refine_iterations = 1
fix_positions = false

[clustering]
contour_factor = 5.0

[output]
directory = "Fits"
formats = ["txt"]
save_simulated = true
save_html_report = true

exclude_planes = []
```

### Validate Input Files

Before fitting, you can validate your input files:

```bash
peakfit validate pseudo3d.ft2 pseudo3d.list
```

### Output Files

After running, the `Fits/` directory will contain:

- **{peak_name}.out** - Individual peak fitting results with intensity profiles
- **shifts.list** - Fitted chemical shift positions
- **simulated.ft2** - Reconstructed spectrum from fitted parameters
- **logs.html** - HTML report with detailed fitting information

### Plotting Results

Generate CEST profiles:

```bash
# Plot all peaks with reference at plane 0
peakfit-plot cest -f Fits/*N-H.out --ref 0

# Plot specific peaks
peakfit-plot cest -f Fits/10N-H.out Fits/15N-H.out --ref 0
```

### Advanced Options

```bash
# Fix peak positions during fitting
peakfit fit pseudo3d.ft2 pseudo3d.list --z-values b1_offsets.txt --fixed

# Use more refinement iterations
peakfit fit pseudo3d.ft2 pseudo3d.list --z-values b1_offsets.txt --refine 3

# Force specific lineshape model
peakfit fit pseudo3d.ft2 pseudo3d.list --z-values b1_offsets.txt --lineshape pvoigt

# Exclude noisy planes
peakfit fit pseudo3d.ft2 pseudo3d.list --z-values b1_offsets.txt --exclude 0 --exclude 127
```

## Peak List Format

The peak list uses Sparky format:

```
Assignment  F1(ppm)   F2(ppm)
2N-H        115.630   6.868
3N-H        117.519   8.693
...
```

## Z-Dimension Values

The Z-values file contains one value per line, corresponding to each plane in the pseudo-3D spectrum:

```
-5000
-4500
-4000
...
```

For CEST experiments, these are typically B1 offsets in Hz.
