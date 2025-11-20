# PeakFit Terminal Output Specification

This document defines the **exact** terminal output format for all PeakFit commands. Every command should follow these specifications for consistent, professional, and user-friendly output.

## Design Principles

1. **Clarity**: Users should immediately understand what's happening
2. **Consistency**: All commands use the same visual language (icons, tables, colors)
3. **Actionability**: Errors include suggestions, successes include next steps
4. **Professionalism**: Clean tables, proper spacing, structured logs
5. **Information Density**: Show important info without overwhelming the user

## Visual Elements

### Icons (Consistent Across All Commands)
- `✓` Success (green)
- `⚠` Warning (yellow)
- `✗` Error (red)
- `ℹ` Info (cyan)
- `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` Spinner (for progress)
- `━` Progress bars
- `│` Separators
- `‣` Bullets

### Table Style
- Box: `ROUNDED` (`┏━┓┃┗━┛┡━┩│─`)
- Header: Bold cyan
- Border: Dim
- Cell alignment: Left for text, right for numbers

---

## Command: `peakfit fit`

### Success Case

```
🎯 PeakFit v2025.11.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Loading Data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Loaded spectrum: pseudo3d.ft2
  ‣ Shape: (20, 128, 2048)
  ‣ Z-values: 20 planes

✓ Noise level: 234567.12

  ‣ Lineshapes: sp1

✓ Loaded 147 peaks
  ‣ Contour level: 1172835.60

✓ Created 45 clusters

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property             ┃ Value                ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Spectrum shape       │ (20, 128, 2048)      │
│ Number of planes     │ 20                   │
│ Number of peaks      │ 147                  │
│ Number of clusters   │ 45                   │
│ Noise level          │ 234567.1200          │
│ Contour level        │ 1172835.6000         │
└──────────────────────┴──────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Fitting Clusters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

────────────────────────────────────────────────────────────
Cluster 1/45 │ Peaks: A45N-HN, A46N-HN
────────────────────────────────────────────────────────────
✓ Converged │ Cost: 3.421e+05 │ Evaluations: 127

────────────────────────────────────────────────────────────
Cluster 2/45 │ Peaks: G10N-HN
────────────────────────────────────────────────────────────
✓ Converged │ Cost: 1.234e+05 │ Evaluations: 89

[... continues for all clusters ...]

────────────────────────────────────────────────────────────
Cluster 23/45 │ Peaks: K15N-HN, K16N-HN, K17N-HN
────────────────────────────────────────────────────────────
⚠ Maximum iterations reached │ Cost: 8.765e+05 │ Evaluations: 1000

╭─ Fitting Challenge Detected ────────────────────────────────╮
│ Cluster 23 did not fully converge.                          │
│                                                              │
│ Suggestions:                                                 │
│   • Try global optimization:                                 │
│     peakfit fit ... --optimizer basin-hopping                │
│   • Increase iterations:                                     │
│     peakfit fit ... --max-iterations 5000                    │
│   • Check peak positions and overlaps manually              │
╰──────────────────────────────────────────────────────────────╯

[... continues with remaining clusters ...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Saving Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Profiles written
  ‣ Fits/*.out
✓ Shifts written
  ‣ Fits/shifts.list
✓ Fitting state saved
  ‣ Fits/.peakfit_state.pkl
  ‣ Use 'peakfit analyze' to compute uncertainties

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric               ┃ Value                ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Total clusters       │ 45                   │
│ Successful fits      │ 43 (95.6%)           │
│ Failed fits          │ 2 (4.4%)             │
│ Total peaks          │ 147                  │
│ Total time           │ 2m 34s               │
│ Time per cluster     │ 3.4s                 │
└──────────────────────┴──────────────────────┘

✓ Fitting complete!

📋 Next steps:
  1. Plot intensity profiles: peakfit plot intensity Fits/
  2. View results: peakfit plot spectra Fits/ --spectrum pseudo3d.ft2
  3. Uncertainty analysis: peakfit analyze mcmc Fits/
  4. Check failed fits in: Fits/peakfit.log

```

---

## Command: `peakfit validate`

### Success Case

```
🎯 PeakFit v2025.11.0 - Input Validation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Validating Input Files
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ℹ  Checking spectrum: pseudo3d.ft2
✓ Spectrum readable - Shape: (20, 128, 2048)

ℹ  Checking peak list: pseudo3d.list
✓ Peak list readable - 147 peaks found

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property                         ┃ Value                 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ Spectrum shape                   │ (20, 128, 2048)       │
│ Dimensions                       │ 3                     │
│ Type                             │ 3D (20 planes)        │
│ Peaks                            │ 147                   │
│ X range (ppm)                    │ 105.23 to 131.78      │
│ Y range (ppm)                    │ 7.12 to 9.54          │
└──────────────────────────────────┴───────────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Check                            ┃ Status  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Peaks within spectral bounds     │ ✓ Pass  │
│ Peak list dimensions match spec  │ ✓ Pass  │
│ No duplicate peaks               │ ✓ Pass  │
│ File permissions                 │ ✓ Pass  │
└──────────────────────────────────┴─────────┘

✓ All validation checks passed!

ℹ  Ready for fitting. Run:
    peakfit fit pseudo3d.ft2 pseudo3d.list
```

### Error Case

```
🎯 PeakFit v2025.11.0 - Input Validation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Validating Input Files
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✗ File not found: specturm.ft2

╭─ Suggestion ────────────────────────────────────────────────╮
│ Did you mean: spectrum.ft2?                                  │
│                                                              │
│ Available .ft2 files in current directory:                   │
│   • pseudo3d.ft2                                             │
│   • test_spectrum.ft2                                        │
│   • cest_data.ft2                                            │
╰──────────────────────────────────────────────────────────────╯

ℹ  Use 'peakfit validate SPECTRUM PEAKLIST' to check inputs before fitting
```

---

## Command: `peakfit plot intensity`

### Success Case

```
🎯 PeakFit v2025.11.0 - Plotting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Generating Intensity Profile Plots
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Found 147 result files

✓ Saving plots to: intensity_profiles.pdf

⠹ Generating plots... ━━━━━━━━━━━━━━━━╸━━━━━ 65% ETA: 0:00:08
   Current: A67N-HN (96/147)

✓ Generated 147 plots in 23.4s

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Output               ┃ Details              ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ PDF file             │ intensity_profiles...│
│ Total plots          │ 147                  │
│ Pages                │ 147                  │
│ File size            │ 4.2 MB               │
└──────────────────────┴──────────────────────┘

✓ Plots saved successfully!

📋 Next steps:
  1. Open PDF: open intensity_profiles.pdf
  2. Plot CEST profiles: peakfit plot cest Fits/
  3. Interactive viewer: peakfit plot spectra Fits/ --spectrum pseudo3d.ft2
```

---

## Command: `peakfit plot cest`

```
🎯 PeakFit v2025.11.0 - CEST Profiles

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Generating CEST Profile Plots
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Found 147 result files

ℹ  Reference points: Auto-detect (|offset| >= 10 kHz)

✓ Saving plots to: cest_profiles.pdf

⠙ Normalizing and plotting... ━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

✓ Generated 147 CEST profiles

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Output               ┃ Details              ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ PDF file             │ cest_profiles.pdf    │
│ Total plots          │ 147                  │
│ Normalization        │ Auto (|ν| >= 10kHz)  │
│ File size            │ 3.8 MB               │
└──────────────────────┴──────────────────────┘

✓ CEST plots saved successfully!
```

---

## Command: `peakfit analyze mcmc`

### Success Case

```
🎯 PeakFit v2025.11.0 - Uncertainty Analysis

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Running MCMC Uncertainty Estimation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Loaded fitting state: Fits/.peakfit_state.pkl
  Clusters: 45
  Peaks: 147
  Parameters: 882

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Configuration        ┃ Value                ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Method               │ MCMC (emcee)         │
│ Walkers              │ 32                   │
│ Steps                │ 1000                 │
│ Burn-in              │ 200                  │
│ Total samples        │ 25,600               │
└──────────────────────┴──────────────────────┘

Cluster 1/45: A45N-HN, A46N-HN
  ⠹ Sampling posterior distribution...

┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Parameter    ┃ Value     ┃ Std Error ┃ 68% CI         ┃ 95% CI         ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ peak1_x0     │ 118.452   │ 0.012     │ [118.44,118.46]│ [118.43,118.48]│
│ peak1_x_fwhm │ 15.234    │ 0.234     │ [15.00, 15.47] │ [14.78, 15.69] │
│ peak1_y0     │ 8.234     │ 0.008     │ [8.226, 8.242] │ [8.218, 8.250] │
│ peak1_y_fwhm │ 12.456    │ 0.156     │ [12.30, 12.61] │ [12.15, 12.76] │
[... more parameters ...]
└──────────────┴───────────┴───────────┴────────────────┴────────────────┘

[... continues for all clusters ...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Summary              ┃ Value                ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Clusters analyzed    │ 45                   │
│ Total samples        │ 1,152,000            │
│ Mean accept. rate    │ 0.52 ± 0.08          │
│ Converged chains     │ 45 (100%)            │
│ Total time           │ 8m 23s               │
│ Time per cluster     │ 11.2s                │
└──────────────────────┴──────────────────────┘

✓ Updated output files with MCMC uncertainties

✓ MCMC analysis complete!

📋 Next steps:
  1. Review updated uncertainties in: Fits/*.out
  2. Plot corner plots: peakfit analyze correlation Fits/
  3. Profile likelihood for specific param: peakfit analyze profile Fits/ --param peak1_x0
```

---

## Log File Format: `Fits/peakfit.log`

```
2024-11-19 14:23:45 | INFO  | ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2024-11-19 14:23:45 | INFO  | PeakFit v2025.11.0 - Fitting Session Started
2024-11-19 14:23:45 | INFO  | ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2024-11-19 14:23:45 | INFO  | Command: peakfit fit pseudo3d.ft2 pseudo3d.list
2024-11-19 14:23:45 | INFO  | Working directory: /home/user/PeakFit/examples
2024-11-19 14:23:45 | INFO  | Python: 3.13.0 | Platform: Linux-4.4.0-x86_64
2024-11-19 14:23:45 | INFO  |
2024-11-19 14:23:46 | INFO  | === LOADING DATA ===
2024-11-19 14:23:46 | INFO  | Spectrum: pseudo3d.ft2
2024-11-19 14:23:46 | INFO  |   - Dimensions: (20, 128, 2048)
2024-11-19 14:23:46 | INFO  |   - Size: 70.0 MB
2024-11-19 14:23:46 | INFO  |   - Data type: float32
2024-11-19 14:23:46 | INFO  | Peak list: pseudo3d.list
2024-11-19 14:23:46 | INFO  |   - Format: Sparky
2024-11-19 14:23:46 | INFO  |   - Peaks: 147
2024-11-19 14:23:46 | INFO  |
2024-11-19 14:23:47 | INFO  | === NOISE ESTIMATION ===
2024-11-19 14:23:47 | INFO  | Method: Median Absolute Deviation (MAD)
2024-11-19 14:23:47 | INFO  | Noise level: 234567.12
2024-11-19 14:23:47 | INFO  |
2024-11-19 14:23:47 | INFO  | === LINESHAPE DETECTION ===
2024-11-19 14:23:47 | INFO  | Detected apodization: SP (sine-bell, power=1)
2024-11-19 14:23:47 | INFO  | Selected lineshape: sp1
2024-11-19 14:23:47 | INFO  |
2024-11-19 14:23:47 | INFO  | === CLUSTERING ===
2024-11-19 14:23:47 | INFO  | Algorithm: DBSCAN
2024-11-19 14:23:47 | INFO  | Contour level: 1172835.60 (5.0 * noise)
2024-11-19 14:23:47 | INFO  | Parameters: eps=5.0, min_samples=1
2024-11-19 14:23:47 | INFO  | Identified 45 clusters
2024-11-19 14:23:47 | INFO  | Cluster size distribution:
2024-11-19 14:23:47 | INFO  |   - Min: 1 peak
2024-11-19 14:23:47 | INFO  |   - Max: 8 peaks
2024-11-19 14:23:47 | INFO  |   - Median: 3 peaks
2024-11-19 14:23:47 | INFO  |
2024-11-19 14:23:47 | INFO  | === FITTING ===
2024-11-19 14:23:47 | INFO  | Optimizer: least_squares (scipy)
2024-11-19 14:23:47 | INFO  | Backend: numba
2024-11-19 14:23:47 | INFO  | Parallel: disabled
2024-11-19 14:23:47 | INFO  | Tolerances: ftol=1e-7, xtol=1e-7
2024-11-19 14:23:47 | INFO  | Max iterations: 1000
2024-11-19 14:23:47 | INFO  |
2024-11-19 14:23:48 | INFO  | Cluster 1/45: A45N-HN, A46N-HN
2024-11-19 14:23:48 | INFO  |   - Peaks: 2
2024-11-19 14:23:48 | INFO  |   - Varying parameters: 40
2024-11-19 14:23:50 | INFO  |   - Status: Converged
2024-11-19 14:23:50 | INFO  |   - Cost: 3.421e+05
2024-11-19 14:23:50 | INFO  |   - Function evaluations: 127
2024-11-19 14:23:50 | INFO  |   - Time: 2.3s
2024-11-19 14:23:50 | INFO  |
2024-11-19 14:23:50 | INFO  | Cluster 2/45: G10N-HN
2024-11-19 14:23:50 | INFO  |   - Peaks: 1
2024-11-19 14:23:50 | INFO  |   - Varying parameters: 20
2024-11-19 14:23:51 | INFO  |   - Status: Converged
2024-11-19 14:23:51 | INFO  |   - Cost: 1.234e+05
2024-11-19 14:23:51 | INFO  |   - Function evaluations: 89
2024-11-19 14:23:51 | INFO  |   - Time: 1.8s
2024-11-19 14:23:51 | INFO  |
[... continues for all clusters ...]
2024-11-19 14:25:12 | WARN  | Cluster 23/45: K15N-HN, K16N-HN, K17N-HN
2024-11-19 14:25:12 | WARN  |   - Peaks: 3
2024-11-19 14:25:12 | WARN  |   - Varying parameters: 60
2024-11-19 14:25:12 | WARN  |   - Status: Maximum iterations reached
2024-11-19 14:25:12 | WARN  |   - Cost: 8.765e+05
2024-11-19 14:25:12 | WARN  |   - Function evaluations: 1000
2024-11-19 14:25:12 | WARN  |   - Time: 4.5s
2024-11-19 14:25:12 | WARN  |   - Suggestion: Try --optimizer basin-hopping or increase --max-iterations
2024-11-19 14:25:12 | INFO  |
[... continues for remaining clusters ...]
2024-11-19 14:26:21 | INFO  |
2024-11-19 14:26:21 | INFO  | === RESULTS SUMMARY ===
2024-11-19 14:26:21 | INFO  | Total clusters: 45
2024-11-19 14:26:21 | INFO  | Successful fits: 43 (95.6%)
2024-11-19 14:26:21 | INFO  | Failed fits: 2 (4.4%)
2024-11-19 14:26:21 | INFO  | Total peaks: 147
2024-11-19 14:26:21 | INFO  | Total time: 154s (2m 34s)
2024-11-19 14:26:21 | INFO  | Average time per cluster: 3.4s
2024-11-19 14:26:21 | INFO  |
2024-11-19 14:26:21 | INFO  | === OUTPUT FILES ===
2024-11-19 14:26:21 | INFO  | Output directory: Fits/
2024-11-19 14:26:21 | INFO  | Profile files: 147 *.out files
2024-11-19 14:26:21 | INFO  | Shifts file: Fits/shifts.list
2024-11-19 14:26:21 | INFO  | State file: Fits/.peakfit_state.pkl
2024-11-19 14:26:21 | INFO  | Log file: Fits/peakfit.log
2024-11-19 14:26:21 | INFO  |
2024-11-19 14:26:21 | INFO  | ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2024-11-19 14:26:21 | INFO  | PeakFit Session Completed Successfully
2024-11-19 14:26:21 | INFO  | ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Enhance `PeakFitUI` class with logging capabilities
- [ ] Add `setup_logging()` method
- [ ] Add `log()` method that outputs to both console and file
- [ ] Add structured table helpers
- [ ] Add progress bar helpers with detailed status

### Phase 2: Command Refactoring
- [ ] Refactor `fit_command.py` to use new output system
- [ ] Refactor `validate_command.py` to use new output system
- [ ] Refactor `plot_command.py` to use new output system
- [ ] Refactor `analyze_command.py` to use new output system

### Phase 3: Testing and Documentation
- [ ] Create test suite for output formatting
- [ ] Test all commands with example data
- [ ] Create BEFORE_AFTER.md with screenshots
- [ ] Update user documentation

### Phase 4: Polish
- [ ] Ensure all error messages have suggestions
- [ ] Ensure all success outputs have next steps
- [ ] Verify log file structure is parseable
- [ ] Check consistency across all commands

---

## Acceptance Criteria

1. **Visual Consistency**
   - [ ] All commands use same table style (ROUNDED)
   - [ ] All commands use same icons (✓ ⚠ ✗ ℹ)
   - [ ] All commands have consistent spacing
   - [ ] All commands show headers with ━ separators

2. **Information Quality**
   - [ ] Progress bars show meaningful context
   - [ ] Tables show all relevant metrics
   - [ ] Errors always include suggestions
   - [ ] Successes always include next steps

3. **Log Files**
   - [ ] Structured format (timestamp | level | message)
   - [ ] All important events logged
   - [ ] Clear section headers (===)
   - [ ] Parseable by standard tools

4. **User Experience**
   - [ ] Clear what's happening at each step
   - [ ] Easy to understand what to do next
   - [ ] Helpful when things go wrong
   - [ ] Professional appearance

---

## Future Enhancements

- [ ] Add `--json` flag for machine-readable output
- [ ] Add `--quiet` flag for minimal output
- [ ] Add `--progress-style` option (auto/plain/fancy)
- [ ] Add color theme customization
- [ ] Export logs to JSON/CSV for analysis
