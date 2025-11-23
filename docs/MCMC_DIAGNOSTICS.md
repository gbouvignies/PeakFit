# MCMC Diagnostics in PeakFit

This document explains how to use and interpret MCMC diagnostics in PeakFit, following the **Bayesian Analysis Reporting Guidelines (BARG)** by Kruschke (2021).

## Overview

PeakFit now includes comprehensive MCMC diagnostics to help you:
- **Assess convergence**: Has the MCMC sampling reached the posterior distribution?
- **Evaluate sample quality**: Are there enough effective samples for reliable estimates?
- **Identify correlations**: Which parameters are correlated in the posterior?
- **Visualize posteriors**: What do the posterior distributions look like?

## Quick Start

### 1. Run MCMC Analysis

```bash
# Basic MCMC analysis (uses adaptive burn-in by default)
peakfit analyze mcmc Fits/

# Custom MCMC settings for better convergence (still uses adaptive burn-in)
peakfit analyze mcmc Fits/ --walkers 64 --steps 2000

# Manual burn-in override (disables adaptive determination)
peakfit analyze mcmc Fits/ --walkers 64 --steps 2000 --burn-in 500

# Disable adaptive burn-in (uses default 200 steps)
peakfit analyze mcmc Fits/ --no-auto-burnin

# Analyze specific peaks only
peakfit analyze mcmc Fits/ --peaks 2N-H 5L-H
```

This command will:
- Sample the posterior distribution using MCMC
- **Automatically determine burn-in** using R-hat convergence monitoring (new!)
- Compute R-hat and ESS diagnostics
- Display convergence statistics
- Save chain data for plotting

**What is Adaptive Burn-in?**

Instead of using a fixed burn-in period (e.g., 200 steps), PeakFit now automatically determines the optimal burn-in by monitoring when R-hat drops below 1.05 for all parameters. This follows modern Bayesian workflow best practices and ensures you don't discard too many or too few samples.

The adaptive burn-in report shows:
- **Convergence step**: When R-hat first reached the threshold
- **Method**: How burn-in was determined (R-hat monitoring or default fallback)
- **Max R-hat**: R-hat value at the convergence point
- **Validation warnings**: If burn-in exceeds 50% of chain (suggests running longer chains)

### 2. Generate Diagnostic Plots

```bash
# Create comprehensive diagnostic PDF
peakfit plot diagnostics Fits/

# Custom output file
peakfit plot diagnostics Fits/ --output my_diagnostics.pdf

# Plot specific peaks only
peakfit plot diagnostics Fits/ --peaks 2N-H
```

The diagnostic PDF includes:
- **Trace plots**: Chain evolution over iterations
- **Corner plots**: Marginal and joint posterior distributions
- **Autocorrelation plots**: Mixing efficiency
- **Posterior summary**: Compact overview of all parameters

## Understanding the Diagnostics

### R-hat (Gelman-Rubin Statistic)

**What it measures**: Ratio of between-chain variance to within-chain variance

**Interpretation**:
- **R-hat ≤ 1.01**: ✓ Excellent convergence
- **1.01 < R-hat ≤ 1.05**: ⚠ Marginal convergence (may be acceptable)
- **R-hat > 1.05**: ✗ Poor convergence (do not trust results!)

**What to do if R-hat is high**:
1. Increase `--steps` (e.g., from 1000 to 5000)
2. Increase `--walkers` (e.g., from 32 to 64)
3. Check if parameters are hitting bounds (may need wider bounds)
4. Try different starting values

### ESS (Effective Sample Size)

**What it measures**: Number of "independent" samples in your MCMC chains

**Interpretation**:
- **ESS_bulk ≥ 100 × chains**: ✓ Good (recommended for stable credible intervals)
- **ESS_bulk ≥ 10 × chains**: ⚠ Marginal (rough estimates only)
- **ESS_bulk < 10 × chains**: ✗ Poor (highly uncertain estimates)

**Example**: With 32 chains:
- ESS ≥ 3,200: Excellent
- ESS ≥ 320: Acceptable for rough estimates
- ESS < 320: Problematic

**What to do if ESS is low**:
1. Increase `--steps` to collect more samples
2. Check autocorrelation plots (high autocorrelation = poor mixing)
3. Consider reparameterization if correlations are very strong

### Trace Plots

**What to look for**:
- ✓ **Good mixing**: Chains overlap and look like "fuzzy caterpillars"
- ✓ **Stationary**: No trends, drifts, or long-term correlations
- ✗ **Bad mixing**: Chains separated, trending, or stuck

**Visual indicators of problems**:
- Chains don't overlap → Poor mixing, increase `--walkers`
- Upward/downward trends → Not converged, increase `--steps`
- Stuck in one region → Trapped in local mode, check initialization

### Corner Plots

**What they show**:
- **Diagonal**: 1D marginal posterior distributions
- **Off-diagonal**: 2D joint distributions showing correlations

**What to look for**:
- **Correlation strength**: If two parameters show elliptical patterns, they're correlated
- **Multimodality**: Multiple peaks may indicate parameter non-identifiability
- **Boundary effects**: Parameters piling up at bounds need wider ranges

**Interpretation of correlations**:
- `|r| < 0.3`: Weak correlation (parameters relatively independent)
- `0.3 ≤ |r| < 0.7`: Moderate correlation (expected for NMR parameters)
- `|r| ≥ 0.7`: Strong correlation (parameters are interdependent)

**Example**: In NMR fitting, you might see:
- Strong correlation between `x0` (chemical shift) and `fwhm` (linewidth) for overlapping peaks
- Weak correlation between well-separated peaks

### Autocorrelation Plots

**What they show**: How correlated each sample is with previous samples

**What to look for**:
- ✓ **Good**: Autocorrelation drops quickly to ~0
- ✓ **Fast mixing**: Autocorrelation crosses 0.1 threshold within 10-20 steps
- ✗ **Poor**: Autocorrelation decays slowly (> 100 steps)

**What slow decay means**:
- Low ESS (fewer independent samples)
- Need more MCMC steps to get reliable estimates
- Consider thinning (but increasing steps is usually better)

## Recommended MCMC Settings

**Note**: All examples use adaptive burn-in by default. Burn-in is automatically determined unless you specify `--burn-in` or `--no-auto-burnin`.

### For Quick Exploratory Analysis
```bash
peakfit analyze mcmc Fits/ --walkers 32 --steps 500
```
- **Use when**: Just checking if MCMC is feasible
- **Time**: Fast (minutes)
- **Quality**: May not converge fully; adaptive burn-in will use fallback if needed

### For Standard Analysis (Recommended)
```bash
peakfit analyze mcmc Fits/ --walkers 32 --steps 2000
```
- **Use when**: Default for most fitting problems
- **Time**: Moderate (10-30 minutes)
- **Quality**: Should achieve R-hat ≤ 1.01 for most parameters; adaptive burn-in typically finds convergence

### For Publication-Quality Results
```bash
peakfit analyze mcmc Fits/ --walkers 64 --steps 5000
```
- **Use when**: Final analysis for publication
- **Time**: Longer (30-60 minutes)
- **Quality**: High ESS, very stable credible intervals; adaptive burn-in ensures optimal sample usage

### For Difficult Problems
```bash
peakfit analyze mcmc Fits/ --walkers 128 --steps 10000
```
- **Use when**: Many overlapping peaks, strong correlations
- **Time**: Extended (hours)
- **Quality**: Maximum reliability; longer chains allow adaptive burn-in to find true convergence point

## BARG Compliance Checklist

When reporting MCMC results in publications, ensure you include:

- [ ] **Number of chains**: Report `--walkers` used
- [ ] **Number of iterations**: Report `--steps` per chain
- [ ] **Burn-in method**: Report whether adaptive (R-hat monitoring) or manual burn-in was used
- [ ] **Burn-in amount**: Report number of steps discarded (PeakFit displays this automatically)
- [ ] **Convergence diagnostics**: Report R-hat for all parameters
- [ ] **Effective sample size**: Report ESS_bulk and ESS_tail for key parameters
- [ ] **Diagnostic plots**: Include trace and/or corner plots in supplementary materials
- [ ] **Credible intervals**: Report 95% CIs (not just standard errors)
- [ ] **Software version**: `peakfit --version`

## Example Workflow

### Step 1: Initial Fit
```bash
# First do a standard least-squares fit
peakfit fit spectrum.ft2 peaks.list --save-state
```

### Step 2: MCMC Analysis
```bash
# Run MCMC to get uncertainties
peakfit analyze mcmc Fits/ --walkers 64 --steps 2000

# Check console output:
# - Are R-hat values ≤ 1.01?
# - Are ESS values > 1000?
# - Any convergence warnings?
```

### Step 3: Generate Diagnostics
```bash
# Create diagnostic plots
peakfit plot diagnostics Fits/

# Open the PDF and check:
# - Trace plots: Do chains mix well?
# - Corner plots: What parameters are correlated?
# - Autocorrelation: Does it decay quickly?
```

### Step 4: Iterate if Needed
If diagnostics show problems:
```bash
# Rerun with more samples
peakfit analyze mcmc Fits/ --walkers 64 --steps 5000 --burn-in 1000

# Regenerate diagnostics
peakfit plot diagnostics Fits/
```

## Troubleshooting

### Problem: R-hat > 1.05 for some parameters

**Possible causes**:
1. Chains haven't converged yet
2. Parameter hitting bounds
3. Multimodal posterior

**Solutions**:
- Increase `--steps` to 5000 or 10000
- Check if parameters are at boundaries (adjust bounds in config)
- Look at trace plots to identify problematic parameters

### Problem: Low ESS (< 1000)

**Possible causes**:
1. High autocorrelation (poor mixing)
2. Not enough samples
3. Strong parameter correlations

**Solutions**:
- Increase `--steps` to collect more samples
- Check autocorrelation plots
- If correlations are very strong, consider reparameterization

### Problem: Chains don't overlap in trace plots

**Possible causes**:
1. Chains started in different modes
2. Not enough walkers
3. Posterior is multimodal

**Solutions**:
- Increase `--walkers` to 64 or 128
- Increase `--burn-in` to allow more time to find the same mode
- Check if there's truly multimodality (may indicate non-identifiable parameters)

### Problem: MCMC is very slow

**Causes**: MCMC evaluates the likelihood function many times

**Solutions**:
- Use fewer walkers for quick tests (but more for final analysis)
- Consider using global optimization first to get better starting point
- Run overnight for publication-quality results

## References

1. **Kruschke, J. K. (2021)**. Bayesian analysis reporting guidelines. *Nature Human Behaviour*, 5(10), 1282-1291. https://doi.org/10.1038/s41562-021-01177-7

2. **Gelman, A., & Rubin, D. B. (1992)**. Inference from iterative simulation using multiple sequences. *Statistical Science*, 7(4), 457-472.

3. **Vehtari, A., et al. (2021)**. Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.

4. **Stan Development Team**. Brief Guide to Stan's Warnings. https://mc-stan.org/misc/warnings.html

## Support

For questions or issues:
- GitHub Issues: https://github.com/gbouvignies/PeakFit/issues
- Check `peakfit --help` for command options
- See examples in `examples/` directory
