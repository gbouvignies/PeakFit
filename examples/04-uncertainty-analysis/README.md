# Example 4: Uncertainty Quantification

## Overview

This example demonstrates how to estimate parameter uncertainties and assess fit quality. Understanding uncertainties is crucial for:
- Determining confidence in fitted parameters
- Comparing results across experiments
- Identifying poorly constrained parameters
- Publishing reliable results

**Note:** This is a template example. Uncertainty analysis features may require additional implementation or external tools.

## Why Uncertainty Analysis?

A fitted parameter without an uncertainty is just a number. Uncertainty analysis tells you:

✅ **How reliable is this fit?**
- Are the parameters well-determined?
- Or are they essentially unconstrained by the data?

✅ **Are parameters correlated?**
- Do changes in one parameter compensate for another?
- Should parameters be fitted independently?

✅ **How do errors propagate?**
- What's the uncertainty in derived quantities?
- Which parameters contribute most to overall uncertainty?

## Methods for Uncertainty Estimation

### 1. Covariance Matrix (Fast)

From the Hessian matrix at the optimum:

**Pros:**
- Fast (computed during optimization)
- Standard approach for least-squares fitting

**Cons:**
- Assumes quadratic cost landscape (Gaussian errors)
- May underestimate uncertainties for non-linear problems
- Unreliable for poorly constrained parameters

**Usage:**
```bash
peakfit fit data/spectrum.ft2 data/peaks.list \
  --output Fits/ \
  --compute-uncertainties
```

### 2. Bootstrap Resampling (Medium)

Resample data and refit multiple times:

**Pros:**
- Non-parametric (few assumptions)
- Works for non-Gaussian errors

**Cons:**
- Computationally expensive (requires many refits)
- Assumes data points are independent

**Usage:**
```bash
# Fit once
peakfit fit data/spectrum.ft2 data/peaks.list --output Fits/

# Bootstrap analysis
peakfit analyze bootstrap Fits/ --nsamples 1000 --output Bootstrap/
```

### 3. MCMC Sampling (Slow but thorough)

Sample the posterior distribution using Markov Chain Monte Carlo:

**Pros:**
- Full posterior distribution (not just mean ± std)
- Handles parameter correlations correctly
- Works for complex, non-Gaussian problems

**Cons:**
- Very computationally expensive
- Requires careful convergence checking
- May need tuning of sampling parameters

**Usage:**
```bash
# Fit once
peakfit fit data/spectrum.ft2 data/peaks.list --output Fits/

# MCMC sampling
peakfit analyze mcmc Fits/ \
  --chains 4 \
  --samples 10000 \
  --output MCMC/
```

## Running This Example

Since uncertainty analysis features may not be fully implemented, here's the general workflow:

### Step 1: Fit the Data

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits/
```

### Step 2: Compute Uncertainties (if available)

```bash
# Covariance-based uncertainties
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --compute-uncertainties \
  --output Fits/
```

Or as a post-processing step:

```bash
peakfit analyze uncertainty Fits/ --output Uncertainty/
```

### Step 3: Analyze Results

Check output files for uncertainty estimates:

```bash
# Chemical shift uncertainties
cat Fits/shifts.list
# Should include columns: assignment, w1, w1_err, w2, w2_err

# Parameter correlations
cat Fits/correlations.txt
# Shows which parameters are correlated
```

## Expected Output

### With Uncertainties

**shifts.list** with error estimates:
```
Assignment  w1_fit    w1_err   w2_fit   w2_err
A1N-HN      115.632   0.002    6.869    0.001
A2N-HN      117.521   0.003    8.694    0.002
...
```

### Interpretation

**Small uncertainties (< 0.01 ppm):**
- Well-determined parameters
- Good signal-to-noise
- Confident in fitted values

**Large uncertainties (> 0.05 ppm):**
- Poorly constrained
- Check fit quality
- May need more data or constraints

**Asymmetric uncertainties:**
- Non-Gaussian posterior
- Consider MCMC for full distribution

## Alternative Approaches

### Manual Uncertainty Estimation

If automated tools aren't available:

1. **Repeat fits with different starting points:**
   ```bash
   for i in {1..10}; do
       peakfit fit data/spectrum.ft2 data/peaks.list --output Fits-$i/
   done
   # Check variability in results
   ```

2. **Jackknife resampling:**
   - Remove one plane at a time and refit
   - Uncertainty from variability across fits

3. **Add noise and refit:**
   - Add synthetic noise matching your data
   - Refit multiple times
   - Uncertainty from spread of results

### External Tools

Use external packages for uncertainty analysis:

**Python (if using PeakFit as a library):**
```python
import emcee  # MCMC sampling
import corner  # Corner plots for posteriors
import scipy.optimize  # Covariance matrices
```

**R:**
```r
library(FME)  # Parameter sensitivity and MCMC
```

## Quality Metrics

Even without formal uncertainty analysis, check:

### 1. Residuals

```bash
# Plot residuals for each peak
# Look for systematic patterns (indicates poor fit)
cat Fits/10N-HN.out | awk '{print $4}' | gnuplot
```

**Good fit:**
- Residuals randomly scattered around zero
- No systematic trends

**Poor fit:**
- Systematic deviations
- Large residuals

### 2. χ² Values

Check the log file:

```bash
grep "chi-squared" Fits/peakfit.log
```

**Interpretation:**
- **χ² ≈ 1:** Good fit (residuals match noise level)
- **χ² << 1:** Overfitting (too many parameters)
- **χ² >> 1:** Poor fit (systematic errors)

### 3. Fit Convergence

```bash
grep "Converged" Fits/peakfit.log | wc -l
grep "Failed" Fits/peakfit.log | wc -l
```

High failure rate suggests:
- Poor initial guesses
- Data quality issues
- Wrong model (lineshape, etc.)

## Visualization

Create plots to assess fit quality:

### Fitted vs. Observed

```python
import matplotlib.pyplot as plt
import numpy as np

# Load data from .out file
data = np.loadtxt('Fits/10N-HN.out', comments='#')
z = data[:, 0]
observed = data[:, 1]
fitted = data[:, 2]
residuals = data[:, 3]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Data and fit
ax1.plot(z, observed, 'o', label='Observed')
ax1.plot(z, fitted, '-', label='Fitted')
ax1.legend()
ax1.set_ylabel('Intensity')

# Residuals
ax2.plot(z, residuals, 'o')
ax2.axhline(0, color='k', linestyle='--')
ax2.set_ylabel('Residuals')
ax2.set_xlabel('Z-value')

plt.tight_layout()
plt.savefig('fit_quality.pdf')
```

## Next Steps

1. **Implement systematic checks:**
   - Run fits with different optimizers
   - Compare results - variability indicates uncertainty

2. **Try global optimization (Example 3):**
   - More robust to local minima
   - May reduce uncertainty

3. **Consider batch analysis (Example 5):**
   - Replicate measurements reduce uncertainty
   - Compare results across samples

4. **Consult literature:**
   - How do others in your field handle uncertainties?
   - Standard practices for your experiment type

## Reference

### Uncertainty Methods Comparison

| Method | Speed | Accuracy | Assumptions |
|--------|-------|----------|-------------|
| Covariance | Fast | Approximate | Gaussian errors |
| Bootstrap | Medium | Good | Independent points |
| MCMC | Slow | Excellent | Fewest assumptions |

### Quick Command Reference

```bash
# With uncertainties during fit
peakfit fit SPECTRUM PEAKS --compute-uncertainties --output DIR

# Post-processing uncertainty analysis
peakfit analyze uncertainty DIR --output UNCERT_DIR

# Bootstrap resampling
peakfit analyze bootstrap DIR --nsamples N --output BOOT_DIR

# MCMC sampling
peakfit analyze mcmc DIR --chains N --samples M --output MCMC_DIR
```

**Note:** Check `peakfit analyze --help` for available analysis commands.

## Additional Resources

- **[Main Examples README](../README.md)** - Overview of all examples
- **[Example 2: Advanced Fitting](../02-advanced-fitting/)** - Get good fits first
- **[Example 3: Global Optimization](../03-global-optimization/)** - Improve convergence
- **[GitHub Issues](https://github.com/gbouvignies/PeakFit/issues)** - Request features

---

**Questions about uncertainties?** Open an issue at https://github.com/gbouvignies/PeakFit/issues
