# Example 3: Global Optimization for Difficult Peaks

## Overview

This example demonstrates global optimization methods for fitting challenging peaks where local optimization fails or produces suboptimal results. Global optimization explores the entire parameter space rather than just locally around the initial guess.

**Note:** This example uses the same CEST data as Example 2 but applies global optimization methods. Compare the results to see if global optimization improves the fits.

## When to Use Global Optimization

Use global optimization when:

❌ **Local optimization fails:**
- Many peaks fail to converge
- Fits get stuck in local minima
- Results are sensitive to initial conditions

❌ **Severe peak overlap:**
- Multiple peaks very close together
- Difficult to separate individual contributions

❌ **Complex lineshapes:**
- Non-standard peak shapes
- Multiple components per peak

❌ **Poor signal-to-noise:**
- Noisy data makes gradient-based methods unstable
- Need robust fitting

✅ **When NOT to use:**
- Well-resolved peaks with good S/N (use basic fitting - faster)
- Very large datasets (global optimization is slow)
- Real-time analysis (too computationally expensive)

## Global Optimization Methods

PeakFit supports two global optimization algorithms:

### 1. Basin-Hopping

**How it works:**
- Randomly "hops" to different starting points
- Performs local optimization from each point
- Keeps the best solution found

**Best for:**
- Moderate-sized problems (< 20 parameters per cluster)
- When you want thorough exploration
- Relatively smooth cost landscapes

**Parameters:**
- `--optimizer basin_hopping`
- Can tune: number of iterations, temperature

### 2. Differential Evolution

**How it works:**
- Maintains a population of candidate solutions
- Evolves population through crossover and mutation
- Gradually converges to global optimum

**Best for:**
- Larger problems
- Highly multimodal landscapes (many local minima)
- When basin-hopping is too slow

**Parameters:**
- `--optimizer differential_evolution`
- Can tune: population size, crossover probability

## Running the Example

### Step 1: Run with Basin-Hopping

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --optimizer basin_hopping \
  --output Fits-BH/
```

**Expected time:** 10-20 minutes (much slower than local optimization)

### Step 2: Run with Differential Evolution

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --optimizer differential_evolution \
  --output Fits-DE/
```

**Expected time:** 15-30 minutes

### Step 3: Compare with Local Optimization

Run the standard fit for comparison:

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits-Local/
```

**Expected time:** 2-3 minutes

### Step 4: Compare Results

```bash
# Compare chemical shifts
diff Fits-Local/shifts.list Fits-BH/shifts.list
diff Fits-Local/shifts.list Fits-DE/shifts.list

# Check success rates in logs
grep "Successful" Fits-Local/peakfit.log
grep "Successful" Fits-BH/peakfit.log
grep "Successful" Fits-DE/peakfit.log

# Compare specific peak profiles
diff Fits-Local/10N-HN.out Fits-BH/10N-HN.out
```

## What to Look For

### Success Rate

Global optimization should have:
- ✓ Higher success rate (more converged fits)
- ✓ Fewer failed clusters
- ✓ More consistent results across runs

### Fit Quality

Check the log files for:
- **χ² values:** Should be similar or lower
- **Residuals:** Should be smaller and less systematic
- **Chemical shifts:** Should be more physically reasonable

### Computational Cost

Trade-offs:
- ⏱️ **Local:** Fast (~2-3 min) but may fail
- ⏱️ **Basin-hopping:** Slower (~10-20 min) but more robust
- ⏱️ **Differential evolution:** Slowest (~15-30 min) but most thorough

## Example Output

### Local Optimization (for comparison)

```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Metric               ┃ Value        ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Total clusters       │ 45           │
│ Successful fits      │ 42 (93%)     │
│ Failed fits          │ 3  (7%)      │
│ Average χ²           │ 1.45         │
│ Total time           │ 2m 34s       │
└──────────────────────┴──────────────┘
```

### Basin-Hopping Optimization

```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Metric               ┃ Value        ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Total clusters       │ 45           │
│ Successful fits      │ 45 (100%)    │  ← Improved!
│ Failed fits          │ 0  (0%)      │
│ Average χ²           │ 1.32         │  ← Better fit!
│ Total time           │ 14m 23s      │  ← Much slower
└──────────────────────┴──────────────┘
```

## Tuning Global Optimization

### Basin-Hopping Parameters

```bash
# More iterations for thorough search (slower)
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --optimizer basin_hopping \
  --bh-niter 200 \
  --output Fits/
```

### Differential Evolution Parameters

```bash
# Larger population for better exploration
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --optimizer differential_evolution \
  --de-popsize 30 \
  --output Fits/
```

**Note:** Check `peakfit fit --help` for all available tuning parameters.

## When Global Optimization Doesn't Help

If global optimization doesn't improve results:

1. **Check data quality:**
   - Low S/N may be the limiting factor
   - No optimizer can fit pure noise

2. **Verify peak list:**
   - Are peak positions approximately correct?
   - Remove obviously bad peaks

3. **Try different lineshapes:**
   - Maybe the lineshape model is wrong
   - Try `--lineshape pvoigt` or `--lineshape gaussian`

4. **Check clustering:**
   - Adjust contour level: `--contour-factor X`
   - Maybe peaks shouldn't be clustered together

## Troubleshooting

### "Global optimization is very slow"

**Expected:** Global optimization is 5-10× slower than local

**Solutions:**
- Reduce number of iterations (if using basin-hopping)
- Reduce population size (if using differential evolution)
- Use parallel processing (automatically enabled when beneficial)
- Process fewer planes at once

### "Results are inconsistent between runs"

Global optimization has some randomness:

**Solutions:**
- Run multiple times and compare
- Increase number of iterations for better convergence
- Set random seed for reproducibility (if option available)

### "Still getting failed fits"

Even global optimization can fail for:
- Extremely poor data quality
- Wrong lineshape model
- Peaks outside spectral region

**Solutions:**
- Check log file for specific errors
- Validate peak list positions
- Try fixing positions: `--fixed`

## Advanced Usage

### Hybrid Approach

Use global optimization only for failed clusters:

1. Run local optimization first
2. Identify failed clusters from log
3. Re-run those clusters with global optimization
4. Combine results

### Selective Global Optimization

Apply global optimization to specific peak ranges:

```bash
# Only fit peaks in a specific region
# (This requires pre-filtering the peak list)
peakfit fit data/pseudo3d.ft2 data/peaks_region1.list \
  --optimizer basin_hopping \
  --output Fits-Region1/
```

## Comparing Methods

Create a comparison table:

```bash
echo "Method          Success   Avg χ²   Time"
echo "---------------------------------------"
grep "Successful" Fits-Local/peakfit.log | awk '{print "Local           " $0}'
grep "Successful" Fits-BH/peakfit.log | awk '{print "Basin-Hopping   " $0}'
grep "Successful" Fits-DE/peakfit.log | awk '{print "Diff-Evolution  " $0}'
```

## Next Steps

After running this example:

1. **Analyze differences:**
   - Which peaks improved with global optimization?
   - Are the differences significant?

2. **Cost-benefit analysis:**
   - Is the extra time worth the improvement?
   - Can you use local optimization for most peaks?

3. **Try uncertainty analysis** (Example 4):
   - Quantify confidence in fitted parameters
   - Understand parameter correlations

4. **Learn batch processing** (Example 5):
   - Apply lessons to multiple datasets
   - Automate global optimization for difficult samples

## Reference

### Quick Command Reference

```bash
# Basin-hopping
peakfit fit SPECTRUM PEAKS --optimizer basin_hopping --output DIR

# Differential evolution
peakfit fit SPECTRUM PEAKS --optimizer differential_evolution --output DIR

# Tuning basin-hopping
peakfit fit SPECTRUM PEAKS --optimizer basin_hopping --bh-niter N

# Tuning differential evolution
peakfit fit SPECTRUM PEAKS --optimizer differential_evolution --de-popsize N

# Help
peakfit fit --help
```

### Performance Tips

- Use global optimization only when needed (not all data requires it)
- Start with basin-hopping (often faster than differential evolution)
- Tune parameters based on problem size
- Use configuration files to document settings
- Consider running overnight for large datasets

## Additional Resources

- **[Main Examples README](../README.md)** - Overview of all examples
- **[Example 2: Advanced Fitting](../02-advanced-fitting/)** - Compare with local optimization
- **[Optimization Guide](../../docs/optimization_guide.md)** - Performance tuning
- **[GitHub Issues](https://github.com/gbouvignies/PeakFit/issues)** - Get help

---

**Questions about global optimization?** Open an issue at https://github.com/gbouvignies/PeakFit/issues
