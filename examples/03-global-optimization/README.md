# Example 3: Global Optimization for Difficult Peaks

## Overview

This example demonstrates global optimization methods for fitting challenging peaks where local optimization fails or produces suboptimal results. Global optimization explores the entire parameter space rather than just locally around the initial guess.

**Note:** This example uses the same CEST data as Example 2 but applies global optimization methods. Compare the structured JSON outputs to evaluate improvement.

## When to Use Global Optimization

[GOOD] **Use global optimization when:**

- Many peaks fail to converge
- Severe peak overlap
- Results are sensitive to initial conditions
- Local optimization gets stuck in local minima

[BAD] **Don't use when:**

- Well-resolved peaks with good S/N (use basic fitting - faster)
- Large datasets where time is critical
- Most peaks converge with local optimization

## Global Optimization Methods

### Basin-Hopping

- Randomly "hops" to different starting points
- Performs local optimization from each
- Best for moderate-sized problems

### Differential Evolution

- Evolves a population of candidate solutions
- Best for larger, multimodal problems

## Running the Example

### Step 1: Run Local Optimization (Baseline)

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits-Local/ \
  --verbosity full
```

### Step 2: Run Basin-Hopping

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --optimizer basin_hopping \
  --output Fits-BH/ \
  --verbosity full
```

### Step 3: Compare Results Using JSON

```python
import json

# Load both results
with open('Fits-Local/fit_results.json') as f:
    local = json.load(f)
with open('Fits-BH/fit_results.json') as f:
    bh = json.load(f)

# Compare summary statistics
print("Method        χ²       Time")
print("-" * 35)
print(f"Local         {local['global_summary']['overall_chi_squared']:.3f}    {local['metadata']['elapsed_seconds']:.1f}s")
print(f"Basin-Hop     {bh['global_summary']['overall_chi_squared']:.3f}    {bh['metadata']['elapsed_seconds']:.1f}s")

# Find clusters that improved
for l_cluster in local['clusters']:
    cid = l_cluster['cluster_id']
    b_cluster = next(c for c in bh['clusters'] if c['cluster_id'] == cid)

    l_chi2 = l_cluster['fit_statistics']['chi_squared']
    b_chi2 = b_cluster['fit_statistics']['chi_squared']

    if b_chi2 < l_chi2 * 0.95:  # >5% improvement
        print(f"Cluster {cid}: {l_chi2:.2f} -> {b_chi2:.2f} (improved)")
```

## Output Structure

Each optimization run produces the standard structured outputs:

```
Fits-Local/
├── fit_results.json    # Complete structured results
├── parameters.csv      # Parameter estimates
├── shifts.csv          # Chemical shifts
├── intensities.csv     # Fitted intensities
├── report.md           # Human-readable report
└── peakfit.log

Fits-BH/
├── fit_results.json
├── parameters.csv
├── shifts.csv
├── intensities.csv
├── report.md
└── peakfit.log
```

### Comparing with CSV

```python
import pandas as pd

local_df = pd.read_csv('Fits-Local/parameters.csv')
bh_df = pd.read_csv('Fits-BH/parameters.csv')

# Merge on cluster_id and compare chi_squared
comparison = local_df.merge(
    bh_df,
    on=['cluster_id', 'peak_name'],
    suffixes=('_local', '_bh')
)

# Find improved fits
improved = comparison[
    comparison['chi_squared_bh'] < comparison['chi_squared_local'] * 0.95
]
print(f"Improved clusters: {len(improved)}")
```

### Comparing with Shell

```bash
# Extract chi-squared values and compare
echo "Cluster | Local χ² | BH χ²"
paste \
  <(jq -r '.clusters[] | "\(.cluster_id) \(.fit_statistics.chi_squared)"' Fits-Local/fit_results.json) \
  <(jq -r '.clusters[] | .fit_statistics.chi_squared' Fits-BH/fit_results.json) \
  | awk '{printf "%7s | %8.3f | %5.3f\n", $1, $2, $3}'
```

## Expected Results

### Local Optimization

```json
{
  "global_summary": {
    "n_clusters": 45,
    "overall_chi_squared": 1.45,
    "failed_clusters": 3
  },
  "metadata": {
    "elapsed_seconds": 154
  }
}
```

### Basin-Hopping

```json
{
  "global_summary": {
    "n_clusters": 45,
    "overall_chi_squared": 1.32,
    "failed_clusters": 0
  },
  "metadata": {
    "elapsed_seconds": 863
  }
}
```

## Tuning Parameters

### Basin-Hopping

```bash
# More iterations for thorough search
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --optimizer basin_hopping \
  --bh-niter 200 \
  --output Fits/ \
  --verbosity full
```

### Differential Evolution

```bash
# Larger population for better exploration
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --optimizer differential_evolution \
  --de-popsize 30 \
  --output Fits/ \
  --verbosity full
```

## Performance Comparison

| Method          | Success Rate | Avg χ² | Time       |
| --------------- | ------------ | ------ | ---------- |
| Local           | 93%          | 1.45   | ~2-3 min   |
| Basin-Hopping   | 100%         | 1.32   | ~10-15 min |
| Diff. Evolution | 100%         | 1.30   | ~15-25 min |

## When Global Optimization Doesn't Help

If global optimization doesn't improve results:

1. **Check data quality** - Low S/N may be the limit
2. **Verify peak list** - Positions approximately correct?
3. **Try different lineshapes** - `--lineshape pvoigt`
4. **Adjust clustering** - `--contour-factor X`

## Next Steps

1. **MCMC Uncertainty** - [Example 4](../04-uncertainty-analysis/)

   - Quantify confidence in parameters
   - Full posterior distributions

2. **Read the output guide** - [docs/output_system.md](../../docs/output_system.md)

---

**Questions?** Open an issue at https://github.com/gbouvignies/PeakFit/issues
