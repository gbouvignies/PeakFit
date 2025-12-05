# Example 4: Uncertainty Analysis with MCMC

## Overview

This example demonstrates how to estimate parameter uncertainties using MCMC (Markov Chain Monte Carlo) sampling. MCMC provides **full posterior distributions**, not just point estimates, giving you deeper insights into parameter reliability and correlations.

The new output system provides comprehensive MCMC diagnostics including convergence metrics, chain storage, and diagnostic visualizations.

## Why MCMC?

| Method     | Speed | What You Get                |
| ---------- | ----- | --------------------------- |
| Covariance | Fast  | Point estimate ± std        |
| MCMC       | Slow  | Full posterior distribution |

MCMC tells you:

- [GOOD] Full shape of parameter distributions
- [GOOD] Parameter correlations
- [GOOD] Confidence intervals (68%, 95%, etc.)
- [GOOD] Convergence diagnostics (R-hat, ESS)

## Running the Example

### Step 1: Run MCMC Fitting

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --method mcmc \
  --save-chains \
  --verbosity full \
  --output Fits/
```

**Expected time:** 10-20 minutes (MCMC is thorough but slow)

### Step 2: Explore the Outputs

```bash
ls -la Fits/
ls -la Fits/mcmc/
```

## Output Structure

MCMC fitting produces enhanced outputs:

```
Fits/
├── README.md                 # Auto-generated directory guide
├── summary/
│   ├── fit_summary.json      # Complete results with MCMC diagnostics
│   ├── analysis_report.md    # Report with convergence status
│   └── quick_results.csv
├── parameters/
│   ├── parameters.csv        # Parameter estimates with uncertainties
│   ├── amplitudes.csv
│   └── parameters.json
├── statistics/
│   ├── fit_statistics.json
│   └── residuals.csv
├── diagnostics/              # MCMC-specific diagnostics
│   ├── mcmc_diagnostics.json # R-hat, ESS, convergence status
│   ├── convergence.csv       # Per-parameter convergence
│   └── warnings.txt          # Collected warnings
├── chains/                   # Full MCMC chains (if --save-chains)
│   ├── mcmc_chains.h5        # HDF5 format (preferred)
│   └── mcmc_chains.npz       # NumPy archive (fallback)
├── figures/
│   ├── profiles/
│   ├── diagnostics/          # MCMC trace plots, autocorrelation
│   └── correlations/         # Corner plots
└── metadata/
    ├── run_metadata.json
    └── configuration.toml
```

## Understanding the Outputs

### summary/fit_summary.json - MCMC Diagnostics

The JSON output includes detailed MCMC diagnostics for each cluster:

```json
{
  "clusters": [
    {
      "cluster_id": 1,
      "peaks": ["2N-HN"],
      "fit_statistics": {
        "chi_squared": 1.12,
        "reduced_chi_squared": 0.98
      },
      "mcmc_diagnostics": {
        "status": "GOOD",
        "n_samples": 10000,
        "n_chains": 4,
        "convergence": {
          "all_rhat_below_threshold": true,
          "effective_sample_size_adequate": true,
          "rhat_max": 1.02,
          "ess_min": 3200
        },
        "parameters": [
          {
            "name": "2N-HN.F1.cs",
            "rhat": 1.01,
            "ess": 4500,
            "converged": true
          },
          {
            "name": "2N-HN.F2.cs",
            "rhat": 1.02,
            "ess": 4300,
            "converged": true
          }
        ]
      },
      "parameters": [
        {
          "name": "cs_F1",
          "value": 115.632,
          "uncertainty": 0.002,
          "ci_lower_95": 115.628,
          "ci_upper_95": 115.636
        },
        {
          "name": "cs_F2",
          "value": 6.869,
          "uncertainty": 0.001,
          "ci_lower_95": 6.867,
          "ci_upper_95": 6.871
        }
      ]
    }
  ]
}
```

### MCMC Status Indicators

| Status       | Meaning              | R-hat    | Action         |
| ------------ | -------------------- | -------- | -------------- |
| `GOOD`       | Well converged       | < 1.05   | Trust results  |
| `ACCEPTABLE` | Minor issues         | 1.05-1.1 | Results usable |
| `MARGINAL`   | Convergence concerns | 1.1-1.2  | Run longer     |
| `POOR`       | Not converged        | > 1.2    | Don't trust    |

### Loading MCMC Chains

```python
import numpy as np
import json

# Load chains (NumPy fallback format)
data = np.load('Fits/chains/mcmc_chains.npz')
chains = data['chains']      # Shape: (n_chains, n_samples, n_params)
param_names = list(data['param_names'])

print(f"Shape: {chains.shape}")
print(f"Parameters: {param_names}")

# Load diagnostics metadata
with open('Fits/diagnostics/mcmc_diagnostics.json') as f:
    diagnostics = json.load(f)
print(f"Samples per chain: {diagnostics['n_samples']}")
print(f"R-hat max: {diagnostics['convergence']['rhat_max']:.3f}")

# Or use HDF5 format (preferred, if h5py available)
import h5py

with h5py.File('Fits/chains/mcmc_chains.h5', 'r') as f:
    chains = f['chains'][:]
    param_names = [name.decode() for name in f['param_names'][:]]
```

### Analyzing Posteriors

```python
import numpy as np
import matplotlib.pyplot as plt

# Load chains
data = np.load('Fits/mcmc/chains.npz')
chains = data['chains']
param_names = list(data['param_names'])

# Flatten chains (combine all chains)
flat_chains = chains.reshape(-1, chains.shape[-1])

# Plot posterior for first parameter
param_idx = 0
samples = flat_chains[:, param_idx]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Trace plot
for i in range(chains.shape[0]):
    axes[0].plot(chains[i, :, param_idx], alpha=0.5)
axes[0].set_xlabel('Sample')
axes[0].set_ylabel(param_names[param_idx])
axes[0].set_title('Trace Plot')

# Histogram
axes[1].hist(samples, bins=50, density=True, alpha=0.7)
axes[1].axvline(np.median(samples), color='r', label='Median')
axes[1].axvline(np.percentile(samples, 2.5), color='r', linestyle='--', label='95% CI')
axes[1].axvline(np.percentile(samples, 97.5), color='r', linestyle='--')
axes[1].set_xlabel(param_names[param_idx])
axes[1].set_title('Posterior Distribution')
axes[1].legend()

plt.tight_layout()
plt.savefig('posterior_analysis.pdf')
```

### Corner Plot for Correlations

```python
import corner
import numpy as np

data = np.load('Fits/mcmc/chains.npz')
chains = data['chains']
param_names = list(data['param_names'])

# Flatten chains
flat_chains = chains.reshape(-1, chains.shape[-1])

# Create corner plot
fig = corner.corner(
    flat_chains,
    labels=param_names,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.3f'
)
plt.savefig('corner_plot.pdf')
```

## Checking Convergence

### From JSON

```python
import json

with open('Fits/fit_results.json') as f:
    results = json.load(f)

# Check all clusters
for cluster in results['clusters']:
    diag = cluster.get('mcmc_diagnostics')
    if diag:
        status = diag['status']
        rhat_max = diag['convergence']['rhat_max']
        ess_min = diag['convergence']['ess_min']
        print(f"Cluster {cluster['cluster_id']}: {status} (R-hat={rhat_max:.3f}, ESS={ess_min})")
```

### From diagnostics.json

```bash
# Quick convergence check
jq '.clusters[] | "\(.cluster_id): \(.status) R-hat=\(.rhat_max)"' Fits/mcmc/diagnostics.json
```

## Interpreting Results

### Good Convergence

- R-hat < 1.05 for all parameters
- ESS > 100 per parameter (preferably > 1000)
- Trace plots show good mixing (no trends)
- Status: `GOOD` or `ACCEPTABLE`

### Poor Convergence

- R-hat > 1.1
- ESS < 100
- Trace plots show drifting or stuck chains
- **Action:** Run longer chains or increase warmup

```bash
# Run longer chains
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --method mcmc \
  --mcmc-samples 20000 \
  --mcmc-warmup 5000 \
  --save-chains \
  --output Fits/
```

## CSV Output with Uncertainties

The CSV includes full uncertainty information:

```csv
cluster_id,peak_name,parameter,value,uncertainty,ci_lower_95,ci_upper_95,rhat,ess
1,2N-HN,cs_F1,115.632,0.002,115.628,115.636,1.01,4500
1,2N-HN,cs_F2,6.869,0.001,6.867,6.871,1.02,4200
1,2N-HN,lw_F1,25.3,1.2,23.0,27.8,1.01,3900
```

### Analyzing with pandas

```python
import pandas as pd

df = pd.read_csv('Fits/parameters.csv')

# Find poorly converged parameters
poor_convergence = df[df['rhat'] > 1.1]
if len(poor_convergence) > 0:
    print("⚠️ Parameters with poor convergence:")
    print(poor_convergence[['cluster_id', 'peak_name', 'parameter', 'rhat']])

# Parameters with large uncertainties
df['rel_error'] = df['uncertainty'] / df['value'].abs() * 100
large_error = df[df['rel_error'] > 10]
print(f"\nParameters with >10% relative error: {len(large_error)}")
```

## Markdown Report

The `report.md` includes a convergence summary:

```markdown
## MCMC Diagnostics Summary

| Status       | Count |
| ------------ | ----- |
| ✓ GOOD       | 42    |
| ⚠ ACCEPTABLE | 2     |
| ✗ MARGINAL   | 1     |

### Cluster 1: 2N-HN

**MCMC Status**: ✓ GOOD

- Samples: 10,000
- Chains: 4
- Max R-hat: 1.02
- Min ESS: 3,200

| Parameter | Value   | 95% CI             | R-hat | ESS  |
| --------- | ------- | ------------------ | ----- | ---- |
| cs_F1     | 115.632 | [115.628, 115.636] | 1.01  | 4500 |
| cs_F2     | 6.869   | [6.867, 6.871]     | 1.02  | 4200 |
| lw_F1     | 25.3    | [23.0, 27.8]       | 1.01  | 3900 |
```

## Performance Tips

### Speed vs Quality

```bash
# Quick check (fewer samples)
peakfit fit ... --method mcmc --mcmc-samples 1000

# Publication quality (more samples)
peakfit fit ... --method mcmc --mcmc-samples 10000 --mcmc-warmup 2000
```

### For Large Datasets

- Start with a subset of peaks to test settings
- Use fewer chains if memory is limited
- Consider running clusters in parallel (at OS level)

## Troubleshooting

### "MCMC not converging"

- Increase warmup: `--mcmc-warmup 5000`
- Increase samples: `--mcmc-samples 20000`
- Check if local optimization succeeds first

### "Chains look stuck"

- Poor initial guess - run local optimization first
- Model mismatch - check lineshape

### "Memory error"

- Reduce number of samples
- Use fewer chains
- Don't save chains: remove `--save-chains`

## Next Steps

1. **Visualize posteriors** - Create corner plots
2. **Check correlations** - Identify dependent parameters
3. **Compare methods** - Run with local optimizer for comparison
4. **Read the docs** - [docs/output_system.md](../../docs/output_system.md)

---

**Questions?** Open an issue at https://github.com/gbouvignies/PeakFit/issues
