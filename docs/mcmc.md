# MCMC Uncertainty Analysis

`peakfit mcmc` estimates parameter uncertainty from an existing fit result. It is
a post-fit workflow, not a fitting optimizer.

## Basic Workflow

```bash
peakfit fit spectrum.ft2 peaks.list --output Fits

LATEST_RUN="$(find Fits -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"

peakfit mcmc "$LATEST_RUN" --peaks 2N-H --walkers 32 --steps 1000
peakfit plot mcmc "$LATEST_RUN" --output "$LATEST_RUN/mcmc_diagnostics.pdf"
```

If `--peaks` is omitted, all clusters are analyzed. A selected peak analyzes its
whole overlap cluster.

## Inputs And Outputs

MCMC reads:

- `summary/fit.json`
- `metadata/fitting_state.pkl`

MCMC writes chain files under:

```text
<run_dir>/chains/cluster_*_chains.h5
```

`peakfit plot mcmc` reads those chain files and creates a diagnostics PDF.

## Runtime Options

- `--walkers N`: number of MCMC walkers. Default: `32`.
- `--steps N`: samples per walker. Default: `1000`.
- `--burn-in N`: manual burn-in. Supplying this disables auto burn-in.
- `--auto-burnin/--no-auto-burnin`: enable or disable automatic burn-in selection.
- `--workers N`: parallel workers; `-1` uses all CPUs.
- `--save-chains/--no-save-chains`: control whether plot-ready chains are written.

Use small runs for feasibility checks and longer runs for final analysis:

```bash
peakfit mcmc "$LATEST_RUN" --walkers 32 --steps 500
peakfit mcmc "$LATEST_RUN" --walkers 64 --steps 5000
```

## Diagnostics

Check the terminal summary and diagnostics PDF before trusting MCMC intervals.

- `R-hat <= 1.01`: strong convergence signal.
- `R-hat <= 1.05`: often usable, but inspect trace plots.
- `R-hat > 1.05`: increase steps, inspect bounds, or reconsider the model.
- Low effective sample size means intervals are unstable; increase `--steps`.
- Separated or drifting trace plots mean chains have not mixed.
- Strong corner-plot correlations are common for overlapping peaks, but boundary
  pileups usually indicate overly tight constraints or non-identifiability.

## Practical Guidance

- Start with a successful least-squares fit.
- Analyze a small peak subset first when exploring.
- Increase `--steps` before trusting uncertain intervals.
- Keep `--save-chains` enabled if you want diagnostics plots later.
- Report walkers, steps, burn-in method, R-hat, effective sample size, and the
  PeakFit version for scientific work.
