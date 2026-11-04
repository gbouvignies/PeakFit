# MCMC Uncertainty Analysis

`peakfit mcmc` estimates parameter uncertainty from an existing fit result. It is
a post-fit workflow, not a fitting optimizer.

[`mcmc-formulation.md`](mcmc-formulation.md) specifies what MCMC samples and
reports, and the command follows it.

## Basic Workflow

```bash
peakfit fit spectrum.ft2 peaks.list --output Fits

LATEST_RUN="$(find Fits -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"

peakfit mcmc "$LATEST_RUN" --peaks 2N-H --walkers 32 --steps 1000
peakfit plot mcmc "$LATEST_RUN" --output "$LATEST_RUN/mcmc_diagnostics.pdf"
```

If `--peaks` is omitted, all clusters are analyzed. A selected peak analyzes its
whole overlap cluster. Sampling covers every fitted nonlinear parameter of the
cluster, including a cluster's fitted phase (`--phx`/`--phy`).

## Inputs And Outputs

MCMC reads:

- `scientific.npz`: published nonlinear values/bounds/freedom, exact terminal
  corrections/mappings and validated model descriptors.
- The verified spectrum for compact results, or retained observations for
  `--detailed-evidence` exports. The original peak/plane text files are unnecessary
  for native continuation.

Results without `scientific.npz` are not read; refit them. Each analyzed cluster
samples its own posterior (`engine.scientific.sampling`):
- Only the analyzed clusters must be usable.
- Before sampling, the likelihood at the fitted values must reproduce the recorded
  chi-squared to a relative 10⁻¹⁰, or the cluster is refused.
- MCMC does not resume a chain or RNG state.

See [storage policy](outputs.md).

MCMC writes chain files under:

```text
<run_dir>/chains/cluster_*_chains.h5
```

Each file holds the whole chain, burn-in included, with the burn-in length and the
run's seed. `peakfit plot mcmc` discards the burn-in once and creates a
diagnostics PDF.

## Runtime Options

- `--walkers N`: number of MCMC walkers. Default: `32`. A cluster needs at least
  twice as many walkers as it has sampled parameters; fewer is an error, not
  silently raised.
- `--steps N`: samples per walker. Default: `1000`.
- `--burn-in N`: manual burn-in. Supplying this disables auto burn-in. It must
  leave the chain long enough for the checks below.
- `--auto-burnin/--no-auto-burnin`: automatic burn-in is 5 τ_max of the whole
  chain.
- `--seed N`: the run's `SeedSequence` entropy. The same seed, inputs and versions
  give identical chains. Every run prints the seed it used.
- `--workers N`: parallel worker processes. Default: serial; `-1` uses all CPUs.
- `--save-chains/--no-save-chains`: control whether plot-ready chains are written.

Walkers start inside the bounds around the fitted values, spread by a tenth of the
fit's standard errors; a draw outside the bounds is drawn again, never clipped.

Use small runs for feasibility checks and longer runs for final analysis:

```bash
peakfit mcmc "$LATEST_RUN" --walkers 32 --steps 500
peakfit mcmc "$LATEST_RUN" --walkers 64 --steps 5000
```

## Diagnostics

Each cluster either **passed the convergence checks** or is **provisional**.
- **Provisional:** the terminal lists the reasons and gives no final intervals.
- **Passed:** a run is never called "converged". Walkers start near the fitted
  values, so a distant mode would not be found.

A run is provisional when any of these hold:
- **Too short:** the kept chain is shorter than 50 integrated autocorrelation times
  (τ), or τ is undefined. Increase `--steps`.
- **Drift:** the medians of the kept chain's two halves disagree beyond their Monte
  Carlo error. Increase `--steps` or the burn-in.
- **Stuck walkers:** a walker accepts under a tenth of the ensemble's median rate.
- **Exchanged peaks:** two peaks with overlapping position bounds exchange order in
  some samples. Both peaks' intervals are withheld; their identities are not
  determined by the data.

R̂ is not reported: ensemble walkers are coupled, so R̂ across them means nothing.
Each parameter shows τ, an approximate effective sample size (walkers × steps / τ)
and the fraction of samples At bound. The diagnostics PDF applies the checks that
need only the chain (length and drift).

## Amplitudes

Amplitude (intensity) intervals include the lineshape uncertainty. They come from
conditional amplitude draws at up to 4,000 kept samples, balanced across walkers
and spread over time. The
terminal states how many samples were used, and the Monte Carlo errors behind
them.
- **Wider or narrower:** they are usually wider than the fit's analytical errors,
  which hold the lineshapes fixed, but need not be.
- **Withheld:** a peak whose amplitude is not determined in some samples (shapes
  that coincide, or are too close to compute reliably) is withheld, with the
  fraction of samples.
- **Read the sum:** two amplitudes with posterior |ρ| ≥ 0.9 in a plane are flagged
  as not separately determined, and their summed amplitude is reported.
- **Read with care (‡):** those peaks' positions, linewidths and per-peak
  amplitudes are still shown, marked ‡ as not separately resolved.
- **Sums of undetermined peaks:** a group of undetermined peaks whose sum is
  determined, such as duplicated peaks, reports its sum too.

Coverage on one peak and on overlapping pairs is recorded in
[validation/mcmc-coverage.md](validation/mcmc-coverage.md).

## Practical Guidance

- Start with a successful least-squares fit.
- Analyze a small peak subset first when exploring.
- Increase `--steps` before trusting uncertain intervals.
- Keep `--save-chains` enabled if you want diagnostics plots later.
- Report walkers, steps, burn-in, the seed, τ, effective sample size and the
  PeakFit version for scientific work.
