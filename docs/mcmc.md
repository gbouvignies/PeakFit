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

Each run writes one immutable record, published atomically once every cluster has
been sampled:

```text
<run_dir>/mcmc/<record id>.npz
```

Record IDs are UTC timestamps, so they sort chronologically; a new run never
replaces an earlier record. A record holds:
- **Its parent:** the fit's run ID, the SHA-256 of its `scientific.npz`, the
  verified spectrum's SHA-256 (or none when the archive retains its observations),
  the terminal correction revision and σ. A run whose `scientific.npz` changed while
  it was sampling is not recorded.
- **Each cluster's run:** the whole chain, burn-in included, with log-probabilities
  and per-sample amplitude identifiability, all lossless.
- **Each cluster's results:** the diagnosis, status, reasons, exchanges, and
  amplitude and summed-amplitude summaries, exactly as the terminal showed them,
  and the parent cluster's reduced χ².
- **What reproduces it:** the seed, the bit generator of each random stream, the
  formulation version, the amplitude-sample cap, and the PeakFit, NumPy and emcee
  versions.

Results read back from the record alone, without the fit or its spectrum.
`peakfit plot mcmc` plots the latest record, or the one given with `--record`, as a
diagnostics PDF. Its report page shows the recorded status, reasons and noise
convention, and its traces mark the recorded burn-in. Every page of a provisional
cluster says so. The marginal distributions are always drawn for inspection, but a
median and 68% interval are marked only where the terminal reports an interval; a
provisional or withheld parameter shows why instead.

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
and the fraction of samples At bound.

## Amplitudes

Amplitude (intensity) intervals include the lineshape uncertainty. They come from
conditional amplitude draws at up to 4,000 kept samples, balanced across walkers
and spread over time. The
terminal states how many samples were used, and the Monte Carlo errors behind
them.
- **Noise:** every MCMC interval is conditional on the recorded noise σ and is never
  rescaled by the reduced χ² (χ²ᵣ). The terminal and the report page show the
  parent cluster's χ²ᵣ as a goodness-of-fit diagnostic. A χ²ᵣ well above 1 means the
  model or σ does not describe the data; the intervals then understate the
  uncertainty. Noise is also assumed independent between points, which zero filling
  and apodization break, and χ²ᵣ does not reveal that.
- **Wider or narrower:** they are usually wider than the fit's *unscaled*
  analytical amplitude errors (`amplitude_standard_errors`), which hold the
  lineshapes fixed, but need not be. The exported `intensity_err` is multiplied by
  √χ²ᵣ when χ²ᵣ exceeds 1 ([outputs.md](outputs.md)), so there it can be wider
  than the MCMC interval without any contradiction; the terminal states that factor.
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
- Report walkers, steps, burn-in, the seed, τ, effective sample size and the
  PeakFit version for scientific work; the record holds them all.
