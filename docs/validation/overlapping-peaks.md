# Overlapping peaks: scientific regression investigation

Validation date: 2026-10-28. Base: `1467eee` (main through PR #58).
Reference source: tag `v0.10.5`. NumPy 2.4.1, SciPy 1.17.0, LMFit 1.3.4,
Python 3.14.5, one BLAS thread. LMFit is installed only in a temporary
comparison environment; it is not a new PeakFit dependency.

**Conclusion:** the CSV mapping, rank-deficient projection, covariance fallback,
shared-parameter Jacobian, and unknown-error readback defects are corrected.
The three experimental collapses are reproducible but are **not explained by
nonexistent QR directions**: their design matrices remain full rank. They are
unstable near-coincident decompositions with large cancelling signed amplitudes.
They remain warned results, not scientifically certified positions. This work
does not justify releasing these assignments as trustworthy or choosing the
historical local minimum as ground truth.

## Reproduction and evidence

Inputs are the unmodified `examples/02-advanced-fitting/data/pseudo3d.ft2`,
`pseudo3d.list`, and `b1_offsets.txt`: 166 peaks, 121 groups, 131 planes.
The actual groups are:

| Cluster | Members (all retained) | Mask points | Observations |
| --- | --- | ---: | ---: |
| 269 | 36N-H, 75N-H, 93N-H | 90 | 11,790 |
| 360 | 5N-H, 10N-H, 76N-H, 105N-H | 106 | 13,886 |
| 424 | 53N-H, 147N-H | 67 | 8,777 |

Each experiment fits **every group**, then updates external-neighbor corrections
only between passes, preserving the production masks. Historical mask hashes
match exactly. Noise is 11922.544916060584 in the current implementation and
11922.550440769748 in the natural release run (relative difference 4.63e-7).
Controlled comparisons supply the current noise to both implementations.

One initial assumption needed correction: the current **bare CLI defaults to two
passes**, while the configuration model and example TOML specify one. The
reported optimalities reproduce with two passes. All tables below label the pass
count explicitly; χ² across passes can change because corrected data change.

[Machine-readable results](overlap-results.json) retain starts, final values,
bounds, errors, termination, optimality, χ², ranks, singular values, raw and
column-scaled Jacobian condition numbers, finite-difference checks, mask hashes,
residual quality, and signed-amplitude summaries for every experiment. The
`current_default` records also retain complete signed amplitudes and residual
RMS across all 131 planes, including neighboring peaks.
[Start sensitivity](overlap-start-sensitivity.json) records 30 additional fits.

## Confirmed defects and their corrections

1. **Serialization:** `write_final_outcome_shifts()` stored `F2_err` but retrieved
   `F2`; likewise for F3. The regression test first failed with shift 11.0/error
   0.5 both exported as 11.0. It now retrieves the error key. A complete actual
   CLI fit exported 166 rows, with all 332 shift-error cells checked against
   their underlying JSON values, including unavailable errors.
2. **Projection at rank loss:** for duplicated Gaussians and data in the second
   QR direction, the original triangular solve returned amplitudes approximately
   ±7.68e14. Its projected residual approached zero although the actual
   rank-aware least-squares residual norm is one. Triangular solves need not
   raise on a nearly singular R. A single factorization now supplies amplitudes,
   the retained column space and the Jacobian pseudoinverse. A small-R SVD uses
   the `lstsq` cutoff `eps * max(shape dimensions) * largest singular value`.
   Ordinary groups retain triangular QR algebra; ill-conditioned groups use the
   retained SVD factors. Every residual is computed from `data - shapes.T @ a`.
   MCMC uses the same actual-model residual, closing the same defect there.
3. **Nonlinear covariance:** `inv(J.T @ J)` squares the condition number and can
   lose substantial precision before it throws. Returning zero on failure was
   false precision. The replacement scales columns before SVD, transforms the
   full-rank covariance back to parameter units, and withholds the whole joint
   covariance if the scaled condition number reaches `1/sqrt(eps)`. This is a
   conservative numerical reliability threshold, not a scientific resolution
   limit. No pseudoinverse diagonal is advertised as uncertainty. Individual
   symmetric errors are also withheld at active bounds or when one standard
   error crosses either parameter bound. The bound check is a warning about a
   local Gaussian approximation, not a constrained confidence interval.
4. **Amplitude covariance:** singular/ill-conditioned amplitudes retain their
   minimum-norm model, but their joint covariance/errors are unavailable. The
   previous normal-equation pseudoinverse fallback was not a covariance for
   individually identifiable amplitudes. Nonfinite numerical overflow remains
   an evaluation failure, distinct from expected unavailable covariance.
5. **Shared nonlinear Jacobian:** the accumulated derivative of a shared
   parameter was multiplied by one peak's amplitude. The correct contribution
   sums each peak derivative weighted by its own signed amplitude, including
   the VarPro correction term. This is independently tested with shared
   linewidths and different signed plane profiles. The reported default fits
   have no shared nonlinear parameters, so this defect did not cause them.
6. **Readback:** JSON `standard_error: null` previously became `stderr = 0` in
   `ResultsLoader`. It now stays unavailable, including for old 4.0.0 documents.

The Jacobian is defined for a locally constant rank. At a rank-changing point
the reduced objective need not have a derivative; the retained-factor formula
must not be interpreted as proving smoothness or identifiability there.

## 36N-H / 93N-H before and after

All positions are unfolded ppm; widths are unapodized FWHM in Hz.

| Peak | Listed F2 / F3 | Before, 2 passes F2 / F3 | After, 2 passes F2 / F3 | After F2 / F3 widths |
| --- | --- | --- | --- | --- |
| 36N-H | 119.783262 / 8.809981 | 119.833975543 / 8.809697675 | 119.833975544 / 8.809697675 | 11.352566 / 18.897941 |
| 93N-H | 119.835983 / 8.810906 | 119.834069882 / 8.809695979 | 119.834069880 / 8.809695979 | 11.345572 / 18.927005 |

F2 bounds are [119.166305622, 120.400218378] and
[119.219026622, 120.452939378], respectively. F3 bounds are
[8.747458156, 8.872503844] and [8.748383156, 8.873428844]. Width bounds remain
[0.1, 200] Hz. No separation constraints were added.

Before the correction, the underlying F2 errors were 0.849345 and 0.852931 ppm;
F3 errors were 0.0152412 and 0.0153480 ppm. These already contradicted precise
individual assignments, independently of the CSV bug. The corrected SVD-based
F2 errors are withheld because they cross parameter bounds; F3 errors are
0.0145109 and 0.0146126 ppm, overwhelmingly larger than the fitted separation
of 0.000001695 ppm. Width errors are withheld. The group carries an explicit
uncertainty warning, and the GUI prevents applying these fitted positions.

The fact that positions barely change is an important **negative finding**:
the exact-rank correction cannot cure a full-rank pathological local solution.

## All three collapsed groups

Fixed current settings, two passes:

| Group | χ² before → after | Shape rank / condition after | Scaled Jacobian condition | Optimality after |
| --- | --- | --- | ---: | ---: |
| 269 | 20805.211620 → 20805.211620 | 3 / 2255 | 9.16e6 | 3864.5 |
| 360 | 18378.445003 → 18378.445003 | 4 / 2780 | 6.71e6 | 6579.5 |
| 424 | 21313.623645 → 21313.623645 | 2 / 6309 | 8.80e7 | 38652.9 |

Every original termination is `ftol` satisfaction; it is not evidence of a
stationary, identifiable optimum. QR-versus-actual-model residual gaps in these
full-rank fits are around 1e-9 in normalized residual norm, far too small to
explain the observed χ². All three groups are now warned. Group 424's entire
nonlinear joint covariance is withheld; the other two have individual errors
that cross bounds even where the numerical joint covariance remains invertible.

The nearly coincident pairs have opposite signed amplitudes in 130 of 131
planes. In group 269 the coefficient ranges are approximately [-1.15e10, 1.46e12]
and [-1.46e12, 1.15e10]; the 75N-H neighbor is retained. Coefficients depend on
lineshape normalization, so absolute magnitudes must not be compared directly
with the old normalized-height convention. Their cancellation is the important
finding. Relative residual norms are 4.93%, 7.06%, and 4.99% for groups 269,
360, and 424. Mean plane-wise residual RMS is 1.30, 1.14 and 1.53 noise units.
These residuals are not compatible with declaring an exact, uniquely resolved
peak decomposition just because the optimizer returned success.

## Controlled historical comparison

The release uses `r2 = 20 s^-1`, with [0.1, 200] s^-1 bounds. Its physically
comparable current settings are `lw = 20/pi = 6.3661977 Hz`, with
[0.1/pi, 200/pi] Hz bounds. Release center windows are ±25 Hz; current defaults
are ±50 Hz. Phase and J fitting are off in all experiments.

Evaluating current shapes at each released optimum, then normalizing each shape
to unit norm, agrees with the actual release shape arrays to at most **3.89e-16**.
The lineshape functions therefore agree for these fits up to amplitude-absorbed
normalization; there is no evidence of a lineshape regression here.

| Group | Current start 20 Hz, ±50 Hz, 1 pass | Only start → 20/pi Hz | Only windows → ±25 Hz | All historical settings, VarPro | Same settings, LMFit/current shapes | Actual release, matched noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 269 | 20797.70518 | 21548.16890 | 20951.66228 | 21548.16885 | 21548.16885 | 21548.16885 |
| 360 | 18428.00488 | 17996.06407 | 18428.00450 | 17996.06408 | 17996.06409 | 17996.06409 |
| 424 | 21456.13539 | 25164.94104 | 21497.51085 | 25164.94102 | 25164.94102 | 25164.94101 |

Entries are actual-model χ². Narrowing center windows alone does not resolve
collapse; changing linewidth initialization selects a distinct basin. Five
shift jitters per group, uniform ±1e-4 ppm with seeds 0–4, give the same
respective basin at each linewidth start. With identical inputs, the fit is
deterministic; it is sensitive to linewidth starts, not random optimization.
The ±1e-4 jitter experiment is a sensitivity probe, not a proof of basin size.

After two matched passes:

| Group | VarPro/current χ² | Actual release χ² |
| --- | ---: | ---: |
| 269 | 21577.285903 | 21577.285684 |
| 360 | 17946.844322 | 17946.844315 |
| 424 | 25070.757396 | 25070.757405 |

For group 269, matched current positions are 119.803023772 / 8.809371269 and
119.845232100 / 8.811151441. The release gives 119.803034823 / 8.809371847 and
119.845264680 / 8.811152590. Differences reflect stopping/numerical derivative
tolerances, not a different numerical objective. Current position errors are
0.00108107 / 0.0000704104 and 0.00360305 / 0.000110062 ppm.

LMFit's default covariance scaling inflates errors by the fitted residual
variance and counts only its varied nonlinear parameters when computing degrees
of freedom. Current nonlinear errors use supplied statistical noise, and the
analytical output statistics also count eliminated amplitudes. To isolate the
covariance calculation, the LMFit/current-shapes control disables `scale_covar`;
its errors agree closely with current errors. For example, released 36N-H F2
error 0.00146133 ppm versus the unscaled control's 0.00107965 ppm is predominantly
residual-variance scaling, not evidence that one solver estimates a different
position. Current exported amplitude errors retain their documented conditional
noise covariance and residual-inflation convention.

For groups 269 and 424, the collapsed basin has *lower actual χ²* than the
historical distinct basin. For 360, the distinct basin fits better. Restoring
old starts, choosing a solution just because it separates peaks, or forcing
positive amplitudes would therefore introduce a scientific modeling decision.
No such decision is hidden in this fix. The unconstrained signed model can
approach derivative-like shapes with cancelling amplitudes as two columns
coincide. At those solutions, the data/model do not support trustworthy
individual positions; alternative model assumptions require separate review.

## Numerical tests, performance and limitations

Focused tests cover exact/near duplicate shapes, identifiable close and
separated Gaussians, independent signed profiles across planes, actual-model
objective/Jacobian agreement, shared linewidth derivatives, unit scaling,
rank-deficient/ill-conditioned covariance, active bounds, noise scaling and
20 seeded noisy ground-truth fits (at least 85% coverage within two reported
standard errors). Real-data tests retain the complete three groups and assert
consistent residuals and warnings; historical-setting tests protect their
separate local basins without demanding artificial separation in degenerate fits.

At collapsed experimental endpoints, a 1e-6 ppm finite-difference step crosses
too much of the separation and can falsely imply a Jacobian error. Sweeping
steps to 1e-8 ppm reduces relative Jacobian differences to about 1.7e-7, 1.0e-7
and 4.8e-6 in first-pass groups 269, 360 and 424. Identifiable historical-setting
fits reach relative agreement around 1e-9. Exact rank changes are excluded from
claims of differentiability.

Nine single-thread sweeps of all 121 real initial group matrices, including
amplitudes, residuals and Jacobian pseudoinverse work, have median times:
original QR 3.382 ms, rank-aware 3.426 ms, direct lstsq/pinv 5.149 ms. A separate
run gave about 5% rather than 1.3% overhead, illustrating microbenchmark noise.
Full two-pass optimizer time, excluding diagnostic probes, was 1.487 s before
and 1.561 s after (about 5%). No new runtime dependencies were introduced.

Local covariance is not a global confidence region, an assignment test, or a
cure for model misspecification. Conditional amplitude errors omit nonlinear
parameter uncertainty. Warnings conservatively reject questionable local
Gaussian errors, while retaining finite models and explicit optimizer success
provenance. This PR fixes numerical correctness and false precision; the
remaining near-coincident experimental assignments need scientific review.

## Reproduce

```bash
# Actual CLI, default two passes, explicit for comparison:
uv run peakfit fit examples/02-advanced-fitting/data/pseudo3d.ft2 \
  examples/02-advanced-fitting/data/pseudo3d.list \
  -z examples/02-advanced-fitting/data/b1_offsets.txt \
  --refine 2 --workers 1 --headless -o /tmp/peakfit-overlap

uv run python tools/validate_overlap.py --passes 2 --output /tmp/current.json
uv run python tools/validate_overlap.py --scenario start --output /tmp/start.json
uv run python tools/validate_overlap.py --scenario bounds --output /tmp/bounds.json
uv run python tools/validate_overlap.py --scenario matched --passes 2 --output /tmp/matched.json
uv run python tools/validate_overlap.py --jitter-seed 0 --output /tmp/jitter-0.json
uv run python tools/benchmark_projection.py > /tmp/performance.json
```

To compare pre-fix main, export its `src/` tree and pass `--source /tmp/main/src`
to `validate_overlap.py`. Export `v0.10.5` to a separate directory, install LMFit
and NetworkX in an isolated validation environment, and run
`validate_release_overlap.py --source /tmp/v0.10.5 --output /tmp/release.json
--noise 11922.544916060584`. Use that same environment with
`validate_overlap.py --source /path/to/current/src --lmfit --scenario matched
--passes 2 --output /tmp/lmfit.json` to isolate optimizer differences. Ensure
the same NumPy/SciPy versions are used (the recorded runs used the current
`.venv` site-packages on `PYTHONPATH`). The release retains LMFit's default
residual covariance scaling; the controlled current-shapes solver disables it.

## Delivery verification

The full suite passes: **292 tests**, including offscreen GUI tests for editing,
Undo/Redo, fit export and model comparison. Ruff lint and format checks, `ty
check --error-on-warning`, both import-linter contracts, and `prek run
--all-files` pass. `uv build --offline` builds the wheel and source archive.
The build emits the existing nonfatal notice that the uv-build requirement
range does not contain the installed uv version 0.12.23; both artifacts build
successfully. `graphify update .` refreshed the AST graph without API use.

The first commit isolates the CSV mapping correction. Numerical/output
corrections and the reproducible scientific evidence follow separately. No
optimizer initialization, bounds, pass defaults, noise settings, signed-amplitude
policy, or peak-separation rule is changed by the correction.
