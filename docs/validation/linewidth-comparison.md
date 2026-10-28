# Two-start CLI validation

Validated 2026-10-28 against main including PRs #59 and #60. The new CLI
composes the unmodified production pipeline twice, retaining both complete
scientific outputs. It does not change the VarPro objective, optimizer scaling,
noise treatment, bounds, signed amplitudes or assignment assumptions.

## Reproduction

```bash
uv run python tools/validate_linewidth_comparison.py --directory /tmp/linewidth-61
```

The driver executes three fresh processes each for the ordinary CLI and
`--compare-linewidth-start 13`, with the bundled pseudo-3D spectrum, peak list
and offsets, all 166 peaks / 121 clusters / 131 planes, two passes, one worker
and one BLAS thread. It retains complete outputs and logs in the requested
fresh directory. On macOS, an `atexit` wrapper measures each CLI process's
own `resource.getrusage(RUSAGE_SELF).ru_maxrss` in bytes. Timing includes process
startup, input loading, both fitting passes, finalization and normal output
writing. No diagnostic finite differences or extra fits are hidden in timing.

[Machine-readable evidence](linewidth-comparison-results.json) contains commands,
input hashes, individual measurements and complete target parameter values,
uncertainties, χ², reduced χ², termination and optimizer provenance for all
repeats. The original base commit is recorded; measurements used the working
implementation of this PR. Full candidate trees were retained under
`/tmp/peakfit-linewidth-61-measurements` during validation; the driver recreates
them without relying on that temporary path.

The driver asserts agreement with the production two-pass records from
[PR #60](linewidth-scaling.md), including its `single_global_13` control:
absolute χ² difference <0.02, every target/neighbor shift difference <2e-5 ppm
and linewidth difference <0.02 Hz. These allow stopping-tolerance/platform
variation in ill-conditioned endpoints while remaining far smaller than the
observed differences between basins. Every baseline target must retain its
uncertainty warning and every alternative target must retain its absence.
All three repeats passed, with identical numerical target values on this host.

## Competing results

These are actual-model χ², evaluated with each candidate's terminal corrections.

| Cluster / pair | Baseline χ² | 13 Hz χ² | Baseline F2/F3 separation, ppm | 13 Hz F2/F3 separation, ppm |
| --- | ---: | ---: | --- | --- |
| 269 / 36N-H, 93N-H | 20805.211620 | 21577.285902 | 0.000094337 / 0.000001695 | 0.042208376 / 0.001780174 |
| 360 / 5N-H, 76N-H | 18378.445003 | 17946.844323 | 0.000055037 / 0.000018663 | 0.031916156 / 0.010069661 |
| 424 / 53N-H, 147N-H | 21313.623645 | 25070.758053 | 0.000036802 / 0.000002830 | 0.089677850 / 0.003187281 |

All six fits terminate on `ftol`. Baseline optimalities are approximately
3865, 6580 and 38653; alternative optimalities are approximately 6.28, 0.952
and 13.6. Every baseline has unavailable parameter uncertainties and an explicit
uncertainty warning; alternative local parameter errors are finite without that
numerical warning. The comparison preserves these distinctions, optimizer
classification and both scientific outputs, rather than declaring a successful
assignment. Its human report names every pair's F2/F3 separations without a
reliability score or a separation threshold; its CSV provides each fitted shift,
linewidth, error and position difference by stable parameter identity.

The separated basins have higher χ² in groups 269 and 424. Their assignment
correctness remains unresolved. Complete multi-pass runs have independently
updated neighboring corrections; these χ² differences must not be confused with
optimizing exactly the same isolated corrected-data objective. Both retain
unconstrained signed amplitude profiles in normal candidate outputs.

## Runtime and memory

Fresh-process medians, three repeats per mode on this host:

| Mode | Total seconds | Peak RSS, MiB |
| --- | ---: | ---: |
| Ordinary single-start CLI | 7.079 | 1078.94 |
| Two-start CLI | 13.089 | 1125.44 |

Comparison adds about 6.01 seconds (85%) and 46.5 MiB (4.3%) peak RSS here.
This includes two full sets of output serialization; optimizer-only timings
from PR #60 are not directly comparable. The original spectra are shared
read-only; candidate peaks, masks/data, corrections and fit state are copied
independently. Retaining both complete results costs additional memory and disk.
These are local measurements, not performance guarantees for other datasets,
worker counts, machines or optional simulated-spectrum outputs.

## Executable protections

`tests/test_linewidth_comparison.py` checks the actual SciPy starts and own-pass
warm starts; explicit values including a value equal to the model default and
file-loaded values; fixed/step-fixed linewidths and bounds; unchanged data and
original corrections; deterministic CSV values; stable cluster/peak/parameter
alignment; unavailable errors, unusable and non-converged outcomes; preservation
of either candidate when the other raises; and ordinary CLI output layout.
Overlapping synthetic Gaussian peaks with known signed profiles recover their
original parameters from both starts. Independent `numpy.linalg.lstsq` model
reconstruction verifies the same residual objective, without changing scientific
constraints. Existing numerical and experimental overlap regressions remain in
place. No runtime dependencies or GUI changes are introduced.

Delivery verification: all **313 tests** pass with offscreen Qt and Agg plotting.
Ruff lint and format checks, `ty check --error-on-warning`, both import-linter
contracts and all `prek` hooks pass. `uv build --offline` builds wheel and source
archives, with the existing nonfatal notice about the installed uv version
falling outside the declared uv-build requirement range. `graphify update .`
refreshes the AST graph without API use. A final real CLI run retains both
complete candidates, 1328 parameter comparison rows and the expected target
warnings/separations.
