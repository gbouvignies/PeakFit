# Testing and Validation Baseline

This page records the repository baseline observed on 2026-07-25. It describes
current checks and evidence; it does not certify scientific validity.

## Intended environment

**Verified.**

- Python: CPython 3.14 or newer (`pyproject.toml` and `uv.lock`).
- Environment and command runner: `uv`.
- Hook runner: `prek`, not `pre-commit`.
- Headless tests: `QT_QPA_PLATFORM=offscreen` and `MPLBACKEND=Agg`.
- Runtime dependencies include NumPy, SciPy, nmrglue, pandas, Pydantic, Typer,
  Rich, Matplotlib, PySide6, emcee, h5py, and threadpoolctl.
- Development dependencies include pytest, Ruff, ty, import-linter, build
  support, coverage, xdist, timeout support, and prek.

The audit synchronized successfully with `uv 0.11.15` on CPython 3.14.5. All
declared required and optional dependencies installed; no dependency was
missing.

## Commands and observed results

| Command | Result |
| --- | --- |
| `uv sync --all-extras` | Passed; resolved 50 packages and installed the project plus 48 packages into a new `.venv`. |
| `uv run ruff check .` | Passed: “All checks passed!” |
| `uv run ruff format --check .` | Passed: 147 files already formatted. |
| `uv run ty check --error-on-warning` | Passed: “All checks passed!” |
| `uv run lint-imports` | Passed: 2 contracts kept, 0 broken; 133 files and 254 dependencies analyzed. |
| `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q -p no:cacheprovider` | Passed: 126 tests in 38.15 seconds. |
| `uv build` | Passed; built `peakfit-2025.11.0.tar.gz` and `peakfit-2025.11.0-py3-none-any.whl`. |

`uv run prek run --all-files` is the aggregate hook command. It runs
whitespace/EOF, YAML, TOML, large-file and conflict checks, then Ruff with
automatic fixes and formatting, ty, and import-linter. Because it is
write-capable, its output must be followed by a worktree review.

CI runs Ruff lint and format checks, ty, import-linter, and pytest with coverage
on Ubuntu and macOS. The package publish workflow builds distributions with
`uv build`.

## Executable protection

**Verified.** The suite contains:

- a full real-data CLI fit fixture reused for output layout, JSON schema,
  public fit/plot/MCMC workflow, and numerical range checks;
- a second real-data golden fit covering summary and parameter-table contracts;
- strict lineshape value and derivative equivalence tests at
  `rtol=1e-12`, `atol=1e-12`;
- representative Gaussian and Lorentzian FWHM half-height checks at
  `rtol=1e-12`, `atol=1e-12`;
- auto-pick candidate, ROI, naming, parameter-sharing, constraint, threshold,
  and orchestration tests;
- output schema, output-plan, CSV layout, Markdown report, run README,
  configuration, scalar parameter, and ResultsLoader path tests;
- authoritative-final-outcome tests covering stable cluster identity, mixed
  classifications, terminal provenance, immutable analytical values, and
  agreement among CLI review, JSON, CSV, Markdown, README, and simulation;
- deterministic intensity, CEST, and CPMG transformation tests.
- unequal point/series cluster tests covering strict construction, cluster
  creation and merging, amplitude counts, fit statistics, uncertainty scaling,
  state/output invalidation, and reconstruction.

Pytest uses strict configuration and registered `slow`, `mcmc`, and `gui`
markers. Warnings are errors except for known nmrglue deprecations and a NumPy
future warning.

## Numerical regression fixtures

**Verified.** `tests/data/golden/baseline.json` contains:

| Value | Current assertion |
| --- | --- |
| `chi_squared = 1829681.0` | New real-data fit must be positive and within 50% relative difference. |
| `n_clusters = 121` | Exact equality. |
| `n_peaks = 166` | Exact equality. |
| `pre_simplification_parameter_rows = 22410` | Current parameter table must be non-empty and smaller. |

The baseline was introduced in commit `ec7b5b4` (“Refactor PeakFit
architecture”, 2026-05-06); the row-count comparison was added in `24487da`
(“Simplify output and workflow architecture”, 2026-06-12).

**Verified.** Tests and comments describe these numbers as a refactoring safety
or compatibility baseline and explicitly allow a broad chi-squared tolerance.
No repository evidence identifies them as independently scientifically
validated reference values. The row count is explicitly historical, not a
scientific expectation.

**Unknown.** Whether the chi-squared value, cluster count, peak list, and example
fit were reviewed against an external scientific ground truth requires
maintainer knowledge.

## Runtime and determinism

- **Verified.** The complete suite took 69.16 seconds in the audited environment;
  real-data subprocess fits dominate the runtime. The integration suite repeats
  substantially the same fit in two module-scoped fixtures.
- **Verified.** The MCMC smoke test uses a short real-data run with unseeded
  walker initialization. It asserts successful workflow completion, not stable
  posterior values.
- **Verified.** Basin hopping supports a seed in TOML configuration but defaults
  to no seed. No basin-hopping numerical regression test exists.
- **Likely.** Runtime will vary materially with CPU count because routine fits
  default to all CPUs and cluster tasks use multiprocessing.
- **Verified.** The golden test comments call nonlinear optimization stochastic
  and use a 50% chi-squared tolerance. The default VARPRO path has no explicit
  random generator, so the exact source of observed variation is not established
  by the repository.

## Important gaps

The following public or scientific contracts lack direct executable protection:

1. equality between optimizer convergence metadata and persisted result
   metadata;
2. scalar/vector `FittingState` parameter consistency;
3. numerical round-trip behavior for pickle state and JSON reconstruction;
4. real CSV, JSON, and Excel peak-list parsing and equivalence with preflight
   validation;
5. plane-value length validation and plane exclusion;
6. NMRPipe 2D/3D dimensional and axis-label variants;
7. config-file/CLI precedence;
8. fixed-seed basin-hopping and MCMC results, saved-chain round trips, and
   burn-in edge cases;
9. automatic-picking end-to-end numerical output on real data;
10. Qt spectrum viewer and interactive automatic-pick stepper behavior;
11. scientifically reviewed expected parameter values, amplitudes, and residual
    quality for a controlled synthetic or experimental reference dataset.

## Completed-result authority checks

The completed-fit contract is protected at the `FinalFitOutcome` seam. Tests
verify that writers and simulation use frozen outcome values, that source-state
mutation cannot change completed output, that nonconsecutive `cluster_id`
values survive every projection, and that unusable outcomes never receive
fabricated numerical values. `FittingState` remains covered separately as a
continuation-state persistence contract; it is not a completed-output input.

**Recommendation.** Add new reference values only after their provenance is
recorded. Label them explicitly as mathematical identities, synthetic
ground-truth expectations, compatibility baselines, or independently reviewed
scientific references.
