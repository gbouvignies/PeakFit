# Development Guide

This guide records the current contributor workflow and coding conventions. Keep
it short; implementation details should live in code and tests.

## Commands

Use `uv` for project commands.

```bash
uv sync --all-extras
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check --error-on-warning
uv run lint-imports
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q
uv run prek run --all-files
uv build
```

Use `prek`, not `pre-commit`, for hooks.

## Testing And Scientific Fixtures

Run the complete suite headlessly with `QT_QPA_PLATFORM=offscreen` and
`MPLBACKEND=Agg`, as shown above. This is the supported environment for
automated tests that may import GUI or plotting dependencies.

Tests protect both numerical/scientific behavior and refactoring compatibility
baselines. Treat golden files, regression data, tolerances, and reference values
as evidence with an explicit provenance and intended claim. Do not silently
regenerate them, weaken their assertions, or change them without explaining and
reviewing the scientific or contract reason.

## Architecture Rules

- Keep numerical code independent from Typer, Rich, Qt, and filesystem side
  effects unless that boundary is intentionally changed.
- Prefer direct functions and explicit data flow over managers, service layers,
  registries, factories, or adapters.
- Keep `engine` focused on numerical/domain primitives.
- Keep `fit` focused on loading, validation, orchestration, and fit-output
  assembly.
- Keep `auto_pick` isolated as an experimental workflow.
- Keep output formats canonical: `summary/fit.json` and CSV tables under
  `tables/`.

The import-linter contracts in `pyproject.toml` enforce the current package
layers.

## Agentic Development

For AI-assisted changes, keep progress reviewable and convergent:

- Change one coherent thing at a time.
- Prefer deleting or merging stale docs over adding new planning files.
- Turn important architecture rules into executable checks when possible.
- Commit after each verified slice.
- Record lasting non-obvious decisions in `docs/decisions.md`.

Before starting another broad refactor, check whether `docs/architecture.md`,
the examples, and output docs still match the code.

## CLI And Terminal Output

PeakFit runs can be long. Terminal output should leave a useful scrollback:

- Show a compact input/config/output summary before work starts.
- Keep routine success terse.
- Make parse errors, failed clusters, boundary hits, and high reduced
  chi-squared easy to find.
- Prefer actionable messages over decorative panels.
- Do not expose internal architecture names in user-facing output.

Shared helpers under `src/peakfit/ui/` are useful only when they reduce
duplication and improve consistency. They may be merged or removed during
simplification.

## Typing

- Do not add `from __future__ import annotations`; Python 3.14 already defers
  annotations.
- Use modern syntax: `X | Y`, `list[str]`, `dict[str, float]`.
- For NumPy-style public inputs, prefer `numpy.typing.ArrayLike` and normalize
  immediately with `np.asarray(...)`.
- Return `numpy.typing.NDArray[...]` for vectorized numerical functions.
- Use overloads only when the public API guarantees scalar-in/scalar-out
  behavior.
- Enforce shape requirements at runtime; do not encode fragile shape types.

## Documentation

Documentation should be consolidated with code changes. Prefer one accurate page
over several overlapping pages. Delete stale notes when they stop matching the
current design.


## GUI responsiveness

PR #53 was profiled natively on macOS with the bundled `pseudo3d.ft2` dataset
(131 × 256 × 546 samples, 166 peaks, 69.85 MiB input), ten contour levels per sign,
and thresholds in multiples of the viewer's first-plane noise. Representative
single-process timings are diagnostic measurements, not portable performance targets.

| Threshold | Contour vertices | Extraction + Qt path construction before | Native paint before | Selected all-plane region | Data + correction arrays before |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5× | 29,206 | 7 ms | 13 ms | 39 samples, 1 peak | 0.08 MiB |
| 3× | — | — | — | 77,874 samples, 166 peaks | 156 MiB |
| 2× | 72,093 | 11 ms | 18 ms | 139,616 samples, 166 peaks | 279 MiB |
| 1× | 229,856 | 23 ms | 49 ms | 139,776 samples, 166 peaks | 279 MiB |
| 0.5× | 489,861 | 41 ms | 100 ms | 139,776 samples, 166 peaks | 279 MiB |
| 0.1× | 1,097,210 | 75 ms | 216 ms | 139,776 samples, 166 peaks | 279 MiB |

All-plane segmentation took about 7 ms, with boundary-component merging under
1 ms. The old `create_clusters()` path took 26–43 ms and materialized every cluster;
at 5× its combined data/correction arrays occupied 12.1 MiB although the selected
group needed only 0.08 MiB. First all-plane noise estimation took about 150 ms.

The prolonged freeze was separately reproduced while replacing a 664-row parameter
table: repeated ten-second stack dumps remained inside `QTableWidget.setItem()` for
more than 30 seconds. Clearing the old rows, using interactive rather than live
content-based column sizing, and sizing once after population reduced repeated
native replacement to 20–21 ms, independently verified with the allocation safeguard
temporarily raised in the diagnostic harness. This table fix is distinct from
background preparation and the contour workload guard.

After correction, a native sweep through 5, 4, 3, 2, 1, 0.5, 0.2, 0.1, and back to
5× exercised panning and plane navigation while a 10 ms Qt heartbeat ran. Maximum
heartbeat gaps were 28–46 ms; transitions settled in 190–240 ms including the 150 ms
debounce. Frame capture/painting was 13–22 ms. Dense regions were reported before
allocating their arrays; dense contours were reported before constructing Qt paths.
No scientifically truncated region or partial contour set was displayed.

Resource budgets deliberately bound both memory and GUI rendering work: 128 MiB
for selected data/correction arrays, 1,000 parameter rows (664 measured rows cost
20 ms), and 100,000 contour vertices (72k measured vertices painted in 18 ms).
They are not minimum intensity thresholds. Increasing the threshold or reducing
contour levels can bring work back within budget; large numerical fits remain a
CLI workflow. Statistical noise is independent of these budgets and the contour
threshold.

Native resident memory was roughly 0.94–0.98 GiB during the corrected sweep; initial
all-plane noise estimation and allocator retention remain expensive. The resource
guards prevent the additional 156–279 MiB cluster arrays and the large repeated
contour allocations, but do not establish a low total-process memory bound.

One representative regression keeps a worker deliberately busy while Qt heartbeats,
selection changes, and slider updates continue. It verifies serialized latest-work
handling, rejection of obsolete results, explicit large-region/contour messages,
and non-waiting close. The original scientific scenario still compares selected
membership, grid indices, data, parameters, and alias signs against the engine path.
