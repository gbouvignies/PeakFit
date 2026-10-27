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

PR #53 was tested natively on macOS with the bundled `pseudo3d.ft2` dataset
(131 × 256 × 546 samples, 166 peaks, 69.85 MiB input), ten contour levels per sign,
and thresholds in multiples of first-plane noise. These are representative local
measurements, not portable performance guarantees.

The prolonged freeze was reproduced while replacing a 664-row parameter table:
repeated ten-second stack dumps remained inside `QTableWidget.setItem()` for more
than 30 seconds. Clearing old rows, using interactive column sizing, and sizing
once after population reduced repeated native replacement to 20–21 ms. This fix
is retained. All-plane segmentation took about 7 ms, including under 1 ms for
boundary merging. Only the selected group is materialized; at 5× its arrays use
0.08 MiB versus the old all-cluster path's 12.1 MiB.

The current UX was compared directly with the original merged viewer at commit
`21e5cfc`, using the same dataset, threshold sequence, window size, and selected
spectral range. Both retain all 20 contour paths at every threshold. The original
inspection workspace has three plots; the revised Fit workspace has scope metadata.
Times below include event processing and frame capture, so visible side panels
account for part of the difference; they are not isolated renderer speedups.

| Threshold | Vertices | Original local pan / zoom | Revised local pan / zoom | Original / revised overview capture |
| --- | ---: | ---: | ---: | ---: |
| 5× | 29,206 | 62 / 94 ms | 23 / 21 ms | 29 / 15 ms |
| 1× | 229,856 | 59 / 58 ms | 28 / 35 ms | 29 / 22 ms |
| 0.1× | 1,097,210 | 86 / 82 ms | 43 / 42 ms | 53 / 43 ms |

The sweep 5 → 2 → 1 → 0.5 → 0.2 → 0.1 → 5× verified full contour visibility,
zoom/pan, recovery to ordinary scope, and unchanged plot/control geometry before,
during, and after threshold updates. At 0.1× the revised update settled in about
0.51 s, versus 0.30 s originally; this includes the Fit workspace's 150 ms debounce
and scope update. The previous contour set remains visible throughout. Dense
overviews still cost more to draw; there is no blanket contour-vertex cutoff.

One serial background lane handles preparation and numerical contour extraction,
with latest-pending requests and revision checks. Qt paths and widgets stay on the
GUI thread. The renderer uses full geometry, and completed sets replace the old
sets together. Fixed-height, width-independent status/metadata fields prevent
progress or warning text from reflowing the plot; full messages are available in
Details. Efficient parameter-table replacement and non-waiting shutdown are retained.

A large region does not suppress visualization. At 0.1×, the exact scope contains
139,776 spectral points and all 166 peaks. Its two grid-index arrays use about
2.13 MiB, versus 279 MiB for dense numerical data/correction arrays. Scope geometry,
all markers, and metadata remain available without allocating the latter. Numerical
arrays are limited to 128 MiB; that limit affects numerical preparation only. `PreparedLocalFit.cluster` and `.parameters` are absent
for a scope-only result, which must not be treated as ready numerical input.

The first all-plane noise estimate still has substantial temporary memory use, and
full low-threshold vector geometry requires more memory and painting time. No low
total-process memory bound or constant frame rate is promised. Regression scenarios
cover exact scientific geometry/data/identity, complete dense contours, stable
layout through progress and numerical limits, latest-work handling, and safe close.
