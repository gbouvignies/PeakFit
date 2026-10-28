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

### Prepared local execution (Step 2A)

Native Cocoa validation on 2026-10-28 used the same bundled experimental data on
macOS arm64/Python 3.14. The selected group contained three overlapping peaks,
75 sampled points per plane, and all 131 planes. It converged in eight function
and eight Jacobian evaluations, with reduced χ² = 1.36219. These are observations
from this machine and harness, not performance guarantees.

| Measurement | Observed |
| --- | ---: |
| VarPro plus terminal analytical evaluation | 24 ms |
| Spawn startup (including imports/handshake) | 831 ms |
| Input serialization plus owned copy | 0.95 ms |
| Serialized input payload | 183,481 bytes |
| End-to-end execution, transfer, and finalization | 862 ms |
| Child peak resident memory | 210.9 MiB |
| Parent high-water resident memory | 981.3 MiB |
| Cancel after child startup through process reaping | 20 ms |

The complete spectral data array is 69.85 MiB; only 78,600 bytes of selected sampled
data plus corrections, model geometry, parameters, and provenance cross into the
child. The parent high-water value includes loading, initial all-plane noise
estimation/preparation, rendering, and screen capture; it is not the incremental
cost of optimization. The existing noise-preparation peak remains substantial.
The child import footprint dominates this small fit. No speculative startup cache,
shared-memory transport, or persistent worker pool was added.

During execution the native harness switched Inspect/Fit, selected plane 25, moved
the crosshair, changed the displayed threshold from 5× to 6×, and requested updated
contours. The submitted threshold remained 60349.7535 while displayed contours used
72419.7042; working peak records stayed identical. Dataset replacement was rejected.
A 10 ms Qt timer continued throughout (251 callbacks; largest gap 178 ms, including
native screen capture/layout/navigation, so this is not an isolated optimizer stall
measurement). Cancellation and close with an active spawned child left zero fit
children. Canceling the native unsaved-result close dialog also kept the application
and completed result intact. Running/completed screenshots are in
`docs/images/local-fit-*.png`.

The two integration checks in `tests/test_gui_fit_execution.py` cover one-pass engine
agreement for overlapping multi-plane signals with duplicate assignments, unfolded
indirect coordinates, fixed linewidths, position bounds, signed amplitudes, and
unaltered inputs; they also exercise actual spawn/transport, provisional/unusable
outcomes, cancellation, child failure, stale delivery, retained-result replacement
guards, navigation/contour responsiveness, and shutdown. Existing GUI tests protect
manual editing and preview behavior. A full offscreen run passed 233 tests. One
earlier full run segfaulted inside native Qt/Python garbage collection; it did not
reproduce on the subsequent full run (clean main: 231 passed). The new lifecycle
test explicitly drains its preview worker and deletes its Qt objects on Qt's thread.

Cancellation terminates the private child; it never attempts to interrupt SciPy
inside a Qt task. A bridge thread owns spawn, serialization, receive, finalization,
and bounded terminate/join/kill cleanup. Application shutdown signals cancellation;
the bridge is non-daemon so interpreter exit cannot abandon a partly spawned child.
Qt never joins it synchronously. Failed/canceled attempts retain the earlier result;
completed numerical values are frozen using the same finalization seam as the CLI.
The 128 MiB preparation budget is not an optimizer/process memory bound: dense
Jacobian and finalization arrays add memory. No samples are dropped to fit a budget.

### PR #54 focused lifecycle review

The intermittent crash was investigated with a bounded comparison on macOS arm64,
CPython 3.14.5, PySide6 6.10.1, and NumPy 2.4.1. All runs enabled
`PYTHONFAULTHANDLER=1` and pytest's 120-second per-test timeout. Immutable source
archives of PR head `71536a4` and main `86a0528` used the same installed interpreter
and dependencies, with alternating native Cocoa full-suite runs:

| Scenario | Runs | Result |
| --- | ---: | --- |
| PR head, native Cocoa full suite | 3 | 233 passed each; 58–59 seconds |
| Main, native Cocoa full suite | 3 | 231 passed each; 56–58 seconds |
| Corrected PR, native Cocoa GUI/lifecycle tests | 3 | 10 passed each; about 6 seconds |
| Corrected PR, original offscreen full-suite invocation | 1 | 233 passed; 69 seconds |

No crash recurred. The saved original macOS report records `EXC_BAD_ACCESS` at
address `0x8` in `SignalManagerPrivate::qtMethodMetacall` during queued Qt dispatch.
It does not identify the receiving application object or establish a root cause.
This is not evidence that the crash was fixed or that it belongs to main, the fit
bridge, or the preview worker; no speculative lifetime refactor was made.

Teardown inspection confirmed that explicit close marks both lanes closed before
destroying the viewer. A temporary diagnostic also exercised the existing
`ViewerComputation` callback in two orders: completion after parent/worker QObject
deletion, and completion queued before deletion but dispatched afterward. Each
order passed five iterations, with invalidated C++ objects, completed callbacks,
and zero delivered UI completions. Temporary runners/logs stayed outside the repo;
no new testing infrastructure was introduced.

Native mouse/keyboard validation used the real bundled 131-plane spectrum: run a
fit, retain its result, make a working edit, start a replacement, switch Inspect/Fit,
and navigate to plane 25. Only the disposable numerical child was temporarily
SIGSTOP-paused to keep the modal sequence deterministic. Answering Yes to close
then Cancel at either the peak-discard or result-discard dialog preserved the same
running child, retained result, dirty peaks, and an unset cancellation flag.
Explicit Cancel fit reaped the child; accepting every close/discard decision on a
subsequent fit exited without remaining validation/fit processes. No numerical or
subprocess architecture changes were needed.

The existing close test now protects those transactional decisions, including a
fit completing during an earlier dialog. The existing spawned lifecycle test
checks direct `set_peaks()` rejection before any history/selection/revision changes.
