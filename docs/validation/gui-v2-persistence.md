# GUI v2 Milestone 3: save, reopen and continue

Native Cocoa validation on 30 October 2026, macOS arm64, Python 3.14.5.
Reproduce with `uv run python tools/validate_gui_persistence.py` on a desktop.
These are one workload's elapsed measurements with normal filesystem caching;
heartbeat gaps are event-loop observations, not input-to-photon measurements.

The real bundled dataset contains 131 planes of 256×546 samples, 166 stable working
peaks and 121 fitted clusters, with SP2 models on both axes and two optimizer passes.
The scenario saves through the editor actions, repeats Save, uses Save As, reopens
verified results, inspects overlay/residual, edits a peak and performs a real fresh
fit. It then demonstrates numerical-only review after explicit verification of an
unavailable source. File chooser results are deterministic test selections; the
workspace, painting, worker processes, numerical fitting and disk I/O are real.

| Operation | Elapsed |
| --- | ---: |
| Initial spectrum load + stable SHA-256 fingerprinting | 0.091 s |
| First Save, including fsync and process transfer/startup | 2.956 s |
| Routine Save | 10.5 ms |
| Save As, copying retained evidence | 53.2 ms |
| Verified reopen, including reconstruction/sample checks | 1.909 s |

The source is **73,244,672 bytes**. The project is **87,646,888
bytes**, containing no `.ft2`/`.ft3` payload members and no complete source or
calculated-plane arrays. Its immutable payloads are a 77,709,752-byte
canonical `summary/fit.json`, a 9,596,634-byte sample-input NPZ, and small
resolved-descriptor/working-state manifests. The project is larger than this source:
canonical JSON analytical sample arrays dominate storage. This is retained scientific
evidence, not an embedded spectrum. A different canonical binary representation is
future work; the current CLI JSON contract is preserved.

Routine Save reads **zero source-content bytes**, rewrites **zero immutable evidence
files**, and writes only the 156,883-byte working manifest. Source metadata
is checked cheaply. The executable reuse test fails if hashing or immutable rewriting
is attempted. Save As copies and verifies evidence while preserving external source
fingerprints and rebasing relative paths. MCMC storage is zero in this milestone.

| Native 10 ms heartbeat phase | p95 gap | Maximum gap |
| --- | ---: | ---: |
| First Save | 10.8 ms | 148.5 ms |
| Routine Save | 9.3 ms | 9.3 ms |
| Save As | 10.7 ms | 10.8 ms |
| Reopen | 11.0 ms | 81.2 ms |
| Refit | 10.8 ms | 153.1 ms |
| Numerical Review | 11.0 ms | 130.9 ms |

The first implementation's long JSON-validation pause was removed by isolating new
archive work in a disposable process. Routine saves avoid process startup. Remaining
peaks include transfer/Qt installation and are visible in the measurements; this is
not a guarantee for larger workloads or slower storage.

![Reopened verified overlay, light](gui-v2-persistence/reopened-light.webp)

![Reopened verified overlay, dark](gui-v2-persistence/reopened-dark.webp)

![Reopened whole-spectrum residual](gui-v2-persistence/residual.webp)

![Numerical review without a verified source](gui-v2-persistence/numerical-only.webp)

## Scientific and failure protection

`tests/test_gui_project.py` covers lossless numerical samples/results and original
bounds/freedom; stable identity/adopted-position validation; reconstruction without
preparation/optimization/amplitude solving; independent historical Currentness;
relative relocation, absolute fallback and identical moved relinking; absent/corrupt/
unreadable sources; same-size/same-mtime content changes; mutation during initial
read; cancellation; unsupported models; corrupt payload checksums/version refusal;
first/interrupted/disk-full saves; repeated saves and bounded/protected/summary-only
history; stale PID locks and trusted spawned-process transport. Existing editor tests
cover the analysis-loss guard, asynchronous scientific publication and fresh fitting.

Final verification: **438 tests passed**, Ruff, formatting, strict type checking,
import-linter and package build passed. Build emits the existing uv-build version-range
warning (installed uv 0.12.23 versus configured `<0.12`); both distributions are produced.
The changed-file Prek hooks pass. `prek run --all-files` is blocked by a pre-existing
read-only `.agents/skills/setup-matt-pocock-skills/triage-labels.md`; it is outside
this change and was not modified.

## Explicit limits

- Native evidence is macOS Cocoa at 1280×800 logical pixels in light/dark modes.
  Windows/Linux, larger fonts and slow/network filesystems need their own acceptance.
- Only the bundled real dataset was benchmarked; substantially larger external sources
  and long protected histories have not been performance-qualified.
- Reader limits: 512 MiB per JSON, 8 GiB total compressed payloads and 8 GiB expanded
  per NPZ. Active/protected evidence is exempt from historical eviction, not from
  reader safety limits. Large JSON evidence may require a chunked format later.
- Latest-ten full history and protected runs are inspectable; older unprotected runs
  are labeled Summary only. Historical storage applies a 1 GiB budget; the active run
  is always retained. Temporary recovery precedes resident-history eviction, but there
  is no automatic recovery browser or background autosave of every edit.
- Exact sample tables show at most 2,000 rows at once; NPZ export preserves all samples.
  The existing one-file NMRPipe reader remains the input scope.
- No MCMC execution/resumption, full refinement UI, source-embedding export or legacy
  local-fit companion conversion. The legacy GUI and canonical CLI exports remain
  unchanged. The new internal dataset handoff moved from `editor.loading` into
  `gui.spectrum` to keep the import graph acyclic.
