# Implemented scientific retention policy — PR #71

> Historical PR #71 validation. The later [linked/portable policy](project-portability.md)
> supersedes default cube embedding and source-free recovery claims below;
> scientific export retention remains unchanged.

Validated on macOS arm64 (Apple M5 Pro, 64 GiB RAM), Python 3.14.5,
NumPy 2.4.1, one BLAS thread. Baseline is **1051ceb**, the completed storage audit;
after is the implementation committed with this report. This supplements the
[original per-array audit](scientific-storage-audit.md) rather than changing its
historical measurements. [Current formats and migration](../outputs.md) are the
user-facing contract.

## Decision and unavoidable size tradeoffs

Default scientific exports retain published results, exact terminal corrections,
exact observation dtype/checksums, masks/plane identities, model descriptors and
small evaluator references, with CSV tables and one short report. Large observed,
model and residual matrices are absent; JSON and executable pickle are no longer
default outputs. Both interfaces use one writer, scientific schema and reader.

The audit's 5.77 MB estimate omitted terminal corrections. Retaining them adds
about 6.16 MB plus the small reference/checksum budget; omitting them would lose
the exact multi-pass parent problem. Corrections cannot be generated from final
neighbor models. Compact continuation reads only observations from a verified
source, checks their exact dtype/byte digests and validates the evaluator/statistics.
Detailed evidence retains observations and historical sampled models; raw and
normalized residuals are regenerated in the original operation order. Reported
parameters, amplitudes, errors/covariance and statistics are never re-solved.

The requested self-contained project must include experimental values **outside**
fit masks. The audit's 16.85 MB estimate retained only fitting samples and could
not provide full-spectrum inspection or subsequent adjusted fitting without its
source. This implementation embeds the complete original-dtype experimental cube
once, checksummed and shared across history. It costs **68.392 MB compressed** in
addition to approximately **16.85 MB of run/working evidence**. No compression
codec or precision change was used to disguise that requirement.

## Real-data measured sizes

131 × 256 × 546 external spectrum, 166 peaks, 121 clusters, 131 retained planes,
664 nonlinear parameters, 21,746 amplitudes, SP2 on both axes, **two optimizer
passes in both versions/interfaces**. These are logical decimal MB, not allocated
filesystem blocks. Project size includes one accepted run; additional retained
history has its own evidence but shares the cube. Temporary raw input copies are
excluded from project size and deleted for the source-free checks.

| Measured artifact | 1051ceb (MB) | Implemented policy (MB) |
| --- | ---: | ---: |
| Default CLI directory | 122.082 | 12.023 |
| Default GUI directory, recorded adoption | 111.966 | 12.061 |
| One-run project | 29.314 | 85.248 |
| Explicit detailed GUI directory | — | 21.802 |

CLI and GUI differ by only their recorded context (GUI working-coordinate
adoption/currentness and operational metadata). All 968 outcome/sample NPY members
in their default archives are byte-identical, outcome metadata agrees, and all
four CSV tables are byte-identical. GUI scientific archive is about 6.90 MB;
its detailed archive is about 16.64 MB. CSV/report add 5.16 MB. GUI defaults shrink
by approximately 89.2%; matched CLI defaults shrink by approximately 90.2%.

The retained shape reference costs roughly 0.12 MB including framing, and lets
continuation detect evaluator drift. Historical observations remain per run so
sample/history review does not require loading the whole experimental cube.
Full-grid simulation stays opt-in and still adds 146.49 MB. Explicit legacy JSON
is supported, but would add about 77.71 MB; it is not a default or an authority.
Exact per-file counts are in [before.json](scientific-storage-policy/before.json)
and [after.json](scientific-storage-policy/after.json).

## Loading, saving and publication

The same driver invokes actual preparation/pipeline, durable `SessionEvidence`
publication through its archive process, Save/Save As, reopen and GUI/CLI writers.
The first Save copies already accepted immutable evidence and, now, the cube.
The initial publication cost is therefore reported separately. Runs are fresh
processes with a warm local filesystem; these single-run timings are not statistical
performance bounds. Metadata IDs/timestamps produce small byte variations.

| Operation | Before (s) | After (s) |
| --- | ---: | ---: |
| Numerical fit | 3.974 | 3.907 |
| Completion → durable acceptance | 1.259 | 2.449 |
| First permanent Save after acceptance | 0.023 | 0.074 |
| Routine Save | 0.006 | 0.006 |
| Verified reopen with spectra | 0.591 | 0.643 |
| Explicit historical-result load | 0.161 | 0.121 |
| CLI default export | 2.342 | 0.426 |
| GUI default export | 2.262 | 0.359 |
| Fit start → durable acceptance | 5.233 | 6.356 |
| Reopen without input files, full spectra | unavailable | 0.614 |

Initial publication is slower because it now durably embeds a complete spectrum;
Save As copies it too. Routine Save still reuses immutable cube/run payloads,
without source reads/hashing or evidence rewrites. Export and historical loading
improve by omitting redundant numerical representations. The fitted-parameter/
uncertainty digest is identical before/after:
`0a779ba505dab4557bdd9f0b20c4a8558be5cf85d8ae45c18d77a814b7395af5`.
The fit algorithm and optimizer settings are unchanged.

## Peak memory, including temporary serialization

Concurrent process-tree RSS is sampled with `ps` every 20 ms plus command overhead.
RSS sums include shared pages more than once, interpreter/allocator retention,
uv wrappers and transient workers; they are not unique physical memory. Shorter
peaks can be missed. These traces are separate fresh runs of the same workload;
subsequent metadata-only observation-digest checks add a few KiB, not large arrays.
The sampler's `gui` role labels the root **uv wrapper**; the actual benchmark/
GUI process is its Python child. Read the command/PID breakdown, not that label.

| Comparable phase peak, whole process tree | Before (GB) | After (GB) |
| --- | ---: | ---: |
| Initial durable publication | 1.282 | 1.550 |
| Default CLI export | 1.292 | 1.132 |
| Default GUI export | 0.732 | 0.636 |

At the publication peak, the Python driver is 979.8 → 1,029.9 MB and the archive
worker 260.3 → 478.0 MB. Most of the increase is temporary worker/IPC handling of
the newly embedded full experimental cube; unchanged-source routine Save does not
repeat that transfer. This is a measured cost of self-containment, not a claim
that lower export size must lower total GUI memory.

The extended native validation peaks at **2.673 GB**, with a 1.730 GB Python driver
and a 0.900 GB fit worker during refit. That driver deliberately still holds
original, reopened, historical and continuation validation records; this is not
a minimal single-workspace GUI benchmark. The before run has no corresponding
source-free GUI refit capability, so comparing its 1.292 GB overall peak to
2.673 GB would be misleading. No total-GUI-memory reduction or persistent leak
absence is claimed. Existing bounded-history tests still enforce one resident
full result; historical records stay lazy. Runtime completed evaluations still
retain their dense arrays for GUI use; on-disk savings are not an in-memory
representation rewrite. No speculative memory refactor is
included. Raw [before](scientific-storage-policy/before-memory.json) and
[after](scientific-storage-policy/after-memory.json) include process/phase attribution.

## Source-free native and scientific validation

The driver copies the bundled inputs to its own temporary directory, fits them,
saves, then deletes **all three input copies**. It never changes bundled files.
With those inputs absent, it:

- Reopens the entire experimental cube byte-identically and verifies the result.
- Reconstructs compact continuation against the verified source before deletion,
  and detailed continuation without sources afterward, preserving all **806,698
  nonzero terminal correction values** and exact corrected observations.
- Reopens a native Cocoa editor, renders the calculated overlay and full-spectrum
  residual, adjusts a working peak, marks the old result Outdated and fits again
  through real Qt controls and the disposable fit/publication workers. The new
  result becomes Current. This native adjusted refit takes about **7.66 s**.
- Checks full scientific array round trips and regenerated residuals against the
  original result. Bit identity on this runtime does not guarantee future
  NumPy/BLAS/libm bit identity; runtime validation uses declared tolerances.

[Native results](scientific-storage-policy/native.json) and the
[source-free overlay capture](scientific-storage-policy/native-source-free.webp)
record this workflow. Unit/integration tests additionally save/reopen the adjusted
result, cover repeated/excluded plane identities, wrong-source fingerprints,
changed reconstructed observations, evaluator drift, corruption/header limits,
atomic failures, cancellation, recovery and bounded/lazy history.

## Compatibility and continuation correction

Genuine fixtures generated by **06157dd** (Milestone 3), **bb77d65** (initial PR context) and **1051ceb** cover old
projects 1.1 and 2.0, plus scientific NPZ 1.0; an uncompressed
1.0-layout regression uses the same genuine numerical evidence. They remain readable. Legacy
JSON/sample evidence has one bounded read-only decoder; new writes use scientific
2.0 and self-contained project 3.0. Saving an old linked project requires its
verified/relinked spectrum, then embeds it. Historical immutable payloads are
preserved, rather than risking an in-place evidence rewrite during migration.
New fits use the compact typed representation. Legacy files cannot supply full
experimental data that they never saved.

`save_state` now explicitly records corrections and its round trip preserves them.
Old pickles that already omitted corrections cannot be repaired alone. Native
scientific evidence takes precedence; legacy multi-pass MCMC refuses missing
corrections, and JSON-only summaries cannot fabricate a likelihood parent from
dummy samples. The old summaries remain readable for numerical review/plots.
Default pickle output is retired only after native parent-input and **actual MCMC
likelihood/amplitude** parity checks pass. Existing sampling/chain features are
unchanged; no chain/RNG continuation or intensity feature was added.

## Reproduction and remaining limits

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python tools/validate_storage_policy.py \
  --output /tmp/policy.json --phase-file /tmp/policy-phase.json --native
uv run python tools/measure_process_memory.py \
  --output /tmp/memory.json --phase-file /tmp/policy-phase.json \
  uv run python tools/validate_storage_policy.py \
  --output /tmp/policy.json --phase-file /tmp/policy-phase.json --native
```

For before measurements run the same driver with `PYTHONPATH` pointing to a
checkout of `1051ceb`'s `src`. The driver deliberately uses the older public
interfaces too; only source-free/detailed native checks depend on project 3.0.
It reports bytes/timing and asserts scientific equality; it is not production
serialization code. The original audit tool is pinned to its historical revision.

Qualification is one bundled real workload/local filesystem/macOS. Other models
and platforms still need their own replay checks. Default compact exports require
a matching source for continuation, while detailed evidence covers samples and
projects cover the full experimental cube. Unsupported/unusable parents cannot
continue without adjustment/refitting. Neither fit evidence nor MCMC's existing
chain files alone promise exact sampler/RNG resumption. Initial cube embedding
and trusted IPC can cost memory/latency; the format is not memory-mapped or chunked
for random-access arrays. No compression tuning, MCMC feature, intensity analysis
or automatic merge is part of this correction.

## Repository checks

The full offscreen suite passes (489 tests in the final run), including real CLI
fit/plot/MCMC smoke tests. Ruff, formatting, strict typing, import-linter and
wheel/sdist builds pass. The build emits the existing uv-build version-range
warning; it still completes. Native Cocoa validation and the before/after
measurements above are separate from headless tests. The AST knowledge graph is
refreshed without API calls. Linux CI initially exposed two fixture migrations
that regenerated synthetic source values with host libm and then required exact
agreement with macOS fixture bytes. The tests now load a canonical source array;
production exact source checks and separate evaluator tolerances are unchanged.
