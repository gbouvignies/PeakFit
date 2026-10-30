# External-source project storage — revised PR #72

A `.peakfit` project always references an independent experimental spectrum.
Portable-project export, embedded-cube serialization/reading and the obsolete
project-format adapter have been removed. The sole project writer uses the shared
compact scientific serializer. Scientific export schemas and their existing
compact/detailed options are unchanged.

## Stored authority and reconstruction

Each retained run contains exact nonlinear values/bounds/freedom/errors, signed
amplitudes, published errors/covariance, optimizer diagnostics/statistics,
identities, starting/optimized/adopted coordinates, settings, provenance, paired
sample coordinates, source-plane identities, calibration, observation dtype/digests
and **exact terminal corrections**. Corrections remain independent inputs captured
before the terminal pass; final neighboring predictions cannot replace them.

There is no experimental cube, extracted observation matrix, sampled fitted-model
matrix or dense residual array in the project. Experimental values are recovered
from the verified external spectrum at the exact paired coordinates and source
planes, cast to their recorded dtype and checked against SHA-256. Repeated plane
values never substitute for plane identity.

The small component-basis evaluator reference is retained for cross-runtime
validation (85,832 raw bytes for this dataset). It has `(peaks, spatial samples)`
shape; fitted spectra have `(spatial samples, planes)` shape and are generated
from the recorded parameters/amplitudes. The reference preserves a numerical
check of evaluator behavior without retaining those predicted intensity matrices.

A reopened result remains compact. Full spectral GUI projections evaluate the
existing bounded model at requested coordinates. Numerical sample review loads
its selected run on the existing worker and temporarily reconstructs observations,
model and residuals. Detailed export and numerical continuation use the same
`restore_completed` reconstruction code. No reconstruction path prepares fitting
defaults, optimizes parameters or solves amplitudes. Published results/uncertainties
remain recorded authority, not outputs of another solve.

Reconstruction validates:

- Observation dtype, ordering and digest exactly, after full-file identity and
  scientific calibration/shape verification.
- Model evaluator against the saved component reference: `rtol=1e-10`,
  `atol=1e-12`, allowing small platform/libm differences while rejecting drift.
- Reconstructed terminal chi-squared: `rtol=1e-8`, `atol=1e-9`. This looser aggregate
  check accounts for multiplication/subtraction, noise scaling and squared-sum
  accumulation; it never changes the published statistic.
- Detailed historical model agreement where such evidence is supplied by a
  scientific export: existing `rtol=1e-10`, `atol=1e-9` checks remain.

These are the existing shared continuation tolerances. Tests accept an evaluator
perturbation of 1e-12 relative and reject 1e-4 drift. This run reconstructed all
121 cluster predictions bitwise; that is not a future-platform bit guarantee.

Missing, changed, corrupt or unreadable sources retain numerical parameters,
uncertainties/covariance, amplitude profiles, diagnostics and provenance. Sample
inspection, spectral views and new fits are disabled until an explicit matching
relink verifies. Compact numerical export remains available; the existing detailed
checkbox is disabled when observations cannot be recovered. No unavailable data
are fabricated. Scientific Current/Outdated status is independent of source access.

Project schema **5.0.0** is the only accepted project format. Earlier experimental
projects are explicitly rejected, including attempts to overwrite them. No silent
conversion discards their embedded data. Existing independent scientific-result
readers are unchanged. Recovery, lazy history, atomic saving, cancellation,
bounded history and failure ownership use this same compact project format.

## Real-dataset measurements

Bundled `pseudo3d.ft2`, `pseudo3d.list`, `b1_offsets.txt`: 166 peaks, 121 clusters,
131 planes and two fitting passes. Original cube `(131,256,546)`, float32,
73,242,624 numerical bytes remains external. macOS arm64, NumPy 2.4.1, one BLAS
thread. Three fresh processes per implementation, sequential local runs. The
baseline is the previous PR #72 revision **9b8e48b**, whose linked projects retained
detailed samples/predictions. Values below are medians, logical decimal MB;
[raw measurements and array groups](external-projects/measurements.json) retain
every run. Temporary copies are deleted; bundled inputs are never altered.

| Operation | Previous linked project | Revised external-only project |
| --- | ---: | ---: |
| One-run project size | 16.856 MB | **7.115 MB** |
| Two independent analyses referencing one source | 33.711 MB | **14.230 MB** |
| Fit completion → durable acceptance | 1.032 s | **0.862 s** |
| First Save after accepted evidence publication | 18.8 ms | **13.2 ms** |
| Routine Save | 8.6 ms | **8.2 ms** |
| Verified reopen | 542 ms | **159 ms** |
| Missing-source numerical reopen | 134 ms | **70 ms** |
| Explicit matching relink | 527 ms | **153 ms** |
| Requested all-cluster sample/model/residual reconstruction | Already retained | **24 ms** |

The project is **57.8% smaller** than the preceding linked implementation. The
reconstructed arrays are temporary, never written back by Save. The original
spectrum is shared by independent references; it contributes no project bytes.
Compared with the earlier merged 85.25 MB embedded default, project size falls
about 91.7%. That implementation used a different ownership policy and is not
the matched timing baseline above.

| Retained typed-array group | Compressed member contribution |
| --- | ---: |
| Exact terminal corrections | 6,143,695 bytes |
| Amplitudes | 176,981 bytes |
| Published errors/covariance/scaled errors | 29,859 bytes |
| Paired sample indices | 27,311 bytes |
| Component evaluator reference | 95,120 bytes |
| Extracted observations | **0** |
| Sampled fitted-model predictions | **0** |
| Dense residuals | **0** |

NPY member headers are included in these contributions. Run/project metadata and
ZIP framing account for the remainder. The preceding linked project additionally
stored 3,549,672 compressed bytes of observations and 6,152,275 bytes of predictions.
No compression setting was changed.

Publication RSS is essentially unchanged: median summed worker/descendant RSS is
**1.239 GB before / 1.238 GB after**, sampled every 25 ms using standard `ps` on
macOS/Linux. The archive child is approximately 260 MB in both versions; the
parent approximately 958 MB, with allocator variability. The first completed fit
still has its numerical arrays in memory and trusted IPC during durable acceptance;
this change removes persisted retention, not every transient fitting allocation.
Summed RSS can double-count shared pages and sampling can miss brief peaks. It is
not a unique-physical-memory or allocation count. The validation tool no longer
imports undeclared `psutil`; no dependency was added to fix CI.

Timing is warm-cache local evidence; filesystem/network latency and different
allocators can change it. No speculative memory-management refactor was added.

## Scientific and GUI validation

Focused regressions cover exact terminal corrections, parameters/amplitudes/
uncertainties/covariance, immutable working transitions, absence of prohibited
arrays, sample digests, source-plane exclusions/repeated values, model drift,
missing/moved/changed/corrupt sources, explicit relinking, multiple analyses
sharing a source, atomic failures, cancellation, history eviction and recovery.

Verified numerical review materializes samples only on request, leaving the
original compact result unchanged. Missing-source review exposes numerical tables
and amplitude profiles while disabling samples. Reopened compact and detailed
exports agree with the same original completed science and CSV tables; shared
GUI/CLI parity and native continuation tests remain in place.

Native Cocoa validation reopened the real compact project, displayed calculated
and residual spectra, adjusted one peak (Outdated) and ran a real new fit (Current)
in **5.98 s**, including durable acceptance. The external spectrum remained
available and verified. Source removal then preserved numerical review, and
explicit relocation/relink restored access. Portable/source-free fitting is not
provided; continuing spectral work requires the independent source.

Reproduce current measurements and native validation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  tools/validate_external_projects.py --output /tmp/external-projects.json --native
```

The same driver runs against revision 9b8e48b via `PYTHONPATH` for the baseline.
It uses the existing native GUI exercise and standard process-table RSS sampling.


Final validation: **493 tests passed** with `QT_QPA_PLATFORM=offscreen
MPLBACKEND=Agg uv run pytest -q`. Ruff, repository formatting, strict typing and
both import-linter contracts pass. Native Cocoa validation is additional to the
offscreen suite. Graphify's code graph was refreshed.
