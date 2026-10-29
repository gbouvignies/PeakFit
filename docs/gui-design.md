# GUI design reference

This is the agreed direction for whole-dataset fitting and scientific review.
It distinguishes the intended experience from the features currently available
in [the desktop guide](gui.md). Implementation details and measured limitations
belong in [architecture](architecture.md), [outputs](outputs.md), and
[development](development.md).

## Scientific and interaction principles

The 2D spectrum remains central. The two synchronized 1D slices stay on the right;
the lower analysis panel starts collapsed. Background preparation, fitting, and
result delivery must never open or resize that panel. Navigation, crosshairs,
plane identity, and visible ppm ranges survive changes of presentation.

Overlay is the default comparison. One coordinated selector exposes Overlay,
Experimental, and Residual across the spectral views. Total-model comparison
continues to include all usable fitted clusters when selection changes. Both
signs of experimental and fitted signals remain visible, in common intensity
units. A model contour is a prediction, not an additional observation.

A useful, collapsible cluster navigator appears on the left when a whole-dataset
result is available. It offers member assignments, numerical status, meaningful
review reasons, assignment search, and a Needs review filter. Selecting a cluster
highlights its members and recorded fitting region without opening details or
silently changing comparison scope. No empty navigator is reserved beforehand.

Details expose parameters with units, bounds and uncertainty availability,
recorded fitting diagnostics, signed amplitudes across planes, and provenance.
Raw crosshair intensity, fitted amplitude, and integrated volume are different
quantities. Repeated or unordered plane values remain distinct observations;
profile points do not imply interpolation or a response law.

Convergence, numerical usability, identifiability, and scientific validity remain
separate. A converged optimizer does not validate an assignment or lineshape.
Unreliable uncertainties and structured residuals remain visible. An unusable
cluster stays in the navigator but supplies no fabricated model or parameter rows.
An incomplete total model is identified as such; an all-unusable run has no model.

Clarity, readability, intuitive interaction, and aesthetic quality are acceptance
criteria for every slice, including light/dark appearance, keyboard and focus
behavior, realistic assignments, and 1280×800 / near-1440×900 native layouts.
Design improvements ship with functionality and continue through later review work.

## Completed result and exact fitting problem

`FinalFitOutcome` is the single authority for completed fitted parameters,
amplitudes, analytical uncertainties, residuals, statistics and optimizer
diagnostics. `CompletedFit`, captured by `complete_fit()`, retains that outcome
by reference together with the following distinct information:

| Information | Current in-memory owner | Meaning |
| --- | --- | --- |
| Evaluable model geometry | `FinalModelSnapshot` | Owned copies of lineshape objects, processing/axis context and peak association. Its parameter-value tuple is a consistency check, not another source of fitted values. |
| Sampled geometry | `FinalClusterSamples.grid_indices` | Exact paired native coordinates, in analytical array row order; not a rectangle or a regenerated segmentation. |
| Original sampled observations | `FinalClusterSamples.data` | Point-by-fitted-series inputs, retained even for unusable clusters. |
| Terminal neighbor corrections | `FinalClusterSamples.neighbor_corrections` | The actual point-by-fitted-series subtraction used by the terminal problem. A correction revision identifies these values but cannot reconstruct them. |
| Analytical model and residual | `FinalFitOutcome` | Frozen numerical output against those terminal corrected observations. |
| Mutable continuation | `FittingState` | Optimizer continuation only; subsequent mutations must not change completed review. |

Sample arrays are copied and read-only. Capture checks identity/revision agreement
and, for usable outcomes, exact consistency of recorded raw residuals with the
retained observations, corrections and model. No optimization or amplitude solve
occurs. This costs two additional float64 point-by-series arrays plus paired
indices per cluster; retain a bounded number of runs rather than an unbounded
history. Model objects are privately copied, not recursively immutable Python
objects, and consumers must treat them as read-only.

This numerical record does not define a persisted run or sampling request.
The GUI binds it to reviewed source fingerprints, resolved descriptors, stable
working-ID/assignment mapping, original coordinates and ordered plane identities.
The first GUI workflow includes every source plane. In general projection `planes`
indexes fitted-series columns; after exclusions those are not source-file indices.

## Two residual meanings

The recorded cluster residual is
`D_cluster - terminal_neighbor_corrections - cluster_model`. Its normalized values
and statistics remain exactly those in the outcome.

The whole-spectrum display residual is `D - sum(usable final cluster models)`.
`project_final_residual()` calculates this second quantity at supplied observations,
including outside fitting masks, optionally divided by the retained statistical
noise. It does not replace recorded residuals or recompute fit-quality statistics.
The caller supplies verified observations in the requested coordinate/plane order.

Terminal corrections precede the last optimizer pass and need not equal the sum
of the final neighbor models. Therefore these residuals can differ on fitted
samples. The main Residual view should display the total-model difference; cluster
details should label recorded optimizer residuals separately. Missing signal
outside masks matters scientifically. The fitting mask remains inspectable, and
its absence never means that a displayed residual was used in optimization.

`tests/test_completed_fit.py` protects these meanings with an independent
Lorentzian calculation, overlapping signed contributions, disjoint irregular
masks in non-raster order, nonzero terminal corrections, mixed usability, repeated
plane values and reordered/repeated requested columns. Existing full-grid and GUI
tests continue to protect model projection and isolated local fitting.

## Measured projection cost

`uv run python tools/benchmark_review_projection.py` fits the bundled experimental
dataset with two passes, then times five uncached projections per operation with
one BLAS thread. [The recorded run](validation/completed-review-benchmark.json)
on 2026-10-29 used macOS arm64, Python 3.14.5, NumPy 2.4.1, 131 planes of
256×546 samples, 166 peaks, 121 clusters, SP2 on both axes, and terminal correction
revision 1. All clusters were numerically converged. These are timings of this
workload, not a responsiveness guarantee or independent scientific validation.

| Projection | Median |
| --- | ---: |
| One complete native model plane | 1.285 s |
| Horizontal native trace, all planes | 19.56 ms |
| Vertical native trace, all planes | 13.72 ms |
| Crosshair, all planes | 8.36 ms |

The profiled plane spent about 1.39 of 1.45 seconds in shape evaluation, including
0.77 seconds in the SP2 kernel and 0.17 seconds in parameter extraction. It made
4,235 cluster evaluations in bounded chunks. Complete-plane work must stay off
Qt; native slice/crosshair requests are much cheaper. No new caches, cutoff rules,
projection contexts or kernel changes were introduced. Profile the smaller repeated
requests before deciding whether shared parameter preparation earns its complexity.

## Persistence direction and compatibility

Keep canonical `summary/fit.json` 4.1.0 and CSV outputs unchanged. Add a versioned
review manifest and separate lossless numeric payload in the persistence slice. Do not
duplicate authoritative parameters or amplitudes in that companion. Bind it to
the canonical summary and numerical file with checksums, explicit array names,
shapes, dtypes, cluster/peak identities and ordered source-plane mapping.

The manifest must preserve resolved per-axis/per-peak model descriptors, axis
and processing metadata, original input identities, effective settings and input
fingerprints. Loading must reconstruct evaluable geometry without rerunning
segmentation, preparing parameter defaults, solving amplitudes, or unpickling
arbitrary continuation objects. Verify reconstructed shapes/model values against
the saved analytical arrays before offering graphical comparison.

The payload must retain exact sampled indices, original sampled observations and
the terminal correction arrays themselves. Do not reconstruct correction values
by subtracting rounded model/residual exports from observations. A future MCMC
request additionally needs validated parameter freedom, bounds, noise, and a
specific parent run/correction context. Opening a review archive does not by
itself authorize or guarantee numerical continuation.

For the measured bundled workload, compressed NPZ is the preferred initial
candidate for the additional arrays: no new dependency, typed lossless arrays,
cluster-level access and substantially less space than JSON. Load with
`allow_pickle=False`. NPZ cannot efficiently slice a compressed member by plane;
HDF5 offers chunked slicing and should be reconsidered if measured larger runs
need it. Exact manifest validation, archive size limits, atomic publication and
compatibility reading of existing local-fit companions belong to the writer/reader
slice. No experimental archive from the benchmark is a supported output format.

The same run compared all additional terminal input arrays (13,005,696 bytes in
memory). Each format was written/read once and checked for exact values and dtypes;
timings include serialization/deserialization and normal OS caching. Canonical
analytical output, model descriptors, checksums, atomic publication and fsync are
excluded from this format experiment.

| Candidate | Bytes on disk | Write | Read all arrays |
| --- | ---: | ---: | ---: |
| JSON numerical lists | 27,986,960 | 329 ms | 163 ms |
| Uncompressed NPZ | 13,136,496 | 8 ms | 14 ms |
| Compressed NPZ | 9,603,894 | 396 ms | 33 ms |
| HDF5 gzip + shuffle | 9,983,371 | 156 ms | 54 ms |

HDF5 wrote faster in this sample, but its partial-read advantage was not benchmarked.
The NPZ preference is a simplicity/size choice for the first review archive,
subject to testing larger workloads before freezing the format.

Existing CLI summaries without reconstructable geometry retain numerical review;
do not turn dummy reconstructed shapes into a purported model. Missing or
mismatched spectrum inputs prevent comparison. Working edits mark retained runs
Outdated and hide comparisons while preserving original numerical inspection and
export. Selection, display changes and navigation do not alter scientific inputs.

## Delivery sequence

The completed-result/projection contracts are established. Product work proceeds
in visible vertical slices; persistence is not a prerequisite for fitting and review.

1. **Fit and review a complete dataset (implemented).** A primary Fit dataset
   action opens actual effective settings, defaults to two passes and separates
   clustering from display contours. The real pipeline runs in a cancellable
   process and supplies whole-model overlay/residual review and synchronized
   native slices. Summaries distinguish convergence, usability, completeness and
   uncertainty. A native compact toolbar, lighter plot decoration and narrower
   adjustable slices improve the same workflow. Details stay collapsed. Canonical
   export and result-loss/outdated safeguards apply to complete results. No
   cluster tree, complete-position application or saved-run reader is included.
2. **Cluster navigation and deeper scientific review.** Add a useful collapsible
   navigator, cluster diagnostics, peak amplitude profiles and further design
   refinement. Populate it from genuine completed results, preserving total-model
   comparison scope. Assess explicit whole-dataset position application here.
3. **Reliable persistence and reopening.** Implement the manifest/payload writer
   and strict reader together, input relinking and lossless scientific review.
   The format direction above remains a proposal, not a supported archive.

Every slice includes native light/dark visual and interaction review, keyboard
focus, resizing and realistic data. Visual design is ongoing product work,
not a finishing milestone. See [native acceptance evidence](validation/gui-dataset-fit.md).

Refinement and MCMC remain separate later workflows with parent-run identity,
explicit neighbor context, validated numerical inputs and scientifically accurate
uncertainty interpretation. They must not mutate an earlier completed outcome.
