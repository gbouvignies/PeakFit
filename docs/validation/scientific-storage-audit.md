# Scientific storage audit — PR #71

Completed at `1051ceb`. The later [implemented policy](scientific-storage-policy.md)
retains terminal corrections in default exports and embeds the full experimental
cube in projects; those requirements cost more than the result-only/sample-only
estimates below. This audit and its array inventory remain baseline evidence.
Reproduce its original output by running its tool at `1051ceb`.


## Recommendation

Keep one shared GUI/CLI scientific contract, but separate **published fitted
results** from **detailed numerical evidence**. Retain exact published numbers;
remove duplicated JSON, legacy executable-state duplication and dense display
caches from the default output path. This report proposes retention changes;
**no scientific implementation or production output has changed**.

For this real dataset, recommend:

- **Default scientific export:** compact `scientific.npz`, existing useful CSV
  tables and a short report: approximately **5.770 MB**, versus
  **111.966 MB** today (**94.8% smaller**). The scientific
  archive itself is **0.610 MB**; omitting optional CSV leaves approximately
  **0.618 MB** including the report. Keep exact fitted values, all published
  uncertainty outputs, diagnostics/provenance, model/processing descriptors,
  initial/fitted/adoption distinctions, stable IDs, assignments and exact sample/
  source-plane mappings. Do not re-solve amplitudes or recompute published errors.
- **Working `.peakfit` project:** retain the same result core, exact terminal
  corrections, sample mappings, GUI working state and a small sampled-shape
  reference; read observations from the verified external source. Estimated
  **7.106 MB**, versus **29.314 MB** (**75.8% smaller**).
  This preserves exact working/result values and exact terminal input values
  with a verified source. It deliberately makes observed/corrected/residual
  sample review source-dependent; source-independent tables/profiles remain.
- **Explicit detailed evidence:** additionally retain sampled observations,
  corrections, shapes and the historical sampled model; derive raw and normalized
  residuals on demand. Estimated scientific archive **16.642 MB**, or
  **21.802 MB** with CSV/report. This preserves source-independent
  terminal sample review and historical model values. The corresponding project
  is **16.852 MB**.

If preserving full source-independent sample review in **every default project**
is mandatory, use the 16.852 MB
project instead of the 7.106 MB linked project. That is a real capability
choice, not a compression setting. Do not advertise a compact result-only export
as a self-contained parent likelihood or complete sample evidence.

The detailed-evidence selection should control the **same named evidence fields**
in both interfaces. GUI-only adoption/currentness facts remain explicitly separate
export context; the CLI must not invent them. Remove the old dense-array JSON
projection rather than make it another default copy. Keep CSV selection and
full-grid simulation explicit. Replace the CLI continuation reader's pickle/JSON
dependence before removing those files; this is existing-workflow migration,
not new MCMC functionality. No compatibility adapter is needed.

## Reproducible measurement

Source revision: `335f569` (the approved typed-array/provenance PR state). Dataset:
`examples/02-advanced-fitting/data/pseudo3d.ft2`, `pseudo3d.list`, `b1_offsets.txt`.
Spectrum SHA-256: `2f91c489f1fe6448c1ead0a417a600a1e401616f76caf1f6cdb3219fbf6ac442`.
The original spectrum is **73.245 MB**, shape `131 × 256 × 546`.
The fitted workload has **166 peaks, 121 clusters, 664 nonlinear parameters,
21,746 fitted amplitudes, 6,158 selected spatial points and 806,698 scalar
observations**. Each cluster has 25–154 selected points and 1–4 peaks.

The audit invokes the real preparation, pipeline, project saver and GUI/CLI
export APIs; it does not construct synthetic numerical arrays. GUI defaults use
two optimizer passes; actual CLI defaults use one. Both are measured, plus CLI
exports of exactly the GUI's completed two-pass result to separate scientific
workload differences from writer overhead. Three uncertainty warnings from the
real two-pass workload are retained in its provenance.

Environment: `macOS-27.2-arm64-arm-64bit-Mach-O`, Python `3.14.5`,
NumPy `2.4.1`, one BLAS thread. File sizes are logical bytes, **decimal MB**;
filesystem allocation, external linked spectrum, source copies and multiple
historical runs are excluded. The project contains one accepted run. Extra
retained runs add their own result evidence; this is not a bounded-history
stress benchmark. IDs/timestamps/timings can change metadata bytes slightly.

Run from the repository root:

```bash
UV_CACHE_DIR=/tmp/peakfit-uv-cache uv run --no-sync python \
  tools/audit_scientific_storage.py \
  --output docs/validation/scientific-storage-audit
```

Artifacts:

- [Exact measurements and per-file directory sizes](scientific-storage-audit/measurements.json).
- [Complete per-array inventory, gzip CSV](scientific-storage-audit/arrays.csv.gz):
  every stored typed ndarray in project and GUI/CLI archives, both CLI pickles,
  corresponding analytical JSON arrays/plane values, and full-grid simulation;
  **11,448 rows**, grouped by artifact and scientific purpose. Columns include
  physical member name, scientific path, exact shape/dtype, numerical bytes,
  stored uncompressed bytes, compressed contribution and exact-duplicate reference.
- [Reproduction tool](../../tools/audit_scientific_storage.py).

The inventory counts physical storage, including arrays repeated across files.
For NPZ, compressed contribution is each ZIP member's measured `compress_size`,
including its NPY header; ZIP structural overhead and JSON metadata are separate.
JSON has no stored dtype: its source-array dtype is identified, and exact text
spans include whitespace. Small scalar/record metadata is budgeted separately.
Pickle is uncompressed: ndarray payloads and remaining object/framing bytes are
reported separately; there is no meaningful independently compressed pickle
member size. The duplicate marker detects identical NPY bytes, not mathematical
reconstructibility or global deduplication opportunities.

## Where the archive bytes go

`q` is a cluster's selected points (25–154); `k` is its peaks (1–4).
Shapes below describe **each** member; the linked inventory lists every exact
cluster shape and name. All floats are little-endian `float64` (`<f8`); mappings
are little-endian `int64` (`<i8`). Counts and sizes below are the two-pass GUI
scientific archive. The project has the same analytical/sample matrices.

| Purpose / stored field | Members | Shape per member | Raw numerical bytes | Compressed member bytes |
| --- | ---: | --- | ---: | ---: |
| `shapes` | 121 | `(k, q)` | 85,832 | 95,120 |
| `amplitudes` | 121 | `(k, 131)` | 173,968 | 176,981 |
| errors, covariance, scaled errors | 363 | `(k)`, `(k,k)`, `(k)` | 5,040 | 29,859 |
| `model_values` | 121 | `(q, 131)` | 6,453,584 | 6,152,275 |
| `raw_residuals` | 121 | `(q, 131)` | 6,453,584 | 6,148,654 |
| `normalized_residuals` | 121 | `(q × 131)` | 6,453,584 | 6,255,884 |
| paired `grid_indices` | 242 | two `(q)` arrays per cluster | 98,528 | 27,311 |
| `data` | 121 | `(q, 131)` | 6,453,584 | 3,549,672 |
| `neighbor_corrections` | 121 | `(q, 131)` | 6,453,584 | 6,143,695 |
| plane values / source-plane indices | 2 | two `(131)` arrays | 2,096 | 725 |
| initial/fitted/adopted positions and geometry | 664 | `(2)` | 10,624 | 56,406 |
| **All numerical arrays** | **2,118** | | **32,644,008** | **28,636,582** |
| Metadata JSON | 1 | scalar/structured records | 535,015 | 86,189 |
| ZIP structural overhead | | | | 380,918 |
| **Whole scientific archive** | | | | **29,103,689** |

Three dense derived matrices (`model_values`, `raw_residuals`,
`normalized_residuals`) alone occupy **18.557 MB compressed**, approximately 64%
of the archive. Accepted amplitudes and all amplitude uncertainty arrays occupy
only **179,008 raw bytes** in total. Nonlinear values/errors, classifications,
statistics and provenance live in the small structured metadata, not another
dense matrix. Covariance is `(k,k)`, shared across planes for this fit's noise and
geometry; it is not a hidden `(k,131,k,131)` tensor.

The repeated raw observations occupy **6.454 MB** in float64, or **3.550 MB
compressed**. They are exact promoted float32 source values; all 121 clusters
survive float64 → float32 → float64 bit-exactly. A source-native observation dtype
could therefore be lossless here, but this recommendation does not depend on
precision changes, index narrowing, array packing or another compression codec.
Float64 corrections and published fitted results must not be narrowed on that
basis.

Small arrays carry measurable NPY/ZIP overhead, but are not the size problem.
There are 664 separate two-coordinate arrays in the GUI export; they cost only
56,406 compressed member bytes plus their headers. More compact packing can wait.
The repeated position views encode distinct initial/fitted/adopted/geometry
semantics and must not be removed merely because this run has equal values.

The project `result.npz` is **28.932 MB**: 1,452 arrays,
32,631,288 numerical bytes, 28,579,451
compressed array bytes, 62,965 compressed metadata
bytes and 289,336 ZIP overhead bytes. Its manifest and
review JSON add **382,122 bytes**. Export adds explicit plane/coordinate context;
that explains the small archive difference. Matched CLI has 166 fewer adopted
coordinate arrays because no GUI adoption was recorded, while using the same
scientific schema and exactly equal outcome/sample arrays.

## Complete output directories

| Output selection | GUI, two passes (MB) | CLI, same completed two-pass result (MB) |
| --- | ---: | ---: |
| Current default: NPZ + JSON + CSV | 111.966 | 122.082 |
| JSON disabled; NPZ + CSV | 34.256 | 44.373 |
| All configurable formats disabled | 29.104 | 39.221 |
| JSON + CSV + text report + simulation | 258.458 | 268.575 |

The **actual one-pass CLI default is 115.812 MB**;
its NPZ is 22.953 MB because terminal corrections are
zero and compress cheaply. It is not the same numerical problem as the two-pass
GUI fit. Compare the matched two-pass columns when assessing writer overhead.

| Component, matched two-pass default | GUI bytes | CLI bytes |
| --- | ---: | ---: |
| `README.md` | 0 | 1,129 |
| `metadata/fitting_state.pkl` | 0 | 10,153,927 |
| `scientific.npz` | 29,103,689 | 29,065,906 |
| `summary/fit.json` | 77,709,753 | 77,708,636 |
| `tables/clusters.csv` | 23,936 | 23,936 |
| `tables/intensities.csv` | 4,920,197 | 4,920,197 |
| `tables/parameters.csv` | 168,719 | 168,719 |
| `tables/shifts.csv` | 39,284 | 39,284 |

Default JSON is **77.710 MB**, of which
**77.192 MB** is analytical numerical arrays
and plane values already available in the archive. In particular the model/raw/
normalized residual arrays consume approximately **76.084 MB of JSON text**.
This is why optimizing only the `.peakfit` directory misses most default export
storage. CSV costs **5.152 MB**, predominantly the 21,746 amplitude rows with
explicit fitting/source-plane identities and repeated contextual columns.
Unlike dense JSON, those standalone tables serve a useful interchange workflow.

`simulated.npy` is **146,485,376 bytes**, shape `(131,256,546)`, dtype float64;
146,485,248 data bytes plus a 128-byte header. It is a derived full-grid model,
not a default output and not original observations. Its float64 payload is twice
the external source's float32 grid payload. Optional text report is just 7,400
bytes. Logical file sizes for every selection are in `measurements.json`.

The matched CLI pickle is **10,153,927 bytes**, including 533 unique ndarray
objects totaling **6,565,392 bytes**: 121 observation arrays (6,453,584 B),
242 index arrays (98,528 B), 166 peak-position vectors (2,656 B) and four
332-element parameter vectors (10,624 B). Remaining object/framing bytes total
**3,588,535 B**. It includes 22,410 scalar parameter records, of which 21,746
computed amplitudes duplicate the published amplitude result. The deep inventory
follows PeakFit object dictionaries/slots and bound methods; it finds no hidden
large ndarray model caches. This is duplicated operational state, not a second
necessary scientific authority.

### Existing continuation limitation found by replay

The real two-pass continuation state has **806,698 nonzero correction values**.
After the existing `save_state` → `load_state` route, it has **zero**. Maximum
change in corrected observations is **16,589.07035125268**. Pydantic
`state.model_dump()` omits `Cluster.corrections` (`init=False`); reconstruction
initializes them to zero. See [state saving/loading](../../src/peakfit/io/state.py)
and [Cluster initialization](../../src/peakfit/engine/domain/cluster.py).
This predates this audit; **the scientific NPZ does retain exact corrections**.
It means the current pickle cannot substantiate exact two-pass parent-likelihood
continuation. Do not retain an extra 10 MB and call it exact scientific evidence,
or delete it before the existing continuation workflow has a correct replacement.
No fix is included in this audit.

## Reconstruction and exactness findings

For all **121/121** real two-pass clusters, the following comparisons were
bit-exact in the measured environment, without refitting or amplitude solving:

- Observations read from the verified source at retained indices/planes.
- Sampled shapes evaluated from final parameters and stored model descriptors.
- Sampled models from stored or regenerated shapes multiplied by **stored** amplitudes.
- Raw residuals from `(data - corrections) - stored_model`, preserving operation order.
- Normalized residuals from `raw.ravel() / noise`.
- Conditional amplitude errors from covariance and scaled errors from saved scale.

These checks establish the practical redundancy of the large caches here.
They do **not** promise universal cross-version/BLAS/libm bit identity for future
model reevaluation. The detailed recommendation keeps sampled model values for
that reason. Even residual recomputation should be validated against the declared
arithmetic/runtime contract; retaining both residual arrays remains available as
a forensic snapshot if historical output bits themselves are contractual.
Keep the tiny published uncertainty arrays even though their formulas replayed
exactly; there is little saving and no benefit to redefining published numbers.

Reconstructing terminal corrections by summing final other-cluster models was
bit-exact for **0/121 clusters**: maximum absolute difference **2,932.629506957645**,
relative L2 difference **6.046%**. Corrections are frozen inputs from the previous
pass, with a different joint amplitude solve. The source plus final fit therefore
cannot recover the exact terminal problem. Do not regenerate them from final
models, or assume inverse floating-point subtraction restores discarded inputs.

Writer parity was checked on the **same** completed two-pass result: all **1,452
outcome/sample NPY members are byte-identical**, outcome metadata is equal,
all four CSV tables are byte-identical, and optional simulations are byte-identical.
GUI adoption/currentness context is intentionally interface-specific evidence,
separate from the common scientific outcome.

## Measured retention budgets and capability tradeoffs

These are measured field-retention projections at **unchanged deflate level 1**,
using the original NPY members, metadata structure and ZIP64 member framing.
No arrays are rounded, resampled or recomputed for sizing. They are deliberately
marked **unsupported audit-only artifacts**, removed with the temporary payloads;
current production readers cannot read these projections. New capability metadata,
manifest hashes and reader behavior would need implementation and validation.
Project estimates include the current 382,122-byte manifest/review budget; they
are not a claim that an edited archive already reopens successfully.

| Retained evidence, in addition to compact fitted core | Scientific archive (MB) | One-run project estimate (MB) | Main limitation |
| --- | ---: | ---: | --- |
| None | 0.610 | 0.820 | Tables/profiles and model descriptors only; no exact terminal inputs |
| Terminal corrections | 6.775 | 6.985 | Requires verified source and supported evaluator |
| Corrections + sampled-shape reference | 6.896 | 7.106 | Recommended linked project; sample observations require source |
| Observations + corrections | 10.342 | 10.552 | Portable parent inputs; sampled model must be reevaluated |
| Observations + corrections + shapes | 10.463 | 10.673 | Portable inputs/basis; sampled model requires multiplication |
| Observations + corrections + shapes + sampled model | 16.642 | 16.852 | Recommended detailed evidence; residuals regenerated |
| All current arrays | 29.104 | 29.314 | Full forensic snapshots; largest storage |

The 10.463 MB portable-input/basis candidate is a defensible lower bound if
historical sampled-model bits are not required; its model multiplication was
bit-exact in this run. Prefer the 16.642 MB detailed option initially because it
keeps the exact historical sampled model and useful standalone sample evidence.
Neither is a promise to reconstruct the entire external experimental cube:
only fitted samples are retained. Full-grid experimental/residual visualization
still requires the original spectrum.

This recommendation removes redundant default storage without changing a single
fitted quantity, amplitude, uncertainty, assignment, sample mapping or correction.
The intentional capability changes are: default exports no longer contain raw
sample evidence; linked default projects require their source for observed sample
review; dense JSON and executable pickle cease to be routine scientific output
after their readers are migrated. Numerical-only review must advertise precisely
which views remain available. A shared compact core plus optional evidence payload
also leaves a clear place for later intensity/MCMC products without implementing
them now. No NPZ/HDF5 comparison or compression retuning is needed to justify
these gains: retention policy dominates the measured costs.

## What is a fitted result, and what is retained evidence?

“Independent result” here means a published quantity from the accepted fit whose
recorded value should remain authoritative, rather than a claim of statistical or
information-theoretic independence from the input spectrum.

| Quantity | Scientific role and dependency | Storage implication |
| --- | --- | --- |
| Final nonlinear values, units, bounds/freedom and uncertainties | Accepted terminal fitted quantities, with explicit identities | Retain exact recorded values; do not refit to recover them. |
| Analytical amplitudes, signed and ordered by peak/plane | Accepted linear fit results; obtaining them from data would require another amplitude solve | Retain their exact recorded values, even though the original fit computed them analytically. |
| Amplitude covariance and reported errors | Covariance is conditional on nonlinear geometry and statistical noise; errors are `sqrt(diag(covariance))`, scaled errors multiply by the recorded uncertainty scale | Retain the published uncertainty result; error arrays are algebraically redundant but small. Recomputing a covariance requires factorization and can change rank/rounding decisions. |
| Sampled shapes | Evaluations of resolved lineshape descriptors, calibration and final nonlinear parameters at the exact paired sample indices | Reconstructible by model evaluation; no fitting or amplitude solve is needed. Exact original array bits are a stronger contract than a scientifically equivalent evaluation. |
| Sampled model values | `shapes.T @ amplitudes` | Reconstructible; storing this matrix is a cache or exact computed-output snapshot, not another independent fitted result. |
| Raw terminal residuals | `(observations - terminal_corrections) - sampled_model` | Reconstructible given those exact inputs and the operation order. Do not confuse them with whole-spectrum display residuals. |
| Normalized terminal residuals | `raw_residuals.ravel() / noise` | Redundant representation of the same sample errors. Division is not generally invertible bit-for-bit; retaining only normalized residuals is not an exact substitute for original observations/corrections. |
| Retained observations | Samples of the verified external spectrum at recorded source planes and paired indices | Duplicated source information when that source remains available and verified. Necessary for those sample views/likelihood inputs to remain self-contained without the source. |
| Terminal neighbor corrections | Frozen numerical inputs submitted to the final optimizer pass | Not reconstructible from the final accepted result alone under the current pipeline; see the correction provenance caveat below. |
| Exact sample indices and source-plane indices | Identity of the fitting problem and the order of every observation/amplitude column | Retain explicitly. A contour threshold or repeated experimental values cannot recover these identities without resegmentation or ambiguity. |
| Model geometry, calibration/processing, initial coordinates and stable-ID/assignment/adoption metadata | Semantic reconstruction context, distinct from fitted parameters and editable export-time state | Retain once in a compact structured form; value tables must not silently relabel fitted coordinates as submitted coordinates. |
| Global parameter tuple / cluster lookup | Duplicate views of cluster-owned parameters | Reconstruct views; current binary storage already avoids these duplicates. |
| Global/cluster statistics, optimizer termination and provenance | Small recorded scientific diagnostics | Keep the recorded values even where formulas could recompute a subset; a successful reconstruction does not replace actual optimizer provenance. |
| Full simulated spectrum | Evaluation of the same stored model at every output-grid point | Optional explicit export/cache; it is unnecessary for storing fitted results and should not be a default project payload. |
| CSV and full JSON numerical projections | Additional representations of already stored scientific values | Count them as additional export size, not as free metadata; full JSON duplicates the large analytical arrays. |
| CLI pickle continuation state | Operational executable state containing model/data structures | Separate from scientific result authority. The current CLI writes it by default, but that is an implementation choice, not a requirement to publish fitted values. |

The dependency formulas above are implemented in
[`evaluation._evaluate_analytical_model`](../../src/peakfit/engine/algorithms/evaluation.py#L110),
[`evaluation._evaluate_model`](../../src/peakfit/engine/algorithms/evaluation.py#L254),
[`linear_algebra.calculate_amplitudes_with_uncertainty`](../../src/peakfit/engine/algorithms/linear_algebra.py#L74),
and the immutable result fields in
[`FinalAnalyticalEvaluation`](../../src/peakfit/engine/scientific/final_outcome.py#L42).
The accepted outcome explicitly freezes these arrays today; that does not make
all of them independent information.

## Workflow requirements

| Workflow | Strict scientific information required | What can be regenerated or omitted? | Exactness / capability qualification |
| --- | --- | --- | --- |
| Read/publish numerical fitted results | Final nonlinear values and uncertainties, amplitudes and uncertainty outputs, classifications/statistics, provenance, stable identities, coordinates, ordered plane mappings and units | Observations, sampled shapes, model/residual matrices and simulated cubes are unnecessary for numerical tables/profiles | Preserve the recorded result numbers exactly. This is not equivalent to retaining every optimizer evidence array. |
| Reopen the same GUI working analysis | Committed working peaks/settings/view state and revision/currentness semantics; immutable result identity, exact initial/fitted/adopted distinctions; scientific result above; linked-source fingerprint/calibration | Display caches and full calculated grids are disposable | Working-state identity and published fitted values can remain exact even when display arrays are reevaluated. Current PR readers still demand the full evidence payload. |
| Verified model/overlay/full-spectrum residual visualization | Verified source for the experimental/residual layers; supported model/processing/calibration descriptors, final nonlinear values, stored amplitudes, source-plane ordering and output-grid identity | Evaluate lineshapes/model on demand; do not solve amplitudes again; the sampled terminal correction matrix is not needed for a whole-spectrum `source - sum(final model contributions)` residual | Reconstructed model samples use declared tolerances, not a universal promise of original BLAS/libm output bits. Unsupported models must disable reconstruction rather than refit. |
| Exact terminal sample selection/review with source available | Verified source, exact paired indices/source-plane order, terminal corrections, fitted results and model descriptors; stored checksums/statistics as appropriate | Original observations can be read from the verified source; shapes/model/residuals can be forward evaluated | If every historical sampled prediction/residual bit must remain identical across runtimes, keep the relevant computed snapshots or define and verify a stricter replay contract. |
| Source-independent parameter/amplitude review | Compact fitted results, uncertainty outputs, coordinates, mappings, settings and provenance | No original sample observations are needed to inspect parameter tables or amplitude profiles | This useful capability is smaller than full sample-evidence review and must be labeled accurately. |
| Source-independent terminal sample/residual review | Exact sample observations or an explicitly sufficient numerical equivalent; exact terminal corrections; paired indices; fitted results/model descriptors | Shapes/model/residual arrays can be reconstructed when descriptors are supported; exact computed snapshots preserve review even if they are not | Without observations/corrections, do not invent residuals or claim full evidence. A compact default may preserve tables/profiles while offering detailed evidence separately. |
| Start a future MCMC analysis of the exact parent problem | Exact terminal corrected observations (or exact observations plus corrections), paired sample/source-plane identities, executable model/processing/calibration, nonlinear values/bounds/freedom, noise, likelihood/prior conventions and parent identity | Final sampled shapes/model/residual matrices are unnecessary likelihood inputs: sampling reevaluates shapes and solves amplitudes at each proposal | Final amplitudes remain valuable published results, but do not replace the parent likelihood inputs. The current archive does not itself guarantee a supported MCMC continuation interface. |
| Resume an existing MCMC chain exactly | Exact parent problem above, chain/walker state, RNG/sampler state and conventions, plus diagnostics/provenance | This is not delivered by a completed fit or saved final arrays | Out of scope for this PR; retaining a fit-evidence NPZ is not a promise of chain resumption. |

Sources: current
[`project.verify_sources`](../../src/peakfit/gui/project.py#L501) verifies source
content/calibration, exact retained source samples and model reconstruction;
[`project_results.verify_model`](../../src/peakfit/gui/project_results.py#L168)
compares re-evaluated shapes at `rtol=1e-10, atol=1e-12` and model values at
`rtol=1e-10, atol=1e-9`;
[`project_final_outcome`](../../src/peakfit/engine/scientific/simulation.py#L107) reevaluates shapes
and multiplies stored amplitudes without an amplitude solve;
[`NumericalReview`](../../src/peakfit/gui/editor/numerical.py#L248) distinguishes
parameter/profile views from observed/corrected/model/residual sample tables.
The approved
[GUI v2 capability contract](../PeakFit_GUI_Design_Specification_v2.0.md#L713)
requires numerical evidence, verified comparison and validated continuation to be
identified separately.

## Correction provenance is essential to the exact parent problem

The final optimizer uses the correction snapshot captured **before** that pass.
Corrections update only **between** passes, and no final update replaces the
terminal snapshot. Consequently a two-pass fit's terminal corrections encode the
previous pass's cross-cluster correction problem, not the final per-cluster model.
See [`run_pipeline_iter`](../../src/peakfit/fit/pipeline.py#L228).

Moreover,
[`update_cluster_corrections`](../../src/peakfit/engine/algorithms/common.py#L53)
creates a joint contributor cluster, sets its corrections to zero, re-solves
contributor amplitudes at the then-current nonlinear values, and projects only
external contributions onto each target's samples. Re-evaluating the final
accepted per-cluster amplitudes is therefore **not** the same operation. The
current completed outcome does not retain every previous-pass generation input.

Thus verified external source + final result is insufficient to recreate the
exact terminal corrections. A compact result may deliberately omit them only if
it explicitly relinquishes exact terminal residual / parent-likelihood replay,
or separately retains a sufficient correction-generation record. Keeping exact
terminal correction arrays is the straightforward evidence choice; replacing
those arrays with a new historical-generation representation needs its own
scientific design and measurement.

The current MCMC likelihood reads
[`cluster.corrected_data`](../../src/peakfit/engine/domain/cluster.py#L77), evaluates
shapes under each proposal, solves amplitudes, and evaluates normalized residuals:
[`_log_likelihood_blobs`](../../src/peakfit/engine/algorithms/mcmc.py#L84).
It does not consume the archived final sampled shape/model/residual arrays.

## Current readers are stricter than the minimal scientific contract

The proposals above are not implemented. Today `write_scientific` and
[`_validate_science`](../../src/peakfit/io/scientific.py#L210) require all retained
sample and analytical arrays and verify their algebra. The project
[`_write_run`](../../src/peakfit/gui/project.py#L177) always writes that full record;
reopening constructs a `CompletedFit` with full sample inputs. Simply deleting
members would make current readers reject the archive. A compact/detailed
contract needs explicit capability-aware readers and shared GUI/CLI exporters,
not missing arrays disguised as a complete existing record.

The CLI additionally writes
[`metadata/fitting_state.pkl`](../../src/peakfit/fit/fitting.py#L381), distinct from
`scientific.npz`. Current
[`run_mcmc_analysis`](../../src/peakfit/mcmc/analysis.py#L71) prefers that pickle,
requires summary JSON and source path resolution, and has not been migrated to a
compact scientific archive. Its JSON-only fallback constructs
[dummy one-point clusters](../../src/peakfit/io/readers/results.py#L139), so it is
not an exact reconstruction of the original parent problem. Do not claim that a
compact result already works with that fallback. A future continuation route must
validate/reconstruct the actual likelihood problem, not silently synthesize it.

## Validation and boundaries of this audit

- The real-data reproduction completed with the actual project/GUI/CLI writers;
  archive contributions reconcile exactly to their measured file sizes.
- `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run --no-sync pytest -q
  tests/test_scientific_storage.py tests/test_export_coordinates.py
  tests/test_gui_fit_export.py`: **34 passed**. An initial invocation without the
  headless settings crashed in native Qt widget construction; the offscreen run
  above completed. This audit does not claim a new native GUI interaction test.
- Ruff check/format for the audit tool and `ty check --error-on-warning` pass.
  Graphify was updated using its AST-only update.
- Production scientific code and schemas are unchanged. The retention estimates
  are measured byte budgets, not implemented round-trip formats. A later change
  must validate capability-aware reopen/recovery, source absence/mismatch, exact
  parent inputs and cross-interface parity before making these defaults real.

“Reproducible numerical results” must distinguish reading the exact published
values from rerunning an optimizer. The compact core preserves the former.
Reproducing the exact terminal likelihood additionally needs the retained inputs
listed above; reproducing an optimizer's trajectory can also depend on initial
state, algorithm/library versions, execution order and random state. Neither a
sampled-model snapshot nor a fit-result archive alone promises that trajectory.
Repeated experimental values must remain separate ordered planes, and exclusions
must remain explicit source-plane mappings/configuration; equal values are never
an invitation to deduplicate amplitude columns or renumber their scientific
identity.
