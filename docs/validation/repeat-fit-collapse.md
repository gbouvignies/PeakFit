# Repeat-fit collapse of resolved overlapping peaks

Investigation: 2026-11-02, base `2113c39` (main through #87). Python 3.14.5,
NumPy 2.4.1, SciPy 1.17.0. Read-only: no production code, default, tolerance or
schema changes. This extends [overlapping peaks](overlapping-peaks.md),
[linewidth scaling](linewidth-scaling.md) and the unmerged
coincident-decompositions proposal (branch `dev/epic-kalam-098f00`).

**Finding.** Refitting an unedited analysis is not reproducible. On the bundled
dataset the second fit collapses the resolved pair 52N-H/102N-H onto one
position. χ²ᵣ rises from 1.53 to 2.32, |ρ| from 0.67 to 1.00, and the cluster
gains the uncertainty warning; every later refit keeps the collapse. The fit
itself is deterministic. The cause is the refit's starting point: completion
adopts fitted **positions** but every **linewidth** restarts at the 20 Hz
default. That combination does not correspond to any previous solution, and
from it the optimizer slides into a degenerate valley. Coincident positions
are then adopted and the collapse becomes permanent.

**Root cause.** Two peaks with near-identical shapes and unconstrained,
opposite per-plane amplitudes approximate one peak plus a lineshape-derivative
term. The objective has no stationary point there. The evidence indicates its
infimum lies in the limit of coincident shapes and diverging amplitudes. Trust-region iterations
stall along that valley and stop on `ftol`/`xtol`, and the result is classified
`converged` even though the first-order optimality is 300–10 000 times that of
any healthy cluster. Whether the stalled point beats the resolved optimum
depends on the pair: 52N-H/102N-H and 5N-H/76N-H are optimizer failures (a
resolved optimum is better), while 53N-H/147N-H and 36N-H/93N-H are
identifiability problems (the degenerate construct fits better than any
resolved two-peak solution found).

**Recommended correction.** Make a refit a continuation. A peak whose working
position is the one adopted from the active run starts from that run's complete
fitted nonlinear state (positions *and* linewidths). Edited and new peaks start
as today. Emulated on the bundled data, this reproduces every cluster of the
previous run and keeps 52N-H/102N-H resolved, while the three run-1 coincident
pairs remain coincident and warned rather than hidden. No minimum separation, peak deletion,
sign constraint or solution selection is involved. Two related decisions are
listed under [Decisions](#decisions).

## Reproduction

Inputs: `examples/02-advanced-fitting/data/pseudo3d.ft2`, `pseudo3d.list`,
`b1_offsets.txt`, peak axes indirect-direct, GUI default
`PeakFitConfig(fitting=FitConfig(refine_iterations=2))`. 166 peaks,
121 clusters, noise 11 922.5.

```python
from copy import deepcopy
from pathlib import Path

from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fit_positions import dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.spectrum import read_dataset

DATA = Path("examples/02-advanced-fitting/data")
ds = read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt")
settings = PeakFitConfig(fitting=FitConfig(refine_iterations=2))


def fit(peaks):
    prepared = prepare_dataset_fit(ds.spectra, tuple(peaks), settings)
    return finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))


def adopt(result):  # the GUI's automatic adoption after every fit
    return dataset_completion(
        result, result.inputs.snapshot, submitted_revision=0, manual_revision=0
    ).working_peaks


run1 = fit(ds.peaks)
run2 = fit(adopt(run1))  # 52N-H/102N-H collapses here
```

Run with one BLAS thread (`OPENBLAS_NUM_THREADS=1`); each whole-dataset fit
takes about 9 s. Single-cluster experiments call `fit_cluster` on the pair's
cluster from `prepare_dataset_fit` with zero corrections, exactly as every
first pass does, after setting starting values with `prepare_cluster_parameters`.

## Evidence

### 1. Separating the candidate causes

| Candidate | Test | Result |
| --- | --- | --- |
| Non-determinism | Same prepared inputs fitted twice | Bit-identical outcomes for all 121 clusters |
| Clustering | Members and fitting samples, runs 1 and 2 | Identical: 52N-H/102N-H, 7 991 samples |
| Neighbor corrections | Pass-by-pass trace | Collapse completes in **pass 1**, whose corrections are zero in both runs |
| Two-pass procedure | Pass 2 of run 2 | Refines inside the collapsed basin (positions change < 2×10⁻⁴ ppm) |
| Amplitude initialization | — | None exists: VarPro solves amplitudes exactly at every iteration |
| Starting positions | Run-1 fitted positions vs listed seeds | Both are separated; see linewidth row |
| Linewidth starts | 20 Hz default vs representative 12 Hz | **Decisive** (tables 2 and 3) |

Starting values of the 52N-H/102N-H cluster:

| Run | 52N-H F2 / F3 ppm | 102N-H F2 / F3 ppm | Linewidth starts | Pass-1 evaluations | Outcome |
| --- | --- | --- | --- | ---: | --- |
| 1 | 115.43725 / 7.65308 (list) | 115.51158 / 7.66088 (list) | 20 Hz all | 16 | Resolved |
| 2 | 115.45445 / 7.65248 (run 1) | 115.50477 / 7.66151 (run 1) | 20 Hz all (fitted 5.9–12.8 Hz discarded) | 135 | Coincident |
| 3 | 115.44193 / 7.65967 (run 2) | 115.44195 / 7.65967 (run 2) | 20 Hz all | — | Coincident |

The run-1 optimum separates the pair by 4.1 Hz (F2) and 7.2 Hz (F3), about
0.4 and 0.8 of the fitted linewidths. Starting both peaks there at 20 Hz
makes their shapes nearly collinear before the widths can narrow.

### 2. Start dependence of pass 1, per pair

`2·cost` is χ² on the cluster's samples with zero corrections. Lower is better.

| Pair | Listed seeds, 20 Hz | Run-1 positions, 20 Hz | Listed seeds, 12 Hz | Run-1 positions, 12 Hz |
| --- | --- | --- | --- | --- |
| 52N-H/102N-H | resolved 11 940 | **coincident 17 782** | resolved 11 940 | resolved 11 940 |
| 5N-H/76N-H | coincident 18 428 | coincident 18 428 | **resolved 17 996** | coincident 18 568 |
| 53N-H/147N-H | coincident 21 456 | coincident 21 456 | resolved 25 165 | coincident 21 456 |
| 36N-H/93N-H | coincident 20 798 | coincident 20 798 | resolved 21 548 | coincident 20 952 |

Two classes follow. For 52N-H/102N-H and 5N-H/76N-H a resolved solution has a
lower χ² than the coincident endpoint (Δχ² = 5 842 and 432): the collapse is an
**optimizer failure**. For 53N-H/147N-H and 36N-H/93N-H the coincident endpoint
has the lower χ² (Δχ² = −3 709 and −750): this is an **identifiability
problem**, as found earlier. No starting point establishes their assignments.

Run 1 already collapsed three of the four pairs, so their adopted positions
are coincident. Seeding the next fit there keeps them coincident even with
12 Hz starts (last column).

### 3. Basin of the collapse

Pair started symmetrically about its listed midpoint, at a fraction *f* of the
listed separation, equal linewidth starts. C = coincident, R = resolved.

| Pair | Start | f = 0.25 | 0.5 | 0.67 | 0.75 | 0.9 | 1.0 | 1.25 | 1.5 | 2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 52N-H/102N-H | 20 Hz | C | R | R | R | R | R | R | R | R |
| 52N-H/102N-H | 12 Hz | R | R | R | R | R | R | R | R | R |
| 5N-H/76N-H | 20 Hz | C | C | C | C | C | C | R | R | R |
| 5N-H/76N-H | 12 Hz | R | R | R | R | R | R | R | R | R |
| 53N-H/147N-H | 20 Hz | C | R | C | R | C | C | R | R | C |
| 53N-H/147N-H | 12 Hz | R | R | R | R | R | R | R | R | R |

Wide starts shrink and fragment the resolved basin; at 12 Hz every tested
start resolves (for 53N-H/147N-H at a higher χ² than the collapse).

### 4. Why the collapse persists: a stalled valley, not a minimum

After adoption, the two peaks start 10⁻⁵ ppm apart. Starting 52N-H/102N-H
coincident at its run-1 position with small ¹⁵N offsets:

| Offset (ppm) | 20 Hz start | 12 Hz start |
| --- | --- | --- |
| 0 | C 17 782 | R 11 940 |
| 10⁻⁶ | C 17 782 | C 17 463 |
| 10⁻⁵ | **C 83 038, stopped after 12 evaluations** | C 17 463 |
| 10⁻⁴ | R 11 940 | C 17 782 |
| 10⁻³ | C 17 782 | C 17 782 |
| 10⁻² | C 17 782 | C 17 782 |

Outcomes flip with 10⁻⁶ ppm changes: the region is chaotic. The 10⁻⁵ case
ends on `xtol` with first-order optimality 7.8×10⁹ at seven times the resolved
χ², and is classified `converged`.

Recorded first-order optimality (`OptimizerProvenance.optimality`) confirms
that the coincident endpoints are not stationary points:

| Run | Unwarned clusters: median / max | Coincident clusters |
| --- | --- | --- |
| 1 (118 unwarned) | 0.57 / 12.9 | 3.9×10³, 6.6×10³, 3.9×10⁴ |
| 2 (117 unwarned) | 0.58 / 13.1 | 1.5×10⁴, 1.6×10⁴, 4.3×10⁴, **1.3×10⁵ (52N-H/102N-H)** |

Each stops on `ftol` because the relative reduction per step becomes small
while amplitudes keep diverging (±6×10¹² for 53N-H/147N-H, against a total of
about 5.5×10⁹). In the limit, two coincident shapes with amplitudes
`(a + b/2ε, −b/2ε)` reproduce `a·f + b·∂f`: one peak plus a per-plane
derivative term. That added freedom is what lowers χ² for 53N-H/147N-H and
36N-H/93N-H. It is a property of the model with free signed amplitudes, not a
numerical defect, and the existing uncertainty warning flags every case.

### 5. Continuation reproduces the previous run

Refitting the unedited run-1 analysis three ways (χ²ᵣ compared cluster by
cluster with run 1; *changed* means a relative difference above 10⁻⁶):

| Refit start | Changed χ²ᵣ | New coincident pair | Warned clusters |
| --- | ---: | --- | --- |
| Adopted positions, 20 Hz widths (current) | 20 of 121 | 52N-H/102N-H | 4 |
| Adopted positions and all fitted nonlinear values | 1 of 121 (146N-H, 2.0970 → 2.0970) | none | 3 (the run-1 pairs) |
| The same, repeated once more | 1 of 121 | none | 3 |

## Root-cause assessment

1. **Model.** With free, signed per-plane amplitudes, two near-coincident peaks
   span one peak plus its derivative. The non-stationary endpoints and the
   diverging amplitudes indicate a degenerate valley whose infimum lies at the
   coincident limit, with infinite amplitudes. This is an
   identifiability property of the two-peak model. No numerical component is
   defective, consistent with [overlapping peaks](overlapping-peaks.md).
2. **Optimizer.** Inside the valley, trust-region steps stall and terminate on
   `ftol`/`xtol` far from stationarity. `converged` means "stopped normally",
   not "at a minimum". The uncertainty warning nonetheless catches every case
   observed here.
3. **Initialization.** A 20 Hz width start for peaks that overlap within one
   linewidth makes their shapes nearly collinear from the first step and
   steers them into the valley. This is the documented start sensitivity
   ([linewidth scaling](linewidth-scaling.md)).
4. **Refit procedure (new).** Automatic adoption moves positions to the
   previous optimum but discards its fitted widths, producing a start that
   belongs to no solution. This can **create** a collapse that the first fit
   avoided (52N-H/102N-H). Adoption of coincident positions then **locks it in**,
   as the coincident-decompositions proposal predicted without testing it.

## Recommended correction

**Make a refit a continuation of the active run** (engine and preparation):

- A working peak whose position equals the position adopted from the active
  run starts from that run's complete fitted nonlinear state. That means its
  positions, linewidths and any other fitted shape parameter, clamped to the
  configured bounds.
- A peak that was added, moved or edited since that run, or any peak after a
  settings change that alters its parameters (lineshape, bounds), starts from
  the configured defaults, as now.
- The prepared problem records which peaks continued from which run, so the
  starts remain reproducible from the project. This touches
  `FitInputSnapshot`, a storage decision.

Effect: an unedited refit reproduces its run, and only edits move the problem.
It does not escape existing coincident pairs, which stay warned. It applies no
separation, deletion, sign restriction or preference between solutions.

## Decisions

1. **Continuation and storage.** Approve recording continued starts in the
   input snapshot (and so in projects), or derive them at preparation time from
   the retained active run. The second needs no format change but makes a
   project's next fit depend on run history being present.
2. **Adoption of coincident pairs.** Today completion adopts both positions of
   a coincident pair. With continuation, the pair would also continue
   coincident. An alternative is to adopt nothing for peaks recorded as
   coincident, retaining their previous starting positions with a stated reason,
   so the next fit is not seeded inside the valley. This needs the coincident-pair record
   from the coincident-decompositions proposal, which this investigation
   supports.
3. **Not recommended now:** changing the 20 Hz default, multi-start fitting or
   reclassifying `ftol` stops. The first changes basins for every dataset and
   needs multi-dataset evidence; the second is a larger design. The third would
   change warning classifications, which stay as they are.

## Regression tests to add with the correction

- **Unedited refit is idempotent:** on the bundled data, fit, adopt and refit;
  every cluster's χ²ᵣ matches within 10⁻⁶ relative and no cluster gains a
  warning. This fails today (20 clusters change; 52N-H/102N-H collapses).
- **Determinism:** identical prepared inputs give bit-identical outcomes
  (passes today; protects the comparison above).
- **Continuation scope:** a moved or added peak starts from defaults; an
  unchanged neighbor continues; a lineshape change resets all starts.
- **Honest persistence:** a coincident run-1 pair stays warned after a
  continued refit, and its classification is unchanged.

## Limits

One experimental dataset and one lineshape (SP2). Starts were varied for one
pair at a time with zero corrections, which matches every first pass. The
continuation emulation set `Parameters` values directly. The production change
must carry starts through preparation and the snapshot rather than patch them.
