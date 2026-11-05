# Default linewidth start of sp1, sp2 and no_apod

Validation date: 2026-11-05 (#135). Tool: `tools/validate_linewidth_start.py`
(one BLAS thread). Regression tests: `tests/test_linewidth_start.py`,
`tests/test_experimental_overlap.py`, `tests/test_coincidence.py`.

## Question

From the default starts, the bundled pair 5N-H/76N-H stopped in the collapsed limit
(coincident positions, diverging opposite amplitudes, uncertainty warning). A
resolved minimum with a lower χ² existed under the same model. Is this specific to
the pair, and what is the simplest correction?

## Cause

All linewidths started at 20 Hz: broader than the intrinsic widths of typical lines
(about 5–15 Hz for the bundled amides). Two peaks less than about one linewidth apart
then start with nearly identical shapes. Their amplitude problem is nearly rank
deficient from the first step, and the optimizer slides into the collapse
([repeat-fit-collapse.md](repeat-fit-collapse.md) §4). That valley holds no
stationary point.

The 20 Hz value was not a measured choice. Earlier releases started SP2 at
R2 = 20 s⁻¹, which is lw = 6.4 Hz; the number survived the change of parameter to
the linewidth in Hz ([optimizers.md](../optimizers.md)).

## Real data: the bundled spectrum from common starts

Whole-dataset fits, two passes, 121 clusters. "Worse" is χ² above the best reached
from any start.

| Start (Hz) | Warned clusters | Worse by > 1 / > 10 | Time (s) |
|---:|---|---|---:|
| 30 | 4 | 19 / 7 | 4.0 |
| 20 | 3: 5N-H/76N-H, 36N-H/93N-H, 53N-H/147N-H | 20 / 5 | 3.7 |
| 15 | 1: 36N-H/93N-H | 14 / 3 | 3.5 |
| 12, 10, 6, 4, 2, 1 | 0 | 9 / 3 | 3.5–3.6 |

Every start from 1 to 12 Hz gives the same fit, cluster by cluster, to 0.006 in χ².

- **5N-H/76N-H and 36N-H/93N-H:** from narrow starts they resolve without a warning,
  with χ² 2.2% and 1.2% below the collapse.
- **53N-H/147N-H:** it also resolves without a warning, with χ² 20% above the
  collapse. The collapsed limit fits better because two coincident shapes with
  diverging opposite amplitudes add a per-plane derivative term the two-peak model
  does not otherwise contain. It is not a stationary point, and its individual
  values mean nothing. The resolved fit is a genuine local minimum; its misfit shows
  in χ²ᵣ and the residuals instead of in the warning.

## Synthetic pairs with known truth

Two SP2 lines on the bundled spectrum's axes with CEST-like profiles, peak SNR 30,
listed positions a tenth of a linewidth off the truth. A fit fails when its χ² exceeds
that of the fit started from the truth by more than 1. Failures of 6 (two intensity
ratios, three noise draws):

| True lw (Hz) | Separation (linewidths) | 20 Hz | 10 Hz | 5 Hz | 2 Hz | 1 Hz |
|---:|---:|---:|---:|---:|---:|---:|
| 2.5 | 1.0 | 3 | 1 | 0 | 0 | 0 |
| 5 | 0.5 | 5 | 5 | 0 | 0 | 0 |
| 5 | 1.0 | 3 | 0 | 0 | 0 | 0 |
| 10 | 0.3 | 3 | 0 | 0 | 0 | 0 |
| 10 | 0.5 | 2 | 0 | 0 | 0 | 0 |
| 2.5–60 | 0.3–1.0, other cases | 0 | 0 | 0 | 0 | 0 |

Over all 108 cases the 20 Hz start used 5 829 function evaluations; 2 Hz used 1 060.

- **Failures need a start broader than the truth.** They appeared when the start was
  at least twice the true width, at any separation up to one linewidth.
- **A narrow start costs nothing for broad lines.** Starts of 1–5 Hz never failed,
  including on 60 Hz lines. The window keeps a narrow apodized shape about as wide
  as the acquisition can resolve, so the optimizer widens it smoothly.
- **Unresolvable pairs stay unresolvable.** Where the fit from the truth itself is
  warned (2.5–5 Hz lines 0.3–0.5 linewidth apart, and some 2.5 Hz lines one
  linewidth apart), the warning reflects the data, whatever the start.

## Decision

The default linewidth start of sp1, sp2, no_apod and their doublets is 2 Hz:
- below the intrinsic linewidths expected of the data PeakFit targets;
- inside the 1–5 Hz range that never failed.

Configured starts and continued runs take precedence, as before. Gaussian,
Lorentzian and pseudo-Voigt keep their 25 Hz start of the apparent width. They were
not tested, and a start narrower than the point spacing would make their sampled
shapes degenerate.

**Golden reference (CLI, one pass).** Only the three formerly warned clusters change:
5N-H/76N-H −2.2% and 36N-H/93N-H −1.2% in χ², 53N-H/147N-H +19.9%; none is warned
now. In the other 118 clusters every parameter moves by at most 0.0042 standard
errors, and no linewidth by more than 0.01%.

## Limits

- One real dataset: every bundled example shares the same spectrum.
- SP2 only in the synthetic study; sp1 and no_apod share the mechanism.
- The narrow start removes the start-induced collapse, not the model's capacity to
  collapse. Pairs that collapse from their own resolved solution, or are closer than
  the data resolve, still carry the warning.
