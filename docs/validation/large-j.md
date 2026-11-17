# Large J couplings: fitting regions, qualifications and phase

Validation date: 2026-11-17 (W2c, #214). Tool: `tools/validate_large_j.py`, records
[large-j.json](large-j.json) and [large-j-confirmation.json](large-j-confirmation.json).
Tests: `tests/test_large_j.py`, `tests/test_gui_couplings.py`, and
`tests/test_large_j_record.py`, which checks that this note quotes the records.

**Verdict.** The gates fixed in advance fail in two places. Neither failure comes
from a large-J region:
- The ¹H singlet linewidth of the SP1 case under-covers. The cause is a bias from
  contour-selected regions, which singlets show too.
- At SNR 8, the "below LW" qualification is raised in 96.4 % of J = ½ LW fits.

Everything the issue asks of large J passes:
- **Regions:** every realization holds both components in one region.
- **¹J recovery:** J, both positions and the ¹⁵N linewidth are calibrated.
- **Phase:** the 10° phase is recovered within its σ, and ρ(J, phase) is reported.
- **J below LW:** the qualification and ρ(J, LW) are reported.

## Scientific impact

Before this change, **a doublet whose splitting spans more than about three points
was fitted without its components**:
- the fitting region was the contour segment holding the peak's centre point;
- the centre of a doublet with large J has no signal;
- with each component its own contour island, the region was the 3 × 3 block at the
  centre.

On the validation geometry, a ¹⁵N ¹J of 92 Hz is 11.8 points. With the former rule,
the two components were in the peak's region in 0.8 % (Lorentzian) and 0 % (SP1) of
realizations.

**Since W2b**, a one-bond coupling can be configured (`doublet = { j = 92 }`). Every
such fit that relied on the region was unusable: J, positions, linewidths and
amplitudes came from noise around the centre.

**Singlet results are unchanged, bit for bit** ([below](#bundled-example)).

## Change

1. **Footprint (`ShapeBase.footprint_i`, `Peak.footprint_i`).**
    - A doublet's position is the multiplet centre (`GLOSSARY.md`), and its
      components sit at cs ± J/2.
    - Its footprint is every point from cs − J_max/2 to cs + J_max/2 on its axis.
    - J_max is J's upper bound after the configured constraints (`prepare_clusters`
      applies `[parameters]` first), the bound the fit itself uses.
    - The footprint therefore holds both components wherever J settles, a fixed J
      included, since a fit step can release it.
    - It wraps across the spectral edge, as an aliased component does.
2. **Segmentation (`segment_data`).**
    - As before, each peak's centre point is marked and dilated by one point.
    - New: each footprint is then marked as it is, undilated, and both are joined to
      the points above the contour.
    - The footprint only connects: its line through both components' maxima joins
      their contour islands to the centre.
    - A footprint inside the centre's 3 × 3 block changes nothing.
3. **J below its linewidth (`engine.diagnostics.couplings`).**
    - A fitted J below its axis's fitted linewidth is qualified "below LW": the
      components overlap at half height, so J and the linewidth are determined
      together.
    - The qualification gives ρ(J, LW) and, where the axis's phase is fitted,
      ρ(J, phase).
    - Both come from the recorded lineshape covariance σ²C_θ, which matches the
      published σ to the precision shown.
    - They are withheld wherever σ(J) is: whenever a varied J has no σ, the W2a rule
      withholds the lineshape factors.
    - It is a qualification, not a numerical warning. It is shown under the peak
      page's Fitted table and in the report's "J Couplings" section.
4. **Naming the covariance's rows (`lineshape_order`).**
    - The recorded parameters are sorted by name, and C_θ's rows were in the order of
      the working parameters (cs, lw, j, phase per axis). Nothing recorded named
      them.
    - The evaluation now builds θ in the order the shapes create their parameters.
      A reader recomputes that order from the run's recorded model geometry.
    - The two orders agreed in every test and every fit of the bundled example
      (checked with an assertion across the full suite and five lineshape setups,
      custom axis labels included), so no archive changes.
    - Reading C_θ in the recorded, sorted order mislabels rows: an early version
      reported ρ(J, phase) = 10⁻¹⁶ for a fit whose ρ is 0.27.

## Protocol

It is fixed in the tool's docstring, committed before the first run (bf35f2a):
- **Model:** a pseudo-3D peak with a doublet on ¹⁵N F2 and a singlet on ¹H F3, in a
  256 × 256 grid at 7.8 and 9.4 Hz per point.
- **Fit:** prepared and fitted as the CLI and GUI do.
- **Clustering:** at a contour of 15 % of the first-plane peak, or 5σ where that is
  higher.
- **Starts:** J starts 20 % low; positions start 0.25 point off.
- **Realizations:** 1000 per case, seeds (2141117, case, index), five blocks.

Gates:
- every fit's region holds both components;
- ±1σ and ±1.96σ coverage of every parameter in the two ¹J cases;
- "below LW" in ≥ 99 % of J = ½ LW fits and ≤ 1 % of ¹J fits;
- no misleading σ(J).

## Results

Second footprint design (the one in this change). In the first table, "Region" is
the fraction of fits whose single region holds both components, and "former rule"
the same fraction under the centre-only rule. "ρ" gives the recorded first-order
value (mean over fits that report it), then the correlation of the estimates across
those fits, with its 95 % interval. Bias, scatter and RMS σ are in Hz (positions too)
and degrees (phase).

| Case | Region (former rule) | Below LW | J bias / scatter (Hz) | σ(J) withheld | Misleading (SE) | ρ(J, LW) recorded / estimates [95 %] | ρ(J, phase) | Gate |
|---|---|---:|---|---:|---|---|---|---|
| Lorentzian, one-bond coupling (J = 92 Hz) | 1.000 (0.008) | 0.000 | -0.009 / 0.153 | 0.000 | 0.048 (-0.3) | -0.00 / -0.03 [-0.09, +0.04] | — | pass |
| SP1, one-bond coupling, 10° phase | 1.000 (0.000) | 0.000 | +0.007 / 0.153 | 0.000 | 0.048 (-0.3) | -0.01 / -0.02 [-0.08, +0.04] | +0.02 / -0.02 [-0.08, +0.04] | FAIL |
| Lorentzian, J = ½ LW | 1.000 (1.000) | 1.000 | -0.078 / 0.479 | 0.002 | 0.036 (-2.0) | -0.79 / -0.80 [-0.82, -0.78] | — | pass |
| Lorentzian, J = ½ LW, low SNR | 1.000 (1.000) | 0.964 | -1.831 / 2.887 | 0.985 | 0.013 (-5.4) | -0.46 / -0.14 [-0.61, +0.40] | — | FAIL |

| Case | Parameter | Bias | Scatter | RMS σ | Withheld | ±1σ (SE) | ±1.96σ (SE) |
|---|---|---:|---:|---:|---:|---|---|
| Lorentzian, one-bond coupling (J = 92 Hz) | J | -0.009 | 0.153 | 0.151 | 0.000 | 0.660 (-1.5) | 0.952 (+0.3) |
| Lorentzian, one-bond coupling (J = 92 Hz) | δ F2 | +0.005 | 0.078 | 0.076 | 0.000 | 0.678 (-0.3) | 0.948 (-0.3) |
| Lorentzian, one-bond coupling (J = 92 Hz) | δ F3 | -0.007 | 0.089 | 0.091 | 0.000 | 0.692 (+0.6) | 0.952 (+0.3) |
| Lorentzian, one-bond coupling (J = 92 Hz) | LW F2 | +0.048 | 0.233 | 0.230 | 0.000 | 0.662 (-1.4) | 0.946 (-0.6) |
| Lorentzian, one-bond coupling (J = 92 Hz) | LW F3 | +0.110 | 0.327 | 0.327 | 0.000 | 0.643 (-2.7) | 0.942 (-1.2) |
| SP1, one-bond coupling, 10° phase | J | +0.007 | 0.153 | 0.155 | 0.000 | 0.691 (+0.6) | 0.952 (+0.3) |
| SP1, one-bond coupling, 10° phase | δ F2 | +0.003 | 0.255 | 0.243 | 0.000 | 0.658 (-1.7) | 0.942 (-1.2) |
| SP1, one-bond coupling, 10° phase | δ F3 | +0.000 | 0.078 | 0.079 | 0.000 | 0.692 (+0.6) | 0.946 (-0.6) |
| SP1, one-bond coupling, 10° phase | LW F2 | +0.029 | 0.330 | 0.327 | 0.000 | 0.681 (-0.1) | 0.946 (-0.6) |
| SP1, one-bond coupling, 10° phase | LW F3 | +0.193 | 0.295 | 0.299 | 0.000 | 0.607 (-5.1) | 0.905 (-6.5) |
| SP1, one-bond coupling, 10° phase | phase F2 | -0.034 | 1.190 | 1.131 | 0.000 | 0.653 (-2.0) | 0.937 (-1.9) |

One gate came within 0.5 SE of its limit; the confirmation run (realizations
1000–4999, rule committed before it in ac376c5) decides it:

| Case | Gate | First run (SE) | Confirmation (SE) | Blocks p | Verdict |
|---|---|---|---|---:|---|
| Lorentzian, one-bond coupling (J = 92 Hz) | coverage_1sigma:A.F3.lw | 0.643 (-2.7) | 0.665 (-2.4) | 0.10 | pass |

### Findings

**One-bond couplings are fitted and calibrated.**
- Every realization holds both components in one region. The former rule did so in
  at most 0.8 %.
- J is unbiased (≤ 0.01 Hz on 92 Hz), and its RMS σ matches its scatter within 2 %.
- Positions (the multiplet centre) and the ¹⁵N linewidth are calibrated. All their
  gates and replications pass.

**The phase is recovered, and ρ(J, phase) is reported.**
- The 10° phase comes out at 10° − 0.03°, with σ 1.13° against a scatter of 1.19°.
- Its ±1.96σ coverage is 0.937 (−1.9 SE).
- ρ(J, phase) is +0.02 recorded, −0.02 across the estimates.
- They are uncorrelated by symmetry: J's derivative is even about the multiplet
  centre and the phase's, a dispersive line at each component, is odd. So a
  resolved doublet's phase trades off with its position, not its J.
- Where the region or the lineshape breaks that symmetry, ρ(J, phase) is not zero.
  `tests/test_large_j.py` checks a J = ¾ LW SP1 doublet whose recorded ρ(J, phase) is
  0.72.

**A J below its linewidth reports its dependence.**
- At J = ½ LW and SNR 50, every fit is qualified "below LW".
- σ(J) is reported in 99.8 % of fits.
- The recorded ρ(J, LW) of −0.79 matches the estimates' −0.80 [−0.82, −0.78].
- J and LW are therefore determined together, and the qualification says so.

**Fails: at SNR 8, the qualification misses 3.6 % of J = ½ LW fits.**
- σ(J) is withheld in 98.5 % of these fits (the W2a rule), and 1.3 % of all fits
  report a ±1.96σ interval that misses (the gate allows 5 %).
- Diagnosis, on the record's own realizations:
    - 959 fits are qualified with σ(J) withheld, and 5 qualified with σ(J) reported;
    - 26 are not qualified but have σ(J) withheld;
    - 10 are neither.
- All 10 miss the truth at ±1.96σ. They land on a narrow resolved doublet
  (Ĵ = 11.8–15.6 Hz, LŴ = 5.9–14.5 Hz), so Ĵ > LŴ and the qualification, which reads
  the fitted values, cannot see them.
- This is the residual that the doublet-J validation found for unresolved couplings
  at low SNR (0.5–2 % there). No fit-local rule removes it; use MCMC.
- The 99 % gate assumed the fitted values would stay on the right side of J = LW.
- ρ(J, LW) there is first-order in a non-linear regime: −0.46 recorded against
  −0.14 [−0.61, +0.40] across the 15 fits that report it.

**Fails: the SP1 ¹H linewidth is biased by its contour-selected region.**
- The ¹H singlet's linewidth is 0.19 Hz high (0.65 of its scatter). Its ±1σ coverage
  is 0.607 and its ±1.96σ coverage 0.905.
- It is not this change. Exploratory diagnosis, 400 realizations per row, seeds
  (1, 99, index), SP1 unless stated:

  | Peak | Region | ¹H LW bias (Hz) | ±1.96σ coverage |
  |---|---|---:|---:|
  | doublet | contour, as in the record | +0.154 ± 0.015 | 0.905 |
  | singlet | contour | +0.169 ± 0.019 | 0.930 |
  | singlet | fixed 41 × 23 box | +0.002 ± 0.016 | 0.960 |
  | doublet | fixed 41 × 23 box | −0.019 ± 0.011 | 0.948 |
  | Lorentzian singlet | contour | +0.068 ± 0.025 | 0.960 |
  | Lorentzian singlet | fixed box | +0.006 ± 0.022 | 0.958 |

- A region chosen where |data| exceeds the contour keeps edge points whose noise
  happens to be positive. The line looks broader at its edges, so its fitted width
  is biased up.
- It affects every peak whose region ends at its contour, singlets included, and
  shows most where the line spans few points at the contour (¹H here).
- The Lorentzian ¹J case shows it too (+0.11 Hz, ±1σ coverage 0.643, −2.7 SE). The
  confirmation passes it at 0.665 (−2.4 SE).
- It needs its own change and validation: a region that does not depend on the noise
  in the data it fits.

## First footprint design (superseded)

The first design dilated the whole span, which gave every point a component can
reach a 3 × 3 neighbourhood. Its run under the same protocol (record at ea028d6)
passed the region and ¹J calibration gates, with the same ¹H linewidth failure.

It also grew the regions of small-J doublets. A J_max/2 of 8 Hz, one point, gave
15.4 points against 9.8. At SNR 8 and J = ½ LW, that raised the fraction of fits
reporting a σ(J) that misses the truth from 1.3 % to 6.7 % in the run.
- On the same realizations, the former rule gives 1.3 %. A confirmation on 4000 more
  gave 7.8 % (+8.2 SE): a regression for couplings small against the point spacing.
- With more points, more fits of an unresolved doublet pass the σ(J)/J rule on the
  wrong mode.
- The second design leaves small-J regions as they were, and the same run passes
  that gate at 1.3 %.

Its larger ¹J regions were more precise for the SP1 case, both versions calibrated:

| SP1 ¹J, RMS σ | first design | second design |
|---|---:|---:|
| phase (°) | 0.73 | 1.13 |
| δ F2 (Hz) | 0.17 | 0.24 |
| region (points) | 77.7 | 57.1 |

The second design was chosen because the issue requires small couplings to stay
unchanged. A ¹J fit loses the noise-dominated points between and around its
components; the components themselves are inside.

**Exploratory runs before the protocol**, not part of the record:
- 10 realizations per case, seed 1. They found that a 15 % contour is below the noise
  at SNR 8, which joined most of the spectrum to the region; the 5σ floor was added
  before the run.
- Before that, a ¹H axis acquired over its whole 128-point width (no zero-filling)
  made the SP1 ¹H line one to two points wide. Its width went to the 0.1 Hz bound, so
  the geometry was changed to 50 ms in 256 points.

## Bundled example

`tools/scientific_digest.py` digests of the bundled pseudo-3D (golden-reference
arguments, `--refine 1`), on this change and on main (245961a), in one environment:

| Fit | context | outcome | samples |
|---|---|---|---|
| `auto`, `lorentzian`, `gaussian`, `sp1`, `sp2` | identical | identical | identical |
| `lorentzian`, `--doublet F2=5,1,10 --doublet F3=5,1,10` | identical | identical | identical |
| `sp1`, `--doublet F3=7` (J in [3.5, 14] Hz) | identical | identical | identical |
| `sp1`, `--doublet F2=92` | changed | changed | changed |

- `metadata.json` differs only by the input files' absolute paths, which name the two
  checkouts.
- **Small couplings are unchanged.** At 5.9 Hz per point, a J_max/2 of 5 or 7 Hz
  stays within the centre's 3 × 3 block, so no region changes.
- **The first design changed them.** It changed 23 of 121 clusters (19 for SP1) by
  one to twelve points (131 observations each). Its largest shifts were where the
  doublet model does not describe these singlet peaks: reduced χ² up to 34, or J
  moving to a bound.
- **A 92 Hz ¹⁵N doublet on this singlet spectrum is meant to change.** Its J_max
  line spans ±15.5 points, which joins neighbouring peaks: 121 clusters become 79.
  The model is wrong for these data either way; on main its regions held no doublet
  components at all.
