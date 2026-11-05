# Acquisition time of the sp1, sp2 and no_apod lineshapes

Validation date: 2026-11-05. Rule: [lineshapes.md](../lineshapes.md). Tool:
`tools/validate_acquisition_time.py` (needs a local NMRPipe; NMRPipe 13.0 rev
2026.072 here). Regression fixtures: `tests/data/nmrpipe/lineshapes`
(`tools/make_nmrpipe_lineshape_fixtures.py`).

## Question

PeakFit's apodized lineshapes integrate the decaying signal under the window over
0 ≤ t ≤ aq. They used aq = `APOD`/`SW` and the recorded window (Q1, Q2). Is that the
time NMRPipe-processed signals actually spend under the window, for:
- short acquisitions;
- the first-point scaling and first-order phase that go with each sampling convention;
- direct and indirect axes, including Bruker data corrected for the digital filter?

## Set-up

Exact noiseless samples of decaying signals with known linewidth, position and
amplitude are converted by NMRPipe (`bin2pipe`, or `bruk2pipe` with digital-filter
metadata) and processed by `nmrPipe`. PeakFit then fits them with its own pipeline
under two rules:
- **recorded:** the earlier aq = `APOD`/`SW` with Q1, Q2 as recorded;
- **signal:** the window over the acquired signal of [lineshapes.md](../lineshapes.md).

The cases:
- **Sampling conventions:** first sample at t = 0 (first point ×½, `P1` 0);
  half-dwell (×1, `P1` −90/180); and a 0.1-dwell delay (×0.6, `P1` 36).
- **Shapes and lengths:** `sp1`, `sp2` and no window; 32, 64 and 256 points;
  linewidths of R2 5, 15 and 40 s⁻¹ at a 2000 Hz spectral width.
- **Bruker direct axis:** the filter emulated as an ideal band-limited delay of
  g = 67.98 points, corrected by FT; with and without `SP -df`; 256–2048 points.
- **Indirect axis:** ¹⁵N States, processed after a transpose; t = 0 and half-dwell;
  32 and 64 increments.

## Results

Linewidth error (%), for the three lines from narrowest to broadest:

| Case | Shape | N | Recorded | Signal |
|---|---|---:|---|---|
| t = 0 | `sp2` | 32 | +237, +75, +24 | −5.8, −4.1, −2.4 |
| | | 64 | +41, +14, +4.3 | −0.6, −0.5, −0.3 |
| | | 256 | +2.3, +0.6, +0.1 | 0.0, 0.0, 0.0 |
| | `sp1` | 64 | +50, +15, +3.7 | −2.4, −0.4, +0.1 |
| Half-dwell | `sp2` | 32 | +49, +12, +3.1 | −1.9, −2.4, −0.7 |
| | | 64 | +11, +2.4, −0.3 | −0.3, −0.2, −0.1 |
| | `sp1` | 32 | +71, +22, +8.4 | −8.9, −2.2, +1.4 |
| | | 64 | +7.1, +3.5, +0.1 | −0.8, −1.6, −0.1 |
| 0.1-dwell delay | `sp2` | 64 | +56, +16, +8.7 | +7.3, +1.7, +4.8 |
| | | 256 | +2.6, +0.7, +1.8 | +0.8, +0.3, +1.8 |
| Bruker filter | `sp2` | 512 | +83, +39, +19 | +1.8, +0.2, +0.5 |
| | | 1024 | +19, +9.1, +4.7 | +0.1, +0.1, +0.3 |
| | | 2048 | +4.5, +2.3, +1.4 | 0.0, +0.1, +0.3 |
| | `sp2 -df` | 512 | +1.8, +0.1, +0.2 | +1.1, −0.1, +0.1 |
| | no window | 512 | +29, +5.1, +0.5 | +4.2, +0.5, +0.4 |
| | | 1024 | +2.0, +0.2, +0.3 | +0.2, +0.1, +0.3 |

**Digital filter corrected during conversion** (`bruk2pipe -DMX`). The header
records no delay (`FDDMXVAL` 0, `FDDMXFLAG` 0), and `TDSIZE` already excludes the
filter points (512 → 443, 1024 → 955). The reader therefore uses g = 0, with
aq = (`TDSIZE` − 1)·dw. With the first point ×½, `sp2` linewidths are within 0.3%
and relative amplitudes within 0.2% at both lengths; this is the regression fixture
`sp2_digital_filter_converted`. With the first point ×1, they are biased +2% to +5%,
as the t = 0 convention predicts.

Indirect axis, two peaks (¹⁵N, then ¹H linewidth error, %):

| Case | N₁ | Recorded | Signal |
|---|---:|---|---|
| t = 0 | 32 | +54, +20; +25, +13 | −0.6, −0.1; −0.2, −0.1 |
| | 64 | +13, +20; +5.6, +13 | −0.1, −0.1; 0.0, −0.1 |
| Half-dwell | 32 | −0.7, +20; −1.5, +13 | +0.3, −0.1; +0.1, −0.1 |

The ¹H errors under the recorded rule come from its 64-point direct axis, which
starts at t = 0.

**Positions:** unchanged by the rule. They are within 0.03 Hz on every bin2pipe
case, and within 0.001 Hz on the indirect axis. The emulated Bruker data carry an
offset of 0.2–1.3 Hz under both rules, which falls with N. Its origin in the
emulation is not established, and the rule does not cause it.

**Amplitudes,** relative between lines of different width: within 0.5% under the
signal rule for every case with N ≥ 64 except the 0.1-dwell delay (+1.9% at 600 Hz,
identical under both rules). At N = 256 on the Bruker axis, where the filter delay is
27% of the acquisition, they are within 10%.

**Real data (GB3 HSQC, Bruker, NMRPipe demo).** The FID was truncated at conversion,
so each truncation is a genuine shorter acquisition. The table gives fitted linewidths
of 25 isolated peaks relative to the full acquisition (median, IQR); a consistent
model gives 1:

| Truncation | Recorded | Signal |
|---|---|---|
| ¹H 1676 → 512 points | 1.54 (1.36–1.70) | 0.79 (0.73–0.90) |
| ¹H 1676 → 256 points | 4.33 (3.62–5.35) | 0.93 (0.77–1.10) |
| ¹⁵N 170 → 64 increments | 0.87 (0.83–1.34) | 0.73 (0.68–1.20) |
| ¹⁵N 170 → 32 increments | 1.82 (1.11–2.29) | 0.84 (0.32–1.22) |

The signal rule removes the gross dependence on acquisition length. The remaining
±15–27% is expected of real amide lines, which are not single Lorentzians: unresolved
³J(HN,Hα) couplings of up to about 10 Hz sit under 7 Hz linewidths. This is
supporting evidence; the synthetic cases above are decisive.

## Conclusions

- **The recorded rule is biased whenever the acquisition is short.** That includes
  half-dwell axes, at the size of the clusters PeakFit actually fits.
- **The signal rule is exact for the two standard conventions,** t = 0 with the first
  point ×½ and half-dwell with it ×1. There the discrete sum is the trapezoid or the
  midpoint rule over the window's span.
- **For other sampling delays it is a first-order approximation.** With a 0.1-dwell
  delay at 64 points, up to +7% remains on the narrowest line.
- **Lines much narrower than the digital resolution** (R2·aq ≈ 0.08) keep up to 9%
  error at 32 points under either convention, where the data barely determine their
  width.
- **The Bruker digital filter** must be counted as missing acquisition time and a
  shifted window. Ignoring it overstated linewidths by 5–83% for 512–1024-point
  direct axes.
- **No window:** the established N·dw is kept. Only the filter delay is subtracted,
  which the evidence supports. For conventional data its errors are dominated by
  truncation ringing and do not favour a change.
- **Inconsistent first-point scaling** (×1 at t = 0, or ×½ with half-dwell) adds a
  constant offset that no rule can absorb. It is documented, not detected.

## Golden reference (CEST example)

The CEST example was acquired on a Varian spectrometer and converted with `var2pipe`.
It carries no digital-filter correction (`FDDMXVAL` 0, `FDDMXFLAG` 0).
- **What it shows:** experimental evidence for the correction on Varian-derived
  spectra, through the two standard conventions: a half-dwell ¹⁵N axis (`P1` 180,
  first point ×1, N 92) and an extracted ¹H axis sampled from t = 0 (`P1` 0, first
  point ×½, N 768).
- **What it does not show:** anything about Bruker digital-filter handling. That
  rests on the NMRPipe/Bruker fixtures above, and on the GB3 Bruker demo.
- Classifications and warnings are unchanged in all 121 clusters.
- Linewidths change by +0.35% median (¹⁵N, −0.08% to +1.08%) and +0.30% (¹H, −0.40%
  to +2.37%).
- Positions shift by at most 0.49 SE.
- Amplitudes of unwarned clusters change by +0.26% to +0.81%. The three warned
  clusters, whose individual values the data do not determine, change more.
- χ² rises by 0.09%. 460 of this comes from one retained pair that now re-collapses
  from default starts, although a resolved fit still has lower χ²
  (`tests/test_coincidence.py`).
