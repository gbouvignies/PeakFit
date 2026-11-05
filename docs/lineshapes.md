# Apodized lineshapes: sp1, sp2 and no_apod

These lineshapes model a processed peak as the Fourier transform of a decaying
complex exponential under the window the NMRPipe header records:

  S(Δν) = ∫₀^aq w(t) e^{−(π·LW + 2πi·Δν)t} dt,

where LW is the intrinsic linewidth (FWHM, R2/π) the line would have without
apodization, w is the sine bell (`sp1`: power 1, `sp2`: power 2) or 1 (`no_apod`),
and aq is the time the acquired signal spends under the window.

## The window over the acquired signal

NMRPipe evaluates the sine bell sin^p(π(Q1 + (Q2 − Q1)·j/(N − 1))) on sample
j = 0 … N − 1. The signal reaches sample j at time (j − g + d)·dw, where:
- **dw** is one acquired dwell, 1/(full spectral width): `SW` × `FTSIZE` / stored
  points, so extraction does not change it;
- **d** = `P1`/360 is the sampling delay the first-order phase removes: 0 when the
  first sample is at t = 0, ½ for a half-dwell delay;
- **g** is the Bruker digital-filter delay (`FDDMXVAL`) when the Fourier transform
  corrected it (`FDDMXFLAG` 1, direct axis). NMRPipe applies the window before that
  correction, so the first g points of the window precede the signal.

`SP -df` instead shifts the window itself by `APODDF` points.

So, with N = `TDSIZE`, PeakFit uses:

| | Acquisition time aq | Window over the signal, in units of Q1…Q2 |
|---|---|---|
| `sp1`, `sp2` | (N − 1 + d − g)·dw | from (g − d − `APODDF`)/(N − 1) to (N − 1 − `APODDF`)/(N − 1) |
| `no_apod` | (N − g)·dw | — |

For example:

| Acquisition and processing | aq for `sp1`/`sp2` | Window start |
|---|---|---|
| First sample at t = 0, first point ×½, `P1` 0 | (N − 1)·dw | Q1 |
| Half-dwell, first point ×1, `P1` 180 | (N − ½)·dw | Q1 − (Q2 − Q1)/(2(N − 1)) |
| Bruker direct axis, filter corrected by FT | (N − 1 − g)·dw | Q1 + (Q2 − Q1)·g/(N − 1) |

`no_apod` keeps its established N·dw, less the digital-filter delay.

## When this is exact

- **First-point scaling must match the sampling delay:** ½ + d, so ½ when the first
  sample is at t = 0 and 1 for half-dwell. Any other scaling adds a constant offset to
  the spectrum that no lineshape parameter absorbs. For example, an unscaled first
  point sampled at t = 0 overstates `sp2` linewidths by about 30% at 64 points.
- **`P1` must be the sampling delay.** A small manual adjustment barely changes aq,
  because it shifts d by `P1`/360 of a dwell.
- **The processed signal must be the acquired decay.** After linear prediction or
  NUS reconstruction, `TDSIZE` and `APOD` give the extended size. The lineshape then
  holds only if the extension reproduces the true decay; see
  [decisions.md](decisions.md) (2026-11-05, reconstruction) and
  [correlated-noise.md §2.5](correlated-noise.md#25-what-the-final-header-cannot-show).

Other windows (`APODCODE` other than 0 or 1) are fitted with the Gaussian,
Lorentzian or pseudo-Voigt lineshapes, which do not use aq.

## Validation

NMRPipe-processed noiseless signals of known linewidth, position and amplitude,
fitted by PeakFit: [validation/acquisition-time.md](validation/acquisition-time.md).
The regression fixtures are in `tests/data/nmrpipe/lineshapes`.
