# Processing quantitative series

This page is for anyone who processes a pseudo-ND series (T1, T2, CPMG, CEST) for
PeakFit and then uses its profile σ downstream, for example in ChemEx. It states
which processing PeakFit's uncertainties are validated for, what is known about the
alternatives, and how to read the empirical checks of `peakfit replicates`. It describes evidence, not rules: PeakFit accepts every processing
below and changes no value or error because of it.

Every number here comes from a recorded study, linked where it is used. Each is a
measurement under the conditions stated with it, not a bound for other data.

## The validated workflow

PeakFit's uncertainties are validated for:
- **uniform sampling** of every indirect dimension;
- **linear NMRPipe processing**: a sine bell (`SP`) or no window, zero fill, Fourier
  transform, with the Bruker digital-filter delay corrected by the Fourier transform
  (`FT`, the default for data converted with `-decim`, `-dspfvs` and `-grpdly`).

For this processing, the noise correlation PeakFit derives from the header matches
NMRPipe-processed noise to 5 × 10⁻⁵, and the `sp1`, `sp2` and `no_apod` lineshapes
are those of the processed peaks
([correlated-noise §2.2](correlated-noise.md#22-correlation),
[lineshapes](lineshapes.md)). In synthetic 2D clusters fitted by PeakFit, the
published errors match the observed scatter
([sandwich errors](validation/sandwich-errors.md)), and the profile σ of peaks without
the correlated-profile flag is within 0.962–1.039 of the scatter
([profile-uncertainty §8.1](profile-uncertainty.md#81-t3-coverage-and-the-correlated-profile-thresholds)).
These studies use the exact lineshape and fits started at the truth.

This is the case PeakFit can check end to end, not the only acceptable processing.
Other methods can have real experimental advantages; what follows says what is known
about each, so that the choice can be made knowingly.

**The header records only part of the history.** PeakFit reads the processing from
the final NMRPipe header. Linear prediction, an earlier window, `SP -size`, `-start`
or `-one`, digital-filter correction during conversion, a solvent filter and NUS
reconstruction all leave a header that looks like ordinary processing, so the run is
recorded as "noise correlation modelled"
([correlated-noise §2.5](correlated-noise.md#25-what-the-final-header-cannot-show)).
That record states what the header shows, not that the processing was the
validated one.

## Three concerns

Each alternative below is described by its evidence on three separate questions:
- **Relative-intensity bias:** does the processing change the intensity ratios
  between planes, which every downstream fit with a free scale depends on?
- **σ calibration:** do the published errors match the actual scatter?
- **Correlation between planes:** does the processing couple the noise of different
  planes? PeakFit's covariance between planes includes only what shared fit
  parameters introduce; it treats the spectral noise of different planes as
  independent.

**Intensity mixing under overlap applies to every section.** Identical linear
processing of every plane does not guarantee unbiased relative intensities. When
the processed lineshape differs from the fitted one, part of each peak's intensity
is assigned to its overlapping partner. For an isolated peak the error is a common
factor that cancels in ratios
([profile-uncertainty §5.2](profile-uncertainty.md#52-the-residual-excess)). For
overlapping peaks whose profiles differ, their amplitude ratio changes from plane to
plane, so the mixing does too. In the known-truth study T7 (conditions under linear
prediction, below), an unmodelled Gaussian decay without LP biased the partner's
relative intensity by 1.0%
([profile-uncertainty §2.3](profile-uncertainty.md#23-which-component-causes-the-large-whitened-χ²)).

## Linear prediction (LP)

LP extends the acquired FID before the Fourier transform. The header then gives the
extended length as if it had been acquired.

**Conditions of the known-truth study T7**
([profile-uncertainty §2.3](profile-uncertainty.md#23-which-component-causes-the-large-whitened-χ²),
[record](validation/profile-uncertainty/lp_known_truth.txt)):
- one processed axis, not a full pseudo-3D;
- 47 acquired points extended to 92 by NMRPipe's `LP -fb -ord 16`, then the bundled
  example's ¹⁵N processing;
- an isolated line, and a pair 3 points apart whose partners decay at rates 1 and 2
  across seven planes;
- SNR 50, 200 realizations, PeakFit's least-squares estimator.

**Relative-intensity bias.**
- **Overlapping pair, exact lineshape:** the partner's relative intensity was biased
  by −11.1% at its worst plane (0.56 SD), and its fitted rate by +6.2% (0.61 SD).
  With an unmodelled Gaussian decay: −4.8% (0.25 SD) and +3.7% (0.36 SD).
- **Isolated line:** no bias above 0.2 SD. LP's error was then a common factor that
  cancelled in the ratios.
- **These are measurements, not bounds.** T7 tested one SNR, one LP order and one
  separation. The cancellation for an isolated line, under these conditions, does
  not establish that LP is safe for isolated peaks in general.
- A likely cause, not tested: LP fits its filter to each plane, so what it
  propagates depends on that plane's signal-to-noise ratio.

**σ calibration.**
- In T7, χ² was 2.8× its expectation for the pair, and 1.15× for the isolated line.
- In an earlier synthetic series (`LP -fb -ord 8`; 8 peaks at SNR about 2–200; 6–12
  realizations), the scatter of fitted parameters exceeded the published errors
  1.5–2.3× for isolated peaks at high SNR, and up to 6.6× for an overlapping pair
  ([correlated-noise §2.5](correlated-noise.md#25-what-the-final-header-cannot-show)).
  LP also changed fitted ¹⁵N linewidths by +24% to +192% at low SNR or in overlap
  ([decisions, 2026-11-05](decisions.md#2026-11-05-nmrpipe-phase-sign-digital-filter-and-header-limits-correlated-noise)).
- On the bundled CEST example, whose ¹⁵N axis was extended by LP from about 47 to 92
  points, the baseline-replicate check gives observed / stated 1.07 (0.89–1.28) over
  164 peaks
  ([profile-uncertainty §5.1](profile-uncertainty.md#51-cest-baseline-planes-as-replicates)).
  This is one dataset, and replicates measure repeatability, not accuracy: a bias
  common to every plane is invisible to them. It is not evidence that LP is safe in
  general.

**Correlation between planes.** LP of one plane uses only that plane's data, so it
does not couple the noise of different planes. Within a plane, the noise correlation
along the extended axis is not the one the header predicts
([correlated-noise §2.5](correlated-noise.md#25-what-the-final-header-cannot-show)).

**The noise-power check** may detect an inconsistency associated with LP under the
tested conditions: NMRPipe's `LP -fb -ord 16` on white noise (32 to 64 points), and
the bundled example's ¹⁵N axis
([correlated-noise §2.6](correlated-noise.md#26-the-noise-power-check)).
- It is not a specific LP detector: an earlier window or `SP -size` give the same
  kind of warning.
- Its silence does not exclude LP. The drop in late noise power it relies on
  weakens as the LP order grows (the T7 record), and the check is inconclusive with
  fewer than 32 noise-only lines.

**Practical advice.** Where peaks overlap and their relative intensities matter,
process the same data without LP and compare the profiles or the downstream rates.

## NUS reconstructed plane by plane

Each plane is reconstructed on its own, and the header then looks like uniform
acquisition at the full size. The evidence calls for more caution here than for LP.

**What was tested**
([correlated-noise §2.5](correlated-noise.md#25-what-the-final-header-cannot-show),
[decisions, 2026-11-05](decisions.md#2026-11-05-nmrpipe-phase-sign-digital-filter-and-header-limits-correlated-noise)):
NMRPipe's IST at its default settings, reconstructing 50% Poisson-gap NUS plane by
plane, in synthetic series of 8 peaks at SNR about 2–200, 6–12 realizations. SMILE,
hmsIST and other reconstruction methods were not tested with PeakFit, and their
behaviour may differ.

**Relative-intensity bias.** Peak areas, and the relative amplitudes of isolated
peaks across planes, were preserved in these tests. Overlapping peaks were not
assessed. Reconstruction broadened the fitted lines by +158% to +190% without noise,
so the processed lineshape differs from the one PeakFit fits, and the intensity
mixing above applies.

**σ calibration.** Parameter scatter exceeded the published errors up to 2.8×.
IST extrapolation of a zero-filled NUS series (NUSZF) gave 1.5–3×, with lines
narrowed by 27–51%.

**Correlation between planes.** Reconstructing each plane on its own uses no other
plane's data, so it does not couple their noise. The reconstruction is nonlinear and
depends on each plane's data, so its errors need not be the same from plane to plane.
The noise-power check was not tested on reconstructed data.

**A complementary check.** Stetz and Wand (2016) acquire a few uniformly sampled
reference planes (the shortest and the two longest delays) alongside the NUS series.
Regressing each peak's NUS heights on the uniformly sampled ones gives a
"non-linearity factor", the absolute intercept, which they relate to the accuracy of
the fitted rates. They developed it with hmsIST, and also compared IRLS and NESTA
reconstructions. It assesses reconstruction nonlinearity on your own data; it says
nothing about PeakFit's σ.

## Joint pseudo-3D reconstruction

East, Delaglio and Lisi (2021) treat the series as 3D time-domain data, with the
delay as a third dimension whose imaginary part is zero, and reconstruct it with 3D
NUS methods (NMRPipe's IST, also hmsIST and SMILE).
- **Accurate rates under specific conditions:** planes sorted by increasing delay,
  no repeated delays, exponentially spaced delays to about twice the decay time,
  Poisson-gap schedules and 9–12 planes, at 25% sampling.
- **Systematic error:** some of the schemes tested overestimated the decay time by
  about 10% (T2, and similarly T1). The conditions above come from the search for
  schemes that avoid it.
- **Their uncertainties** came from Monte Carlo trials of the exponential fit. The
  article does not compare them with the observed scatter, so their calibration was
  not assessed.

PeakFit has not been tested on such data.

**Relative-intensity bias.** Not measured with PeakFit; the conditions above are the
published evidence. The intensity mixing above applies wherever the reconstructed
lineshape differs from the fitted one.

**σ calibration and correlation between planes.** The reconstruction couples the
planes, so it introduces correlation into the spectral noise between planes.
PeakFit's covariance between planes includes the correlation from shared fit
parameters, not this one. Its σ are therefore not calibrated after joint
reconstruction, and the size and sign of the error are unknown.

**Duplicate planes.** Duplicates reconstructed together share the reconstruction, so
comparing them is not a calibrated check of the σ; the method also asks for no
repeated delays. Independent acquisitions, each reconstructed separately, do give
information on repeatability.

Stetz and Wand's reference-plane validation was made for their reconstruction
setting. Nothing shows that it carries over to joint reconstruction.

## Other processing

### Measured miscalibration

These operations leave a header that passes every check. NMRPipe 13.0
was run on complex white noise and on synthetic pseudo-3D series that PeakFit then
fitted, and the amplitude SD was compared with the published error
([correlated-noise §2.5](correlated-noise.md#25-what-the-final-header-cannot-show)):

| Processing | Published amplitude error |
|---|---|
| An earlier window (`EM` then `SP`, `SP` then `SP`) | understated 6–16% |
| `SP -size`, `-start`, `-one` | understated up to 16% (`-size`), overstated up to 72% (`-start`) |
| Digital-filter correction during conversion (`bruk2pipe -DMX`) | understated up to 13% |
| Solvent filter (`SOL`) | calibrated beyond about 200 points from the carrier; nearer, understated up to 24% or overstated 11% |

- **Relative-intensity bias:** not measured. The `sp1` and `sp2` lineshapes model
  only the recorded window, so after an earlier or shortened window the fitted
  lineshape differs from the processed one, and the intensity mixing above applies.
  For `bruk2pipe -DMX` the lineshape's acquisition time is handled
  ([lineshapes](lineshapes.md#the-window-over-the-acquired-signal)).
- **Correlation between planes:** each operation acts on every plane separately, so
  none couples planes.
- **The noise-power check** detected an earlier window and `SP -size` in its tests;
  conversion-time correction and `SOL` were not among them.

### Predicted miscalibration

Windows the noise model does not cover (`EM`, `GM`, composite `SP -elb`, `-glb`,
`-goff`, `SP -df`) and input that is not NMRPipe, such as TopSpin-processed spectra,
leave the run "noise correlation not modelled", with the reason
([correlated-noise §2.3](correlated-noise.md#23-header-fields-and-consistency)).
- **Mechanism:** the errors then fall back to independent spectral points. Where zero
  filling and apodization correlate neighbouring points, independent-point errors
  are too small, by a factor that depends on the window, the zero fill and the
  linewidth ([correlated-noise §3](correlated-noise.md#3-size-of-the-effect) gives it
  for sine bells). It was not measured for these windows or for TopSpin, so no size
  is attributed to them.
- **Lineshape:** other windows are fitted with the Gaussian, Lorentzian or
  pseudo-Voigt lineshapes ([lineshapes](lineshapes.md)), so the intensity mixing
  above applies to overlapping peaks.
- **The noise-power check** runs only on modelled axes, so it gives no warning here.
- **Correlation between planes:** not assessed.

### Untested: baseline correction

Baseline correction is outside the noise model, and its effect on relative
intensities, σ and correlation between planes is untested. Data-adaptive variants,
such as automatic baseline correction, fit each plane's own data, so they need not
act identically on every plane. Neither safety nor harm is established. The
intensity mixing above applies as everywhere.

## Interpreting the empirical checks

`peakfit replicates <run>` compares a published run's errors with the scatter of
its own data. It reads the run without refitting, changes no value or error, and
names the method that ran.

**Duplicate planes** (any series; the default when the run has them). Planes whose
recorded plane values are exactly equal are copies. For each peak, the differences
between copies are standardized by the uncertainty the published covariance
predicts for them, which accounts for the correlation from shared lineshape
parameters
([profile-uncertainty §5.3](profile-uncertainty.md#53-duplicate-planes)). The
result is an RMS of standardized contrasts, reported separately for peaks without
and with the correlated-profile flag.
- **About 1:** the differences between copies scatter as the published covariance
  predicts. In synthetic series with the validated processing, the check gives 1
  within its sampling error
  ([profile-uncertainty §8.2](profile-uncertainty.md#82-t8-the-duplicate-plane-check)).
- **Clearly above 1:** the copies differ more than the errors allow. Possible
  causes: errors that are too small for the processing used, copies that are not
  equivalent (drift, temperature, a change between acquisitions), or noise
  anticorrelated between copies.
- **Clearly below 1:** the copies agree better than the errors allow: errors that
  are too large, or noise positively correlated between copies, as after a
  reconstruction or processing that couples them. In the synthetic study, a
  correlation of +0.3 between copies gave an RMS of 0.83, and −0.3 gave 1.14.
- **How clearly:** with n contrasts the sampling SE of the RMS is at least
  1/√(2n); the output warns when that exceeds 0.1 (fewer than 50 contrasts).
  Contrasts from peaks of one cluster are correlated, so the real SE is larger.
- **No duplicate planes:** the output says so. Planes with nearly equal values are
  not paired.

**CEST baseline planes** (`--baseline`, or when the run has no duplicates). Planes
far from a peak's own offset are approximate replicates. The robust scatter of
their second differences is compared with the published profile σ, as the median
of observed / stated
([profile-uncertainty §5.1](profile-uncertainty.md#51-cest-baseline-planes-as-replicates)).
Drift, residual saturation and minor-state dips can only raise it.

**What neither check shows.**
- **Repeatability, not accuracy.** An error common to every copy, or to every
  baseline plane, is invisible: a lineshape misfit, or the bias LP caused for
  overlapping peaks in T7. A result of 1 is not evidence against bias.
- **Contrasts, not marginal σ.** A scale uncertainty shared by the copies cancels in
  their differences, so the duplicate check says nothing about the marginal σ of
  absolute amplitudes.
- **Raw amplitudes, not normalized profiles.** Both checks read amplitudes; the
  ratios and R2eff derived from them are not checked directly.
- **Equal plane values do not prove independent, equivalent acquisitions.** PeakFit
  does not know how the copies were acquired, nor in which order, so it reports no
  signed or drift statistic. Check this yourself.
- **Copies reconstructed jointly** (joint pseudo-3D NUS) share the reconstruction,
  so the check is not calibrated for them. Copies acquired independently and
  reconstructed separately give repeatability information.

**Alongside ChemEx's `duplicates` error option.** With `error = "duplicates"`,
ChemEx replaces the errors in the file with one σ per profile estimated from the
duplicates; with the default `global_error = true`, that σ is shared by every
profile of the experiment
([profile-uncertainty §6a](profile-uncertainty.md#what-chemex-does-today)).
- Both use the same duplicates, so neither is independent evidence for the other,
  and both measure repeatability only.
- If PeakFit's duplicate check is close to 1, its profile σ (`error = "file"`)
  agree with the duplicates on average, and they keep what one shared σ discards:
  the larger errors of overlapping or weaker peaks.
- If it is clearly above 1, the profile σ understate the scatter between copies,
  and the processing sections above may say why. ChemEx's estimate follows the
  observed scatter, but as one uniform σ, estimated from few copies per profile when
  `global_error` is off.
- The RMS pools many peaks and does not say which ones are off. PeakFit does not
  rescale any error by it.

## References

- K. W. East, F. Delaglio and G. P. Lisi (2021). A simple approach for
  reconstruction of non-uniformly sampled pseudo-3D NMR data for accurate
  measurement of spin relaxation parameters. *Journal of Biomolecular NMR* 75,
  213–219. doi:10.1007/s10858-021-00369-7
- M. A. Stetz and A. J. Wand (2016). Accurate determination of rates from
  non-uniformly sampled relaxation data. *Journal of Biomolecular NMR* 65, 157–170.
  doi:10.1007/s10858-016-0046-9
