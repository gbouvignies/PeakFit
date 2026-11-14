# PeakFit

PeakFit fits lineshape models to peaks in pseudo-ND NMR spectra to estimate
lineshape parameters and plane-wise amplitudes.

## Spectra and peaks

**Pseudo-ND spectrum**:
A series of NMR planes with shared spectral axes and an experimental series
dimension rather than another conventional frequency axis.

**Series dimension**:
The non-spectral dimension that orders the planes in a pseudo-ND spectrum.
_Avoid_: Z-dimension

**Plane**:
One spectrum in a pseudo-ND series, identified by its index and plane value.

**Plane value**:
The experiment-specific coordinate associated with a plane. Its physical meaning
depends on the experiment and input configuration. The GUI labels it "Plane
value" (or "Plane number" without values) unless the user supplies a display name
and unit; PeakFit never infers either.
_Avoid_: Z-value when a more specific experiment-defined name is known

**Spectral axis**:
A frequency dimension of each plane on which peak positions and lineshapes are
defined.

**Peak**:
An NMR peak with a name, a position, and a lineshape on each spectral axis.

**Peak list**:
The set of named peaks supplied as prior positions for a fit.

**Peak cluster**:
A group of peaks assigned to one contour-connected spectral segment and fitted
together. Its data matrix has canonical shape `(n_points, n_series)`.
_Avoid_: Treating every peak as an independent fit

**Clustering contour**:
The signal threshold used to segment the spectral axes into peak clusters.

## Models and fitting

**Lineshape**:
A mathematical model of a peak's shape across the spectral axes.

**Chemical shift**:
A peak position along a spectral axis, expressed in parts per million (ppm).

**Linewidth**:
The full width at half maximum (FWHM), in hertz, for one peak axis. For an
apodized lineshape, it is the FWHM the model would have without apodization.

**Fitting parameter**:
A named model quantity that may be fixed, constrained, varied, or computed
during a fit.

**Parameter constraint**:
A bound or fixed/vary rule applied to selected fitting parameters.

**Amplitude**:
The fitted contribution of one peak in one plane.

**Intensity profile**:
The sequence of fitted amplitudes for a peak across plane values.
_Interpretation note_: PeakFit output calls these values intensities; no broader
physical equivalence between fitted amplitude and experimental intensity is
established by the repository.

**Fit**:
The estimation of lineshape parameters and plane-wise amplitudes for one or more
peak clusters.

**Fit step**:
A named stage that specifies which fitting parameters are fixed or varied and
how many iterations run.

**Refinement iteration**:
One optimizer pass in the cross-talk refinement schedule. With
`refine_iterations = N`, PeakFit performs exactly `N` passes and updates
cross-talk corrections only between passes.

**Cross-talk correction**:
The estimated contribution from peaks outside a peak cluster that is subtracted
from that cluster's data before fitting.

**Fit run**:
One completed fitting workflow and its associated results.

**Near-collinear fitted shapes**:
Two peaks whose recorded fitted shapes over a cluster's fitting samples satisfy
1 − |cos| ≤ 10⁻⁴. Their per-plane amplitude split is ill-conditioned, so their
individual amplitudes and the positions that produced them are poorly
constrained; it says nothing about how many resonances are present.

**Identifiable amplitude**:
A member's per-plane amplitude that the recorded fitted shapes determine: its unit
coefficient vector has no component in the null space of the shape matrix, under
the numerical rank of the factorization that solved the amplitudes. A member can be
identifiable inside a rank-deficient cluster. An identifiable amplitude whose value
depends on an ill-conditioned direction, or whose shape is near-collinear with
another member's, is *not established* rather than unidentifiable. The property is
conditional on the recorded fitted lineshapes; it does not establish identifiability
when nonlinear lineshape parameters vary.
_Avoid_: calling a near-collinear or ill-conditioned amplitude "unidentifiable"

**Continued start**:
A peak's starting nonlinear values taken from the active run's fitted state,
because its working position is still the one that run adopted. Other peaks start
from the configured defaults; each run records which peaks continued and from which
run.

**Final fit outcome**:
The immutable authoritative scientific result assembled once every terminal
cluster optimizer result has been classified under the frozen final correction
revision. It is distinct from mutable fitting continuation state.

**Final cluster outcome**:
The immutable terminal outcome for one `cluster_id`: converged, usable
non-converged, or unusable. Unusable outcomes retain only identity,
classification, reason, correction revision, and optimizer provenance.

**Numerical usability**:
Whether the shared analytical evaluation can supply finite amplitudes, model,
residuals, statistics, and uncertainty inputs. It is independent of optimizer
convergence.

**Recorded optimizer residual**:
The residual of one peak cluster on its fitting samples after subtracting the
terminal cross-talk corrections and the cluster model, as stored in the final fit
outcome. Reduced chi-squared derives from it.
_Avoid_: Calling it the spectrum residual

**Largest recorded residual**:
The maximum absolute recorded optimizer residual of one cluster, in units of the
statistical noise, over its corrected fitting samples. Computed once at
finalization and retained at every evidence level.
_Avoid_: Deriving it from the whole-spectrum residual

**Whole-spectrum residual**:
The difference between the experimental plane and the sum of all usable cluster
models at every native point, divided by the statistical noise. It includes points
outside fitting samples and is used for display and to locate unexplained signal.

**Numerical warning**:
An objective, engine-recorded condition of a cluster outcome: unusable, not
converged, unreliable uncertainty, or a varied parameter at a bound.

**At bound**:
The review warning for a varied parameter whose distance from its nearest bound is
at most 10⁻⁶ of its bound interval (`engine.diagnostics.bounds.at_bound`). Fixed
parameters, and parameters without two finite bounds, are never at bound. It does
not decide whether uncertainties are withheld; see active bound.

**Active bound**:
A bound within an absolute 10⁻⁷ of a varied value at the VarPro optimum; its
symmetric standard error is withheld. A scientific uncertainty decision, separate
from the At bound review warning.

**Review measure**:
A descriptive number about a completed cluster fit, such as reduced chi-squared,
the largest recorded residual or the amplitude correlation, shown with its
definition and never combined into a quality score.

**Review heuristic**:
A descriptive highlight with an adjustable display threshold, such as a linewidth
outlier. It is never a numerical warning.

**Highlight**:
The subdued mark (◇) raised by a review heuristic or by a display threshold on a
review measure, labelled with its rule and value. It changes no results, warnings
or run state.

**Amplitude correlation**:
The correlation between cluster members' amplitudes derived from the recorded
amplitude covariance. It is conditional on the fitted lineshape parameters, and
follows the sandwich covariance where the noise correlation is modelled.

**Noise correlation**:
The correlation R between the noise of neighbouring spectral points that
apodization, zero filling and the Fourier transform create from white
time-domain noise. Cov = σ²R: σ is the noise level, and R has a unit diagonal.

**Noise model**:
A run's record of the noise correlation its published errors assume: either
"modelled", from the processing its NMRPipe header records, or "not modelled",
with the reason, in which case the errors assume independent points. It changes
errors only, never fitted values.

**Noise-power check**:
A run's per-axis comparison of the time-domain noise power of the spectrum's
noise-only lines with the power the recorded processing predicts. A mismatch
reveals processing the header does not record, such as linear prediction or an
earlier window. It is recorded with the noise model as a warning only: it changes
no value and no error.

**Late power**:
The noise-power check's measure: the noise power of the late FID relative to the
early FID, measured over predicted; 1 when the processing is as recorded.

**Sandwich error**:
A standard error from the covariance of the least-squares estimates under
correlated noise, (JᵀJ)⁻¹JᵀRJ(JᵀJ)⁻¹ scaled by σ². It is first order: exact in the
locally linear regime.

**Conditional amplitude error**:
An amplitude's standard error with the fitted lineshape parameters held fixed, or
**conditional σ** for short; published as `intensity_err_conditional`. Planes
share no lineshape term, so it gives no covariance between planes.

**Total amplitude uncertainty**:
An amplitude's uncertainty including the variability of the lineshape
parameters. Its first-order analytical form is the marginal σ; MCMC estimates it
under an independent-noise likelihood.

**Marginal σ**:
The first-order total uncertainty of one absolute amplitude, √Σ_ss, with Σ the
covariance between planes; published as `intensity_err_marginal`. It is the right
error for a single amplitude; used as independent per-plane errors it overstates
downstream errors, because its lineshape part is correlated across planes (`docs/profile-uncertainty.md` §3.3).

**Covariance between planes**:
The first-order covariance Σ of one peak's amplitudes over the planes, from the
sandwich over all parameters. Its lineshape part correlates the planes; reference
ratios and R2eff propagate it as gᵀΣg (`docs/profile-uncertainty.md` §4).

**Covariance factors**:
The per-cluster quantities that, with the fitted amplitudes, rebuild the first-order
covariance of the amplitudes between any two planes: the conditional amplitude
covariance, two lineshape-propagation tensors and the lineshape-parameter covariance
(`docs/profile-uncertainty.md` §3.1). Their size does not grow with the number of
planes.

**Relative-profile covariance**:
Σ_rel = cI + PLP for one peak, with c its conditional variance, L = Σ − cI the
lineshape part and P the projector orthogonal to its profile. It is exactly
equivalent to Σ for every downstream analysis that fits a free scale, equals cI
for an isolated peak, and has no scale component.
_Avoid_: common scale uncertainty (not uniquely defined, not reported)

**Profile σ**:
The per-plane error of an intensity profile, √(Σ_rel)_ss, published as
`intensity_err`. It equals the conditional σ for an isolated peak, and is valid as
independent per-plane errors only where Σ_rel is close to diagonal.

**Baseline replicates**:
CEST planes far from a peak's own offset, references excluded, treated as
approximate replicates: the robust scatter of their second differences checks the
profile σ (`peakfit replicates --baseline`, and the default for a run without
duplicate planes). They measure repeatability, not accuracy.

**Duplicate planes**:
Planes of one run whose recorded plane values are exactly equal: copies of one
condition, in any series. `peakfit replicates` standardizes each peak's
orthonormal contrasts between copies by its conditional variance and pools them
into an RMS of standardized contrasts, 1 when the conditional σ is calibrated. The
lineshape-parameter part of the uncertainty cancels between copies, so they test
the conditional σ only. Like baseline replicates, they measure repeatability, not
accuracy; they check contrasts, not the marginal σ.
_Avoid_: replicate pairs (copies may be three or more), near-duplicates (no
tolerance matching)

**Standardized contrast**:
A contrast is a difference between copies of duplicate planes: each group of m
copies gives m − 1 orthonormal contrasts, k per peak over all groups. A peak's
contrasts d are standardized by its conditional variance c, giving q = |d|²/c,
whose expected value is k at first order. The RMS of standardized contrasts,
√(Σq / Σk) over a pool of peaks, is 1 when the conditional σ is calibrated; its
sampling SE is at least 1/√(2Σk).

**Correlated-profile flag**:
A per-peak mark that the largest correlated mode of the relative-profile
covariance exceeds τ·c (`CORRELATED_MODE_THRESHOLD`, 0.1), or that its within-plane
amplitude correlation with a cluster partner exceeds `PARTNER_CORRELATION_THRESHOLD`
(0.15) in magnitude, both set by validation T3: independent per-plane errors then
cannot describe the profile's uncertainty. It states a limit of that approximation,
not the quality of the data; published as `correlated_profile`.

**Reference-normalized profile**:
A display of an intensity profile divided by the mean amplitude of explicitly
chosen reference planes, with propagated uncertainty. It never replaces the
recorded amplitudes.

**Cluster focus**:
The spectral view reframed in place on one recorded peak cluster: its displayed
members ± max(3 fitted FWHM, 8 native samples), with residual lanes. Leaving it
restores the overview's view exactly and never reverts edits or fits.

**Native slice**:
The row or column of one plane through the sample, drawn as a line through its
native samples, with the total model at the same samples. It shares its frequency
axis with the 2D plot.

**Region intensity scale**:
The shared intensity range of both slices: min(0, lowest) to max(0, highest) of the
data over the visible 2D region and every plane, padded by 5 % of the span. It
changes only on a deliberate view change; a manual scale replaces it until reset.

**Residual lane**:
A strip under a native slice showing the whole-spectrum residual
`(D − M) / σ_stat` at the same samples, on one shared ±N σ_stat scale.

**MCMC uncertainty analysis**:
Post-fit sampling used to estimate parameter uncertainty for fitted peak
clusters. It is not a fitting optimizer. Its formulation is in
`docs/mcmc-formulation.md`.

**MCMC record**:
The immutable file one MCMC run writes under its fit's `mcmc/` folder: bound to that
parent fit, with each cluster's whole chain and complete summaries, readable
without the parent or its spectrum.

**Profile likelihood**:
The likelihood of a cluster's nonlinear parameters with the amplitudes at their
conditional optimum: exp(−½χ²) of the VarPro residual. MCMC samples it; it is the
integrated likelihood under the conditional Jeffreys amplitude prior.

**Conditional amplitude draw**:
An amplitude sample drawn, for one sample of the nonlinear parameters, from the
amplitudes' Gaussian distribution given those parameters. MCMC amplitude intervals
come from these draws and so include lineshape uncertainty.
