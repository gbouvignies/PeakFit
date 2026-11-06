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
another member's, is *not established* rather than unidentifiable.
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

**Sandwich error**:
A standard error from the covariance of the least-squares estimates under
correlated noise, (JᵀJ)⁻¹JᵀRJ(JᵀJ)⁻¹ scaled by σ². It is first order: exact in the
locally linear regime.

**Conditional amplitude error**:
An amplitude's standard error with the fitted lineshape parameters held fixed:
the published `intensity_err`.

**Total amplitude uncertainty**:
An amplitude's uncertainty including the variability of the lineshape
parameters. PeakFit does not publish it analytically; MCMC estimates it under an
independent-noise likelihood.

**Reference-normalized profile**:
A display of an intensity profile divided by the mean amplitude of explicitly
chosen reference planes, with propagated uncertainty. It never replaces the
recorded amplitudes.

**Cluster lens**:
The GUI view that juxtaposes experimental, overlay and whole-spectrum residual
panels of one peak cluster with its profiles and parameters.

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
