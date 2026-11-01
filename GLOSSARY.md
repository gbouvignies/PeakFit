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

**Whole-spectrum residual**:
The difference between the experimental plane and the sum of all usable cluster
models at every native point, divided by the statistical noise. It includes points
outside fitting samples and is used for display and to locate unexplained signal.

**Numerical warning**:
An objective, engine-recorded condition of a cluster outcome: unusable, not
converged, unreliable uncertainty, or a varied parameter at a bound.

**Review measure**:
A descriptive number about a completed cluster fit, such as reduced chi-squared,
the largest recorded residual or the amplitude correlation, shown with its
definition and never combined into a quality score.

**Review heuristic**:
A descriptive highlight with an adjustable display threshold, such as a large
position shift or a linewidth outlier. It is never a numerical warning.

**Highlight**:
The subdued mark (◇) raised by a review heuristic or by a display threshold on a
review measure, labelled with its rule and value. It changes no results, warnings
or run state.

**Amplitude correlation**:
The correlation between cluster members' amplitudes derived from the recorded
amplitude covariance. It is conditional on the fitted lineshape parameters.

**Reference-normalized profile**:
A display of an intensity profile divided by the mean amplitude of explicitly
chosen reference planes, with propagated uncertainty. It never replaces the
recorded amplitudes.

**Cluster lens**:
The GUI view that juxtaposes experimental, overlay and whole-spectrum residual
panels of one peak cluster with its profiles and parameters.

**Automatic peak picking**:
Experimental peak discovery from residual signal when no peak list is supplied.

**MCMC uncertainty analysis**:
Post-fit sampling used to estimate parameter uncertainty for fitted peak
clusters. It is not a fitting optimizer.
