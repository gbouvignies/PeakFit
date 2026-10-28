# Numerical scaling: primary-source notes

These notes support the linewidth-initialization investigation. They do not
recommend a production change. The inspected environment used SciPy 1.17.0.

## Initialization and acquisition handling in the source

The model `param_specs` functions supply starts and bounds, identically for
singlets and corresponding doublets:

| Models | Linewidth start | Linewidth bounds | Additional shape parameters |
| --- | ---: | ---: | --- |
| Lorentzian, Gaussian | 25 Hz | 0.1–200 Hz | None |
| Pseudo-Voigt | 25 Hz | 0.1–200 Hz | `eta=0.5`, bounds −1 to 1 |
| NoApod, SP1, SP2 | 20 Hz | 0.1–200 Hz | Phase 0°, bounds −15° to 15° |
| All doublet variants | As above | As above | `j=5 Hz`, bounds 1–10 Hz |

Sources: [Lorentzian](../../src/peakfit/engine/lineshapes/lorentzian/model.py),
[Gaussian](../../src/peakfit/engine/lineshapes/gaussian/model.py),
[Pseudo-Voigt](../../src/peakfit/engine/lineshapes/pvoigt/model.py),
[NoApod](../../src/peakfit/engine/lineshapes/no_apod/model.py),
[SP1](../../src/peakfit/engine/lineshapes/sp1/model.py), and
[SP2](../../src/peakfit/engine/lineshapes/sp2/model.py).

Positions start at supplied peak-list centers in ppm. The model position
half-window is 50 Hz converted to ppm using that dimension's spectrometer
frequency, with a 0.1 ppm fallback without a grid
([estimate_cs_bounds_ppm](../../src/peakfit/engine/lineshapes/utils.py)).
Phase parameters are only created for configured `fit_phase` axes and are
shared by cluster and axis, rather than distinct per-peak values
([ShapeBase](../../src/peakfit/engine/lineshapes/base.py)).

SP2 is acquisition aware: `make_state` uses acquisition time and the two
dimension-specific sine-bell settings. Its kernel uses
`z = aq * pi * (lw + 2j * dw_hz)`, so intrinsic exponential decay rate is
`pi * lw` in inverse seconds. The model is the finite-acquisition,
squared-sine-apodized shape, with optional phase rotation; its fitted `lw` is
the **underlying unapodized Lorentzian FWHM**, not the observed processed
FWHM ([SP2 kernel](../../src/peakfit/engine/lineshapes/sp2/kernel.py)).
For Fourier-transformed spectral dimensions the reader obtains acquisition
time as NMRPipe `APOD / SW` and retains `APODQ1`, `APODQ2`, `APODQ3`, and
`APODCODE`, mapping storage dimensions through `FDDIMORDER`
([spectrum reader](../../src/peakfit/io/readers/spectrum.py)).
The phased templates cache state from that dimension's metadata; an explicit
`aq` constructor override is also possible
([templates](../../src/peakfit/engine/lineshapes/templates.py)).
F2 is the first spectral axis after the pseudo axis; F3 is the direct axis.
Gaussian/Pseudo-Voigt phenomenological widths should not automatically be
interpreted as intrinsic SP2 widths.

Routine fitting builds `Parameters.from_peaks`, applies configured constraints,
then step fix/vary flags
([pipeline](../../src/peakfit/fit/pipeline.py)). Constraint resolution starts
from the model value, applies matching default patterns, position windows,
and per-peak constraints; file values are used when the resolved value still
equals the model/current value. Values are clamped to resolved bounds;
explicit `vary` settings are respected
([constraints](../../src/peakfit/engine/domain/constraints.py)). Any future
initialization strategy should operate within these existing preparation
functions and preserve explicit user starts, fixed values, and bounds.

Automatic peak picking already provides a small reusable idea:
`initialize_new_peak_from_median` groups previous fitted parameters by
`(axis, label)`, excludes positions and amplitudes, and clips the median to
the new parameter's bounds. `initialize_existing_params_from_previous`
warm-starts existing parameters, filtering nonfinite values. The median
helper itself does **not** filter nonfinite previous values or check
identifiability/fit quality, so it is not directly a robust experimental
linewidth estimator
([auto-pick parameters](../../src/peakfit/auto_pick/parameters.py)).

Auto-pick's trial loop also ties linewidths and scalar couplings within each
axis, fits amplitudes with nonlinear parameters fixed, releases non-position
parameters, then releases positions within linewidth-based windows. These
staging/tying rules differ from routine fitting and must not be imported into
the controlled comparison
([_fit_trial_state](../../src/peakfit/auto_pick/algorithm.py)). Reuse the
per-axis median concept and the production VarPro/model evaluation, rather
than creating a second optimizer or copying auto-pick's scientific constraints.

The analytical Jacobian uses physical parameter units: derivatives with
respect to `cs` include the negative Hz-per-ppm chain rule, `lw` derivatives
include `pi * aq`, and phase derivatives include `pi / 180`. Residual and
Jacobian rows are divided by noise, so their columns have units inverse ppm,
inverse Hz, and inverse degree respectively
([templates](../../src/peakfit/engine/lineshapes/templates.py),
[phase derivatives](../../src/peakfit/engine/lineshapes/utils.py),
[VarPro](../../src/peakfit/engine/algorithms/varpro.py)).

## Nonlinear parameter scaling

PeakFit's `fit_cluster` supplies an analytical Jacobian, bounds, `method="trf"`,
and tolerances to `least_squares`, without `x_scale`. In SciPy 1.17 that means
unit scaling. The documented interpretation is a characteristic change in each
parameter: scaled variables are `x / x_scale`, and the trust region changes
accordingly. This leaves the objective and its physical units intact. The
`"jac"` option derives scales from inverse Jacobian-column norms.
See [SciPy 1.17 least_squares documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.least_squares.html),
and [PeakFit's call](../../src/peakfit/engine/algorithms/varpro.py).

The precise implementation matters: `compute_jac_scale` substitutes one for a
zero initial norm; subsequently it takes the maximum of the current column norm
and the previous inverse scale. Thus characteristic scales cannot grow as
optimization progresses. This history makes `"jac"` more than independently
normalizing the current Jacobian at every iteration. See
[SciPy 1.17 common.py, compute_jac_scale](https://github.com/scipy/scipy/blob/v1.17.0/scipy/optimize/_lsq/common.py#L550-L561).

**Inference:** column equilibration can improve unit imbalance, but cannot make
linearly dependent columns independent. Nearly cancelling amplitudes can create
large derivative norms and shrink subsequent characteristic steps; experiments
must measure this behavior rather than presume improvement. Compare singular
values of column-normalized Jacobians as well as raw condition numbers, because
raw conditioning depends on parameter units.

SciPy's bounded TRF implementation combines user scaling with distances to
bounds. It initializes the trust radius using `x0`, computes the dense step via
an SVD in transformed coordinates, and checks step termination in original
coordinates. It reports bound-aware gradient optimality. See
[SciPy 1.17 trf.py](https://github.com/scipy/scipy/blob/v1.17.0/scipy/optimize/_lsq/trf.py).
**Inference:** keeping physical chemical shifts unchanged avoids adding a
reparameterization confound, but an `x_scale` choice independent of absolute
chemical-shift values does not make the whole TRF algorithm translation
invariant. Compare termination reason and actual objective alongside optimality.

A dimensional experimental scale can use a characteristic frequency change
`delta_nu` in Hz, converted to shift units by `delta_nu / spectrometer_MHz`;
a linewidth scale in Hz; and a phase scale in degrees. Choose these before
examining the problematic groups. The scale is a step size, not a prior,
constraint, or uncertainty estimate. Compute final covariance using the
Jacobian in original physical coordinates, as production does.

## Intensity normalization: exact-arithmetic derivation

Write the masked spectrum as a point-by-plane matrix `Y`, shape columns as
`Phi(theta)`, and least-squares amplitudes as `A = Phi^+ Y`. PeakFit evaluates
the actual model and standardized residual
`r = (Y - Phi A) / sigma`; both residual and analytical Jacobian divide by the
same scalar noise. See
[VarPro evaluation](../../src/peakfit/engine/algorithms/varpro.py) and
[rank-aware amplitude solver](../../src/peakfit/engine/algorithms/linear_algebra.py).

For a positive common factor `c`, replacing `(Y, sigma)` by `(Y/c, sigma/c)`
gives `A/c`, the same `r`, the same nonlinear Jacobian, and therefore the same
chi-square and nonlinear covariance. Original amplitudes and their conditional
errors are recovered by multiplying by `c`. This is an algebraic consequence
of the existing objective, not an extra statistical procedure.

Finite precision may change optimization trajectories near poorly identified
solutions even when fixed-parameter residuals agree. Test invariance first at
identical parameters, then compare optimized models and objectives. A very
small parameter difference is not a universal valid requirement when the
decomposition is unidentifiable. Scaling only data, or scaling planes without
their corresponding noise, would change the objective. Independent plane
normalization is outside this experiment.

## Optional shape-column normalization: limitations

For a positive diagonal matrix `D` of column norms, use
`Phi_scaled = Phi D^-1`, solve for `B`, and reconstruct `A = D^-1 B`.
This preserves the attainable physical model space exactly. For full column
rank, it also preserves the unique least-squares amplitudes; transform a
conditional covariance as `Cov(A) = D^-1 Cov(B) D^-1`.

At rank loss the fitted model remains unique, but coefficients do not:
minimizing the Euclidean norm of `B` generally selects a different member of
the solution family than minimizing the norm of `A`. Finite-precision numerical
rank can also change because PeakFit's cutoff is relative to the largest
singular value. Consequently a shape-normalization experiment must explicitly
record rank, actual-model residuals, and amplitude reconstruction; identical
minimum-norm amplitudes cannot be promised at rank deficiency. These statements
follow directly from the change of variables and the implementation of
[factorize_shapes](../../src/peakfit/engine/algorithms/linear_algebra.py).

If normalization depends on `theta`, its derivative must be included when
forming derivatives of normalized shapes, or an equivalent calculation using
the original physical shapes and amplitudes must be retained. Merely scaling
the linear solve and reusing an inconsistent pseudoinverse in the analytical
VarPro Jacobian is insufficient. This additional complexity argues for keeping
column normalization secondary unless measurements show a clear benefit.
