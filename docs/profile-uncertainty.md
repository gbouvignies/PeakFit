# Estimator and uncertainty of intensity profiles

**Status:** accepted, revision 3 ([ADR 0003](adr/0003-ols-and-profile-uncertainty.md)).
Implemented, not yet released: every ticket of the implementation (§9) is done. The
covariance factors of §3.1 are computed in the engine
(`engine.algorithms.covariance_factors`, validated by T1) and stored by archive
12.0.0 and project 15.0.0. `intensity_err` is the profile σ, with the marginal and
conditional σ in their own columns and the correlated-profile flag beside them (§6).
Ratios and R2eff (GUI ratio mode, `peakfit plot cest|cpmg`) propagate each peak's
covariance between planes through `profiles.normalize` (§4). T3 passes under
criterion (b) of §10 item 5 (§8.1) and sets the correlated-profile thresholds. The
duplicate-plane check of `peakfit replicates` (§5.3) passes its validation, T8 (§8.2).

**Builds on:** [correlated-noise.md](correlated-noise.md) and
[ADR 0002](adr/0002-processing-derived-sandwich-errors.md).

**Evidence:** exploratory scripts with fixed seeds, in
[`tools/profile_uncertainty/`](../tools/profile_uncertainty/README.md). Their outputs
are recorded in [`validation/profile-uncertainty/`](validation/profile-uncertainty/),
and every number below comes from them. The README maps each script to its section.
The implementation's validation tools (T3–T8) supersede them.

This specification answers three separate questions:
1. Which estimator suits realistically processed spectra (§2)?
2. How should uncertainty propagate into relative intensities and the physical
   parameters derived from them (§3–§4)?
3. How well does the result describe real experimental variability (§5)?

**The scientific target is the relative intensity profile.** CEST, CPMG, T1 and T2
models fit a free overall scale per peak, so only the uncertainty that varies from
plane to plane matters downstream.

## 1. Summary

1. **The example's ¹⁵N axis was linearly predicted, invisibly to the header.**
   - Its time-domain noise power drops at t ≈ 46–48 of the 92 points the header
     records as acquired. The data fit about 47 acquired points, extended by linear
     prediction (LP).
   - The ¹H axis matches its processing model.
   - So R was wrong on ¹⁵N (§2.2).
2. **Correcting R in noise-only regions does not rescue GLS.**
   - With the measured ¹⁵N correlation, GLS's whitened χ²ᵣ rises from 10.9 to 18.4.
   - The failure comes from the late, LP-extended part of the FID, which GLS
     up-weights. There, neither the signal model nor the noise covariance near peaks
     is known.
   - A known-truth test (T7) separates the causes: signal-dependent LP noise
     reproduces the failure with an exact lineshape and an exact extrapolation
     (§2.3).
3. **Real replicates favour OLS.** On CEST baseline planes:
   - OLS gives the smallest plane-to-plane scatter of relative intensities.
     Shrinkage gives 1.07–1.34× that scatter, and GLS 1.30–1.95×; for overlapping
     peaks GLS reaches 1.9–2.7×.
   - The OLS conditional sandwich σ matches the observed scatter: empirical / stated
     1.06 under the header R, 1.01 under the measured R (§5).
4. **For an isolated peak, the lineshape uncertainty is exactly a common scale
   error.**
   - Every lineshape term of the covariance is proportional to a_s a_t (§3.2), so it
     cancels in relative intensities.
   - Adding it to per-plane σ makes a downstream relaxation fit overstate its rate
     error 1.08–1.47× (§4).
5. **Only overlap leaves a lineshape uncertainty that changes the profile's shape:**
   intensity moves between peaks whose profiles differ. It is correlated across
   planes, so a per-plane σ cannot show it.
   - On the example, its largest correlated mode exceeds 10% of the per-plane noise
     variance for 12 of 74 overlapping peaks, and exceeds the full variance for 2.
   - The per-plane diagonal hides this: it rises by at most 1.23×.
6. **Three distinct quantities, defined in §3.3:**
   - the *marginal σ* of an absolute amplitude;
   - the *relative-profile covariance*, exact for any downstream fit with a free
     scale;
   - the *profile σ*, its diagonal. The profile σ suffices on its own only where a
     criterion on the correlated modes says so.

   A "common scale uncertainty" has no unique definition and is not reported.
7. **The residual excess (2.2×) is mostly misfit proportional to the amplitude** and
   harmless to relative intensities. Inflating the errors is not justified (§5.2).

## 2. Estimator

### 2.1 Theory

- **Information:** for white FID noise, a spectrum apodized by w(t), zero filled at
  least 2× and kept real still holds the FID's information (Bartholdi and Ernst 1973).
- **The two estimators over the full spectrum:**
  - OLS on the spectrum is the FID fit with weights w(t)².
  - GLS under R⁻¹ removes them, which is maximum likelihood when the model and R are
    exact (Aitken 1935).
- **GLS gains on the late-FID components the apodization suppressed.** Model
  inadequacy concentrates in exactly those components:
  - non-exponential decay;
  - LP-extrapolated signal;
  - truncation and the sidelobes of peaks outside the cluster, which a local fit
    cannot model.
- **So GLS undoes the localization that makes local cluster fitting work.** It weights
  the data as if they were unapodized, and so inherits the drawbacks of unapodized
  fitting (truncation and outside sidelobes) without their model.

**Synthetic, exact model and exact R** (one axis, sine-bell² offset 0.4, 92 → 256
points, SNR 30, 2,000 realizations):

| Case | GLS SD / OLS SD |
|---|---|
| Isolated line (amplitude · position · R2) | 0.89 · 0.69 · 0.66 (narrow); 0.95 · 0.95 · 0.92 (broad) |
| Pair 3 points apart (amplitude · position) | 0.50 · 0.35 |
| Unmodelled Gaussian decay (amplitude bias / SD) | OLS 0.64; GLS 1.22 |

Every estimator is calibrated under its own sandwich when the model is exact. With 5×
zero fill, R on a 25-point window has numerical rank 18 of 25.

### 2.2 Linear prediction on the example

The ¹⁵N axis was tested on 25,849 noise-only columns (every point < 4.5σ). The FFT
of a processed axis is its time-domain noise, so its power profile is checked against
the header's prediction. Both are normalized over the first 40 points:

| t (points) | 4–45 | 46 | 47 | 48 | 56 | 64 | 80 |
|---|---|---|---|---|---|---|---|
| Measured / predicted power | 0.94–1.02 | 0.74 | 0.51 | 0.44 | 0.21 | 0.14 | 0.27 |

- **First point:** at t = 0 the ratio is 1.22. That single point is the first-point
  scaling's, and it plays no part in the LP diagnosis.
- **Best fit:** about 47 acquired points, extended to 92. The mismatch is 0.055
  against 0.132 for the header's 92.
- **Lag correlations, measured against predicted:**
  - ¹⁵N: 0.811 · 0.384 · 0.002 against 0.751 · 0.263 · −0.050;
  - ¹H, on 3,147 noise-only rows: 0.650 · 0.064 · −0.204 against
    0.656 · 0.043 · −0.233.
- **This revises `correlated-noise.md` §2.4:** with a strict noise-only selection, the
  ¹H model agrees. The ¹⁵N discrepancy is LP.
- **A diagnostic for any spectrum:** the time-domain noise-power profile detects LP
  and other unrecorded processing from the data alone (§6).

### 2.3 Which component causes the large whitened χ²?

| | Header R | Measured ¹⁵N R |
|---|---|---|
| OLS χ² / E[χ² ∣ R] | 2.22 (1.67–3.51) | 2.26 (1.71–3.58) |
| Same, after removing one amplitude-proportional misfit shape per peak | 1.61 (1.48–1.80) | 1.64 (1.51–1.84) |
| Shrinkage α = 0.1, whitened χ²ᵣ | 1.38 | 1.81 |
| GLS, whitened χ²ᵣ | 10.9 (5.2–206) | 18.4 (9.1–163) |

Values are medians, with the 10th–90th percentile in parentheses.

What these data establish:
- **Numerics are not the cause.** Shrinkage bounds the conditioning, yet it still
  loses to OLS on real replicates (§5).
- **Noise-only covariance is not the cause.** The ¹⁵N correlation measured in
  signal-free columns corrects the noise there, and makes GLS worse.
- **The late, LP-extended part of the FID is the cause.** The signal model, the noise
  covariance near peaks, or both, are wrong there, and GLS gives that part the most
  weight.

What they cannot establish:
- **Which of the two is responsible.** LP fits its coefficients to the data. Near a
  resonance, the prediction filter follows the signal's poles, and it propagates the
  acquired noise into the extended points with a correlation that depends on the
  signal. The covariance in noise-only columns therefore says nothing about the
  covariance under a peak.
- **The causes cannot be separated here.** Three effects act in the same late-FID
  directions:
  - lineshape misfit;
  - deterministic LP extrapolation error;
  - signal-dependent LP noise.
- **Consistent, but not proof:** most of the OLS excess is one fixed shape per peak,
  scaled by its amplitude (2.22 → 1.61). That pattern is what a systematic signal
  error looks like, whether from the lineshape or from extrapolation. Signal-dependent
  noise changes with every plane's realization, so it would sit in the remaining
  1.6×, together with t1 noise and signal from neighbouring clusters.

**Known truth separates them (T7).** Synthetic planes on the LP axis: 47 acquired
points extended to 92 by NMRPipe's `LP -fb -ord 16`, then the example's ¹⁵N
processing. The example's processing script is not recorded; of orders 12–20,
order 16 best matches its noise-power signature (0.57 · 0.47 · 0.22 · 0.12 at
t = 47 · 48 · 56 · 64, against 0.51 · 0.44 · 0.21 · 0.14). An isolated line and a
pair 3 points apart whose partners decay at rates 1 and 2 across seven planes,
SNR 50, 200 realizations. The identity LP(s + n) = s + [LP(s) − s] +
[LP(s + n) − LP(s)] needs only the processing after LP to be linear, so it splits
NMRPipe's LP into extrapolation error and LP noise:

| Pair, exact lineshape | χ² / E[χ²] | Rate SD, B | Rate bias / SD, B |
|---|---|---|---|
| 47 points, no LP: OLS · GLS header R | 0.96 · 1.00 | 11.4% · 7.3% | −0.06 · −0.17 |
| LP: OLS · GLS header R · GLS measured R | 2.82 · 4.61 · 21.3 | 10.1% · 20.3% · 185% | 0.61 · 1.08 · 0.67 |
| s + LP(n), signal-free LP noise: same three | 0.95 · 0.55 · 1.00 | 5.3% · 3.9% · 2.2% | 0.00 · 0.06 · 0.06 |
| s + LP(s+n) − LP(s), signal-dependent LP noise | identical to LP | identical to LP | identical to LP |

- **Signal-dependent LP noise explains the GLS behaviour on the example.** With an
  exact lineshape and no extrapolation error, it alone reproduces every feature:
  - OLS χ² is 2.8× its expectation (2.22 on the example);
  - GLS is worse under the measured R than under the header R (whitened χ² 21 against
    4.6; on the example 18.4 against 10.9);
  - GLS scatter exceeds OLS, more for overlapping peaks: rates 2.0× (header R) and
    18× (measured R) for the pair, relative intensities 1.3× and 4.0× for the
    isolated line.
- **Extrapolation error is negligible:** NMRPipe's LP extends the noise-free FID
  exactly, with the exact lineshape and with an unmodelled Gaussian decay.
- **Signal-free LP noise is benign:** with it, the measured R is exact and GLS under
  it beats OLS. Noise-only columns measure this noise, not the noise under a peak.
- **Lineshape misfit without LP does not cause the failure:** χ² / E stays at
  0.95–1.26 for every estimator. Misfit does add a deterministic bias for the pair
  that GLS amplifies (relative intensity of B: OLS 1.0%, GLS header R −5.8%).
- **LP also biases OLS for overlapping peaks with differing profiles:** relative
  intensity of B −11.1% at its worst plane (0.56 SD), rate +6.2% (0.61 SD); with the
  Gaussian misfit, −4.8% (0.25 SD) and +3.7% (0.36 SD). With the exact lineshape,
  OLS shows no bias above 0.2 SD without LP or for the isolated line. A likely cause, not tested: LP fits
  its filter to each plane, so what it propagates depends on that plane's
  signal-to-noise ratio and does not cancel in ratios.
- **A NumPy `LP -fb` reproduces NMRPipe's** to 10⁻⁵ σ on the same data
  (`lp_extend`). A first version of this study, run with it alone on other noise
  (400 realizations), reached the same conclusions.
- **Limits:** one processed axis rather than a pseudo-3D; one SNR and one LP order.
  NMRPipe 13.0 output is committed as fixtures, so the study reruns without NMRPipe.
  Record: [`lp_known_truth.txt`](validation/profile-uncertainty/lp_known_truth.txt).

### 2.4 Conclusion

**OLS with sandwich errors stays.** The reasons are now empirical:
- the smallest real scatter of relative intensities;
- calibrated errors on real replicates;
- no inverse of an ill-conditioned R;
- no tuning constant.

GLS would be reconsidered only for spectra whose late FID the model describes, for
example without LP and with a well-modelled decay. The replicate test of §5 decides,
whatever the processing.

## 3. Covariance of the fitted amplitudes

### 3.1 First-order covariance

For a cluster, write:
- Φ, the shapes (L × n);
- M = (ΦΦᵀ)⁻¹Φ and P⊥ = I − ΦᵀM;
- E_l, the derivatives of peak l's shape with respect to the lineshape parameters θ;
- D_s = Σ_l a_{l,s} E_l;
- H = Σ_s D_sᵀP⊥D_s and T_s = H⁻¹D_sᵀP⊥.

Then:

  Cov(a_s, a_t) = σ²[δ_st MRMᵀ − MRT_sᵀG_tᵀ − G_sT_tRMᵀ + G_s C_θ G_tᵀ],

with G_s = M D_s and C_θ = Σ_u T_u R T_uᵀ.

- **It is the amplitude block of the OLS sandwich over all parameters.** It matches a
  direct computation to 1.4 × 10⁻¹⁴.
- **When R = I,** the cross terms vanish (10⁻¹⁸ in the check).
- **Storage:** the covariance is fixed by MRMᵀ, the tensors M E_m and M R P⊥ E_m H⁻¹,
  and C_θ, together with the stored amplitudes. That is O(L²n_θ) numbers per cluster (L peaks, n_θ lineshape parameters),
  whatever the number of planes, and reopening a project never refits.
- **C_θ is not the published lineshape covariance.** The published errors use the
  exact Jacobian of the projected residual, which carries a term proportional to the
  residuals; C_θ, as the sandwich over all parameters, does not. On the example:
  - isolated peaks agree within 0.2% for every lineshape combination, and their
    profile σ cannot change, since the lineshape term is a pure scale (§3.2);
  - for overlapping clusters the SD ratio is within 1.5% at the 90th percentile;
  - two strongly overlapped clusters reach 14%. There the marginal σ of two peaks, whose
    variance is 17–19× the conditional variance, moves by up to 12%, and the profile σ
    by at most 4%. These are the clusters where a first-order result is least
    reliable either way.

### 3.2 Isolated peaks: the lineshape term is a pure scale error

With one peak, D_s = a_s E, so T_s and G_s are proportional to a_s. Every term except
the first is therefore proportional to a_s a_t:

  Cov(a_s, a_t) = δ_st c + a_s a_t k.

- **What it means:** the lineshape uncertainty moves all of a peak's amplitudes by the
  same factor. Relative intensities, and anything fitted with a free scale, see only
  c, today's conditional σ².
- **On the example:** for all 92 isolated peaks, the relative-profile covariance of
  §3.3 equals cI to round-off (correlated modes ≤ 4 × 10⁻¹⁶ c). Over all peaks, the
  marginal/conditional ratio at plane 1 is 1.03 (median), 1.12 (90th percentile) and 4.4
  (maximum). For isolated peaks all of it is in the scale term.

### 3.3 Marginal σ, relative-profile covariance and profile σ

For one peak, let:
- Σ be the S × S covariance of its amplitudes over the planes (§3.1);
- c be its conditional variance, the same in every plane, since planes have
  independent noise;
- L = Σ − cI be the lineshape part, including the cross terms;
- u = a/‖a‖ be its unit profile, and P = I − uuᵀ.

**Marginal σ.** σ_s = √Σ_ss is the uncertainty of one absolute amplitude on its own.
It is correct, and it is the right error for a single amplitude. Treating the planes'
σ_s as independent is what overstates downstream errors (§4); the marginal σ itself is
not wrong.

**Relative-profile covariance.** A downstream model with a free scale fits a ≈ A·f(p).
- **First-order response:** to first order, at the fit, f ∝ a. Eliminating A then
  gives δp̂ = K δa with K a = 0, whatever the downstream weights.
- **What follows:** Cov(p̂) = KΣKᵀ is unchanged by Σ → Σ + a vᵀ + v aᵀ, for any
  vector v. Covariances that differ by such terms are equivalent for every analysis
  with a free scale.
- **Representative:** this specification uses

    Σ_rel = cI + PLP.

  It differs from Σ only by terms of that form, because
  L − PLP = (I − P)L + PL(I − P) and I − P = uuᵀ. So it is exactly equivalent to Σ
  for those analyses.
- **Properties:** it is positive definite. It equals cI for an isolated peak (§3.2).
  It contains no scale component.
- **Without a free scale,** for example a ratio to given reference planes, the
  analysis needs Σ itself. Ratios use Σ (§4).

**Profile σ.** √(Σ_rel)_ss, the diagonal of Σ_rel.
- **Valid alone only where Σ_rel ≈ diag:** used as independent per-plane errors, it is
  exact only where Σ_rel is close to diagonal. Off the diagonal, PLP holds modes
  correlated across planes, smooth distortions that resemble the partner peak's
  profile.
- **The diagonal is a poor test of this.** A mode of variance λ spread over S planes
  adds only about λ/S to each diagonal element.
- **The criterion is the largest eigenvalue of PLP relative to c.**

| Case | Largest correlated mode / c | Largest diagonal increase / c | Downstream rate error, profile σ: scatter / stated |
|---|---:|---:|---:|
| Synthetic pair, 3 points apart | 0.26 | 0.07 | 1.05 |
| Synthetic pair, 2 points apart | 1.75 | 0.46 | 1.39 |
| Example, overlapping peaks (median · 90th percentile · maximum) | 0.0004 · 0.38 · 5.6 | up to 0.52 | — |

- **On the example:** 12 of 74 overlapping peaks have a correlated mode above 0.1c,
  and 2 above c. They are larger than in the synthetic case where per-plane errors
  failed.
- **The earlier draft understated this.** It judged the transfer by the diagonal
  alone.

**Common scale uncertainty.** Splitting Σ into a scale part and a remainder is not
unique: any a vᵀ + v aᵀ can be moved between the two. Downstream analyses with a free
scale do not need it, and those without one need Σ. So it is not reported.

**Compact form.** PLP has rank at most twice the number of nonlinear parameters, so
Σ_rel = cI + Σ_k λ_k v_k v_kᵀ over a few modes. Every quantity above follows from the
tensors of §3.1.

## 4. Downstream parameters

The test is a relaxation decay, I_s = A·e^(−R·t_s) over 8 planes, fitted to a peak's
fitted amplitudes. The fit is weighted least squares with independent per-plane σ_s,
as downstream programs do. 1,500 realizations; each cell gives scatter / stated SD of
R, then 95% coverage.

| Per-plane σ | Isolated | Pair 3 points apart | Pair 2 points apart |
|---|---|---|---|
| Conditional | 1.01 · 0.946 | 1.08 · 0.930 | 1.61 · 0.794 |
| Marginal | 0.93 · 0.965 | 0.73 · 0.993 | 0.68 · 0.994 |
| Profile σ | 1.01 · 0.946 | 1.05 · 0.935 | 1.39 · 0.851 |
| Full covariance (GLS downstream fit) | 1.01 · 0.947 | 0.95 · 0.956 | 1.00 · 0.965 |

In the pairs, the partner peak decays at 0.2 instead of 1.0.

- **Marginal σ treated as independent overstates downstream errors,** by up to 1.47×.
  The common scale error is fully correlated across planes, but independent errors
  count it as fresh noise in every plane.
- **Profile σ is exact for isolated peaks and close for moderate overlap.**
- **Strong overlap with differing profiles leaves correlated modes in Σ_rel (§3.3).**
  - Any per-plane σ is then wrong, in one direction or the other.
  - Σ, or equivalently Σ_rel, is calibrated in every case.
- **Ratios (CEST, CPMG I/I₀, R2eff)** are to use the full covariance, through
  `profiles.normalize` extended with a covariance (§9); today it takes independent
  errors. The scale term cancels there.
  - In the 5-plane study, today's independent conditional ratio errors were within
    16% for a pair 3 points apart.
  - Marginal σ treated as independent overstated them up to 2.2×.
  - The full covariance was calibrated: 95% coverage 0.95–0.96.

## 5. Real experimental variability

### 5.1 CEST baseline planes as replicates

- **Planes used:** those more than 400 Hz from the peak's own ¹⁵N offset, excluding
  the −12 kHz reference.
- **Scatter:** the robust SD of second differences across offsets, divided by √6.
  Second differences remove slow trends and common-mode errors, and the robust SD
  limits the influence of unknown minor-state dips.
- **Coverage:** 164 peak profiles. The OLS relative scatter is 2.8% (median).

| Estimator | Scatter / OLS scatter, all peaks | Multi-peak clusters | Empirical / own stated σ |
|---|---|---|---|
| OLS, conditional sandwich (header R) | 1 | 1 | 1.06 (0.89–1.28) |
| OLS (measured ¹⁵N R) | 1 | 1 | 1.01 (0.85–1.22) |
| α = 0.1, header R | 1.07 (0.88–1.26) | 1.09 | 1.19 |
| GLS, header R | 1.30 (0.99–6.2) | 1.89 | 1.48 |
| α = 0.1, measured R | 1.34 (0.99–1.76) | 1.43 | 1.53 |
| GLS, measured R | 1.95 (1.35–11.9) | 2.73 | 2.29 |

- **The conditional sandwich σ that OLS publishes today is calibrated for relative
  intensities on real data.** Correcting R for LP moves the median from 1.06 to 1.01.
- **Tool (T4):** `peakfit replicates <run>` applies this method to any published CEST
  run, read from `scientific.npz` without refitting, against the published profile σ
  (`engine.scientific.replicates`). On the bundled example, observed / stated is 1.07
  (0.89–1.28) over 164 peaks; two peaks with a non-positive baseline are skipped.
- **Caveats:**
  - Common-mode errors are invisible by construction. They are also irrelevant
    downstream (§3.2).
  - Drift, residual saturation and minor-state dips can only inflate the empirical
    scatter.
  - This is one dataset.
  - **Replicates measure repeatability, not accuracy.**
    - A systematic error that is the same in every baseline plane is invisible here,
      for example a misfit, or a bias from LP extrapolation.
    - For isolated peaks such an error is a common factor and cancels.
    - For overlapping peaks with differing profiles it can distort relative
      intensities. Only a test with known truth measures that: in T7, LP biases a
      partner's relative intensity by up to 0.6 SD at SNR 50 (§2.3).

### 5.2 The residual excess

- **Composition:** it is mostly a fixed misfit shape per peak, proportional to the
  amplitude. Such misfit biases every plane's amplitude by the same factor, so it
  cancels in relative intensities for isolated peaks.
- **Inflating σ by the excess would be wrong:** it would make the errors about 1.5×
  too large against the replicates.
- **As a diagnostic,** the proportional part measures how well the lineshape fits. It
  could replace χ²ᵣ in review, whose expectation under correlated noise varies by
  cluster. That replacement is not planned.

### 5.3 Duplicate planes

Any series (T1, T2, CPMG, CEST) may contain *duplicate planes*: planes whose recorded
plane values are exactly equal. `peakfit replicates <run>` checks the published
conditional σ against them, and runs the baseline method of §5.1 only where there are
none or with `--baseline` (`engine.scientific.replicates`). It reads the run without
refitting and changes no error.

- **Matching:** exact equality of the recorded plane values; no tolerance and no
  manual pairing. A run without repeats reports "no duplicate planes".
- **Statistic, per peak:** for groups of m_g copies, all k = Σ_g (m_g − 1)
  orthonormal within-group contrasts C are stacked and standardized by the peak's
  conditional variance c_ll, the diagonal of σ²MRMᵀ (§3.1):
  q = |Ca_l|²/c_ll, with E[q] = k at first order.
  - **Why the conditional variance.** In the covariance of §3.1, every lineshape
    term enters through G_s = M D_s and T_s, linear in the cluster's amplitudes on
    plane s. True copies have equal true amplitudes, so the orthonormal contrasts
    annihilate those terms exactly and the true contrast covariance is c_ll·I:
    no lineshape term, no correlation between groups through shared parameters.
    Built from the fitted amplitudes instead, CΣCᵀ brings those terms back driven
    by noise, along the observed contrast d itself: V ≈ c_ll(I + βddᵀ/c_ll), so
    dᵀV⁻¹d = x/(1 + βx) with x = |d|²/c_ll, biased low by about β(k + 2). That was
    the statistic of the first implementation, and its bias explains the shortfall
    T8 resolved in it (§8.2).
  - q depends neither on the contrast basis nor on the plane order.
- **What it validates:** the conditional σ only, the noise part of the published
  uncertainty. It is blind to the lineshape-parameter part of the marginal and
  profile σ, which cancels between copies at first order: a miscalibration of that
  part goes undetected.
- **Result:** the pooled RMS √(Σq / Σk), 1 when the conditional σ is calibrated,
  separately for peaks without and with the correlated-profile flag
  (§6), with each pool's counts of contrasts, peaks and clusters.
  - Contrasts from peaks of one cluster are correlated, so the effective sample is
    smaller than the count and the result is descriptive.
  - Below 50 contrasts the output warns that the independent-case SE, 1/√(2n),
    exceeds 0.1; it is a lower bound on the sampling uncertainty of the RMS.
  - No robust companion, confidence interval or signed statistic: PeakFit does not
    know the acquisition order.
- **Exclusions,** each listed with its reason and count, nothing regularized. A peak
  takes the first reason that applies, in this fixed order:
  1. an unusable cluster;
  2. a non-finite amplitude on any plane of the peak;
  3. a conditional variance withheld or not positive. The fit publishes the
     conditional covariance only where the shapes are full rank with
     s_min > √ε·s_max (`linear_algebra.factorize_shapes`), and otherwise withholds
     all of it as NaN: a coincident pair is excluded this way. A published
     conditional is computed from the SVD without squaring the condition number, so
     it is numerically accurate whenever finite; no further threshold applies.
  4. a lineshape covariance withheld: the cluster's lineshape covariance factors
     (§3.1) are withheld by the fit's own rule, its rank and conditioning rule on
     the projected Jacobian (`covariance_factors`).
  - **Why the fourth reason (decided 2026-11-14).** It is a conservative
    fit-quality safeguard, not a mathematical requirement of the statistic: q uses
    only c_ll, which stays finite and accurate. But numerical accuracy is not
    first-order validity, and a finite c_ll alone does not make a fit trustworthy,
    least where the fit has difficulty separating overlapping peaks. A truly
    coincident pair that the fit separates has a finite conditional, yet its
    contrasts exceed c_ll by about 50% (§8.2); with its lineshape factors also
    withheld, its flag is not assessed and it would join the unflagged pool. The
    rule reuses the fit's existing withholding criterion and introduces no new
    threshold. Being flagged excludes nothing: the flagged and unflagged pools stay,
    and a near-coalescent pair whose factors are available stays in its pool, with
    its departure from the first order visible (§8.2).
- **Limits:**
  - repeatability, not accuracy: an error common to the copies is invisible;
  - the conditional σ only: the lineshape-parameter part of the marginal and
    profile σ cancels between copies;
  - contrasts, not marginal σ: a scale uncertainty shared by the copies cancels;
  - raw amplitudes, not normalized profiles directly;
  - equal plane values do not prove independent, equivalent acquisitions;
  - not calibrated for copies reconstructed jointly, which share the
    reconstruction's noise ([processing guide](processing-quantitative-series.md#joint-pseudo-3d-reconstruction)).
- **Validation:** T8 (§8.2). **Remaining gap:** no real dataset with duplicate planes
  has been checked. One is to be checked before release if it becomes available;
  release does not depend on it.

## 6. Reporting

Each quantity of §3.3 is reported under its own name:
- **Marginal σ:** the error of one absolute amplitude, in the amplitude table and the
  inspector.
- **Profile σ:** the per-plane error of the profile.
  - It goes in the per-peak profile files that downstream programs read.
  - It is the error bar on profiles in the GUI.
- **The conditional σ** stays an export column, as a diagnostic.
- **Ratios and R2eff:** the full Σ, in the GUI and the CLI alike.
- **Correlated-profile flag:** set where the largest correlated mode of Σ_rel exceeds
  τ·c, so that per-plane errors cannot describe the profile's uncertainty.
  - τ = 0.1, set by T3 (§8.1).
  - The flag also covers strong covariance with a cluster partner (§6a): a
    within-plane |ρ| above 0.15. No covariance is exported until a consumer exists.
  - Below both thresholds, independent profile σ misstate no downstream σ of the
    T3 sweep by more than 3.8%; the tolerance is 5%.
  - The flag describes a limit of the approximation, not the quality of the data
    (§6a).

**Which name downstream programs see** was settled as option (a) of §10: many read
one error column by habit, so `intensity_err` carries the profile σ.
- **Noise-model check:** the measured time-domain noise-power profile of each axis is
  compared with the header's prediction. A mismatch, such as unrecorded LP or an
  earlier window, gives a recorded warning.
- **Storage:** the tensors of §3.1 and the check. The flag is derived when a run
  is read, from the stored factors and amplitudes, and is not stored (#164). This is a deliberate format
  change: the next major version of the scientific archive and of the analysis
  project. The factors went into archive 12.0.0 and project 15.0.0; they add
  0.19 MB to the example's archive, whatever the number of planes.
- **GUI:**
  - error bars use the profile σ, and ratios use the covariance;
  - the flag is shown on the profile, the tiles and the cluster page;
  - there is no uncertainty mode.

## 6a. Consumption by ChemEx

### What ChemEx does today

ChemEx is read at `d0ba6e34`.
- **Input:** one profile per spin system, with columns (metadata, intensity, error).
  CEST and CPMG profiles are raw intensities, reference planes included.
- **Free scale:** each profile has one, solved analytically with weights 1/err²
  (`Data.scale`).
- **Residuals:** (scale·calc − exp)/err, independent.
- **Parameter errors:** from the Jacobian, with err taken as absolute; there is no χ²ᵣ
  rescaling.
- **Monte Carlo and bootstrap:** both assume independent points.
- **Error options:** `error = "scatter"` or `"duplicates"` replaces the file errors
  with one estimated σ per profile. With the default `global_error = true`, that σ is
  shared by every profile of the experiment.

**Consequences:**
- PeakFit's ratios are for display only. ChemEx normalizes through its own free
  scale.
- The per-plane error ChemEx needs is exactly the relative-profile one (§3.3).

### What independent per-plane errors lose

The test is a ChemEx-like global fit: two peaks of one cluster, 13 planes (one
reference and ν_CPMG from 25 to 1,000 Hz), fast-exchange R2eff, a free scale per
profile, and kex shared. Reference-plane SNR is 150, with 1,500 realizations. Each
cell gives scatter / stated SD, then SD relative to the joint-covariance fit.

| Error input | Δ = 6 points: kex | Δ = 3 points (ρ = −0.16): kex · Φ₂ | Δ = 2 points (ρ = −0.43): kex · Φ₂ |
|---|---|---|---|
| Marginal σ, independent | 0.92 · 1.00 | 0.68 · 0.73 (SD 1.03) | 0.35 · 0.50 (SD 1.10) |
| Profile σ, independent | 0.97 · 1.00 | 0.95 · 1.05 (SD 1.01) | 0.87 · 1.20 (SD 1.09) |
| Conditional σ, independent | 0.97 · 1.00 | 0.96 · 1.05 (SD 1.01) | 0.90 · 1.26 (SD 1.08) |
| Covariance per peak | 0.97 · 1.00 | 0.96 · 1.02 (SD 1.01) | 0.90 · 0.99 (SD 1.09) |
| Joint covariance across both peaks | 1.00 · 1.00 | 1.00 · 1.02 (SD 1) | 1.00 · 1.01 (SD 1) |

In every row, the bias is at most 0.06 SD.

- **Marginal σ in the error column is wrong for ChemEx.** It overstates kinetic errors
  by 1.08× for resolved peaks, 1.4–1.5× for moderate overlap and 2–3× for strong
  overlap, and gains no precision.
- **Profile σ is calibrated within 5%** for resolved and moderately overlapping peaks.
- **In strong overlap, two kinds of information are lost:**
  - the correlated modes within a profile, so Φ₂'s error is about 20% too small;
  - the covariance between the two peaks, which costs about 9% of kex precision. The
    covariance within each peak alone does not recover it.
- **Fitted values were not biased by any choice in these simulations.** This is a
  finding for these cases, not a guarantee. Covariance-aware weighting changes the
  weights, and can move fitted values when the model is nonlinear enough, the data are
  sparse or the profiles are strongly overlapped.
- **On the example:**
  - 7 of 29 multi-peak clusters have a within-plane |ρ| above 0.3 (maximum 0.77);
  - 12 of 74 overlapping peaks have correlated modes above 0.1c.

  So the losses concern a minority of clusters, about 6% of all of them.

### Interface

- **The error column gets the profile σ.**
  - For isolated peaks it equals today's `intensity_err`, so nothing changes for most
    peaks.
  - The marginal σ goes in its own column.
- **A per-peak flag marks where independent errors are inadequate:** correlated modes
  above τ·c, or within-plane |ρ| with a cluster partner above a threshold.
  - T3 sets them with the global fit above among its downstream fits: τ = 0.1 and
    |ρ| > 0.15 (§8.1).
  - ChemEx needs no change.
  - The flag states a limitation of the independent-error approximation. It does not
    say the peak's data are bad, and it does not imply discarding or down-weighting
    the peak; that judgment stays with the user.
- **No covariance export yet.**
  - Using one would change ChemEx's residual layer: groups of profiles whitened
    jointly, with coupled scales.
  - The gain is at most about 10% in precision and 20–25% in error calibration, on a
    few clusters.
  - The archive keeps the tensors (§3.1), so such an export can be added when a
    consumer exists.
- **ChemEx `scatter` with `global_error`** averages one σ over every profile. It
  therefore discards the larger σ of overlapping peaks. With PeakFit's errors
  calibrated against baseline scatter (§5.1), `error = "file"` is the better choice.
  That is a usage note, not a ChemEx change.

## 7. MCMC

- **No likelihood to sample.** OLS is not a likelihood estimator under R, and the GLS
  likelihood is contradicted on real data, with or without the LP correction.
- **The formulation stays deferred,** and MCMC stays out of the GUI.
- **When to revisit:** if the target becomes the relative profile, a likelihood could
  be built on plane-to-plane variability, which the replicates (§5) can validate. That
  question is separate.

## 8. Validation plan

| ID | Test | Gate |
|---|---|---|
| T1 | The factor formula against the full sandwich, including shared and fixed parameters; R = I; the proportional structure for a single peak | 10⁻¹⁰ relative |
| T2 | The profile σ of an isolated peak against today's `intensity_err` | Equal to round-off |
| T3 | Engine coverage (2D, NMRPipe-equivalent processing, `sp1`, `sp2`, `no_apod`): marginal σ, profile σ, ratios, R2eff, and downstream decay, global CPMG and CEST fits with profile σ and with the covariance; overlap from resolved to strong (§8.1) | Where no flag is raised, scatter / stated within 0.95–1.05, and 95% coverage within ±3 SE for a single quantity and ±0.01 for quantities pooled over planes (§10 item 5); also sets the flag thresholds |
| T4 | The baseline-replicate tool on the bundled example, and on any further CEST datasets | Empirical / stated within 0.9–1.15 (median) |
| T5 | The noise-power check: LP, an earlier window and `SP -size` (`correlated-noise.md` §2.5) on NMRPipe-processed noise | Detects each case; no false alarm on correctly recorded processing |
| T6 | The estimator comparison (§2), rerunnable | Reproduces §2 and §5 |
| T7 | Known truth through real processing: synthetic pseudo-3D FIDs (isolated and overlapping peaks, differing profiles) processed by NMRPipe with and without LP, fitted by OLS and GLS | Bias and scatter of relative intensities and downstream rates; separates lineshape, LP-extrapolation and LP-noise effects |
| T8 | The duplicate-plane check (§5.3): duplicates through the exact processing operators and the engine fit, two and three copies, isolated and overlapping peaks, a shared parameter, exclusions; the stated contrast covariance c_ll·I against its Monte Carlo value; unmodelled correlation between copies, recorded (§8.2) | Mean q/k over realizations within ±3 SE of 1, SE from the realizations; contrast variances within ±3 SE of c_ll |

### 8.1 T3: coverage and the correlated-profile thresholds

Tool: `tools/profile_uncertainty/profile_coverage.py`. Record:
[`profile_coverage.json`](validation/profile-uncertainty/profile_coverage.json)
(4,000 realizations per case, seed 20261113).

**Set-up.**
- **Clusters:** 2D pseudo-3D, drawn with PeakFit's own `sp1`, `sp2` and `no_apod`
  lineshapes (equal to NMRPipe's processing, `tests/test_lineshapes_equivalence.py`).
  The noise is complex white FID noise through the exact processing operators of
  both axes: sine bell (or none), first point, zero fill, Fourier transform, P1.
  Linewidths 20 Hz, noise 0.02 of the larger peak's reference height.
- **Fit:** the engine fits each realization from the truth (`fit_cluster`,
  `evaluate_analytical_model` under the processing-derived R).
- **Profiles:** 1 plane; 3 planes of a decay; 30 CPMG-like planes (one reference, 29
  ν_CPMG); 30 CEST-like planes (two references, 28 offsets, one dip per peak). The
  partners' profiles differ.
- **Overlap:** an isolated peak, and pairs 4, 3, 2.5 and 2 points apart on both axes;
  a cluster phase shared by both peaks; a near-coalescent pair 1 point apart. PeakFit
  has no linewidth shared between peaks, so the shared phase is the shared parameter.
- **Quantities:** the marginal σ; the profile σ, against the scatter of the
  amplitudes with their own profile direction removed, which has covariance
  Σ_rel − c·uuᵀ; ratios and R2eff through `profiles.normalize`; downstream fits with
  the profile σ as independent errors: A·e^(−R·t) per peak, a global fast-exchange
  CPMG fit with kex shared and a free scale per profile, and a Lorentzian CEST dip
  per peak. A fit with the joint covariance of both peaks is the control.

**Thresholds, by downstream consequence.** A sweep of 2,583 configurations (5,166
peaks: every lineshape and downstream profile, pairs 1–6 points apart diagonally or
along either axis, partners of equal or 0.3× height, a shared phase) gives, at first
order, each downstream σ's misstatement by independent profile σ:
√(KΣKᵀ / (JᵀWJ)⁻¹), with Σ the covariance between planes of both peaks.

| Largest correlated mode / c | < 0.05 | 0.05–0.1 | 0.1–0.13 | 0.13–0.15 | 0.15–0.2 | 0.2–0.3 | 0.3–0.5 | 0.5–1 | > 1 |
|---|---|---|---|---|---|---|---|---|---|
| Worst misstatement, own parameters | 3.5% | 4.8% | 4.9% | 6.1% | 7.7% | 11% | 18% | 31% | 54–154% |

| Within-plane \|ρ\| with a partner (mode ≤ 0.1) | < 0.05 | 0.05–0.1 | 0.1–0.15 | 0.15–0.2 | 0.2–0.25 |
|---|---|---|---|---|---|
| Worst misstatement, shared kex | 1.2% | 2.5% | 3.6% | 4.7% | 6.0% |

- **Two mechanisms.** Correlated modes misstate each peak's own parameters (R, Φ,
  R20, the CEST dip); the tolerance of 5% is first crossed at a mode of 0.134c. The
  covariance between partners, which independent errors drop, misstates the
  parameters a joint fit shares (kex); among peaks below that mode it is first
  crossed at |ρ| = 0.205.
- **Chosen:** τ = 0.1 and |ρ| > 0.15, below both boundaries. The margin covers
  configurations between the sweep's points and profiles it does not contain. Below
  both, the worst misstatement in the sweep is 3.8%. 1,764 of its 5,166 peaks are
  flagged, by construction of a sweep concentrated on overlap.
- **Consistent with earlier estimates:** PeakFit's 1D Gaussian decay pair
  (`tests/test_profile_sigma_calibration.py`) gives 1.044 at 0.12c and 1.09 at 0.26c.
  τ sits at the low end of the 0.1–0.25 this specification expected.

**Gate, cases without a flag: passed.** 48 of the 68 cases raise no flag
(noise-free diagnostics), with 270 gated quantities. The criterion is §10 item 5.
- **Scatter / stated:** every one within 0.962–1.039.
- **95% coverage:** every single quantity within ±3 SE (largest deviation 2.5 SE), and
  every quantity pooled over planes within ±0.01 (largest deviation 0.0063).
- **Information, the strict ±2 SE per quantity:** 20 of the 270 lie outside, against
  12.3 expected by chance alone, since each ±2 SE test fails 4.6% of the time with
  exact σ.
  - A second seed (7) also gives 20, on other quantities, except R2eff.
  - Excluding R2eff, 14 of 258 lie outside (11.7 expected), on both sides, with
    standardized deviations of mean 0 and SD 1.0: chance.
  - R2eff over-covers in 6 of 12 cases (5 at seed 7), by 0.11–0.19 percentage points
    (coverage 0.9511–0.9519). Pooling 29 planes makes the SE 0.0006, so this small
    effect is resolved. Its scatter / stated is 0.96–1.03. σ(R2eff) = σ_r/(r̂T) uses
    the fitted ratio, so the stated σ grows where R2eff errs high: first-order
    propagation through the logarithm, not an error in the covariance.
  - The joint-covariance control exceeds ±2 SE for 2 of its 138 quantities.

**Flagged cases confirm the flag.** Independent profile σ misstate downstream σ as
the first-order sweep predicts; the joint covariance stays calibrated (scatter /
stated 0.976–1.034, coverage 0.942–0.957).

| Case (flagged by) | Independent profile σ: scatter / stated, 95% coverage |
|---|---|
| `sp2` decay, 2 points (mode 0.43c) | R: 1.13 · 0.927 |
| `no_apod` decay, 2 points (mode 0.12c) | R: 1.03 · 0.940 |
| `sp2` CPMG, 2.5 points (\|ρ\| 0.21, mode 0.07c) | kex: 0.95 · 0.963 |
| `sp2` CPMG, 2 points (both) | kex: 0.89 · 0.972; Φ_B: 1.08 · 0.934 |

**Near-coalescent cases, reported, not gated** (`sp2`, 1 point apart, mode 3.8–4.9c,
|ρ| 0.79): the first-order description fails, as the resolution study found for
positions.
- The ratio is withheld by `normalize`'s reliability limit in 3,997–4,000 of 4,000
  realizations.
- Even the marginal σ is short: scatter / stated 1.04–1.14, coverage 0.69–0.94.
- Downstream σ are off by −28% to +72%.

**Other observations.**
- With one plane, pairs 2 points apart (|ρ| 0.19–0.37, flagged) show marginal σ
  short by 4–6% (coverage 0.928–0.939): nonlinearity, which first order cannot show.
- Near a threshold the flag varies between realizations: at |ρ| 0.138 (`sp1`, one
  plane, 2.5 points) it is raised in 20% of them.

**Limits:** the exact lineshape (misfit and LP are T7's); one noise level; fits
started at the truth; first-order thresholds from one family of profiles each.

### 8.2 T8: the duplicate-plane check

Tool: `tools/profile_uncertainty/duplicate_check.py`. Record:
[`duplicate_check.json`](validation/profile-uncertainty/duplicate_check.json)
(40,000 realizations per case, seed 20261114). Fast cases, with amplitudes drawn
from the stated covariance, run in `tests/test_duplicate_planes.py`.

**Set-up.** As T3 (§8.1): 2D clusters with the `sp1`, `sp2` or `no_apod`
lineshapes, white FID noise through the exact processing operators, the engine fit
from the truth, noise 0.02 of the reference height. The profiles are decays
A·e^(−R·t), R = 1 and 2 for the two peaks of a pair: t = 0, 0.1, 0.1, 0.3, 0.6,
0.6, 1.0 (two pairs, k = 2) or t = 0, 0.2, 0.2, 0.2, 0.5, 1, 1, 1 (two triples,
k = 4). Each realization is checked by `duplicate_plane_check` with the flags
derived from its fit, as `peakfit replicates` reads a run.

**Criterion.** The mean of q/k = Σq/Σk over realizations within ±3 SE of 1, the SE
from the realizations' spread; and, the analytical check, each contrast's Monte
Carlo variance against the c_ll the check states (averaged over realizations)
within ±3 SE, SE √(2/(N − 1)), with the off-diagonal covariances recorded against
0. A gated case passes only if both hold: the gate is
stricter than the criterion of spec #172, which asks for the mean q/k alone.

**Confirmation protocol (fixed 2026-11-14, before the run).** The first record
(4,000 realizations) left `sp2`, isolated, two copies at −2.97 SE, at the edge of
the gate. To tell sampling variability from miscalibration without choosing seeds
or changing the criterion:
- every case, gated or not, runs 40,000 realizations at the record seed 20261114.
  Realization i of a case draws from the seed (20261114, case, i), so realizations
  0–3,999 are the first record and 4,000–39,999 are new, independent draws;
- the gate above applies, unchanged, to all 40,000;
- diagnostics, per case: the 10 disjoint blocks of 4,000, each block's deviation in
  its own SE and the χ² of the block means about their pooled mean (9 degrees of
  freedom); and the gate on blocks 2–10 alone, an independent replication of the
  first record;
- decision rule: sampling variability if the case passes on all 40,000, the
  replication passes and the blocks are consistent (p > 0.01); systematic
  miscalibration if the replication deviates beyond 3 SE in the same direction as
  the first record.

**Statistic change (2026-11-14).** The first implementation standardized each
peak's contrasts by CΣCᵀ built from the fitted amplitudes; §5.3 explains why that
biased q low and why q = |Ca_l|²/c_ll replaced it. The protocol above was re-applied
unchanged (same cases, seeds, N, gate and diagnostics) after the statistic change on
2026-11-14, and again, unchanged, after the lineshape-covariance exclusion of
§5.3 (2026-11-14). The table is that last record. Realization i of a case draws the
same noise in every record, so a case without that exclusion reproduces its
previous numbers exactly: only the exclusions case changed.

**Gate: passed** on all 40,000 realizations of every gated case. The
replication is blocks 2–10 alone (36,000 realizations); the block χ² has 9 degrees
of freedom. The worst contrast variance is against c_ll.

| Case | Pool | Mean q/k | SE | Deviation | Replication | Block χ² (p) | Contrast variances, worst |
|---|---|---|---|---|---|---|---|
| `sp2`, isolated, two copies | all | 0.989 | 0.005 | −2.2 SE | 0.993, −1.3 SE | 20.3 (0.02) | 2.2 SE |
| `sp2`, isolated, three copies | all | 1.001 | 0.004 | +0.2 SE | 1.001, +0.3 SE | 4.4 (0.88) | 2.1 SE |
| `sp1`, isolated, two copies | all | 1.007 | 0.005 | +1.3 SE | 1.008, +1.5 SE | 9.0 (0.44) | 2.0 SE |
| `no_apod`, isolated, two copies | all | 1.003 | 0.005 | +0.7 SE | 1.003, +0.6 SE | 6.5 (0.68) | 1.4 SE |
| `sp2`, pair 3 points, two copies | all | 1.000 | 0.004 | +0.1 SE | 0.999, −0.2 SE | 3.3 (0.95) | 1.5 SE |
| `sp2`, pair 3 points, three copies | all | 1.000 | 0.003 | −0.1 SE | 1.000, +0.0 SE | 3.6 (0.94) | 1.7 SE |
| `sp2`, pair 3 points, shared phase, three copies | all | 1.000 | 0.003 | +0.2 SE | 1.000, +0.1 SE | 9.9 (0.36) | 1.3 SE |
| `sp2`, pair 2 points, two copies (flagged in every realization) | flagged | 1.009 | 0.004 | +2.4 SE | 1.007, +1.8 SE | 9.4 (0.40) | 1.8 SE |
| `sp2`, isolated peak with a coincident pair, two copies | unflagged | 0.994 | 0.005 | −1.2 SE | 0.991, −1.6 SE | 8.8 (0.45) | 2.3 SE |

- **The shortfall is gone.** The first implementation's record (the previous
  version of `duplicate_check.json`) left six of the seven `sp2` gated cases below 1, an inverse-variance weighted mean q/k of 0.995 ± 0.0013
  (−4.0 SE). Its mechanism is the bias of §5.3: the lineshape terms of CΣCᵀ,
  evaluated at the fitted amplitudes, return along the observed contrast and lower
  q by about β(k + 2), most where the lineshape term is largest (the pairs and the
  shared phase, −2.3 to −2.6 SE in that record). With q = |Ca_l|²/c_ll the same
  seven cases give 1.0005 ± 0.0013 (+0.4 SE), and the pairs and the shared phase lie
  within 0.2 SE of 1; the `sp1` and `no_apod` cases give 1.005 ± 0.004.
- **The edge case: sampling variability, by the protocol's rule.** `sp2`, isolated,
  two copies passes on all 40,000 (−2.2 SE); its replication passes (−1.3 SE); its
  blocks are consistent (p = 0.02 > 0.01). Block 1, the first 4,000-realization
  record, is at −2.9 SE; the other nine lie between −2.4 and +1.4 SE. For an
  isolated peak the old bias was smallest, so the case barely moved.
- **Exclusions,** end to end, by reason (realizations with at least one peak
  excluded): conditional variance withheld, 36,007 of 40,000 (the coincident pair's
  shapes too ill-conditioned); lineshape covariance withheld, 797 (both peaks of
  the pair, 1,594 peaks). No other reason occurred in any case. In the remaining
  3,196 realizations the pair fitted apart with its lineshape factors available and
  was pooled: in the flagged pool in 3,165 (6,276 peaks, pooled q/k 1.91) and in the
  unflagged pool in 85 (116 peaks, pooled q/k 1.47), one peak in each in 54. The gated unflagged pool is now
  the isolated peak plus those 116 peaks (1.003 peaks per realization; 1.043
  before the exclusion), and passes at −1.2 SE. The unusable-cluster and
  non-finite-amplitude reasons have no end-to-end case (the fits start at the truth
  and succeed): they are covered by the tests of `tests/test_duplicate_planes.py`
  only.
- **What the lineshape-covariance exclusion hides.** The 797 excluded fits are a
  truly coincident pair that the fit separated, with a finite conditional variance
  but lineshape parameters the fit's own rule cannot identify; the flag is then not
  assessed, so before the exclusion both peaks joined the unflagged pool. The tool
  still computes their q from c_ll: pooled q/k 1.51 (1,594 peaks in 797
  realizations), their contrasts about 50% larger than c_ll states. With them, the
  gated statistic would be 1.001 ± 0.005 (+0.3 SE), the previous record's value;
  without them, 0.994 ± 0.005 (−1.2 SE). Diluted by 40,000 isolated peaks, they did
  not fail the gate, but they are not calibrated: the exclusion removes them as a
  fit-quality safeguard (§5.3). The previous record listed the same fits as 797 of
  882 realizations that put the pair in the unflagged pool (1,710 peaks, pooled q/k
  1.51).
- **Remaining coincident-pair fits are not calibrated either.** The 85 unflagged
  realizations whose pair fitted apart with available factors give pooled q/k 1.47,
  and the 3,165 flagged ones 1.91: a coincident pair fitted apart is beyond the
  first order even when its lineshape parameters are identified. The exclusion does
  not reach them; in this case they are too few to move the gated pool, and the
  flagged ones stay in the flagged pool, which this case does not gate.
- **The flagged pool** passes too: the conditional variance does not depend on the
  flag.
- **Off-diagonal contrast covariances** agree with 0, as c_ll·I states, to within
  0.014 of the variances in every gated case.

**Recorded, not gated.**
- **Near-coalescent pair** (`sp2`, 1 point apart, flagged): mean q/k 1.059 ± 0.005
  (+12.4 SE), unchanged by the lineshape-covariance exclusion, which excludes none
  of its 80,000 peaks; its contrast variances lie 5–7% above c_ll, the worst 9.3 SE. The
  first-order cancellation of the lineshape terms fails this close to coalescence,
  as the first order does for the marginal σ in §8.1. The first implementation gave
  1.001 here, its downward bias offsetting this excess.
- **Unmodelled correlation between planes:** the noise of each group's copies
  correlated with ρ, which the model omits. q/k is then about 1 − ρ:

| Case | ρ | Mean q/k | RMS |
|---|---|---|---|
| `sp2`, isolated, two copies | +0.3 | 0.698 ± 0.004 | 0.835 |
| `sp2`, isolated, two copies | −0.3 | 1.301 ± 0.007 | 1.141 |
| `sp2`, pair 3 points, three copies | +0.3 | 0.700 ± 0.002 | 0.837 |
| `sp2`, pair 3 points, three copies | −0.3 | 1.299 ± 0.003 | 1.140 |

  Positive correlation pushes the RMS below 1, negative above. This illustrates
  unmodelled covariance in general; it is not a model of joint pseudo-3D NUS
  reconstruction, whose correlation structure was not simulated.

**Limits:** synthetic data with the exact lineshape; one noise level; fits started
at the truth; decays only. No real dataset with duplicate planes has been checked
(§5.3).

## 9. Implementation sequence

1. **Engine and tools, unpublished:**
   - the covariance tensors, marginal σ, Σ_rel, profile σ and the flag;
   - `profiles.normalize` with a covariance;
   - the noise-power check;
   - T1 and T2, and the tools for T3–T7.
2. **Publishing:**
   - formats and columns;
   - the GUI and CLI;
   - docs: `outputs.md` and `gui.md`. The glossary terms and ADR 0003 are already
     recorded with this specification.
3. **Release gate:** the new reporting is not released until the full T3 coverage
   validation passes, even if the earlier steps merge. T3 passes under the criterion
   of §10 item 5 (§8.1).

## 10. Decisions

1. **Estimator:** OLS with sandwich errors. Agreed on 2026-11-13, on the replicate
   evidence.
2. **Reporting interface:** the three quantities of §3.3, with (a) below. Agreed on
   2026-11-13.
   - (a) `intensity_err` becomes the profile σ in profile files, with the marginal σ
     in a separate column. This keeps downstream programs correct by default.
   - (b) `intensity_err` is the marginal σ everywhere, and downstream users must pick
     the profile column.

   §6a decides for (a): ChemEx reads one error column, and the marginal σ there
   overstates kinetic errors by up to 3×.
3. **When the noise-power check finds a mismatch:** warn only; on this dataset the
   measured correlation changes calibration by 5%. Replacing that axis's correlation
   with the measured one was not chosen. Agreed on 2026-11-13.
4. **Open, optional, outside the current implementation: a decisive test of
   processing.** If the raw FID of this dataset
   is available, process it with and without LP, at two apodizations, and compare
   the baseline-replicate scatter. That would measure the trade-off between LP,
   apodization and local fitting directly.
5. **The T3 coverage criterion:** criterion (b). Agreed on 2026-11-13.
   - **Scatter / stated** within 0.95–1.05, unchanged.
   - **95% coverage** within ±3 SE for a single quantity (each downstream parameter;
     the amplitude of an isolated peak on one plane), and within ±0.01 absolute for
     quantities pooled over planes (marginal σ, profile σ, ratios, R2eff).
   - **Why:** "within ±2 SE" per quantity is one test among many. Over T3's 270
     quantities about 12 fail it by chance; T3 finds 20, 6 of them a 0.1–0.2 point
     over-coverage of R2eff (§8.1). Pooling 29 planes makes the SE 0.0006, which
     resolves effects of no practical weight. The largest deviations seen were 2.5 SE
     and 0.0063.
   - **Not chosen:** (a) the criterion as first written, which T3 does not meet and
     which no change to the estimator or the reported quantities would meet; (c) as
     (b), with R2eff coverage reported but not gated.
   - The strict ±2 SE count stays in the record, as information.

## References

- A. C. Aitken (1935). On least squares and linear combination of observations.
  *Proceedings of the Royal Society of Edinburgh* 55, 42–48.
- E. Bartholdi and R. R. Ernst (1973). Fourier spectroscopy and the causality
  principle. *Journal of Magnetic Resonance* 11, 9–19.
- G. H. Golub and V. Pereyra (1973). The differentiation of pseudo-inverses and
  nonlinear least squares problems whose variables separate. *SIAM Journal on
  Numerical Analysis* 10, 413–432.
- J. A. Hausman (1978). Specification tests in econometrics. *Econometrica* 46,
  1251–1271.
- H. White (1980). A heteroskedasticity-consistent covariance matrix estimator and a
  direct test for heteroskedasticity. *Econometrica* 48, 817–838.
