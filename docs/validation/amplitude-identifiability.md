# Amplitude identifiability on the bundled data

`tools/validate_amplitude_identifiability.py` fits the bundled CEST dataset (131
planes, 166 peaks, two passes) and classifies every recorded member amplitude with
`engine/diagnostics/identifiability.py`. The record is
[amplitude-identifiability.json](amplitude-identifiability.json).

## Method

Per plane, a cluster's amplitudes solve Φᵀa ≈ d for the recorded (peak, sample)
shape matrix Φ. A coefficient aᵢ is estimable exactly when eᵢ has no component in
the null space of Φᵀ. The classification reuses `factorize_shapes`, which solved the
recorded amplitudes, and its conventions:

| Quantity | Convention | Source |
| --- | --- | --- |
| Null space | singular directions below ε·max(m, n)·s₀ | rank cutoff of `lstsq(rcond=None)` |
| Ill-conditioned directions | s ≤ √ε·s₀ | where the joint covariance is withheld |
| Non-zero projection of eᵢ | above ε·max(m, n) | the rank cutoff's relative tolerance |
| Near-collinear shapes | 1 − \|cos\| ≤ 10⁻⁴ | the existing position-adoption rule |

The classification is conditional on the recorded fitted lineshapes, like the
analytical standard errors: it concerns the linear amplitudes with positions and
linewidths held at their fitted values, and does not establish identifiability when
those nonlinear parameters vary.

A member with a null-space component is **not identifiable**. One projecting onto an
ill-conditioned direction, near-collinear with another member, or without a finite
σ is **not established**; only the last is still drawn (without an error bar).

## Results

| Fit | Identifiable | Not established | Not identifiable |
| --- | ---: | ---: | ---: |
| Default linewidth start | 166 | 0 | 0 |
| Former 20 Hz start | 160 | 6 | 0 |
| 147N-H moved to 115.155 / 8.3488 ppm, refit | 164 | 2 | 0 |

The six members not established at 20 Hz are exactly the three collapsed pairs
of [repeat-fit-collapse.md](repeat-fit-collapse.md): 36N-H/93N-H, 5N-H/76N-H and
53N-H/147N-H, each by the near-collinear rule (1 − |cos| between 3.3×10⁻⁸ and
4.4×10⁻⁷). Their cluster partners 75N-H, 10N-H and 105N-H stay identifiable. Every
collapsed cluster keeps full numerical rank, with s_min/s₀ between 1.3×10⁻⁴ and
4.7×10⁻⁴ (√ε ≈ 1.5×10⁻⁸), and every amplitude σ is finite. Calling these amplitudes
unidentifiable would be false; their recorded split, which reached ±6.3×10¹² for
the moved 147N-H refit, is withheld from the profile as not established.

## Synthetic controls

`tests/test_identifiability.py` uses compact-support shapes so that "no
projection" is exact:

- two identical shapes and an independent third: the pair is not identifiable,
  the third is identifiable although the matrix is rank deficient;
- s₃ = s₁ + s₂ with no collinear pair: all three not identifiable;
- full rank with s_min/s₀ ≈ 10⁻¹⁰: not established, never not identifiable;
- whenever `factorize_shapes` withholds the covariance, at least one member is not
  identifiable or not established.

Smooth shapes have tiny but real couplings: a Gaussian six widths from an
ill-conditioned pair projects 2.9×10⁻¹⁴ onto its weak direction, just below the
8.9×10⁻¹⁴ tolerance for 400 samples. Such a member is classified by the same
numerical-zero convention as rank itself.
