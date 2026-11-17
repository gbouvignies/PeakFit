# GUI product review, November 2026

A product-level review of the desktop GUI: features, workflows, information
architecture, vocabulary and consistency. It describes commit `6a5dc18`, main on
17 November 2026, and changed no code. This note keeps its findings and evidence.
The decisions that followed and the implementation plan are in
[#209](https://github.com/gbouvignies/PeakFit/issues/209).

Figures are cited as Fig. N and measurements as Mn; both are listed at the end.

## Method

- **Reading:** the design specification
  (`docs/PeakFit_GUI_Design_Specification_v2.0.md`), `docs/gui.md`,
  `docs/gui-design.md`, `GLOSSARY.md`, spec #178 and the
  [audit follow-up validation](gui-audit-followup.md).
- **Driving the real window** on the bundled `02-advanced-fitting` CEST dataset
  (M1) with `tools/validate_gui_layout.py` and a script on its
  `Session`, now `tools/capture_gui_review.py`. The states: the fit setup, a
  simulated failed first fit, the GUI's own two-pass fit, the overview, peak and
  cluster pages, cluster focus in Overlay and Residual, Profiles, the
  unexplained-signal list, the Outdated state after moving 22N-H, a refit, the Open
  dataset, Display, Analysis history and keyboard-shortcut dialogs, window sizes
  from 1160×720 to 1920×1080, and both appearances.
- **A read-only pass** over the engine, CLI and persistence for the inventory.
  Citations are to `src/peakfit/` at `6a5dc18`.

**Evidence.** [gui-product-review/](gui-product-review/) holds the cited figures
and `measurements.json`. `tools/capture_gui_review.py` wrote them in one run at
`6a5dc18` on a native macOS window: Cocoa, DPR 2.0, Apple silicon, one built-in
display (M16). That run repeats the review session's states, so run IDs differ
from the ones the review saw. The display caps the window's height, so the
1920×1080 window has the recorded height in M15. Sizes are Qt logical
pixels; inspector positions are in its content coordinates, as in the audit
follow-up.

### Not verified

- **Linux** (X11, Wayland) and Windows.
- **Screen readers:** nothing was heard through VoiceOver or another reader.
- **The native macOS menu bar:** the menus were read from code.
- **Moderated sessions with researchers.** Where a finding depends on how
  researchers work, #209 records the question and the user's decision.
- Large system fonts were not repeated; the audit follow-up validated them.

## What works and should stay

- **Three regions, spectrum in the centre.** The cluster is the unit of review and
  the peak the unit of editing, which maps onto navigator, centre and inspector.
- **Native slices, residual lanes and focus in place** (Fig. 11). This is the
  strongest scientific view: it carries the lineshape evidence that the coarse 2D
  view cannot.
- **Interaction states.** Hover, selection, pin and group commit are separate
  states. Undo is exact, and there is no Apply step.
- **Honest scientific state.** Current and Outdated, historical comparison,
  near-collinear vetoes, "not established" amplitudes, covariance-correct
  normalized ratios, no composite score, and highlights kept apart from warnings.
  The refit showed it: one edit produced new warnings and the changed measures
  (M13).
- **The whole-dataset fit** with its clustering preview before Fit (M1). A failed
  fit says that no result exists (M2), and the window keeps working.
- **Profiles as a centre view.** Its tiles follow the cluster list's sort and
  filter.
- **Accessibility foundations:** focus ring, contrast tokens, a minimum-size policy
  that never enlarges the window, and clipped-label detection in CI.

## Critical findings

1. **The model cannot be configured.**
    - One lineshape menu covers both axes. An explicit choice applies to every
      dimension, in the engine as well as the GUI (`engine/domain/spectrum.py:264`).
    - J coupling is reachable only as a doublet model on both axes, with fixed
      1–10 Hz bounds (`engine/lineshapes/templates.py:64`) and no GUI control. Those
      bounds cannot represent a one-bond ¹J(NH) of about 92 Hz.
    - Phase fitting, a manual noise level, an explicit clustering level, parameter
      rules, per-peak constraints and plane exclusion exist in the engine or CLI but
      not in the GUI, and MCMC is CLI-only. The fit setup shows four scientific
      choices and nothing else (Fig. 4). Workflows that need per-axis J, phase or
      sampling leave the GUI for a TOML file and the CLI, whose results the GUI
      cannot open.
2. **The inspector is three pages at once.** With nothing selected, one scrolling
   page holds the run report, the noise-check explanation, run changes, the
   unexplained-signal tool and the editable fit setup. After a fit the setup starts
   at the fold (M4); after a refit it starts well below it (M14, Fig. 17). Settings
   read as an appendix.
3. **Explanations crowd out results.**
    - The noise-check finding is a long label on the Analysis page and recurs inside
      the explanations of the peak and cluster pages (M5). "About these error bars"
      alone has the length in M8.
    - With their disclosures open, the peak and cluster pages are about two thousand
      pixels tall (M9, Figs. 9, 10).
    - The peak page puts editing first, then the profile, then explanation. The
      Fitted table starts near the bottom of the viewport, and its amplitude row is
      below the fold (M6, Fig. 8).
4. **Number formatting contradicts the specification where it matters most.** The
   amplitude row prints the value and its σ in full digits, without ×10ⁿ (M7).
   Spec §5 asks for σ to two significant figures and one shared exponent. χ²ᵣ has
   different precision in the peak page's Cluster review and in the cluster list,
   and Provenance prints the noise without the ×10ⁿ the fit setup uses.
5. **Switching Overlay ↔ Residual moves the plot.** Residual hides the lanes, and
   the 2D plot grows (M10, Figs. 11, 12), so every contour moves. Comparing by
   toggling needs identical geometry in every mode.
6. **Settings live in six places with no organizing rule.** Highlight thresholds
   change the cluster list's counts and marks, so they are review rules, not
   display.

    | Where | Holds | Kind |
    | --- | --- | --- |
    | Inspector · Fit setup | Lineshape, clustering factor, passes | Scientific model |
    | Fit setup · Advanced | Iterations, tolerance, solver label | Scientific model |
    | Display dialog (Fig. 5) | Contours, labels, stroke, residual contour, unexplained threshold, highlight rules | Display mixed with review rules |
    | View ▸ Slices | Slice sizes, gains, lanes | Display |
    | Profile quantity chip | Reference planes | Review |
    | Peak page · Starting values | Starts and bounds, read-only | Scientific model |

7. **Seven names for the ¹H axis.**

    | Where | Text |
    | --- | --- |
    | Plot axis | H · F3 (ppm) |
    | Position editor | H · F3 ppm |
    | Fitted and starts tables | δ H ppm, LW H Hz |
    | Peaks list | H ppm |
    | Cluster members | δ H, LW H, with the caption "δ in ppm · LW in Hz" |
    | History tables | Direct / ppm, Indirect / ppm |
    | Noise check, highlights | F2, (¹H) |

## Warnings

**Workflows**

- **Peaks and clusters are linked only after a fit, one at a time.** Before a fit
  the cluster evidence is a count and an optional region tint. After it the Peaks
  tab still shows only names and positions, with no cluster, warning or fitted
  state (Fig. 16).
- **The plane axis has no name or unit.** "Plane value" and an unlabeled profile
  axis stand for B₁ offsets in Hz (spec §5). The default plane is the
  off-resonance reference, which the profile reports as off scale.
- **The fit setup has dead and duplicated controls** (Fig. 4). Advanced holds a
  non-editable Solver row. The tolerance is in e-notation, which the spec forbids.
  An "N-step schedule (imported)" label cannot be reached, because nothing imports
  a configuration. "Show sampled fitting regions" and the focus bar's "Fitting
  region" pill control one state under two names.
- **The settings chip drops axis names** ("SP2 × SP2"), so it cannot show a
  per-axis model.
- **One quantity, two encodings.** Residual contours and lanes are vermilion and
  teal. Residual-mode slices draw the same residual in the neutral foreground
  (Fig. 12).
- **The corner readout is noisy:** full precision and mixed exponents over several
  lines, floating over the slice corner (Fig. 6).
- **Peak and cluster pages disagree.** The section is "Cluster review" on one and
  "Review" on the other, with different rows. The members table lacks amplitude ±
  σ, the main result (Fig. 10).
- **The peak page repeats itself** (Fig. 8). After a fit its three full-width edit
  fields repeat the fitted position at lower precision, and the profile note takes
  three lines for the bars' meaning, the noise check and off-scale points.
- **Small defects:** "▲ 1 clusters"; the legend breaks a name mid-word and leaves
  its negative entry unnamed; Profiles tile headers truncate measures that the list
  beside them already shows; fit times are in UTC; the comparison selector changes
  width when a checked button turns bold; the status strip stays empty at 1280 px
  and wider.

**Information architecture and naming**

- **The run line strings several states into one sentence** (M3, Fig. 6). The
  "Modified" label before it follows every fit, because the run is unsaved, which
  dilutes the word.
- **Unexplained signal is valuable and buried** (M11, Fig. 7). The strongest
  regions are most likely unpicked side-chain NH₂ peaks: the most actionable finding
  a fit produces. The inspector lists a few of them. There is no way to see the
  rest, no mark on the spectrum and no "Add peak here".
- **History is a data dump, not navigation** (Fig. 15). It shows ISO timestamps
  with microseconds, 32-hex stable IDs, full-precision floats, "Indirect / ppm"
  axes and overflowing tabs. An earlier run can be exported, but not made active,
  compared or overlaid. Run identity is an 8-hex hash.
- **Profiles shows the selected cluster three times:** in the list, as a tile, and
  as the cluster page with a larger copy of the profile (Fig. 13).
- **Vocabulary drift.** "Analysis" names the dataset page, a menu and the saved
  project. "Fit setup", "Fit settings" and "settings chip" name one thing. The
  Active run row "Model: Complete" reads like a lineshape model.

**Controls and layout**

- **One look, several behaviours.** Rounded pills serve as a menu (Raw amplitudes),
  a toggle (Show all points, Fitting region), a command (Focus cluster), navigation
  (‹ Spectrum) and a status summary (settings chip). Bordered buttons and links are
  mixed in the same way.
- **The keyboard reference is incomplete and alarming** (Fig. 2). Help ▸ Keyboard
  shortcuts is a bold message box with an alert icon. It omits Fit (⌘↩), E/O/R,
  N/Shift+N, [ ], Save and Escape to leave focus, and writes "Cmd/Ctrl" instead of
  the platform's glyphs.
- **The toolbar changes shape** with mode and width. Profiles hides the comparison
  and canvas tools, so Fit and the chip shift left. In a narrow window the chip
  becomes "Fit settings" (`gui/editor/workspace.py:486`).
- **The inspector's measure is narrow for prose and does not grow.** Extra window
  width goes to the spectrum (M15, Fig. 14).
- **Qualifiers that change a value's meaning** ("conditional on lineshapes", "not
  established") use the caption style, the smallest size (M17). The body size
  follows the macOS system font and suits a dense desktop tool.

## Feature inventory: what the GUI cannot reach

Capabilities a researcher cannot use from the GUI at `6a5dc18`. ◐ partial or
read-only · ○ engine or CLI only · ✕ missing everywhere.

| Capability | Engine, CLI or configuration | GUI | Status |
| --- | --- | --- | --- |
| Explicit lineshape per axis | The explicit name is repeated for every dimension (`engine/domain/spectrum.py:264`) | One menu for both axes | ✕ |
| J coupling (doublets) | Six doublet models; J start 5 Hz, bounds 1–10 Hz, fitted (`engine/lineshapes/templates.py:64`); fixing it needs a TOML constraint | Doublet entries in the one menu, so a doublet on both axes; no J start, bounds or fix | ◐ |
| J on one chosen axis, or large one-bond couplings | Not representable | – | ✕ |
| Phase fitting, per cluster and axis | `fit_phase`, `--phx`/`--phy` (`engine/lineshapes/base.py:69`) | – | ○ |
| Pseudo-Voigt η start or fix | Constraint patterns only | – | ○ |
| Statistical noise | Estimated, or `noise_level`/`--noise` | Value shown, no override | ◐ |
| Clustering threshold | Factor × noise, or explicit `contour_level`/`--contour` | Factor editable; an explicit level is shown and replaced on edit | ◐ |
| Manual cluster merge or split | – | – | ✕ |
| Fix, vary and bounds by axis and parameter | `[parameters].defaults` (`engine/domain/constraints.py:179`) | Accepted from a project, but nothing creates them; starts and bounds read-only | ○ |
| Per-peak constraints, position windows | `parameters.peaks`, `position_window(s)` | Rejected by the GUI preparation (`gui/fitting.py:278`) | ○ |
| Multi-step fit schedules | `[[fitting.steps]]` | A read-only label that nothing can reach | ◐ |
| Plane exclusion | `exclude_planes`, `--exclude` | Rejected (`gui/fitting.py:303`) | ○ |
| Worker count | `--workers` | – | ○ |
| More than two spectral axes | Readers up to F4 | Two axes | ○ |
| Fitted J, η and phase values | In results | Generic rows; phase is per cluster and absent from the peak page | ◐ |
| Unexplained signal beyond the first rows | GUI-side, read-only | The first rows per plane, then a count | ◐ |
| Replicate-based σ check | `peakfit replicates` | – | ○ |
| MCMC sampling and diagnostics | `peakfit mcmc`, `plot mcmc` | Spec §10 designs it | ○ |
| CEST, CPMG and intensity plots | `peakfit plot …` | Profiles PDF only | ○ |
| Export formats, `simulated.npy` | `-f`, `[output]` | The detailed-evidence checkbox only | ◐ |
| Make an earlier run active; compare two chosen runs | Ten runs' evidence retained | Read-only history and export; changes against the parent only | ✕ |
| Open a CLI result directory | Same writer | – | ○ |
| Plane-axis name and unit (spec §5) | – | "Plane value", no unit | ✕ |

Everything else the engine offers for model setup and review is in the GUI:
automatic lineshapes per axis with their source, passes, iteration limit and
tolerance, Overlay and Residual with slices and lanes, fitted values with marginal
and profile σ, the correlation matrix, every warning family, reference-normalized
profiles, and save, reopen, relink and recovery.

A side finding outside the GUI: with `--config`, `peakfit fit` ignores
`--lineshape`, `--refine`, `--phx`, `--phy` and `--exclude`
(`cli/commands/fit_setup.py:38`). It is tracked separately from #209.

## Measurements

From `gui-product-review/measurements.json`. Line counts are rendered lines of a
label at the inspector's width.

| | Measurement | Value |
| --- | --- | --- |
| M1 | Dataset; preparation before the first fit | 166 peaks · 131 planes; 121 clusters · 6 158 samples per plane |
| M2 | Failed first fit: run line | ▲ Failed · no result yet |
| M3 | Run line after the fit | Run 11f05f4b · Current · ▲ Noise check (F2) · all 121 clusters usable |
| M4 | Fit setup top / inspector viewport, after the fit | 626 / 687 px |
| M5 | Lines of the labels with the noise-check finding: Analysis, peak, cluster page | 18, 37, 39 |
| M6 | 53N-H Fitted table top, bottom / viewport | 616, 724 / 687 px |
| M7 | 53N-H amplitude row | Amplitude plane 1 3103917717 ± 132348379 |
| M8 | About these error bars, lines | 37 |
| M9 | Whole pages, disclosures open: peak 53N-H, cluster 22N-H | 1989, 1916 px |
| M10 | Focus 22N-H: 2D plot, Overlay → Residual | 384×350 → 428×394 |
| M11 | Unexplained signal on plane 1: regions, rows listed, strongest (σ_stat, ppm) | 125, 8, +33.2 at F3 7.593, F2 111.79 |
| M12 | Run line after moving 22N-H | Run 11f05f4b · Outdated · ▲ Noise check (F2) · all 121 clusters usable |
| M13 | Changes from parent run after the refit | ▲ Near-collinear shapes now reported · ▲ Uncertainty unreliable now reported · ◇ Amplitude correlation now reported · ◇ Linewidth outlier now reported · χ²ᵣ 3.04 → 5.04 · \|r\|max 9.2 → 10.7 σ · \|ρ\| 0.41 → 1.00 · With ▲ warnings 0 → 1 |
| M14 | After the refit: Analysis page height, fit setup top / viewport | 1349, 1063 / 687 px |
| M15 | Window sizes (logical): inspector width, overview plot | 1160×720: inspector 314, plot 394×404; 1280×800: inspector 320, plot 482×468; 1440×900: inspector 320, plot 614×546; 1920×921: inspector 320, plot 1008×562 |
| M16 | Screen (logical) | 1512×982 |
| M17 | Type scale: body, small, caption | 13, 12, 11 px |

## Figures

Native macOS, 1280×800 logical @ DPR 2.0 and the light appearance unless stated.
Window captures are downscaled to 1600 px wide; inspector pages are full height.

1. [Open dataset](gui-product-review/dialog-open-dataset.webp): an explicit
   peak-axis order is required.
2. [Keyboard shortcuts](gui-product-review/dialog-shortcuts.webp).
3. [Before the first fit](gui-product-review/fit-setup.webp): the fit setup leads
   the inspector; the cleanest state in the product.
4. [Whole fit setup, Advanced open](gui-product-review/page-fit-setup.webp).
5. [Display dialog](gui-product-review/dialog-display.webp): contour display mixed
   with review and diagnostic thresholds.
6. [Overview after the first fit](gui-product-review/overview.webp): run line,
   settings chip, corner readout, noise-check explanation, the setup at the fold.
7. [Analysis page with the unexplained-signal list](gui-product-review/page-analysis-unexplained.webp).
8. [Peak page of 53N-H](gui-product-review/peak-53N-H.webp).
9. [Whole peak page of 53N-H, disclosures open](gui-product-review/page-peak-53N-H.webp).
10. [Whole cluster page of 22N-H](gui-product-review/page-cluster-22N-H.webp).
11. [Focus on 22N-H, Overlay](gui-product-review/focus-overlay.webp).
12. [Focus on 22N-H, Residual, dark](gui-product-review/focus-residual-dark.webp).
13. [Profiles with 22N-H selected](gui-product-review/profiles.webp).
14. [Overview at 1160×720](gui-product-review/overview-1160x720.webp).
15. [Analysis history](gui-product-review/dialog-history.webp).
16. [22N-H moved by +0.02 ppm in ¹H](gui-product-review/outdated-peak.webp): the
    run is Outdated; the historical overlay and the × mark work as designed.
17. [Analysis page after the refit](gui-product-review/page-analysis-refit.webp).

## Reproducing the evidence

```bash
uv run python tools/capture_gui_review.py --output <directory outside the repository>
```

It runs on a desktop, or headless with `QT_QPA_PLATFORM=offscreen` (offscreen
captures are not native evidence), in a few minutes including the two fits. The
record was made by copying the tool into a checkout of `6a5dc18`, which predates
it. On a later commit the tool describes that commit, and the GUI work planned in
#209 changes the figures by design.
