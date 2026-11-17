# GUI consistency: before and after (#211)

Visual and geometric record of issue #211: one number formatter, one axis name,
one look per kind of control, stable plot geometry and the small defects the
product review found. Display only: no stored or exported value changes.

Sizes are Qt logical pixels with the device-pixel ratio, written
`W×H logical @ DPR`; the recorded size is what the window manager gave.

## How it was run

- **Before:** `main` at e4556a0. **After:** branch `dev/w1-consistency-3d5e66` at
  29d5732 (the review fixes included). PySide6/Qt 6.10.1, Fusion style.
- **Machine:** macOS 27.2.1 on Apple silicon, built-in Retina display, DPR 2.0. The
  screen clamps a 1920×1080 request to 1920×921.
- **Tool:** `uv run python tools/validate_gui_layout.py --output <dir> --sizes 1160x720
  1280x800 1920x1080` (both appearances, native Cocoa), once per commit: 72 captures
  each, from fit setup to Profiles, after the GUI's own two-pass fit of the bundled
  `02-advanced-fitting` dataset (131 planes, 166 peaks, 121 clusters).
- **Review:** I looked at every capture state in both appearances and at all three
  sizes. Three problems found on the way were fixed before the final run (below).

Evidence in [gui-consistency/](gui-consistency/): `measurements-before.json`,
`measurements-after.json` (local output paths replaced by `<output>`), and four
states before and after: focus in Overlay and in Residual (1280×800, light), the
peak page of 53N-H (1280×800, light) and the cluster page of 22N-H (1160×720, dark).

## Stable plot geometry

Light and dark give identical geometry at every size. Lanes are the row lane's
height.

| Recorded size | Focus Overlay: 2D plot, lane | Focus Residual: 2D plot, lane | Overview, every mode |
| --- | --- | --- | --- |
| Before, 1160×720 @ DPR 2.0 | 360×314, 24 | **360×338, none** | 384×404 |
| After, 1160×720 @ DPR 2.0 | 360×318, 24 | 360×318, 24 | 384×404 |
| Before, 1280×800 @ DPR 2.0 | 384×350, 44 | **428×394, none** | 482×468 |
| After, 1280×800 @ DPR 2.0 | 384×354, 44 | 384×354, 44 | 482×468 |
| Before, 1920×921 @ DPR 2.0 | 858×434, 44 | **902×478, none** | 1008×562 |
| After, 1920×921 @ DPR 2.0 | 858×440, 44 | 858×440, 44 | 1008×562 |

- **Overlay ⇄ Residual no longer moves the plot.** Before, Residual hid the lanes and
  the plot grew by 44 px each way. Now the lanes keep their place in every mode once
  a run exists. In Residual they repeat the slice's residual; in Experimental their
  room stays empty, since Experimental draws no fitted contribution.
- **Focus plot 4–6 px taller.** The focus bar is slightly shorter now that
  "‹ Spectrum" is a link and "Fitting regions" a checkbox, not two 20 px chips.
- **Covered by test.** `tests/test_gui_consistency.py` asserts a pixel-identical
  plot rectangle across Experimental, Overlay and Residual, in the overview and in
  focus, at 1160×720, 1280×800 and 1920×1080 (offscreen).

## What changed on screen

| Before | After |
| --- | --- |
| Amplitude plane 1: `3584878297 26438558` | Amplitude (× 10⁹): `3.585 0.026`; the caption starts "Plane 1 · ± σ is the marginal σ …" |
| Axis named `H · F3 (ppm)`, `H · F3 ppm`, `δ H ppm`, `H ppm`, `δ H`, `Direct / ppm`, `F2` | `H F3` everywhere, with `(ppm)` or `(Hz)` where a unit is due; `N F2` in the run line and noise check |
| χ²ᵣ `3` on the peak page, `3.00` in the list | `3.00` in both |
| Readout `I 7.30759 ×10⁵`, coordinates to 4 and 3 decimals | `I 7.308 × 10⁵`, `7.989 · 120.199 ppm` |
| Noise `1.192 ×10⁴` in the setup, `11922.5 a.u.` in provenance; tolerance `1e-08` | `1.192 × 10⁴` in both; tolerance `1 × 10⁻⁸` |
| Fitted `2026-11-17 11:00 UTC` | `2026-11-17 12:00 CET` |
| `▲ 1 clusters` | `▲ 1 cluster` |
| Rounded chips for a menu, a toggle, a command and navigation | Raw amplitudes ▾ (menu), Show all points and Fitting regions (checkboxes), Focus cluster and Export PDF… (commands), ‹ Spectrum (link) |
| Bold checked segment, so the segments' widths jumped | Same weight; the fill marks the choice |
| Shortcuts: a fixed message box without Fit, E/O/R, N, [ ] or Save | Generated from the menus (every key, platform notation) plus the single keys |
| Members table: `δ H · LW H`, "δ in ppm · LW in Hz" | The axis name over its columns, `δ · LW` under it, "δ (ppm) · LW (Hz)" |

## Found while looking, and fixed

- **The run line cut a word at 1160 px.** It read "… all 121 clusters u…". It now
  drops whole phrases when narrow, as the slice caption does, and "all clusters
  usable" no longer repeats the count (the tooltip and the Clusters tab give it).
  No label is clipped in any capture, before or after.
- **The inspector scrolled sideways.** A one-line "Amplitude plane 1 (× 10⁹)" and the
  unit headers of the members table made the peak page 12 px and the cluster page
  24 px wider than a 310 px inspector. The plane moved into the caption and the units
  under the table. Both pages now fit (minimum widths 280 and 298 px).
- **The Fitted table slid below the fold.** That title wrapped to two lines and the
  profile's menu button now has standard control height. With the shorter title,
  158EN-H's table ends at 685 px against the 687 px viewport at 1280×800, so it is
  visible again.

| Fitted table bottom / inspector viewport | 158EN-H | 53N-H | 22N-H |
| --- | --- | --- | --- |
| Before, every size | 681 | 724 | 740 |
| After, every size | 685 | 728 | 744 |
| Viewport: 1160×720 / 1280×800 / 1920×921 | 607 / 687 / 808 | | |

The 4 px is the profile's Raw amplitudes ▾ button, now 26 px high like every
command; it was a 20 px chip.

## Intentional choices to review

- **ppm at 3 decimals everywhere.** This includes unfocused coordinate fields, which
  showed one decimal beyond the digital resolution (four in ¹H). Focus still reveals
  full precision.
- **σ to two significant digits.** The issue's example shows three, "(3.104 ±
  0.132)"; its rule says two, and the rule is what the formatter applies.
- **Lanes in Residual repeat the slice.** This follows the plan's default: lanes in
  every mode once a run exists.
- **Plane values without a label read "Plane value".** The issue says "Plane"
  otherwise. Imported values are not indices, so only a series without values reads
  "Plane".
- **Nucleus as recorded.** The bundled headers say "H" and "N", so the axes read
  "H F3" and "N F2"; "¹H F3" needs a header that gives the mass number. Tests cover
  ¹H–¹⁵N, ¹H–¹³C, homonuclear ¹H–¹H and three-axis names.
- **E/O/R, [ ] and Shift+N.** The specification (§5) and `docs/gui.md` promised these
  keys but they were not bound. They are now, so the shortcuts list is complete and
  true.

## Not verified

- **Linux, DPR 1 and large system fonts.** Same gaps as the
  [configuration-matrix note](gui-audit-followup.md).
- **Native 1920×1080 at full height.** The screen clamps it to 921 px; the offscreen
  geometry test covers the full height.
- **The failure page scrolls sideways.** Its Log row shows a long unbreakable path.
  This is pre-existing (identical before) and not addressed here.
