# GUI audit follow-up: validation across the configuration matrix

Validation record of the GUI audit follow-up (spec #178, piece 7, #192): the
selection, visibility, proportional-slice and adaptive-Profiles changes measured on
the real window at the agreed window sizes, appearances, scale factors and font
sizes. It also closes the visual acceptance that #190 (slice shares), #191 (tile
limits) and #185 (type sizes) left open.

Every size is in Qt logical pixels with its device-pixel ratio (DPR), written
`W×H logical @ DPR`. A requested size is what the tool asked for; the recorded size
is what the window manager gave and is authoritative.

## How it was run

- Code: integration branch `gui-audit-followup` at b66691a plus this ticket's tool
  changes. PySide6/Qt 6.10.1, Fusion style.
- Machine: macOS 27.2 on Apple silicon, one built-in Retina display, 1512×982
  logical @ DPR 2.0 (available area 1512×949 below the menu bar). No external monitor.
- Tool: `uv run python tools/validate_gui_layout.py` (see `docs/development.md`,
  "GUI acceptance captures"): the bundled `02-advanced-fitting` CEST dataset
  (131 planes, 166 peaks), a simulated first-fit failure, then the GUI's own
  two-pass fit (121 clusters). `--body-px 17` simulates a larger system font.
- Native runs use the Cocoa plugin (`qt_platform: cocoa`). Runs marked
  *offscreen* use `QT_QPA_PLATFORM=offscreen` (and `QT_SCALE_FACTOR` for 1.25 and
  1.5): they measure Qt's layout at sizes and scale factors this screen cannot
  provide, but they are **not** native acceptance evidence (no window manager, no
  native font rendering, a Qt menu bar inside the window).
- Spot checks of #180, #181, #183 and #188 used a scratch script driving the same
  session (results in [spot-checks.json](gui-audit-followup/spot-checks.json)).

Evidence in [gui-audit-followup/](gui-audit-followup/): `measurements.json`
(native, 168 captures: 7 sizes × 2 appearances × 12 states),
`measurements-offscreen-dpr1.json`, `measurements-offscreen-dpr1.25.json`,
`measurements-offscreen-dpr1.5.json`, `measurements-body17.json` and a few
representative WebP captures. The full capture sets (about 450 images) were kept
outside the repository, as the tool requires.

## Matrix coverage

| Configuration | Platform / plugin | Recorded size | Status |
| --- | --- | --- | --- |
| 1280×800, light and dark | macOS, cocoa | 1280×800 logical @ DPR 2.0 | Verified |
| 1366×768, light and dark | macOS, cocoa | 1366×768 logical @ DPR 2.0 | Verified |
| 1440×900, light and dark | macOS, cocoa | 1440×900 logical @ DPR 2.0 | Verified |
| 1160×720 (target), light and dark | macOS, cocoa | 1160×720 logical @ DPR 2.0 | Verified |
| 1512×982, light and dark | macOS, cocoa | 1512×921 logical @ DPR 2.0 | Verified at the clamped height |
| 1920×1080, light and dark | macOS, cocoa | 1920×921 logical @ DPR 2.0 | Width only: the screen clamps the height |
| 2560×1440, light and dark | macOS, cocoa | 2560×921 logical @ DPR 2.0 | Width only: the screen clamps the height |
| 1280×800, 1366×768, 1920×1080, 2560×1440 at DPR 1 | offscreen | as requested @ DPR 1.0 | Layout geometry only |
| 1280×800, 1920×1080 at DPR 1.25 and 1.5, light | offscreen, `QT_SCALE_FACTOR` | as requested @ DPR 1.25 / 1.5 | Layout geometry only |
| Larger system font (body 17 px), 1280×800 and 1512×982, light and dark | macOS, cocoa | 1280×800 / 1512×921 logical @ DPR 2.0 | Verified (simulated font) |

### Not verified

| Configuration | Why not | What it needs |
| --- | --- | --- |
| Native DPR 1.0 | The only display is a 2× Retina panel | A 1× external monitor on macOS, or Linux at 100 % |
| Native 1920×1080 and 2560×1440 at full height | The 982-px-tall screen clamps the window to 921 px | A monitor at least 1080 (or 1440) logical px tall |
| Linux X11 and Wayland, any size | No Linux desktop on this machine | A Linux session (e.g. the Ubuntu workstation) running the same tool natively, `xcb` and `wayland` plugins |
| Linux fractional scaling 1.25 and 1.5 | Offscreen `QT_SCALE_FACTOR` exercises Qt's scaling of the layout, not a compositor's fractional scaling or font hinting | Ubuntu with GNOME fractional scaling at 125 % and 150 % (X11 and Wayland) |
| Linux large system font | `--body-px 17` installs the type scale a 17 px system font would produce; it does not change the platform font, so native widgets that read the system font directly were not exercised | Ubuntu with Large Text or a 13–14 pt interface font |
| Screen-reader output, real-mouse gestures, PDF output | Out of scope here (spec #178) | Moderated walkthrough |

## Measurements, native (macOS, cocoa, DPR 2.0, body 13 px)

Light and dark give identical geometry at every size; one row per size. Spectral
sizes are width×height; the row slice shares the plot's width and the column slice
its height, so only the free dimension is quoted.

| Recorded size | Overview: 2D plot, row slice h, column slice w | Focus Overlay: 2D plot, row h, column w (lanes 44) | Focus Residual: 2D plot |
| --- | --- | --- | --- |
| 1160×720 logical @ DPR 2.0 | 384×404, 134, 100 | 360×314, 174, 100 | 360×338 |
| 1280×800 logical @ DPR 2.0 | 482×468, 152, 122 | 384×350, 198, 176 | 428×394 |
| 1366×768 logical @ DPR 2.0 | 552×442, 144, 138 | 448×328, 190, 200 | 492×372 |
| 1440×900 logical @ DPR 2.0 | 614×546, 174, 152 | 502×420, 228, 218 | 546×464 |
| 1512×921 logical @ DPR 2.0 | 672×562, 178, 164 | 556×434, 236, 238 | 600×478 |
| 1920×921 logical @ DPR 2.0 | 1008×562, 178, 238 | 858×434, 236, 344 | 902×478 |
| 2560×921 logical @ DPR 2.0 | 1532×562, 178, 352 | 1332×434, 236, 510 | 1376×478 |

| Recorded size | Fitted table bottom / inspector viewport: 158EN-H, 53N-H, 22N-H | Profiles: viewport w, columns, tile |
| --- | --- | --- |
| 1160×720 logical @ DPR 2.0 | 681, 724, 740 / 607 (none fully visible) | 537, 2, 267×192 |
| 1280×800 logical @ DPR 2.0 | 681, 724, 740 / 687 (158EN-H visible) | 644, 3, 214×156 |
| 1366×768 logical @ DPR 2.0 | 681, 724, 740 / 655 (none fully visible) | 730, 3, 242×174 |
| 1440×900 logical @ DPR 2.0 | 681, 724, 740 / 787 (all visible) | 804, 3, 267×192 |
| 1512×921 logical @ DPR 2.0 | 681, 724, 740 / 808 (all visible) | 876, 4, 218×157 |
| 1920×921 logical @ DPR 2.0 | 681, 724, 740 / 808 (all visible) | 1284, 6, 213×156 |
| 2560×921 logical @ DPR 2.0 | 681, 724, 740 / 808 (all visible) | 1924, 9, 213×156 |

Fitted positions are in inspector content coordinates (the convention of the
14 November 2026 audit, whose 1280×800 baseline was 709/817/820 against 687). At
1280×800 the multi-member pages need 37–53 px of scrolling, which spec #178 accepts.

### Offscreen: full-height large sizes and scale factors

| Recorded size | Overview: 2D plot, row h, column w | Focus Overlay: 2D plot, row h, column w | Fitted bottom / viewport | Profiles columns, tile |
| --- | --- | --- | --- | --- |
| 1280×800 logical @ DPR 1.0 | 482×450, 146, 122 | 384×334, 192, 176 | 681, 724, 740 / 665 | 3, 214×156 |
| 1366×768 logical @ DPR 1.0 | 552×426, 140, 138 | 446×312, 182, 200 | 681, 724, 740 / 633 | 3, 242×174 |
| 1920×1080 logical @ DPR 1.0 | 1006×668, 208, 238 | 856×530, 276, 344 | 681, 724, 740 / 945 | 6, 213×156 |
| 2560×1440 logical @ DPR 1.0 | 1532×950, 288, 352 | 1330×782, 384, 510 | 681, 724, 740 / 1305 | 9, 213×156 |
| 1280×800 logical @ DPR 1.25 and @ DPR 1.5 | 482×450, 146, 122 | 384×334, 192, 176 | 681, 724, 740 / 665 | 3, 214×156 |
| 1920×1080 logical @ DPR 1.25 and @ DPR 1.5 | 1006×668, 208, 238 | 856×530, 276, 344 | 681, 724, 740 / 945 | 6, 213×156 |

Offscreen heights are 16–22 px smaller than native at the same size because the
offscreen window draws its own menu bar. Logical geometry does not depend on the
scale factor: DPR 1.0, 1.25 and 1.5 give identical layouts, as they should.

### Contrast (WCAG 2.2 ratios, both appearances)

| Pair | Light | Dark |
| --- | --- | --- |
| text on panel / chrome / plot | 15.30 / 14.89 / 15.82 | 13.07 / 13.65 / 14.34 |
| secondary on panel / chrome | 5.33 / 5.19 | 7.28 / 7.60 |
| faint on panel / chrome | 5.08 / 4.95 | 5.12 / 5.34 |
| outline (inputs, checkboxes) on panel / chrome / plot | 3.53 / 3.44 / 3.65 | 3.34 / 3.49 / 3.67 |
| border (buttons, tooltips) on panel / chrome | 1.50 / 1.46 | 1.68 / 1.76 |

Secondary and faint text reach 4.5:1 and editable-control boundaries 3:1 in both
appearances. Button borders stay quiet; buttons are identified by their fill and
label, not by the border.

## Constants after visual acceptance

**Slice shares (#190): kept.** Overview 22 % of the view's height for the row slice
and 18 % of its width for the column slice; cluster focus 30 % / 26 %.

- At 1280×800 the overview gives a 482×468 2D plot with a 152 px row slice and a
  122 px column slice: the dense 166-peak map stays legible and both slices resolve
  individual peaks. At 1920×1080 (offscreen, full height) they are 208 and 238 px.
- In focus at 1280×800 the plot is 384×350, only 24×50 px above its 360×300 floor,
  with 198/176 px slices plus 44 px lanes. Larger focus shares would push the plot
  to its floor at the most common laptop size; smaller ones would bring the slices
  under 200 px there. At 1160×720 the column slice already compresses to 100 px
  and the plot sits at its floor, as the space policy intends.
- At 2560×1440 (offscreen) focus gives 1330×782 for the plot, 384 px and 510 px for
  the slices: all three remain useful; nothing is wasted on empty margins.

**Tile limits (#191): kept.** Minimum width 210 px, maximum 320 px, height 0.72 ×
width, at least 156 px.

- On screen the tile is 213–267 px wide (3 columns at 1280×800, 4 at 1512, 6 at
  1920, 9 at 2560); the maximum applies only to one-column widths of 321–419 px,
  and the height floor sets the height up to about 217 px wide.
- CEST tiles, raw and normalized to plane 1: the narrow dips of 32N-H and 39N-H
  near the edge of the offset range, the overlapping two-member clusters
  (53N-H/147N-H, 128N-H/138N-H) and the four-member 22N-H cluster remain
  distinguishable at 214×156. The off-resonance reference plane (−12000 Hz) lies
  outside the plotted offset range; the inspector reports it as "1 point off scale
  (plane 1)".
- Synthetic short series (`cpmg_like_dataset`, 20 planes with an outlier;
  `decay_dataset`, 10 planes) drawn natively at 210, 260 and 320 px wide: the
  outlier and both members are clear at 210 px; wider tiles only add white space.
- The multi-member header elides the measures at 213 px ("χ²ᵣ 3.04 · |r|max 9.2 · …"),
  with the full text in the tooltip, as decided in #191.

## Larger system font (#185), simulated body 17 px

`--body-px 17` (caption 14, small 16, title 20 px), 1280×800 and 1512×921 logical
@ DPR 2.0, both appearances. Text grows everywhere, and the spectral and Profiles
geometry stays within its rules (overview plot 472×460, focus plot 374×342 at
1280×800; Profiles 3 columns of 214×156). Fixed sizes that clip or crowd text at
1280×800:

1. **Inspector width.** The inspector keeps its 300–320 px default width, so its
   content is cut at the right edge without a scroll bar: the members table loses
   its LW N column, the Name/ppm fields and "Review cluster ›" are cut, and long
   caption lines lose their last characters.
2. **Navigator cluster column.** Cluster names are elided in the middle ("15…-H",
   "16…-H") because the column width does not follow the font.
3. **Focus lane caption.** "Lanes ±8 σ_stat · 8 data samples clipped · 9 model
   samples clipped" needs 428 px in a 374 px label and is cut without elision.
4. **Profile plot axis.** The fixed 40 px left axis lets the "×10⁹" exponent
   label touch the tick labels.
5. **Peak page.** The Fitted table moves down by about 70–90 px (bottoms 751/806/826
   against a 687 px viewport).

The run line in the toolbar is elided with an ellipsis by design and is not a
clipping finding. Row height (28 px) and the 156 px tile minimum held at 17 px:
navigator rows and tile headers were not cut vertically. At 1512×921 the clipping
detector reports nothing.

## Spot checks (1280×800 logical @ DPR 2.0, light and dark)

- **#180 Profiles keeps the selection.** With 53N-H selected, opening Profiles and
  returning both kept 53N-H selected; its tile was marked.
- **#181 Centring within the data.** At full spectrum (data 6.50–10.50 ppm ¹H,
  110.70–129.34 ppm ¹⁵N), choosing the corner cluster 126EN-H from the list left
  the view unchanged. At a quarter-size zoom the same choice gave
  9.50–10.50 × 110.70–115.36 ppm: centred as far as the data allow, clamped to the
  extent on both axes ([capture](gui-audit-followup/edge-cluster-zoomed-1280x800-light.webp)).
  The aliased 39N-H (¹⁵N 133.42 ppm) was centred at its displayed, folded position.
- **#183 Focus ring.** Fit, a mode segment, the navigator list, the 2D canvas, the
  Review cluster link and the plane field each show the ring in both appearances;
  on Fit it sits outside the accent fill; on lists and the canvas it follows the
  view's edge ([navigator](gui-audit-followup/focus-navigator-1280x800-light.webp)).
  On a joined mode segment the ring overlaps half of each neighbour's border but
  stays whole and readable. A mouse click also shows the ring (Qt gives focus on
  click; there is no keyboard-only "focus-visible" distinction). Pressing the link
  with the mouse shows the ring, then the page changes; no transient 2 px flash was
  seen. The keyboard walk Fit → Clusters tab → cluster list → 2D canvas →
  inspector worked, but **Tab from the inspector crashed the application** (below).
- **#188 Not-in-model mark.** After placing a peak on a fitted analysis the run
  becomes outdated, the legend reads "☆ not in this model" and the new peak carries
  the ☆ marker; the glyph renders in both appearances
  ([capture](gui-audit-followup/added-peak-1280x800-dark.webp)). Clipped samples
  are counted in words in the slice caption.
- **#186 Failure state.** A first-fit failure reads "▲ Failed · no result yet" in
  the error style in the run line and leads the Analysis page with Stage, Reason and
  Copy details ([capture](gui-audit-followup/failure-1280x800-dark.webp)). The log
  row is absent in these captures because the tool builds the window without
  `launch`, which installs the log file; the log path itself was not checked here.
- **Both appearances.** Every state was captured in light and dark at every native
  size; geometry is identical between them.

## Issues found (to be filed as tickets, not fixed here)

1. **Crash: Tab out of the inspector segfaults on macOS.** With keyboard focus on
   the inspector scroll area, Tab crashes PeakFit (SIGSEGV in
   `QWidget::raise_` called from `FocusRing._place`). Reproduced natively four
   times, with and without a fit; not reproduced offscreen. The stack shows the
   ring being re-placed by a Move/Resize event of a watched widget while
   `FocusRing._follow` is still reparenting the ring (`ring.setParent(host)`);
   suppressing placement during the reparent avoided the crash in a local
   experiment (not committed). One run of the full native matrix also ended with a
   segmentation fault after the 1160×720 focus captures, without a trace; a rerun
   with fault handler enabled completed. Severity: high (data loss for unsaved work).
2. **Larger system font: inspector content clipped at the default width** (finding 1
   above), plus the navigator name column (2), the focus lane caption (3) and the
   profile axis width (4).
3. **Failure log row not covered by the acceptance tool**: the tool should install
   the log file as `launch` does, so captures show the log path.
4. **Focus ring after mouse clicks** (#183): the ring appears on mouse focus as
   well as keyboard focus. Spec §12 does not say whether this is intended.
