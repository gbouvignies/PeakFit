# Cluster diagnostics in the cluster list: before and after (#217)

Visual record of W5 as rescoped on 17 November 2026: the cluster list finds every
per-cluster diagnostic that the cluster and peak pages show. Display only, apart
from the warning counts noted in [decisions](../decisions.md): no stored or
exported value changes.

## How it was run

- **Before:** `main` at 245961a. **After:** branch `dev/w5-findings-6db2d8`.
  PySide6/Qt 6.10.1, Fusion style, native Cocoa.
- **Machine:** macOS 27.2 on Apple silicon, built-in Retina display, DPR 2.0.
- **States:** the real window on the bundled `02-advanced-fitting` dataset after the
  GUI's own two-pass fit, through `tools/validate_gui_layout.Session`. At 1160×720,
  1280×800 and 1920×1080, light and dark: the full list after N, the open Show
  menu, the ↔ choice after N twice, and Profiles.
- **Review:** I looked at every capture. Two findings, both already on `main`, are
  below; neither comes from this change.

## What the example shows

| | Before | After |
| --- | --- | --- |
| Rows with a mark | 3 (◇ linewidth) | 16: 13 ↔, 3 ◇ |
| Show menu | All, Any warning (0), Any highlight (3), LW outside 0.5–2 × median (3) | the same, then **↔ Profile errors correlated (13)** |
| N in the full list | does nothing | says there is no warning and to choose a mark in Show |
| N with ↔ chosen | (no such choice) | 22N-H, 53N-H, … in list order; footer "13 of 121 clusters · N: next" |
| Toolbar ▲ count | 0 | 0 (no decomposition warning on this dataset) |

The marks agree with the pages: 53N-H's cluster page lists both members with ↔ and
Review reads "Profile errors ↔ correlated across planes · 53N-H, 147N-H". A row
shows one mark (▲, then ↔, then ◇), and Profiles tiles show the same one. The tests
cover a decomposition warning on a real degenerate fit (two peaks on one
position, "Ill-conditioned shapes"), which the example does not contain.

Figures in [cluster-diagnostics/](cluster-diagnostics/): the navigator before and
after, the Show menu and the ↔ choice (1280×800, light); the ↔ choice in the whole
window (1920×1080, dark); Profiles (1280×800, light).

## Findings already on main

- **A filtered Profiles view enlarges the window.** With a Show choice of 3 or 13
  clusters, Profiles raises the window's minimum width to 1253 or 1330 logical px
  at a 1160×720 window, and the window grows. Specification §2 says PeakFit never
  enlarges the window. Reproduced identically on `main` with "Any highlight". The
  captures show Profiles unfiltered for that reason.
- **Long cluster names elide at 1160×720.** With all 121 rows and the scroll bar,
  "158EN-H" reads "15…-H". The same on `main`; 1280×800 and wider show full names.
