# 0004: Per-axis lineshape models

**Date:** 2026-11-17 · **Status:** accepted · **Issue:** #212 (W2b of #209)

## Decision

**Model.** Each spectral axis has three independent properties:
- **family:** `auto`, `gaussian`, `lorentzian`, `pvoigt`, `sp1`, `sp2` or `no_apod`;
- **splitting:** none, or a doublet with a J start, minimum, maximum and fitted or
  fixed state;
- **phase:** fitted or not.

The combined names (`lorentzian_doublet` and the others) are gone. `auto` still
takes each axis's family from its NMRPipe apodization.

**Rules.**
- **J start.** A doublet needs an explicit J start in Hz. There is no default and no
  per-nucleus default.
- **J bounds.** Without explicit bounds, J may move within [J/2, 2J]
  (`DOUBLET_RANGE`). This range is symmetric on a log scale, so it suits a 7 Hz ³J
  and a 92 Hz ¹J alike. The start must lie within its bounds.
- **Fixed J.** A fixed J (`vary = false`) is held at its start. As for any fixed
  parameter, a fit step's `vary` pattern can release it, and a
  `[parameters.defaults]` rule such as `*.F2.j` still applies after the axis model.
- **Phase.** Phase needs a phased family (`sp1`, `sp2`, `no_apod`). Phase on any
  other family is refused, including a family that `auto` detected. Before, it was
  silently ignored.
- **Axes.** An axis named in the configuration but absent from the spectrum is
  refused.

**Engine.** `FitConfig` holds the default `family` and `axes`, a map from axis
label to `AxisModel`. `axis_lineshapes(config, spectra)` resolves them, with
`auto` applied, into one `AxisLineshape` per axis. Each shape takes the model of
its own axis. A doublet's J spec (start, bounds, freedom) comes from that model.
`AxisLineshape` is the one model that the CLI, the GUI and stored results build.

**TOML.**

```toml
[fitting]
family = "auto"                 # every spectral axis

[fitting.axes.F2]               # one axis; unset fields keep the defaults
family = "sp1"
phase = true
doublet = { j = 92.0 }          # bounds [46, 184], fitted

[fitting.axes.F3]
doublet = { j = 7.0, min = 3.0, max = 12.0, vary = false }
```

**CLI.** Each option repeats, once per axis:
- `--family NAME` sets every axis; `--family F2=NAME` sets one.
- `--doublet F2=J` or `--doublet F2=J,MIN,MAX` adds a doublet, in Hz.
- `--fix-j F2` holds that doublet's J at its start.
- `--phase F2` fits that axis's phase.

These replace `--lineshape`, `--phx` and `--phy`. On a pseudo-3D, `--phx` was F3
and `--phy` was F2.

**GUI (until W4).** The lineshape control sets one family on every axis. It keeps
the doublets and phases of an imported TOML, and shows them in the resolved
per-axis summary. When a TOML gives an axis its own family, the control reads
**Per axis** until the user picks a family. The GUI has no doublet choice before
W4, which adds per-axis controls with J entry.

**Storage.** One commit changes three formats:
- **Scientific archive 13.0.0:**
    - its recorded configuration has `family` and `axes`;
    - each model descriptor records `family` and `doublet`;
    - reopening rebuilds the shapes from the recorded configuration and
      calibration, then checks them against the descriptors.
- **Fit JSON 10.0.0:** its recorded configuration changes the same way.
- **Analysis project 16.0.0:** run records drop `shape_names`. Each run's resolved
  models come from its recorded configuration and axes.

## Why

- **The combined names cannot express the agreed model.** Decision 1 of #209 asks
  for a doublet on one axis with its own J range. One name repeated on every axis
  forced a doublet on all of them. It also gave every J the same 5 Hz start in
  [1, 10] Hz, which excludes every one-bond coupling.
- **Phase had a separate list of axis labels.** That list was not tied to the
  family, so a phase on a real family vanished without warning. With phase on the
  axis model, the axis owns the setting and the resolution can refuse a phase that
  has no meaning.
- **No J default.** A coupling's size depends on the spin system, not on the
  nucleus alone. A silent default would start fits far from the truth with no sign
  that it had.
- **Default plus per-axis overrides** (chosen by the user over per-axis tables
  only). A configuration stays independent of the spectrum's dimensionality when
  no axis needs its own settings. `--family lorentzian` and the GUI control work
  without knowing the axis labels.

## What breaks

- **Old TOML.** `[fitting] lineshape` and `fit_phase` are refused, with their
  replacement in the message: `family`, `[fitting.axes.<axis>] doublet = { j = … }`
  and `phase = true`. They are never reinterpreted.
- **Old CLI options.** `--lineshape`, `--phx` and `--phy` are refused, with the
  replacement options in the message.
- **Old files.** Scientific archives before 13.0.0, fit JSON before 10.0.0 and
  analysis projects before 16.0.0 are refused. For the formats written just before
  this change (12.0.0, 9.0.0 and 15.0.0), the message names PeakFit commit
  `e4556a0`, which opens them. Refit with this version.
- **Python API.**
    - `get_shape_names` is replaced by `axis_lineshapes`.
    - `create_shapes` and the peak-list readers take resolved `AxisLineshape`s, not
      shape names and a config.
    - Shape constructors take an `AxisLineshape`.
    - `ShapeBase.shape_name` is the family; `doublet` says whether the shape is a
      doublet.

## Consequences

- **Numerical results do not change.**
    - Singlet fits are bit-identical.
    - The old all-axes doublet, written as a doublet on each axis with J 5 Hz in
      [1, 10] Hz, is bit-identical too.
    - The doublet-J validation record reruns byte for byte.
    - Evidence is in the PR for #212.
- **Large J values (W2c, #214).** The doublet model now expresses them, but
  fitting regions and clustering do not yet cover both components of a large
  splitting.
