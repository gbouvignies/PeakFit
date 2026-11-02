# Decisions Log

Short records of non‑obvious design choices and trade‑offs.

## 2026‑01‑19: Introduced AI workflow guardrails
- Added AI contract, architecture boundaries, and project map to prevent misplacement and local hacks.
- Future behavior changes should update the smallest relevant documentation and tests.

## 2026-06-19: Keep documentation consolidated
- Prefer one concise current guide over several historical notes.
- Keep AI instructions in `AGENTS.md`, package boundaries in `docs/architecture.md`,
  and contributor workflow in `docs/development.md`.
- Delete or merge stale documents when refactors make them misleading.

## 2026-06-19: Make agentic development converge
- AI-assisted work should advance through small verified commits rather than
  long-lived planning documents.
- Durable guidance belongs in `AGENTS.md`, `docs/development.md`, executable
  checks, or focused user/developer guides.
- Experimental workflows should stay isolated and clearly labeled so they do not
  complicate the main fitting path.

## 2026-10-31: One set of agent instructions, no project-local agent tooling
- `AGENTS.md` is the only project instruction file; Claude Code reads it directly,
  so there is no `CLAUDE.md`.
- Agent skills (Matt Pocock's) are installed globally, not vendored in `.agents/`.
- Graphify was removed; the code, `GLOSSARY.md` and `docs/` are the navigation
  sources.
- The current focus is the scientifically rigorous desktop GUI, built on the
  completed scientific storage architecture.

## 2026-10-31: Track work in GitHub Issues
- GitHub Issues is the only issue tracker; agent skills read
  `docs/agents/issue-tracker.md`.
- Specifications and decisions stay in version-controlled Markdown under `docs/`.
- `.scratch/` was removed. Its two completed efforts (final fit outcome,
  point/series axis contract) remain in Git history:
  `git log --diff-filter=D --name-only -- .scratch`.
- `CONTEXT.md` was renamed `GLOSSARY.md`, the name the skills expect.

## 2026-10-31: `peakfit gui` is the only desktop entry point
- `peakfit gui` launches the v2 application (`gui.editor.window.launch`). The
  `peakfit-editor` script, `python -m peakfit.gui.editor` and the original viewer
  (`gui.app`, `gui.viewer`) were removed, with no alias.
- `peakfit gui PATH` opens an NMRPipe spectrum or a saved `.peakfit` analysis.
  `--peaks` now requires `--peak-axis-order {indirect-direct,direct-indirect}`; no
  axis order is guessed. Plane values keep `--z-values` / `-z`, matching
  `peakfit fit`. Companion options are rejected with a `.peakfit` analysis.
  Migration: add `--peak-axis-order indirect-direct` to reproduce the old
  `peakfit gui spectrum --peaks list` mapping.
- Local fit and Apply fitted positions were retired, as the v2 design specified:
  whole-dataset fitting with automatic adoption is the one refinement path. Earlier
  local-fit exports remain readable.

## 2026-10-31: Review workspace direction (GUI specification revision 2.1)
- Keep PySide6/Qt Widgets/PyQtGraph. The weaknesses found in a design review were
  information design and visual craft; invest in an owned presentation layer
  (tokens, proxy style, icons, plot chrome) rather than a framework change.
- Three columns: cluster navigator, spectrum, inspector. A Cluster lens gives
  side-by-side judgment of one cluster. No lower analysis panel.
- Review separates numerical warnings, review measures and labeled heuristics; no
  composite quality score.
- Fit settings are an integrated setup with an exact clustering preview, always
  visible as a toolbar summary; no first-fit modal dialog.
- Reference-normalized profiles are display-only, with full-gradient uncertainty
  propagation and explicit unavailable cases.
- No storage change is authorized; four storage items are deferred (specification §14).

## 2026-10-31: Backward compatibility is not a requirement
- APIs, project formats, data structures and GUI architecture may change whenever
  that yields a cleaner design.
- No compatibility layers, migration mechanisms or legacy adapters without a
  compelling reason; obsolete code is removed rather than preserved.
- Breaking changes are stated explicitly. Numerical behavior still changes only for
  an explained scientific reason protected by tests.

## 2026-11-01: Automatic lineshape fallback is a caution, not a block
- When `auto` finds no recognized NMRPipe apodization on an axis, the engine uses
  its pseudo-Voigt default, as the CLI does. The GUI fit setup labels that axis
  and shows a visible caution, but Fit stays available.
- Blocking would make such datasets unfittable until a model is chosen explicitly;
  the scientist judges suitability.

## 2026-11-01: No normalized profile for clusters with unreliable uncertainties
- On the bundled dataset the degenerate pairs (|ρ| = 1.00) record amplitudes near
  ±6×10¹² with σ/|A| ≈ 0.014, like well-determined peaks. The σ_R/|R| limit
  therefore passes, and the ratio cancels the arbitrary scale, giving a plausible
  profile with small error bars for undetermined amplitudes.
- Reference normalization is withheld as unreliable, with the reason, for clusters
  carrying the engine's uncertainty warning; the raw profile, which shows the
  inflated amplitudes, remains. This is distinct from an undefined ratio (zero or
  non-finite reference, unavailable σ).
- Approved in review of #82 as conservative behavior; revisit when the engine
  provides more specific identifiability diagnostics.

## 2026-11-01: Position movement is not a review diagnostic
- A peak may start far from its true position; moving toward it is what the
  optimizer is for. The "position shift" highlight (|fitted − initial| over the
  fitted linewidth) is removed with its threshold. Initial and fitted positions
  remain visible on each peak page.
- Run-to-run changes no longer report positions that moved by more than σ: the
  recorded position σ (about 10⁻⁵ ppm on the bundled data) flagged 113 of 166 peaks
  after a routine threshold change. No replacement threshold is introduced; the
  largest position differences are listed as neutral information, in ppm and in
  fitted linewidths.
- Run-to-run comparison keeps numerical warnings, highlights and cluster membership,
  matched by stable peak ID; a warning that disappears because a cluster was
  regrouped is not described as resolved.

## 2026-11-02: Unexplained-signal regions are signed, diagonally connected and measured in samples
- Positive and negative excursions are labelled separately, so a dispersive residual
  next to a misplaced peak reads as two regions rather than one cancelling blob;
  diagonal neighbours are connected so a tilted ridge stays one region.
- Distance to the nearest working peak is Euclidean in native samples: the two axes
  have no common ppm scale, and samples are also the unit of the region size. The
  list shows the ppm location of each region's extreme sample.
- The list is computed only for a Current run: an Outdated run's model no longer
  corresponds to the working peaks it would be measured against.

## 2026-11-02: The lens keeps its region for peaks the run has not fitted
- A peak just added belongs to no recorded cluster. Following only the selected
  peak's cluster would close the lens at the moment of the correction, so a
  selected unfitted peak keeps the lens on the cluster in view; selected from
  elsewhere, it opens the cluster with its nearest member in native samples.
- After a refit the regrouping line compares the followed peak's previous and new
  clusters by stable ID. A newly added peak has no previous cluster, so the line
  says which peaks it is now fitted with instead of reporting the context cluster's
  members as having left.
- Undo back to a run's exact inputs still leaves that run Outdated, as before:
  returning values through Undo does not recertify a run (docs/architecture.md).

## 2026-11-02: A refit continues unchanged peaks from the active run
- Adopting fitted positions while restarting every linewidth at 20 Hz gave starts
  that matched no solution; on the bundled data the second fit collapsed the
  resolved pair 52N-H/102N-H (χ²ᵣ 1.53 → 2.32) and later fits kept it
  ([repeat-fit collapse](validation/repeat-fit-collapse.md)).
- Unchanged peaks now start from the run's complete fitted nonlinear state, by
  stable ID; added, moved and incompatible peaks start from defaults. An unedited
  refit reproduces every cluster (largest relative χ²ᵣ change 2×10⁻⁶).
- The exact starts stay in the existing `initial_parameters`; only their provenance
  (`continuation`) is new. A rename keeps a peak continuing, since it changes no
  model input. Coincident pairs from earlier runs continue coincident and warned;
  no separation, deletion, basin choice or classification change was introduced.

## 2026-11-02: Near-collinear fitted shapes are not adopted or continued
- On the bundled data three pairs (53N-H/147N-H, 36N-H/93N-H, 5N-H/76N-H) end
  with recorded shapes at 1 − |cos| ≤ 6.4×10⁻⁷, contributions cancelling ×320
  to ×4 000; every other pair has 1 − |cos| ≥ 0.33. A real opposite-sign peak
  0.5 linewidth away stays resolved (0.136, ×2.95); at 0.25 linewidth the fit
  collapses although two resonances are present.
- The criterion is near-collinearity alone, 1 − |cos| ≤ 10⁻⁴ (equal shapes to about
  1 % of a linewidth): a conservative operational threshold, not a physical
  boundary. It does not use the uncertainty warning, cancellation or optimality,
  and cannot tell an optimizer failure from an identifiability problem, so it
  treats both alike and never selects another solution.
- Both members keep their working positions and start the next fit from defaults;
  the fitted result is untouched. On the bundled data the first refit then
  reaches the better resolved fit of 5N-H/76N-H (χ² 18 378 → 17 947), while the
  other two pairs re-collapse in place; later refits reproduce.
- Reopening validates the recorded adoption, so runs recorded before this rule,
  which adopted such peaks, still open and export as recorded.
