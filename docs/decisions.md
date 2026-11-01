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
