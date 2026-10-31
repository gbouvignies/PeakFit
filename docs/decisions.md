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
