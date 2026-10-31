# AGENTS.md

Project instructions for every coding assistant. This is the single source; there
is no separate `CLAUDE.md`.

## Mission

PeakFit fits lineshape models to pseudo-ND NMR spectra through a Python package,
a CLI and a desktop GUI.

The current priority is a **scientifically rigorous, intuitive, polished desktop
GUI** built on the existing scientific core. Engine, CLI and output work serves
that GUI or fixes scientific correctness.

## Scientific rigor

Scientific correctness outranks convenience, speed and visual polish.

- Change numerical behavior only deliberately: explain the scientific reason and
  protect it with a test.
- Golden files, reference values and tolerances are evidence. Change them only
  with an explained, reviewed scientific reason (see `docs/development.md`).
- Name things with the vocabulary in `GLOSSARY.md`. When a new domain term
  settles, add it there.
- The GUI presents published scientific results exactly as recorded. Values come
  from the scientific core, never from GUI-side approximations or refits.

## Scientific storage (complete)

The storage architecture is finished and validated. Treat it as the foundation:

- `FinalFitOutcome` is the sole completed scientific result; `FittingState` is
  continuation-only.
- One writer, schema and reader serve CLI and GUI exports (`docs/outputs.md`).
- Projects link external spectra and never embed them; reopening never refits or
  recovers amplitudes.

Design and evidence: `docs/architecture.md` ("External-source analysis
persistence") and `docs/validation/scientific-storage*.md`,
`docs/validation/external-projects.md`. Change formats or retention only when the
task is explicitly a storage change.

## Architecture

- `engine` stays pure: no Qt, Typer, Rich or filesystem side effects. The GUI is a
  presentation layer above the scientific code. `uv run lint-imports` enforces the
  layers.
- Prefer direct functions and explicit data flow over managers, services,
  registries, factories or adapters.
- `auto_pick` is experimental and stays isolated from the routine workflow.
- Package map and boundaries: `docs/architecture.md`.

## Focused development

- Make the change the task asks for, in its smallest coherent form. Refactor code
  only where that makes the requested change simpler or correct.
- Add a feature, option or setting only for a concrete user workflow.
- Match planning to risk: go straight to code for clear changes; for a genuinely
  ambiguous scientific, UX or compatibility trade-off, ask one focused question.
- Make breaking changes explicit: what breaks, why, and how to migrate.

## GUI

Visual design, readability, intuitiveness and consistency are product
requirements, not finishing touches.

- Design baseline: `docs/PeakFit_GUI_Design_Specification_v2.0.md`.
- User workflow: `docs/gui.md`.
- Long work runs off the UI thread; see "GUI responsiveness" in
  `docs/development.md`.
- Errors say what went wrong and what the user can do next.

## Testing

Test in proportion to risk: numerical and storage changes need scientific tests;
GUI changes need behavior tests of the workflow they touch.

Use `uv`. Run tests headless:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q tests/<relevant>
```

Run the focused tests for each change. Before finishing a change that crosses
packages, also run the full suite, `ruff check`, `ruff format --check`,
`ty check --error-on-warning` and `lint-imports`. The full command list is in
`docs/development.md`. Report failures with their output, including pre-existing
ones.

## Documentation and records

- Update the one document that describes the changed behavior; delete stale text
  rather than layering corrections.
- Short decisions go in `docs/decisions.md`; hard-to-reverse ones in `docs/adr/`.
- Long-form scientific specifications live in version-controlled Markdown under
  `docs/`; issues link to them.

## Agent skills

### Issue tracker

GitHub Issues on `gbouvignies/PeakFit`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single context: `GLOSSARY.md` and `docs/adr/`. See `docs/agents/domain.md`.

## Final report

Report what changed, behavior or compatibility changes, checks run with their
results, and remaining risks.
