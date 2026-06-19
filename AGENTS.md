# AGENTS.md

## Mission

PeakFit is a Python package and CLI for fitting pseudo-ND NMR spectra.

The current priority is **major simplification**: architecture, code, CLI, terminal UI, output model, documentation, and developer workflow may all be changed when this makes the project easier to understand, maintain, test, or use.

Do not preserve the current design just because it exists. Preserve scientific correctness, useful user workflows, and tested behavior unless an intentional change is clearly proposed.

## Source of truth for AI agents

This file is the main instruction file for AI-assisted work.

Files in `docs/` are useful context, but they are not immutable constraints.
They may be simplified, merged, rewritten, or deleted if their content becomes
redundant or misleading.

When changing architecture or behavior, update the relevant documentation so the repository remains coherent.

## What may be changed

Agents may propose and implement changes to:

- package architecture and module boundaries;
- CLI commands, options, defaults, and help text;
- terminal UI/UX and Rich output;
- output files and report structure;
- internal data models and naming;
- fitting orchestration and pipeline structure;
- documentation and AI guardrails;
- tests and development tooling.

Breaking changes are allowed when they are justified by a simpler and better design, but they must be explicit. Explain what breaks, why it is worth it, and how a user or developer should migrate.

## Current project shape

The current code is roughly organized as:

- `src/peakfit/cli/` — Typer command-line interface.
- `src/peakfit/ui/` — Rich terminal UI helpers.
- `src/peakfit/fit/` — fitting workflow orchestration.
- `src/peakfit/auto_pick/` — experimental automatic peak-picking workflow.
- `src/peakfit/mcmc/` — MCMC uncertainty workflows.
- `src/peakfit/plot/` — plotting and spectrum viewer.
- `src/peakfit/io/` — parsing, serialization, and file-format handling.
- `src/peakfit/engine/` — domain models, algorithms, lineshapes, and numerical computation.
- `src/peakfit/shared/` — small shared helpers.
- `tests/` — test suite.
- `docs/` and `examples/` — user/developer documentation and runnable workflows.

This structure is a starting point, not a constraint. Prefer a simpler structure if the evidence supports it.

## Design principles

Prefer:

- fewer concepts;
- fewer files;
- fewer user-facing options;
- one obvious way to do each task;
- explicit scientific/domain names;
- direct data flow;
- small cohesive modules;
- simple functions over frameworks or managers;
- deleting, merging, or renaming before adding abstractions;
- tests that protect scientific behavior.

Avoid:

- preserving abstractions only because they already exist;
- generic service layers, registries, managers, factories, or adapters without clear benefit;
- duplicated representations of the same concept;
- UI/CLI conditionals that hide domain rules;
- output formats or configuration options that are not clearly useful;
- large rewrites that cannot be reviewed safely.

## Stable invariants

Even during simplification, protect these invariants unless explicitly changing them:

- numerical/scientific behavior should remain correct and tested;
- input parsing errors should be clear and actionable;
- long-running CLI jobs should produce understandable progress and failure information;
- pure numerical code should not depend on terminal UI, Rich, Typer, Qt, or filesystem side effects unless a new architecture deliberately changes that boundary;
- public behavior changes should be documented.

## Working mode

For small bug fixes, make the smallest clear change and verify it.

For architecture, refactoring, UI/UX, output, or simplification work:

1. Inspect the relevant code before editing.
2. Identify the real problem with concrete file/function evidence.
3. Separate essential scientific complexity from accidental implementation complexity.
4. Propose the simplest target design.
5. State what will be deleted, merged, renamed, or intentionally broken.
6. Implement in small reviewable steps.
7. Verify with tests or explain why verification was not possible.

Do not ask for permission to simplify current architecture. The project goal is simplification. Ask only when scientific behavior, user expectations, or compatibility trade-offs are genuinely ambiguous.

## Agentic development loop

Long-running AI-assisted development should converge through living contracts, not
through accumulating plans. Prefer this loop:

1. Make one coherent simplification or behavior change.
2. Update the smallest relevant documentation in `docs/`, examples, or this file.
3. Add or adjust executable protection when possible: tests, schemas, import-linter,
   CLI help checks, or golden-output checks.
4. Run the smallest meaningful verification set.
5. Commit the reviewable slice.

Avoid creating new planning documents unless they will stay useful after the
current change. If a plan is implemented, merge the lasting decision into
`docs/architecture.md`, `docs/development.md`, `docs/decisions.md`, or the relevant
user guide, then delete the stale plan.

Every few refactor slices, do a short architecture audit:

- Are package boundaries in `docs/architecture.md` still true?
- Did an experimental workflow leak into the routine user path?
- Did a helper become a manager, service layer, or framework?
- Are examples still teaching the recommended workflow?
- Are output files still canonical and documented?

## UI/UX simplification

The CLI and terminal UI should make the main workflows obvious.

Prefer:

- clear defaults;
- fewer prompts/options;
- concise progress information;
- actionable warnings and errors;
- summaries that help users decide what to do next;
- output that is useful in logs and long-running jobs.

Avoid:

- decorative complexity;
- exposing internal implementation details;
- multiple equivalent ways to do the same thing;
- verbose output that hides failures or outliers;
- UI components that exist only because the code structure is complicated.

UI/UX changes should explain the user workflow improvement, not just the visual change.

## Testing and verification

Use `uv` for project commands.

Common checks:

```bash
uv sync --all-extras
uv run pytest
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check --error-on-warning
uv run lint-imports
uv run prek run --all-files
uv build
```

Before finishing a code-changing task, run the smallest relevant check. For broad refactors, run tests, Ruff, formatting check, type check, and import-linter when possible.

If a check fails because of a pre-existing or unrelated issue, report it clearly with the relevant output.

## Documentation rules

Documentation should be simplified together with the code.

Prefer one accurate document over several overlapping documents. Remove or merge stale docs when they conflict with the current design.

For major changes, document:

- what changed;
- why it is simpler;
- what behavior changed, if any;
- how to migrate, if needed;
- what tests protect the change.

## Final response format for agents

When finishing a task, report:

- what changed;
- what became simpler;
- behavior changes or compatibility breaks;
- tests/checks run;
- remaining risks or recommended next step.
