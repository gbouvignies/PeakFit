# AI Contract (PeakFit)

Non‑negotiable rules for AI‑assisted changes in this repository.

## General Rules
- No one‑off fixes for specific values or datasets. Express behavior as rules in domain logic.
- Business rules live in Core or Services, never in CLI/UI conditionals.
- Core is pure computation: no I/O, no CLI/UI imports, no file system side effects.
- IO layer parses/serializes only; it must not contain fitting logic.
- Config flows one‑way: CLI → Services → Core.
- Public APIs and CLI behavior require tests and docs updates.

## Change Workflow Requirements
- Read [ARCHITECTURE.md](ARCHITECTURE.md) and [projectmap.md](projectmap.md) before coding.
- Create or update a spec in specs/ for any core change.
- Implement in ordered slices: Core/Domain → Services → CLI/UI → Tests → Docs.
- Add tests that prevent shortcuts (anti‑hack tests) for new rules.
- Run the hard gates (tests, lint, type checks) and report results.

## Prohibited Patterns
- UI/CLI logic that encodes domain rules.
- Core code that reads/writes files or uses `rich`/console output.
- “Magic” special cases for particular unit names, peak IDs, or files.
- Silent behavior changes without updating docs and tests.
