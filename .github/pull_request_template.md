## Summary

What changed?

## Rationale

Why is this simpler, more correct, or more useful?

## Behavior Changes

List user-facing or compatibility changes. Write "None" if there are none.

## Verification

Check what was run:

- [ ] `uv run ruff check src tests`
- [ ] `uv run ruff format --check src tests`
- [ ] `uv run ty check`
- [ ] `uv run lint-imports`
- [ ] `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q`
- [ ] `uv run prek run --all-files`
- [ ] `uv build`

## Documentation

- [ ] Updated relevant docs, examples, or `AGENTS.md`
- [ ] Removed stale docs when applicable
