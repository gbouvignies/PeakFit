# Refactoring Guidelines

- **Goal:** Simplify and consolidate. Less code is better.
- **Style:** Prefer "Vertical Slice" architecture (feature-based) over "Layered" (tech-based).
- **Safety:** never delete a file without verifying it is unused.
- **Testing:** Always run `uv run pre-commit run` and `uv run pytest` after moving files.
