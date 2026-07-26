# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root.
- **`docs/adr/`** for ADRs that touch the area you're about to work in.

If either location doesn't exist, **proceed silently**. The `/domain-modeling`
skill creates domain terms and decisions lazily when they are resolved.

## File structure

PeakFit uses a single-context layout:

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/
│       └── NNNN-short-title.md
└── src/
```

## Use the glossary's vocabulary

When your output names a domain concept, use the term defined in `CONTEXT.md`.
Don't drift to synonyms the glossary explicitly avoids.

If a required concept isn't in the glossary, reconsider the language or note the
gap for `/domain-modeling`.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-NNNN — but worth reopening because…_
