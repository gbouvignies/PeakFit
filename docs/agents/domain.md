# Domain docs

Single context: the glossary is `GLOSSARY.md` at the repo root, and ADRs live in
`docs/adr/`. Short decisions are logged in `docs/decisions.md`.

## Before exploring, read these

- **`GLOSSARY.md`** for the domain vocabulary.
- **`docs/adr/`**: ADRs that touch the area you're about to work in.
- **`docs/decisions.md`**: short decisions relevant to that area.

If an ADR or glossary entry you'd expect doesn't exist, proceed silently. The
`/domain-modeling` skill adds entries when terms or decisions actually get
resolved.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal,
a hypothesis, a test name), use the term as defined in `GLOSSARY.md`. Don't drift
to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, either you're inventing
language the project doesn't use (reconsider) or there's a real gap (note it for
`/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR or decision, surface it explicitly
rather than silently overriding:

> _Contradicts ADR-0007 (…), but worth reopening because…_
