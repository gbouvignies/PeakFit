# Issue tracker: Local Markdown

Issues and specs (you may know a spec as a PRD) for this repo live as markdown files in `.scratch/`.
Do not create or depend on remote issues, gists, or hosted tracker services.

## Conventions

- Active work lives one feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01` — never a single combined tickets file
- Triage state is recorded as a `Status:` line near the top of each issue file (see `triage-labels.md` for the role strings)
- Comments and conversation history append to the bottom of the file under a `## Comments` heading
- Move a completed effort to `.scratch/archive/<feature-slug>/` only after implementation, validation, and human acceptance.
- Archived efforts are historical context, not an active task queue. Do not resume or implement archived tickets unless explicitly instructed; consult them only for rationale and provenance.
- Current architecture and operational truth remain in code and authoritative documentation, not archived records.

## When a skill says "publish to the issue tracker"

Create a new file under an active `.scratch/<feature-slug>/` directory (creating
the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue number directly. Archived records may be read for context but are never a frontier for implementation.

## Wayfinding operations

Used by `/wayfinder` for active efforts. The **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` — the Notes / Decisions-so-far / Fog body.
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`); a `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked, and unclaimed; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append the relevant repository-local context path to the map's Decisions-so-far in `map.md`.
