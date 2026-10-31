# Issue tracker: GitHub

Issues live in GitHub Issues on `gbouvignies/PeakFit`. Use the `gh` CLI.

- **Publish a ticket**: `gh issue create --title "..." --body "..."`.
- **Fetch a ticket**: `gh issue view <n> --comments`.
- **Comment / close**: `gh issue comment <n>` / `gh issue close <n>`.

Issues and pull requests share one number sequence. A trailing `(#NN)` on a merge
commit names its pull request; `Closes #NN` or `Fixes #NN` names an issue.
Resolve any other `#NN` with `gh pr view NN`, falling back to `gh issue view NN`.

Use the existing labels when they fit. Long-form specifications and decisions
live in version-controlled Markdown under `docs/`; issues link to them.
