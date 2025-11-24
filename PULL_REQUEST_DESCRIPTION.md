This PR removes the deprecated `--backend` CLI option and performs a repository-wide cleanup to remove in-code parallelization and related documentation and tests.

Summary of changes in this PR

- Remove `--backend` CLI option and backend initialization logic (NumPy-only runtime).
- Remove CLI flags that attempted to toggle parallel execution (`--parallel`) and to tune worker count (`--workers`).
- Completely remove the `peakfit.fitting.parallel` module (deleted from the repo).
- Adjust `run_fit` to run sequentially (no parallel code paths) and update configuration logging and CLI help text.
- Remove parallel benchmarking and parallel tests; update profiling utilities to work in sequential-only mode.
- Update README, documentation, examples, and changelog to reflect the removal of in-code parallelism.
- Update and/or remove related tests; added a test to confirm the `peakfit.fitting.parallel` module is no longer importable.

Notes

- OS-level parallelization (e.g., using GNU `parallel`, `xargs`, or CI matrixing) remains a supported approach for running multiple independent `peakfit` processes in parallel. This change only removes the internal, in-code parallel cluster fitting support, simplifying the runtime model and reducing maintenance burden.

Validation & Tests

- Full test suite ran locally: **285 tests passed, 0 failed**.
- CLI help was updated and verified to no longer list `--parallel`/`--workers`.

Rationale

- Internal parallelization was fragile and introduced complexity (BLAS thread oversubscription, GIL contention, cross-platform process spawn overhead). The simplified, sequential runtime reduces the surface area for such issues and focuses performance work on vectorized NumPy code and benchmarking.

Please review and let me know if you'd prefer a separate PR that re-introduces a developer-only profiling parallel module (out-of-tree), or if we should create a short migration section in the README to guide users that previously relied on `--parallel` or `--workers`.

Commit/PR checklist

- [x] Deleted `peakfit.fitting.parallel` file and references
- [x] Updated CLI help, README, and docs
- [x] Updated tests and re-ran the test suite
- [x] Updated CHANGELOG to reflect removed features

If you'd like to split this change into smaller PRs for easier review, let me know and I can split into: (1) CLI and minor code cleanups, (2) full module removal and tests, (3) docs cleanup.
