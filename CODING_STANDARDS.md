# Coding standards

Review rules for every change, to be applied to the diff. Ruff, ty, import-linter
and the prek hooks enforce the mechanical rules; this file holds the judgement
calls they cannot check. Scientific-rigor and storage rules live in `AGENTS.md`.

## Architecture

- **Pure engine.** `engine` holds numerical and domain primitives, with no Qt,
  Typer, Rich or filesystem side effects. `fit` loads, validates, orchestrates and
  assembles fit outputs. Package map: `docs/architecture.md`.
- **Direct functions.** Prefer direct functions and explicit data flow over
  managers, service layers, registries, factories or adapters.
- **One scientific authority.** `scientific.npz`, the CSV tables under `tables/`
  and a short report come from one writer for CLI and GUI (`docs/outputs.md`).
  JSON is explicit interoperability. The GUI shows values computed by the
  scientific core, never GUI-side approximations or refits.
- **One home per concept.** A domain check, threshold or constant is defined once
  and reused: one predicate (e.g. "factors withheld", "profile flagged"), one
  constant (e.g. CEST reference offset, flag thresholds), and user-facing text
  formatted from that constant, not hard-coded.

## Scientific review

Apply these to any change in a statistic, estimator, uncertainty or diagnostic.
Matching the spec is not evidence of correctness: the spec may be wrong.

- **Independent expectation.** Check the quantity's expected value against known
  truth with a simulation you write yourself, from the model and not from the
  change's own helpers. For a standardized statistic, check that E[q] = k.
- **Plug-in independence.** A variance used to standardize a quantity must not
  depend on the noise in that same quantity. Watch for covariance terms evaluated
  at fitted values that carry the observation being standardized. Example: the
  duplicate-plane check's first version built lineshape gains from fitted
  amplitudes, which put the observed contrast back into its own variance and
  biased q low (`docs/decisions.md`, 2026-11-14).
- **Spec as hypothesis.** Verify every mathematical claim the change relies on
  (a cancellation, an independence, a first-order expectation). Treat a claim
  quoted from a spec or issue as unverified.
- **Validation design.** Validation tools follow the protocol in
  `docs/development.md` ("Validation tools").

## GUI

- **Off the GUI thread.** Scientific and disk work runs off the Qt event loop, in
  the mechanisms listed under "GUI responsiveness" in `docs/development.md`, even
  when it is cached or fast on the bundled example. Work that scales with the
  number of clusters, peaks or planes counts as scientific work.
- **Design baseline.** Layout, wording and visual hierarchy follow
  `docs/PeakFit_GUI_Design_Specification_v2.0.md`; the user workflow is
  `docs/gui.md`.
- **Actionable errors.** Errors say what went wrong and what the user can do next.
- **Logic on values.** Branch on data or types, never on display text.

## Types

- **Statuses as Enums.** A closed set of states (a verdict, a flag status) is an
  `Enum`, consistent across modules, with stored strings kept stable. Small
  records are dataclasses, not unnamed tuples or ad-hoc dicts.
- **NumPy.** Accept `numpy.typing.ArrayLike` and normalise immediately with
  `np.asarray(...)`; return `numpy.typing.NDArray[...]` from vectorised
  functions. Enforce shapes at runtime rather than in fragile shape types. Use
  overloads only when the public API guarantees scalar in, scalar out.

## CLI output

PeakFit runs can be long, so terminal output leaves a useful scrollback: a
compact input, config and output summary before work starts; terse routine
success; parse errors, failed clusters, boundary hits and high reduced χ² easy to
find; actionable messages over decorative panels; no internal architecture names.

## Documentation

- **One document per behaviour.** A change updates the document that describes
  the changed behaviour and deletes stale text rather than layering corrections.
  Changed behaviour with an unchanged document is a finding. Where each behaviour
  is described:

  | Behaviour | Document |
  |---|---|
  | Exported columns, file formats and versions | `docs/outputs.md` |
  | GUI workflow and wording | `docs/gui.md` |
  | GUI design, including error display (§9 for profiles and ratios) | `docs/PeakFit_GUI_Design_Specification_v2.0.md` |
  | Uncertainty of amplitudes, profiles, ratios and R2eff | `docs/profile-uncertainty.md` |
  | Noise model and processing correlation | `docs/correlated-noise.md` |
  | Which processing the uncertainties are validated for (user guidance) | `docs/processing-quantitative-series.md` |
  | Package layers and storage design | `docs/architecture.md` |
  | Commands, fixtures, GUI threading | `docs/development.md` |
  | Domain terms | `GLOSSARY.md` |

- **Status lines.** A specification's status line matches what is implemented
  and released.
- **Records.** Short decisions go in `docs/decisions.md`; hard-to-reverse ones in
  `docs/adr/`. A breaking change states what breaks and why.
