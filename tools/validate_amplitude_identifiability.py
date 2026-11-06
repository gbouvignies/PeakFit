"""Classify every recorded amplitude of real bundled fits as identifiable or not.

Fits the bundled CEST dataset from the default linewidth start and from the former
20 Hz start, which reproduces three real near-collinear collapses
(docs/validation/repeat-fit-collapse.md), plus the refit of a hand-moved 147N-H that
produced a degenerate 53N-H/147N-H decomposition in the GUI assessment. Records, per
cluster that is not fully identifiable, the status of each member and the numbers it
rests on. Writes docs/validation/amplitude-identifiability.json.

Run from the repository root: ``uv run python tools/validate_amplitude_identifiability.py``.
"""

import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from peakfit.engine.diagnostics.identifiability import amplitude_identifiability
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fit_positions import continuation_source, dataset_completion
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.spectrum import read_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/02-advanced-fitting/data"
OUTPUT = ROOT / "docs/validation/amplitude-identifiability.json"
BROAD_START = 20.0  # Hz, the former default linewidth start


def fit(spectra, peaks, settings, completion=None, start=None):
    """Fit; peaks that do not continue a run start at linewidth ``start`` when given."""
    source = continuation_source(completion) if completion is not None else None
    prepared = prepare_dataset_fit(spectra, tuple(peaks), settings, source)
    if start is not None and prepared.continuation is not None:
        initialized = set(prepared.continuation.initialized)
        for identity in prepared.identities:
            if identity.working_id in initialized:
                for axis in ("F2", "F3"):
                    prepared.parameters[f"{identity.engine_name}.{axis}.lw"].value = start
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return dataset_completion(
        result, result.inputs.snapshot, submitted_revision=0, manual_revision=0
    )


def classify(completion):
    """Status counts over every usable member, and details of clusters not fully identifiable."""
    names = {i.engine_name: i.assignment for i in completion.result.inputs.identities}
    counts: Counter[str] = Counter()
    clusters = []
    for cluster in completion.result.outcome.clusters:
        evaluation = cluster.analytical_evaluation
        if not cluster.usable or evaluation is None:
            continue
        members = amplitude_identifiability(
            evaluation.shapes, evaluation.amplitudes, evaluation.amplitude_standard_errors
        )
        counts.update(m.status.value for m in members)
        if any(m.status.value != "identifiable" for m in members):
            clusters.append(
                {
                    "members": [names[n] for n in cluster.peak_names],
                    "rank": members[0].rank,
                    "condition_smin_over_s0": members[0].condition,
                    "amplitude_sigma_finite": [
                        bool(np.isfinite(v)) for v in evaluation.amplitude_standard_errors
                    ],
                    "uncertainty_warning": evaluation.uncertainty_warning is not None,
                    "status": [
                        {
                            "peak": names[n],
                            "status": m.status.value,
                            "cause": m.cause.value,
                            "null_projection": m.null_projection,
                            "weak_projection": m.weak_projection,
                            "partners": [names[cluster.peak_names[j]] for j in m.partners],
                            "dissimilarity": m.dissimilarity,
                        }
                        for n, m in zip(cluster.peak_names, members, strict=True)
                    ],
                }
            )
    return {"counts": dict(counts), "not_fully_identifiable": clusters}


def main():
    """Fit, classify and write the record."""
    dataset = read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt")
    settings = PeakFitConfig(fitting=FitConfig(refine_iterations=2))
    report = {}
    default = fit(dataset.spectra, dataset.peaks, settings)
    report["default_start"] = classify(default)
    broad = fit(dataset.spectra, dataset.peaks, settings, start=BROAD_START)
    report["broad_20Hz_start"] = classify(broad)
    # The GUI assessment's correction: 147N-H moved onto a +3σ residual, then refit.
    moved = tuple(
        replace(p, positions=(115.155, 8.3488)) if p.name == "147N-H" else p
        for p in default.working_peaks
    )
    report["moved_147N-H_refit"] = classify(fit(dataset.spectra, moved, settings, default))
    OUTPUT.write_text(json.dumps(report, indent=1) + "\n")
    for name, part in report.items():
        print(name, part["counts"], [c["members"] for c in part["not_fully_identifiable"]])  # noqa: T201


if __name__ == "__main__":
    main()
