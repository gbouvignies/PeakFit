"""Check that a desktop fit excluding planes equals ``peakfit fit --exclude`` (#216).

Run with ``uv run python tools/validate_gui_plane_exclusion.py``. It fits the bundled
example with the golden reference's arguments (``write_golden_reference.py``) and
``--exclude`` for each plane of ``validate_plane_exclusion.EXCLUDED``, then prepares
and fits the same dataset as the desktop does (``read_dataset``,
``prepare_dataset_fit``, ``compute_dataset_fit``) with the configuration that CLI run
recorded. The desktop names peaks by safe IDs (``p_0`` …), so peak i of the list is
compared with ``p_i``.

It prints a JSON record: for each compared quantity, the largest relative difference
and whether the two fits agree bit for bit. The gate, fixed before the first run, is
bit-for-bit agreement everywhere and identical clusters, classifications and peak
order. The exit code is 1 when the gate fails.
"""

import argparse
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from validate_plane_exclusion import EXCLUDED, Comparison, _fit
from write_golden_reference import DATA, FIT_ARGUMENTS, ROOT

from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.spectrum import read_dataset
from peakfit.io.config import stored_config
from peakfit.io.scientific import read_scientific


def desktop_fit(configuration: dict):
    """The desktop's preparation and fit of the bundled example."""
    data = ROOT / DATA
    dataset = read_dataset(data / "pseudo3d.ft2", data / "pseudo3d.list", data / "b1_offsets.txt")
    prepared = prepare_dataset_fit(dataset.spectra, dataset.peaks, stored_config(configuration))
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return [peak.name for peak in dataset.peaks], result


def _renamed(name: str, rename: dict[str, str]) -> str:
    peak, dot, rest = name.partition(".")
    return f"{rename.get(peak, peak)}{dot}{rest}"


def compare(cli, desktop, names: list[str]) -> dict:
    """The record comparing the CLI archive's outcome with the desktop's."""
    a, b = desktop.outcome, cli
    rename = {f"p_{i}": name for i, name in enumerate(names)}
    clusters = [(c.cluster_id, [rename[n] for n in c.peak_names]) for c in a.clusters]
    if clusters != [(c.cluster_id, list(c.peak_names)) for c in b.clusters]:
        raise RuntimeError("Clusters or their peaks differ.")
    if [c.classification for c in a.clusters] != [c.classification for c in b.clusters]:
        raise RuntimeError("Classifications differ.")
    if a.excluded_planes != b.excluded_planes:
        raise RuntimeError("Excluded planes differ.")
    comparison = Comparison()
    comparison.add("noise", a.noise, b.noise)
    for name in ("chi_squared", "reduced_chi_squared", "n_observations", "degrees_of_freedom"):
        comparison.add(f"global {name}", getattr(a.statistics, name), getattr(b.statistics, name))
    for ca, cb in zip(a.clusters, b.clusters, strict=True):
        # Parameters are ordered by name, and p_12 sorts apart from its assignment.
        ours = {_renamed(p.name, rename): p for p in ca.final_nonlinear_parameters}
        theirs = {p.name: p for p in cb.final_nonlinear_parameters}
        if ours.keys() != theirs.keys():
            raise RuntimeError(f"Cluster {ca.cluster_id}: parameters differ.")
        for field in ("value", "standard_error", "min", "max", "vary"):
            comparison.add(
                f"parameter {field}",
                [getattr(ours[name], field) for name in theirs],
                [getattr(theirs[name], field) for name in theirs],
            )
        ea, eb = ca.analytical_evaluation, cb.analytical_evaluation
        if ea is None or eb is None:
            if (ea is None) != (eb is None):
                raise RuntimeError(f"Cluster {ca.cluster_id}: one fit has no evaluation.")
            continue
        for field in ("amplitudes", "amplitude_standard_errors", "amplitude_covariance"):
            comparison.add(field, getattr(ea, field), getattr(eb, field))
        comparison.add("cluster chi_squared", ea.statistics.chi_squared, eb.statistics.chi_squared)
    return comparison.quantities


def main() -> None:
    """Fit both ways and print the record; exit 1 when the gate fails."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    exclude = [argument for plane in EXCLUDED for argument in ("--exclude", str(plane))]
    with tempfile.TemporaryDirectory() as directory:
        run = _fit([*FIT_ARGUMENTS, *exclude], Path(directory) / "cli")
        cli, _samples, context = read_scientific(run / "scientific.npz")
    configuration = context["metadata"]["configuration"]
    names, desktop = desktop_fit(configuration)  # p_i is the list's peak i
    quantities = compare(cli, desktop, names)
    passed = all(q["bit_identical"] for q in quantities.values())
    record = {
        "excluded": list(EXCLUDED),
        "clusters": len(cli.clusters),
        "gate": "bit-identical",
        "passed": passed,
        "quantities": quantities,
    }
    print(json.dumps(record, indent=2))  # noqa: T201
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
