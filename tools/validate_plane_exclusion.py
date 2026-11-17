"""Check that excluding planes equals fitting the spectrum with those planes removed.

Run with ``uv run python tools/validate_plane_exclusion.py``. It fits the bundled
example twice with the golden reference's arguments (``write_golden_reference.py``):
once with ``--exclude`` for each plane of ``EXCLUDED``, and once on an NMRPipe file
and plane-value list from which those planes were physically removed. The reduced
file differs from the original only in its plane count (``FDF3SIZE``); the reader
checks that its planes are bit for bit the original's remaining planes.

It prints a JSON record: for each compared quantity, the largest relative
difference and whether the two fits agree bit for bit. The gate, fixed before the
first run, is a relative difference of at most ``RTOL`` everywhere and identical
classifications, cluster and peak identities. The exit code is 1 when the gate fails.
``EXCLUDED`` was also fixed in advance: the off-resonance reference plane, two
adjacent offsets and the last plane.

The record also gives the noise estimated from every plane, which the excluded fit
must not use: a fit of the reduced spectrum could not see those planes.
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import nmrglue as ng
import numpy as np
from write_golden_reference import DATA, FIT_ARGUMENTS, ROOT

from peakfit.engine.algorithms.covariance_factors import (
    marginal_standard_errors,
    profile_standard_errors,
)
from peakfit.engine.algorithms.noise import estimate_noise
from peakfit.io.readers.spectrum import read_spectra
from peakfit.io.scientific import read_scientific

EXCLUDED = (0, 40, 41, 130)
RTOL = 1e-12
SPECTRUM = ROOT / DATA / "pseudo3d.ft2"
PLANE_VALUES = ROOT / DATA / "b1_offsets.txt"


def _fit(arguments: list[str], output: Path) -> Path:
    command = [sys.executable, "-m", "peakfit", "fit", *arguments]
    command += ["--output", str(output), "--headless"]
    subprocess.run(command, check=True, cwd=ROOT, capture_output=True)
    (run,) = (path for path in output.iterdir() if path.is_dir())
    return run


def _reduced_inputs(directory: Path, kept: list[int]) -> tuple[Path, Path]:
    """The example's spectrum and plane values with the excluded planes removed."""
    dic, data = ng.pipe.read(str(SPECTRUM))
    header = dict(dic)
    header["FDF3SIZE"] = float(len(kept))  # the plane count of a pseudo-3D stream
    spectrum = directory / "reduced.ft2"
    ng.pipe.write(str(spectrum), header, data[kept], overwrite=True)
    original, reduced = read_spectra(SPECTRUM), read_spectra(spectrum)
    if not np.array_equal(reduced.data, original.data[kept]):
        raise RuntimeError("The reduced spectrum's planes differ from the original's.")
    if reduced.params[1:] != original.params[1:]:
        raise RuntimeError("The reduced spectrum's spectral axes differ from the original's.")
    values = directory / "reduced_offsets.txt"
    lines = PLANE_VALUES.read_text().splitlines()
    values.write_text("\n".join(lines[i] for i in kept) + "\n")
    return spectrum, values


class Comparison:
    """Largest relative difference and bit-for-bit agreement, per quantity."""

    def __init__(self) -> None:
        self.quantities: dict[str, dict[str, float | bool]] = {}

    def add(self, name: str, actual, expected) -> None:
        """Fold one more pair of values of ``name`` into its entry."""
        a = np.asarray(actual, dtype=float)
        b = np.asarray(expected, dtype=float)
        if a.shape != b.shape:
            raise RuntimeError(f"{name}: shapes {a.shape} and {b.shape} differ")
        same_nan = np.isnan(a) == np.isnan(b)
        finite = ~np.isnan(a) & ~np.isnan(b)
        difference = np.abs(a[finite] - b[finite])
        scale = np.abs(b[finite])
        relative = np.where(difference == 0, 0.0, difference / np.where(scale > 0, scale, 1.0))
        entry = self.quantities.setdefault(name, {"max_relative": 0.0, "bit_identical": True})
        entry["max_relative"] = max(
            float(entry["max_relative"]),
            float(relative.max(initial=0.0)) if same_nan.all() else float("inf"),
        )
        entry["bit_identical"] = bool(entry["bit_identical"]) and bool(
            np.array_equal(a, b, equal_nan=True)
        )


def compare(excluded: Path, removed: Path) -> dict:
    """The record comparing the two runs' scientific archives and intensity tables."""
    a, _, context = read_scientific(excluded / "scientific.npz")
    b, _, _ = read_scientific(removed / "scientific.npz")
    identities = [(c.cluster_id, c.peak_names, c.classification.value) for c in a.clusters]
    if identities != [(c.cluster_id, c.peak_names, c.classification.value) for c in b.clusters]:
        raise RuntimeError("Cluster identities or classifications differ.")
    warnings = [
        c.analytical_evaluation.uncertainty_warning
        for c in (*a.clusters, *b.clusters)
        if c.analytical_evaluation is not None
    ]
    half = len(warnings) // 2
    comparison = Comparison()
    comparison.add("noise", a.noise, b.noise)
    for name in ("chi_squared", "reduced_chi_squared", "n_observations", "degrees_of_freedom"):
        comparison.add(f"global {name}", getattr(a.statistics, name), getattr(b.statistics, name))
    for ca, cb in zip(a.clusters, b.clusters, strict=True):
        if [p.name for p in ca.final_nonlinear_parameters] != [
            p.name for p in cb.final_nonlinear_parameters
        ]:
            raise RuntimeError(f"Cluster {ca.cluster_id}: parameter names differ.")
        comparison.add(
            "parameter values",
            [p.value for p in ca.final_nonlinear_parameters],
            [p.value for p in cb.final_nonlinear_parameters],
        )
        comparison.add(
            "parameter standard errors",
            [p.standard_error for p in ca.final_nonlinear_parameters],
            [p.standard_error for p in cb.final_nonlinear_parameters],
        )
        ea, eb = ca.analytical_evaluation, cb.analytical_evaluation
        if ea is None or eb is None:
            continue
        comparison.add("amplitudes", ea.amplitudes, eb.amplitudes)
        comparison.add(
            "conditional sigma", ea.amplitude_standard_errors, eb.amplitude_standard_errors
        )
        comparison.add(
            "profile sigma",
            profile_standard_errors(ea.covariance_factors, ea.amplitudes),
            profile_standard_errors(eb.covariance_factors, eb.amplitudes),
        )
        comparison.add(
            "marginal sigma",
            marginal_standard_errors(ea.covariance_factors, ea.amplitudes),
            marginal_standard_errors(eb.covariance_factors, eb.amplitudes),
        )
        comparison.add("cluster chi_squared", ea.statistics.chi_squared, eb.statistics.chi_squared)
    with (excluded / "tables/intensities.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    marked = {int(r["source_plane_index"]) for r in rows if r["plane_excluded"] == "yes"}
    empty = all(r["intensity"] == "" for r in rows if r["plane_excluded"] == "yes")
    passed = (
        warnings[:half] == warnings[half:]
        and all(q["max_relative"] <= RTOL for q in comparison.quantities.values())
        and a.excluded_planes == EXCLUDED
        and marked == set(EXCLUDED)
        and empty
    )
    return {
        "excluded_planes": list(EXCLUDED),
        "rtol": RTOL,
        "clusters": len(a.clusters),
        "classifications_identical": True,
        "uncertainty_warnings_identical": warnings[:half] == warnings[half:],
        "recorded_excluded_planes": list(a.excluded_planes),
        "intensity_rows_marked_excluded": sorted(marked),
        "excluded_rows_without_intensity": empty,
        "noise_fitted_planes": a.noise,
        "noise_all_planes": float(
            estimate_noise(read_spectra(SPECTRUM).data),
        ),
        "source_plane_values": len(context["source_plane_values"]),
        "quantities": comparison.quantities,
        "passed": passed,
    }


def main() -> None:
    """Fit both ways and print the comparison record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=Path, help="directory to keep both runs in")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        root = args.keep or Path(temporary)
        root.mkdir(parents=True, exist_ok=True)
        n_planes = read_spectra(SPECTRUM).data.shape[0]
        kept = [i for i in range(n_planes) if i not in EXCLUDED]
        spectrum, values = _reduced_inputs(root, kept)
        exclusions = [option for plane in EXCLUDED for option in ("--exclude", str(plane))]
        excluded = _fit([*FIT_ARGUMENTS, *exclusions], root / "excluded")
        reduced = [str(spectrum), FIT_ARGUMENTS[1], "--z-values", str(values)]
        removed = _fit([*reduced, *FIT_ARGUMENTS[4:]], root / "removed")
        record = compare(excluded, removed)
    print(json.dumps(record, indent=1))  # noqa: T201
    sys.exit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
