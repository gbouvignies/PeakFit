"""Qt-free synthetic inputs and completed results shared by scientific and GUI tests."""

from dataclasses import replace

import numpy as np

from peakfit.engine.algorithms.noise_correlation import not_modelled
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, LineshapeName, PeakFitConfig
from peakfit.engine.domain.fit_steps import FitStep
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.fit.export import write_completed_outputs
from peakfit.fit.pipeline import run_pipeline
from peakfit.gui.dataset_fit import DatasetFitAttempt, finish_dataset_fit
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import WorkingPeak
from peakfit.io.writers.config import WriterConfig


def overlapping_gaussians(amplitudes=None, z_values=None):
    """Two overlapping synthetic Gaussians, by default across three planes with repeated values.

    ``amplitudes`` (2 × S) and ``z_values`` (S) replace the default three-plane series.
    """
    if amplitudes is None:
        amplitudes = np.array([[5.0, -3.0, 2.0], [3.0, -2.0, 4.0]])
    if z_values is None:
        z_values = np.array([2.0, 0.0, 2.0])
    n_planes = len(z_values)
    axes = [
        SpectralParameters(
            size=size,
            sw=500,
            obs=500,
            car=0,
            aq_time=0.03,
            apocode=0,
            apodq1=0,
            apodq2=0,
            apodq3=0,
            p180=not direct,
            direct=direct,
            ft=True,
        )
        for size, direct in ((n_planes, False), (32, False), (32, True))
    ]
    spectra = Spectra(
        dic={},
        data=np.zeros((n_planes, 32, 32)),
        z_values=np.asarray(z_values, dtype=float),
        params=axes,
    )
    config = PeakFitConfig(
        noise_level=0.002,
        fitting=FitConfig(lineshape="gaussian", steps=[FitStep(fix=["*.*.lw"])]),
    )
    config.parameters.position_window = 0.04
    positions = [
        (axes[1].pts2ppm(14.2) + 1.0, axes[2].pts2ppm(14.4)),
        (axes[1].pts2ppm(15.5), axes[2].pts2ppm(15.7)),
    ]
    peaks = [
        Peak(
            f"p_{i}",
            np.array(pos),
            create_shapes(spectra, config.fitting, f"p_{i}", pos, ["gaussian", "gaussian"]),
        )
        for i, pos in enumerate(positions)
    ]
    indices = [index.ravel() for index in np.indices((32, 32))]
    shapes = Cluster.evaluate_peaks(peaks, Parameters.from_peaks(peaks), indices)
    rng = np.random.default_rng(42)
    spectra.data[:] = (shapes.T @ amplitudes).T.reshape(n_planes, 32, 32)
    spectra.data += rng.normal(0, 0.002, spectra.data.shape)
    working = tuple(
        WorkingPeak(str(i), "Duplicate", (pos[0] + 0.003, pos[1] - 0.003))
        for i, pos in enumerate(positions)
    )
    return spectra, working, config


def two_cluster_dataset(amplitudes=None, z_values=None, remote=None):
    """The overlapping pair plus a remote peak, prepared as more than one cluster.

    ``amplitudes`` (2 × S), ``z_values`` (S) and the remote peak's ``remote`` (S)
    replace the default three-plane series.
    """
    spectra, peaks, config = overlapping_gaussians(amplitudes, z_values)
    if remote is None:
        remote = np.array([4.0, -2.0, 3.0])
    n_planes = len(remote)
    pos = tuple(
        axis.pts2ppm(point)
        for axis, point in zip(
            spectra.spectral_params,
            (6.2, 25.4),
            strict=True,
        )
    )
    shape = Peak(
        "remote",
        np.array(pos),
        create_shapes(
            spectra,
            config.fitting,
            "remote",
            pos,
            ["gaussian", "gaussian"],
        ),
    )
    coordinates = [index.ravel() for index in np.indices((32, 32))]
    values = Cluster.evaluate_peaks([shape], Parameters.from_peaks([shape]), coordinates)[0]
    spectra.data += (np.asarray(remote)[:, None] * values).reshape(n_planes, 32, 32)
    peaks += (WorkingPeak("remote", "Duplicate", pos),)
    config.fitting.refine_iterations = 2
    config.fitting.steps = []
    config.clustering.contour_level = 0.1
    prepared = prepare_dataset_fit(spectra, peaks, config)
    assert len(prepared.clusters) > 1
    return spectra, peaks, config, prepared


def cpmg_like_dataset():
    """The two-cluster dataset over 20 planes of a CPMG-like relaxation-dispersion series.

    Amplitudes fall from a reference plane toward a plateau as the CPMG frequency rises,
    as in an effective-relaxation-rate experiment; the first plane is the reference
    (no relaxation delay), far above the others. One plane of one member is an outlier.
    """
    frequencies = np.array([0.0, *np.linspace(25.0, 1000.0, 19)])
    dispersion = 0.45 + 0.25 * (1 - np.exp(-frequencies / 200.0))
    series = np.where(frequencies == 0.0, 1.0, dispersion)
    amplitudes = np.vstack([5.0 * series, 3.0 * series**1.4])
    amplitudes[1, 11] *= 1.35  # an outlier the tile must keep recognisable
    return two_cluster_dataset(amplitudes, frequencies, 4.0 * series**0.8)


def decay_dataset():
    """The two-cluster dataset over 10 planes of an exponential decay (relaxation delays)."""
    delays = np.linspace(0.0, 0.18, 10)
    amplitudes = np.vstack([5.0 * np.exp(-delays / 0.08), 3.0 * np.exp(-delays / 0.05)])
    return two_cluster_dataset(amplitudes, delays, 4.0 * np.exp(-delays / 0.12))


def completed_result(formats, save_simulated, detailed=False):
    """Fit the two-cluster dataset once, as both the CLI pipeline and the GUI record it."""
    spectra, _peaks, config, prepared = two_cluster_dataset()
    config.output.formats = formats
    config.output.save_simulated = save_simulated
    config.output.detailed_evidence = detailed
    prepared = replace(
        prepared, snapshot=replace(prepared.snapshot, configuration=config.model_dump_json())
    )
    pipeline = run_pipeline(
        config,
        prepared.clusters,
        prepared.noise,
        prepared.parameters,
        noise_model=not_modelled("synthetic spectrum"),
    )
    gui = finish_dataset_fit(prepared, DatasetFitAttempt(pipeline.completed_fit, (), 1.0))
    return spectra, config, pipeline, gui


def peak_pair(
    separation: float,
    *,
    noise: float = 0.0,
    lineshape: LineshapeName = "gaussian",
    fit_phase: tuple[str, ...] = (),
    amplitudes: np.ndarray | None = None,
):
    """One cluster of two 1D peaks ``separation`` points apart, over four planes.

    ``fit_phase`` adds the cluster's phase on those axes (for lineshapes that have one),
    a single parameter shared by both peaks. ``amplitudes`` (2 × S) replaces the
    default four-plane profiles.
    """
    if amplitudes is None:
        amplitudes = np.array([[2.0, -1.0, 3.0, 0.5], [-1.0, 3.0, 0.5, 2.0]])
    n_planes = amplitudes.shape[1]

    def axis(size, direct):
        return SpectralParameters(
            size=size,
            sw=500,
            obs=500,
            car=0,
            aq_time=0.01,
            apocode=0,
            apodq1=0,
            apodq2=0,
            apodq3=0,
            p180=False,
            direct=direct,
            ft=True,
        )

    spectra = Spectra(
        dic={},
        data=np.zeros((n_planes, 101)),
        z_values=np.arange(float(n_planes)),
        params=[axis(n_planes, False), axis(101, True)],
    )
    peaks = []
    for name, point in zip(("A", "B"), (50 - separation / 2, 50 + separation / 2), strict=True):
        center = float(spectra.spectral_params[0].pts2ppm(point))
        peak = Peak(
            name=name,
            positions=np.array([center]),
            shapes=create_shapes(
                spectra,
                FitConfig(lineshape=lineshape, fit_phase=list(fit_phase)),
                name,
                [center],
                [lineshape],
            ),
        )
        peak.set_cluster_id(1)
        peaks.append(peak)
    params = Parameters.from_peaks(peaks)
    cluster = Cluster(1, peaks, [np.arange(101)], np.zeros((101, n_planes)))
    cluster.data = cluster.evaluate(params).T @ amplitudes
    cluster.data += noise * np.random.default_rng(59).normal(size=cluster.data.shape)
    return cluster, params


def write_as_cli(result, spectra, config, directory, metadata):
    """Call the shared writer with the arguments ``write_fit_run_outputs`` passes.

    It adds the GUI's peak identities, which the CLI does not record (it names peaks by
    engine name), so CLI- and GUI-argument exports of one run compare byte for byte.
    It does not exercise ``write_fit_run_outputs`` itself.
    """
    write_completed_outputs(
        result.completed_fit,
        directory,
        WriterConfig(formats=tuple(config.output.formats), precision=17),
        metadata=metadata,
        data_shape=spectra.data.shape,
        spectral_parameters=[axis.model_dump() for axis in spectra.params],
        plane_values=spectra.z_values,
        source_plane_indices=spectra.source_plane_indices,
        peak_identities=[
            {"engine_name": i.engine_name, "peak_id": i.working_id, "assignment": i.assignment}
            for i in result.inputs.identities
        ],
        save_simulated=config.output.save_simulated,
        detailed_evidence=config.output.detailed_evidence,
    )
