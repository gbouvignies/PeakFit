"""Qt-free synthetic inputs and completed results shared by scientific and GUI tests."""

from dataclasses import replace

import numpy as np

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, LineshapeName, PeakFitConfig
from peakfit.engine.domain.fit_steps import FitStep
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.fit.pipeline import run_pipeline
from peakfit.gui.dataset_fit import DatasetFitAttempt, finish_dataset_fit
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import WorkingPeak


def overlapping_gaussians():
    """Two overlapping synthetic Gaussians across three planes with repeated values."""
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
        for size, direct in ((3, False), (32, False), (32, True))
    ]
    spectra = Spectra(
        dic={},
        data=np.zeros((3, 32, 32)),
        z_values=np.array([2.0, 0.0, 2.0]),
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
    amplitudes = np.array([[5.0, -3.0, 2.0], [3.0, -2.0, 4.0]])
    rng = np.random.default_rng(42)
    spectra.data[:] = (shapes.T @ amplitudes).T.reshape(3, 32, 32)
    spectra.data += rng.normal(0, 0.002, spectra.data.shape)
    working = tuple(
        WorkingPeak(str(i), "Duplicate", (pos[0] + 0.003, pos[1] - 0.003))
        for i, pos in enumerate(positions)
    )
    return spectra, working, config


def two_cluster_dataset():
    """The overlapping pair plus a remote peak, prepared as more than one cluster."""
    spectra, peaks, config = overlapping_gaussians()
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
    spectra.data += (np.array([4.0, -2.0, 3.0])[:, None] * values).reshape(3, 32, 32)
    peaks += (WorkingPeak("remote", "Duplicate", pos),)
    config.fitting.refine_iterations = 2
    config.fitting.steps = []
    config.clustering.contour_level = 0.1
    prepared = prepare_dataset_fit(spectra, peaks, config)
    assert len(prepared.clusters) > 1
    return spectra, peaks, config, prepared


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
    )
    gui = finish_dataset_fit(prepared, DatasetFitAttempt(pipeline.completed_fit, (), 1.0))
    return spectra, config, pipeline, gui


def peak_pair(
    separation: float,
    *,
    noise: float = 0.0,
    lineshape: LineshapeName = "gaussian",
    fit_phase: tuple[str, ...] = (),
):
    """One cluster of two 1D peaks ``separation`` points apart, over four planes.

    ``fit_phase`` adds the cluster's phase on those axes (for lineshapes that have one),
    a single parameter shared by both peaks.
    """

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
        data=np.zeros((4, 101)),
        z_values=np.arange(4.0),
        params=[axis(4, False), axis(101, True)],
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
    cluster = Cluster(1, peaks, [np.arange(101)], np.zeros((101, 4)))
    amplitudes = np.array([[2.0, -1.0, 3.0, 0.5], [-1.0, 3.0, 0.5, 2.0]])
    cluster.data = cluster.evaluate(params).T @ amplitudes
    cluster.data += noise * np.random.default_rng(59).normal(size=cluster.data.shape)
    return cluster, params
