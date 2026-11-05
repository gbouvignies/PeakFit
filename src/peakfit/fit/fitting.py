"""Fit workflow orchestration for PeakFit.

Public API:
- LoadedData: Container for all loaded fitting data
- load_data: Function to load spectrum, peaks, and compute clusters
- FitRun: Result of a fitting operation
- run_fit: Execute the fitting pipeline
- write_fit_run_outputs: Write outputs from a FitRun
- ProgressStart: Event emitted at pipeline start with total steps
- ClusterReview: Data for a cluster that needs review
- find_review_clusters: Identify clusters needing review
"""

from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING, Any

from peakfit.engine.algorithms.clustering import prepare_clusters
from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.engine.diagnostics.bounds import at_bound
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.spectrum import get_shape_names
from peakfit.engine.results import HIGH_REDUCED_CHI_SQUARED
from peakfit.fit.export import write_completed_outputs
from peakfit.fit.pipeline import PipelineResult, run_pipeline_iter
from peakfit.fit.run_models import ClusterReview, FitRun, LoadedData, ProgressStart
from peakfit.io.output_metadata import capture_output_metadata
from peakfit.io.readers.peaks import read_list
from peakfit.io.readers.spectrum import read_noise_model, read_spectra
from peakfit.io.writers.config import WriterConfig
from peakfit.shared.paths import format_path

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.engine.scientific.final_outcome import FinalFitOutcome
    from peakfit.shared.reporter import Reporter


def load_data(
    spectrum_path: Path,
    peaklist_path: Path,
    z_values_path: Path | None,
    config: PeakFitConfig,
) -> LoadedData:
    """Load all data required for fitting.

    Args:
        spectrum_path: Path to spectrum file
        peaklist_path: Path to peak list file
        z_values_path: Path to Z-series file
        config: Full configuration object

    Returns:
        LoadedData container with spectra, peaks, noise, clusters, etc.
    """
    # 1. Load Spectra
    spectra = read_spectra(spectrum_path, z_values_path, config.exclude_planes)

    # 2. Detect Lineshapes
    shape_names = get_shape_names(config.fitting, spectra)

    # 3. Load Peaks
    peaks = read_list(peaklist_path, spectra, shape_names, config.fitting)

    # 4. Noise, clustering threshold and clusters, shared with the GUI
    noise_val, contour, clusters = prepare_clusters(spectra, peaks, config)
    noise_source = "user-provided" if config.noise_level is not None else "estimated"

    return LoadedData(
        spectra=spectra,
        peaks=peaks,
        noise=noise_val,
        noise_source=noise_source,
        noise_model=read_noise_model(spectra.dic, spectra.data.shape),
        shape_names=shape_names,
        contour_level=contour,
        clusters=clusters,
    )


def find_review_clusters(fit_outcome: FinalFitOutcome) -> list[ClusterReview]:
    """Project review rows from ordered final cluster outcomes."""
    reviews: list[ClusterReview] = []
    for outcome in sorted(fit_outcome.clusters, key=lambda cluster: cluster.cluster_id):
        evaluation = outcome.analytical_evaluation
        redchi = evaluation.statistics.reduced_chi_squared if evaluation is not None else None
        at_bounds = [
            parameter.name
            for parameter in outcome.final_nonlinear_parameters
            if at_bound(parameter)
        ]
        non_converged = outcome.classification is not FitOutcomeClassification.CONVERGED
        uncertain = evaluation is not None and evaluation.uncertainty_warning is not None
        needs_review = (
            non_converged
            or uncertain
            or (redchi is not None and redchi > HIGH_REDUCED_CHI_SQUARED)
            or at_bounds
        )

        if not needs_review:
            continue

        if outcome.classification is FitOutcomeClassification.UNUSABLE:
            reason = "unusable"
        elif outcome.classification is FitOutcomeClassification.USABLE_NON_CONVERGED:
            reason = "not_converged"
        elif uncertain:
            reason = "uncertainty"
        elif redchi is not None and redchi > HIGH_REDUCED_CHI_SQUARED:
            reason = "high_chi"
        else:
            reason = "at_bounds"

        reviews.append(
            ClusterReview(
                cluster_id=str(outcome.cluster_id),
                peak_names=outcome.peak_names,
                classification=outcome.classification,
                reason=reason,
                redchi=redchi,
                at_bounds=at_bounds,
                unusable_reason=outcome.unusable_reason,
                termination_message=outcome.optimizer_provenance.termination_message,
            )
        )

    return reviews


# =============================================================================
# Fitting Pipeline
# =============================================================================


def run_fit(
    data: LoadedData,
    config: PeakFitConfig,
    output_dir: Path,
    *,
    workers: int = -1,
    reporter: Reporter | None = None,
    progress_callback: Callable[[Any], None] | None = None,
) -> FitRun:
    """Execute the fitting pipeline.

    Args:
        data: Loaded spectrum, peaks, clusters, etc.
        config: Fitting configuration
        output_dir: Directory for outputs
        workers: Number of parallel workers (-1 = all CPUs)
        reporter: Optional reporter for headless progress
        progress_callback: Optional callback for interactive progress.
            Receives ProgressStart at start, then FitResult for each fit.

    Returns:
        FitRun with fitting state, results, and summary
    """
    params = Parameters.from_peaks(data.peaks, fixed=False)
    n_workers = workers if workers > 0 else multiprocessing.cpu_count()

    if reporter:
        reporter.info(f"Fitting {len(data.clusters)} clusters with {n_workers} workers.")
        reporter.info(data.noise_model.description)

    # Emit progress start event with total steps
    if progress_callback:
        total_steps = _calc_total_steps(data, config)
        progress_callback(ProgressStart(total_steps, len(data.clusters), n_workers))

    items = _iter_pipeline(config, data, params, n_workers)

    pipeline_result = _consume_pipeline(items, progress_callback)

    if pipeline_result is None:
        raise RuntimeError("Pipeline produced no result")

    return FitRun(
        completed_fit=pipeline_result.completed_fit,
        output_dir=output_dir,
        spectra=data.spectra,
    )


def _calc_total_steps(data: LoadedData, config: PeakFitConfig) -> int:
    """Calculate total steps for progress display."""
    n_clusters = len(data.clusters) if data.clusters else 0
    if config.fitting.steps:
        n_passes = sum(step.iterations for step in config.fitting.steps)
    else:
        n_passes = config.fitting.refine_iterations
    return n_clusters * n_passes


def _iter_pipeline(
    config: PeakFitConfig,
    data: LoadedData,
    params: Parameters,
    n_workers: int,
) -> Iterator[Any]:
    """Iterate over pipeline, optionally in parallel."""
    if n_workers > 1:
        with multiprocessing.Pool(processes=n_workers) as pool:
            yield from run_pipeline_iter(
                config,
                data.clusters,
                data.noise,
                params,
                noise_model=data.noise_model,
                executor=pool.imap_unordered,
            )
    else:
        yield from run_pipeline_iter(
            config,
            data.clusters,
            data.noise,
            params,
            noise_model=data.noise_model,
            executor=None,
        )


def _consume_pipeline(
    items: Iterator[Any],
    callback: Callable[[Any], None] | None,
) -> PipelineResult | None:
    """Consume yielded progress items and return the final pipeline result."""
    result = None
    for item in items:
        if isinstance(item, PipelineResult):
            result = item
        elif callback is not None:
            callback(item)
    return result


# =============================================================================
# Output Writing
# =============================================================================


def write_fit_run_outputs(
    fit_run: FitRun,
    config: PeakFitConfig,
    input_paths: dict[str, Path],
    reporter: Reporter | None = None,
) -> None:
    """Write outputs from a FitRun.

    Args:
        fit_run: The fitting result to write
        config: Configuration used for the fit
        input_paths: Dictionary of input file paths
        reporter: Optional reporter for progress updates
    """
    output_dir = fit_run.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    writer_config = WriterConfig(formats=tuple(config.output.formats), precision=17)
    metadata = capture_output_metadata(config.model_dump(), input_paths)
    if reporter:
        reporter.action("Writing outputs...")

    spectra = fit_run.spectra
    write_completed_outputs(
        fit_run.completed_fit,
        output_dir,
        writer_config,
        metadata=metadata,
        data_shape=spectra.data.shape,
        spectral_parameters=[axis.model_dump() for axis in spectra.params],
        plane_values=spectra.z_values,
        source_plane_indices=spectra.source_plane_indices,
        save_simulated=config.output.save_simulated,
        detailed_evidence=config.output.detailed_evidence,
    )

    if reporter:
        reporter.success(f"Results written to [path]{format_path(output_dir)}[/path]")


__all__ = [
    "ClusterReview",
    "FitRun",
    "LoadedData",
    "ProgressStart",
    "find_review_clusters",
    "load_data",
    "run_fit",
    "write_fit_run_outputs",
]
