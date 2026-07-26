"""Fit workflow orchestration for PeakFit.

Public API:
- LoadedData: Container for all loaded fitting data
- load_data: Function to load spectrum, peaks, and compute clusters
- RunSummary: Statistics for a fitting run
- FitRun: Result of a fitting operation
- run_fit: Execute the fitting pipeline
- write_fit_run_outputs: Write outputs from a FitRun
- ProgressStart: Event emitted at pipeline start with total steps
- ClusterReview: Data for a cluster that needs review
- find_review_clusters: Identify clusters needing review
"""

from __future__ import annotations

import logging
import multiprocessing
from typing import TYPE_CHECKING, Any

from peakfit.auto_pick import auto_pick_peaks
from peakfit.auto_pick.logging import log_auto_pick_cycle
from peakfit.auto_pick.types import AutoPickCycleAction
from peakfit.engine.algorithms.clustering import create_clusters
from peakfit.engine.algorithms.evaluation import FitOutcomeClassification
from peakfit.engine.algorithms.noise import prepare_noise_level
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.spectrum import get_shape_names
from peakfit.fit.pipeline import PipelineResult, run_pipeline_iter
from peakfit.fit.results import capture_output_metadata
from peakfit.fit.run_models import ClusterReview, FitRun, LoadedData, ProgressStart, RunSummary
from peakfit.io.readers.peaks import read_list
from peakfit.io.readers.spectrum import read_spectra
from peakfit.io.state import default_state_path, save_state
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.orchestrator import write_fit_outputs
from peakfit.io.writers.run_files import write_readme, write_simulated_spectra
from peakfit.shared.exceptions import DataIOError
from peakfit.shared.paths import format_path

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from peakfit.auto_pick.types import AutoPickCycleReport
    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.shared.reporter import Reporter

    type AutoPickCallbackBuilder = Callable[
        [Spectra, float], Callable[[AutoPickCycleReport], AutoPickCycleAction]
    ]


def load_data(
    spectrum_path: Path,
    peaklist_path: Path | None,
    z_values_path: Path | None,
    config: PeakFitConfig,
    reporter: Reporter | None = None,
    auto_pick_callback_builder: AutoPickCallbackBuilder | None = None,
) -> LoadedData:
    """Load all data required for fitting.

    Args:
        spectrum_path: Path to spectrum file
        peaklist_path: Path to peak list file (optional)
        z_values_path: Path to Z-series file
        config: Full configuration object
        reporter: Optional progress reporter for auto-pick logs
        auto_pick_callback_builder: Optional callback builder for step-wise auto-pick control

    Returns:
        LoadedData container with spectra, peaks, noise, clusters, etc.
    """
    # 1. Load Spectra
    spectra = read_spectra(spectrum_path, z_values_path, config.exclude_planes)

    # 2. Estimate Noise
    computed_noise = prepare_noise_level(config.noise_level, spectra)
    if computed_noise is None:
        raise DataIOError("Could not estimate noise level.")

    noise_val = float(computed_noise)
    noise_source = "user-provided" if config.noise_level and config.noise_level > 0 else "estimated"

    # 3. Detect Lineshapes
    shape_names = get_shape_names(config.fitting, spectra)

    # 4. Contour Level
    contour = (
        config.clustering.contour_level
        if config.clustering.contour_level is not None
        else config.clustering.contour_factor * noise_val
    )

    # 5. Load or Auto-Pick Peaks
    if peaklist_path is None:
        if not config.auto_peak.enabled:
            raise DataIOError("No peak list provided and automatic peak picking is disabled.")
        if reporter is not None:
            reporter.action("Running automatic peak picking (no peak list provided)...")

        user_cycle_callback = (
            auto_pick_callback_builder(spectra, contour)
            if auto_pick_callback_builder is not None
            else None
        )

        def _cycle_callback(cycle: AutoPickCycleReport) -> AutoPickCycleAction:
            if reporter is not None:
                log_auto_pick_cycle(reporter, cycle)
            if user_cycle_callback is None:
                return AutoPickCycleAction()
            return user_cycle_callback(cycle)

        auto_result = auto_pick_peaks(
            spectra,
            shape_names,
            noise_val,
            contour,
            config,
            cycle_callback=_cycle_callback if (reporter or user_cycle_callback) else None,
        )
        peaks = auto_result.peaks
        if reporter is not None:
            stop_msg = " (stopped by user)" if auto_result.diagnostics.stopped_by_user else ""
            reporter.success(
                "Automatic peak picking completed"
                f"{stop_msg}: peaks={len(peaks)} "
                f"accepted_rois={auto_result.diagnostics.accepted_rois} "
                f"rejected_rois={auto_result.diagnostics.rejected_rois}"
            )
        if not peaks:
            if auto_result.diagnostics.stopped_by_user:
                raise DataIOError(
                    "Automatic peak picking was stopped before any peak was accepted."
                )
            raise DataIOError(
                "Automatic peak picking found no peaks above threshold. "
                "Lower auto_peak.start_threshold_sigma or provide a peak list."
            )
    else:
        peaks = read_list(peaklist_path, spectra, shape_names, config.fitting)

    # 6. Create Clusters
    clusters = create_clusters(spectra, peaks, contour)

    return LoadedData(
        spectra=spectra,
        peaks=peaks,
        noise=noise_val,
        noise_source=noise_source,
        shape_names=shape_names,
        contour_level=contour,
        clusters=clusters,
    )


HIGH_REDCHI = 5.0


def find_review_clusters(result: FitRun) -> list[ClusterReview]:
    """Project review rows from ordered final cluster outcomes."""
    reviews: list[ClusterReview] = []
    for outcome in sorted(result.outcome.clusters, key=lambda cluster: cluster.cluster_id):
        evaluation = outcome.analytical_evaluation
        redchi = evaluation.statistics.reduced_chi_squared if evaluation is not None else None
        at_bounds = [
            parameter.name
            for parameter in outcome.final_nonlinear_parameters
            if _is_at_boundary(parameter.value, parameter.min, parameter.max)
        ]
        non_converged = outcome.classification is not FitOutcomeClassification.CONVERGED
        needs_review = non_converged or (redchi is not None and redchi > HIGH_REDCHI) or at_bounds

        if not needs_review:
            continue

        if outcome.classification is FitOutcomeClassification.UNUSABLE:
            reason = "unusable"
        elif outcome.classification is FitOutcomeClassification.USABLE_NON_CONVERGED:
            reason = "not_converged"
        elif redchi is not None and redchi > HIGH_REDCHI:
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
    optimizer: str = "varpro",
    workers: int = -1,
    reporter: Reporter | None = None,
    progress_callback: Callable[[Any], None] | None = None,
) -> FitRun:
    """Execute the fitting pipeline.

    Args:
        data: Loaded spectrum, peaks, clusters, etc.
        config: Fitting configuration
        output_dir: Directory for outputs
        optimizer: Optimizer name (varpro or basin_hopping)
        workers: Number of parallel workers (-1 = all CPUs)
        reporter: Optional reporter for headless progress
        progress_callback: Optional callback for interactive progress.
            Receives ProgressStart at start, then FitResult for each fit.

    Returns:
        FitRun with fitting state, results, and summary
    """
    logger = logging.getLogger("peakfit")
    prev_level = logger.level
    logger.setLevel(logging.CRITICAL)

    try:
        params = Parameters.from_peaks(data.peaks, fixed=False)
        n_workers = workers if workers > 0 else multiprocessing.cpu_count()

        if reporter:
            reporter.info(f"Fitting {len(data.clusters)} clusters with {n_workers} workers.")

        # Emit progress start event with total steps
        if progress_callback:
            total_steps = _calc_total_steps(data, config)
            progress_callback(ProgressStart(total_steps, len(data.clusters), n_workers))

        items = _iter_pipeline(config, data, params, optimizer, n_workers)

        pipeline_result = _consume_pipeline(items, progress_callback)

        if pipeline_result is None:
            raise RuntimeError("Pipeline produced no result")

        outcome = pipeline_result.final_outcome
        if outcome is None:
            raise RuntimeError("Pipeline completed without a final fit outcome.")

        return FitRun(
            outcome=outcome,
            continuation_state=pipeline_result.state,
            output_dir=output_dir,
            spectra=data.spectra,
        )
    finally:
        logger.setLevel(prev_level)


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
    optimizer: str,
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
                data.peaks,
                optimizer=optimizer,
                executor=pool.imap_unordered,
            )
    else:
        yield from run_pipeline_iter(
            config,
            data.clusters,
            data.noise,
            params,
            data.peaks,
            optimizer=optimizer,
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


def _is_at_boundary(value: float, minimum: float, maximum: float, tol: float = 1e-6) -> bool:
    """Match scalar parameter bound checks using immutable final values."""
    return abs(value - minimum) < tol * (1 + abs(value)) or abs(value - maximum) < tol * (
        1 + abs(value)
    )


# =============================================================================
# Output Writing
# =============================================================================


def write_fit_run_outputs(
    fit_run: FitRun,
    spectra: Spectra,
    config: PeakFitConfig,
    input_paths: dict[str, Path],
    reporter: Reporter | None = None,
) -> None:
    """Write outputs from a FitRun.

    Args:
        fit_run: The fitting result to write
        spectra: Spectra data for output
        config: Configuration used for the fit
        input_paths: Dictionary of input file paths
        reporter: Optional reporter for progress updates
    """
    output_dir = fit_run.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    writer_config = WriterConfig(formats=tuple(config.output.formats))
    metadata = capture_output_metadata(config.model_dump(), input_paths)
    if reporter:
        reporter.action("Writing outputs...")

    write_fit_outputs(
        fit_run.outcome,
        output_dir,
        writer_config,
        metadata=metadata,
        z_values=spectra.z_values,
        summary=fit_run.summary,
    )

    if config.output.save_simulated:
        write_simulated_spectra(
            output_dir,
            spectra,
            fit_run.state.clusters,
            fit_run.state.scalar_params,
            reporter,
        )

    state_file = default_state_path(output_dir)
    save_state(state_file, fit_run.state)
    write_readme(output_dir, fit_run.summary)

    if reporter:
        reporter.success(f"Results written to [path]{format_path(output_dir)}[/path]")


__all__ = [
    "ClusterReview",
    "FitRun",
    "LoadedData",
    "ProgressStart",
    "RunSummary",
    "find_review_clusters",
    "load_data",
    "run_fit",
    "write_fit_run_outputs",
]
