"""Assemble structured fit outputs from pipeline state."""

import hashlib
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.engine.algorithms.common import residuals
from peakfit.engine.algorithms.linear_algebra import calculate_amplitudes_with_uncertainty
from peakfit.engine.results import (
    compute_chi_squared,
    compute_degrees_of_freedom,
    compute_reduced_chi_squared,
)

try:
    __version__ = metadata.version("peakfit")
except metadata.PackageNotFoundError:
    __version__ = "unknown"

if TYPE_CHECKING:
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.param_id import ParameterId
    from peakfit.engine.domain.params_scalar import Parameter, Parameters
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.engine.domain.state import FittingState
    from peakfit.shared.typing import FloatArray

_AMPLITUDE_PARAM_PATTERN = re.compile(r"\.I\d+$")
_ZERO_VALUE_THRESHOLD = 1e-15
_POOR_RELATIVE_ERROR_THRESHOLD = 0.5

_RHAT_EXCELLENT_THRESHOLD = 1.01
_RHAT_ACCEPTABLE_THRESHOLD = 1.05

_ESS_EXCELLENT_THRESHOLD = 10000
_ESS_GOOD_THRESHOLD = 1000
_ESS_ACCEPTABLE_THRESHOLD = 400
_ESS_MARGINAL_THRESHOLD = 100

_RECOMMENDED_ESS_PER_CHAIN = 100
_LOW_ESS_PER_CHAIN = 10


class ParameterCategory(StrEnum):
    """Categories of fitting parameters for grouping and formatting."""

    LINESHAPE = "lineshape"
    AMPLITUDE = "amplitude"
    EXCHANGE = "exchange"
    RELAXATION = "relaxation"
    GLOBAL = "global"


class ConvergenceStatus(StrEnum):
    """MCMC convergence status categories."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    POOR = "poor"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ResidualStatistics:
    """Statistics computed from fit residuals."""

    raw_residuals: FloatArray | None = None
    normalized_residuals: FloatArray | None = None
    n_points: int = 0
    n_params: int = 0
    noise_level: float = 1.0

    @property
    def dof(self) -> int:
        """Degrees of freedom."""
        return compute_degrees_of_freedom(self.n_points, self.n_params)

    @property
    def sum_squared(self) -> float:
        """Sum of squared normalized residuals."""
        if self.normalized_residuals is None:
            return 0.0
        return compute_chi_squared(self.normalized_residuals)

    @property
    def rms(self) -> float:
        """Root mean square of raw residuals."""
        if self.raw_residuals is None or len(self.raw_residuals) == 0:
            return 0.0
        return float(np.sqrt(np.mean(self.raw_residuals**2)))

    @property
    def mean(self) -> float:
        """Mean of raw residuals."""
        if self.raw_residuals is None or len(self.raw_residuals) == 0:
            return 0.0
        return float(np.mean(self.raw_residuals))

    @property
    def std(self) -> float:
        """Standard deviation of raw residuals."""
        if self.raw_residuals is None or len(self.raw_residuals) == 0:
            return 0.0
        return float(np.std(self.raw_residuals))


@dataclass(slots=True)
class FitStatistics:
    """Fit quality statistics used by output writers."""

    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    aic: float | None = None
    bic: float | None = None
    log_likelihood: float | None = None
    n_data: int = 0
    n_params: int = 0
    residuals: ResidualStatistics = field(default_factory=ResidualStatistics)
    fit_converged: bool = True
    n_function_evals: int = 0
    fit_message: str = ""

    @property
    def dof(self) -> int:
        """Degrees of freedom."""
        return compute_degrees_of_freedom(self.n_data, self.n_params)


@dataclass(slots=True)
class ParameterEstimate:
    """A fitted parameter value with uncertainty and output metadata."""

    name: str
    value: float
    std_error: float
    unit: str = ""
    category: ParameterCategory = ParameterCategory.LINESHAPE
    ci_68_lower: float | None = None
    ci_68_upper: float | None = None
    ci_95_lower: float | None = None
    ci_95_upper: float | None = None
    min_bound: float = field(default_factory=lambda: -np.inf)
    max_bound: float = field(default_factory=lambda: np.inf)
    is_fixed: bool = False
    is_global: bool = False
    posterior_samples: FloatArray | None = None
    param_id: ParameterId | None = None

    @property
    def has_asymmetric_error(self) -> bool:
        """Check if asymmetric confidence intervals are available."""
        return self.ci_68_lower is not None and self.ci_68_upper is not None

    @property
    def relative_error(self) -> float | None:
        """Relative uncertainty, or None if value is zero."""
        if abs(self.value) < _ZERO_VALUE_THRESHOLD:
            return None
        return abs(self.std_error / self.value)

    def is_at_boundary(self, tolerance: float = 1e-6) -> bool:
        """Check if value is at or near fitting bounds."""
        if np.isinf(self.min_bound) and np.isinf(self.max_bound):
            return False
        at_min = abs(self.value - self.min_bound) < tolerance
        at_max = abs(self.value - self.max_bound) < tolerance
        return at_min or at_max

    @property
    def is_problematic(self) -> bool:
        """Check if parameter has potential reporting issues."""
        if self.is_fixed:
            return False
        if self.is_at_boundary():
            return True
        if self.std_error <= 0:
            return True
        rel_err = self.relative_error
        return bool(rel_err is not None and rel_err > _POOR_RELATIVE_ERROR_THRESHOLD)


@dataclass(slots=True)
class AmplitudeEstimate:
    """Fitted amplitude for one peak at one plane."""

    peak_name: str
    plane_index: int
    z_value: float | None
    value: float
    std_error: float
    ci_68_lower: float | None = None
    ci_68_upper: float | None = None


@dataclass(slots=True)
class ClusterEstimates:
    """Parameter and amplitude estimates for one fitted cluster."""

    cluster_id: int
    peak_names: list[str]
    lineshape_params: list[ParameterEstimate]
    amplitudes: list[AmplitudeEstimate] = field(default_factory=list)
    correlation_matrix: FloatArray | None = None
    correlation_param_names: list[str] = field(default_factory=list)

    @property
    def n_peaks(self) -> int:
        """Number of peaks in cluster."""
        return len(self.peak_names)


@dataclass(slots=True)
class ParameterDiagnostic:
    """Convergence diagnostics for a single MCMC parameter."""

    name: str
    rhat: float | None = None
    ess_bulk: float | None = None
    ess_tail: float | None = None
    status: ConvergenceStatus = ConvergenceStatus.UNKNOWN
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_values(
        cls,
        name: str,
        rhat: float | None,
        ess_bulk: float | None,
        ess_tail: float | None = None,
        n_chains: int = 4,
    ) -> ParameterDiagnostic:
        """Create diagnostics with automatic status and warnings."""
        warnings = []
        status = ConvergenceStatus.UNKNOWN

        if rhat is None or ess_bulk is None:
            return cls(name=name, rhat=rhat, ess_bulk=ess_bulk, ess_tail=ess_tail)

        if rhat <= _RHAT_EXCELLENT_THRESHOLD and ess_bulk >= _ESS_EXCELLENT_THRESHOLD:
            status = ConvergenceStatus.EXCELLENT
        elif rhat <= _RHAT_EXCELLENT_THRESHOLD and ess_bulk >= _ESS_GOOD_THRESHOLD:
            status = ConvergenceStatus.GOOD
        elif rhat <= _RHAT_ACCEPTABLE_THRESHOLD and ess_bulk >= _ESS_ACCEPTABLE_THRESHOLD:
            status = ConvergenceStatus.ACCEPTABLE
        elif rhat <= _RHAT_ACCEPTABLE_THRESHOLD and ess_bulk >= _ESS_MARGINAL_THRESHOLD:
            status = ConvergenceStatus.MARGINAL
        else:
            status = ConvergenceStatus.POOR

        if rhat > _RHAT_EXCELLENT_THRESHOLD:
            warnings.append(
                f"R-hat = {rhat:.4f} (should be <= {_RHAT_EXCELLENT_THRESHOLD:.2f}). "
                "Chains have not mixed well."
            )
        if rhat > _RHAT_ACCEPTABLE_THRESHOLD:
            warnings.append(
                f"R-hat = {rhat:.4f} is very high (> {_RHAT_ACCEPTABLE_THRESHOLD:.2f}). "
                "Results should not be trusted."
            )

        recommended_ess = _RECOMMENDED_ESS_PER_CHAIN * n_chains
        if ess_bulk < recommended_ess:
            warnings.append(
                f"ESS_bulk = {ess_bulk:.0f} (recommended >= {recommended_ess:.0f}). "
                "Consider more samples."
            )
        if ess_bulk < _LOW_ESS_PER_CHAIN * n_chains:
            warnings.append(
                f"ESS_bulk = {ess_bulk:.0f} is very low. Posterior estimates are highly uncertain."
            )

        return cls(
            name=name,
            rhat=rhat,
            ess_bulk=ess_bulk,
            ess_tail=ess_tail,
            status=status,
            warnings=warnings,
        )


@dataclass(slots=True)
class MCMCDiagnostics:
    """MCMC diagnostics embedded in fit output summaries."""

    n_chains: int
    n_samples: int
    burn_in: int
    parameter_diagnostics: list[ParameterDiagnostic] = field(default_factory=list)
    overall_status: ConvergenceStatus = ConvergenceStatus.UNKNOWN
    burn_in_method: str = "manual"
    burn_in_details: dict[str, Any] = field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        """Total number of post-burn-in samples across all chains."""
        return self.n_chains * self.n_samples

    @property
    def converged(self) -> bool:
        """Check if MCMC has converged."""
        return self.overall_status in (
            ConvergenceStatus.EXCELLENT,
            ConvergenceStatus.GOOD,
            ConvergenceStatus.ACCEPTABLE,
        )

    @property
    def all_warnings(self) -> list[str]:
        """Collect all warnings from all parameters."""
        warnings = []
        for diag in self.parameter_diagnostics:
            warnings.extend(diag.warnings)
        return warnings

    def update_overall_status(self) -> None:
        """Recompute overall status from parameter diagnostics."""
        if not self.parameter_diagnostics:
            self.overall_status = ConvergenceStatus.UNKNOWN
            return

        status_order = [
            ConvergenceStatus.EXCELLENT,
            ConvergenceStatus.GOOD,
            ConvergenceStatus.ACCEPTABLE,
            ConvergenceStatus.MARGINAL,
            ConvergenceStatus.POOR,
        ]

        worst_idx = 0
        for diag in self.parameter_diagnostics:
            if diag.status in status_order:
                worst_idx = max(worst_idx, status_order.index(diag.status))

        self.overall_status = status_order[worst_idx]

    def get_problematic_parameters(self) -> list[ParameterDiagnostic]:
        """Get parameters with poor or marginal convergence."""
        return [
            d
            for d in self.parameter_diagnostics
            if d.status in (ConvergenceStatus.POOR, ConvergenceStatus.MARGINAL)
        ]


@dataclass
class RunMetadata:
    """Metadata about a fitting run for reproducible outputs."""

    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    software_version: str = __version__
    git_commit: str | None = None
    python_version: str = sys.version
    platform: str = field(default_factory=platform.platform)
    input_files: dict[str, dict[str, str]] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    command_line: str = ""
    run_duration_seconds: float | None = None

    @classmethod
    def capture(cls, config: dict[str, Any] | None = None) -> RunMetadata:
        """Capture current environment metadata."""
        git_commit = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()[:12]
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass

        return cls(
            timestamp=datetime.now(UTC).isoformat(),
            software_version=__version__,
            git_commit=git_commit,
            python_version=sys.version,
            platform=platform.platform(),
            configuration=config or {},
        )

    def add_input_file(self, name: str, path: Path) -> None:
        """Add an input file with its checksum."""
        if path.exists():
            self.input_files[name] = {
                "path": str(path.name),
                "checksum_sha256": _compute_file_checksum(path),
            }


@dataclass
class FitResults:
    """Complete output data from a fitting run."""

    metadata: RunMetadata = field(default_factory=RunMetadata)
    method: str = "least_squares"
    clusters: list[ClusterEstimates] = field(default_factory=list)
    statistics: list[FitStatistics] = field(default_factory=list)
    global_statistics: FitStatistics | None = None
    mcmc_diagnostics: list[MCMCDiagnostics] = field(default_factory=list)
    z_values: FloatArray | None = None

    @property
    def n_clusters(self) -> int:
        """Number of fitted clusters."""
        return len(self.clusters)

    @property
    def n_peaks(self) -> int:
        """Total number of peaks across all clusters."""
        return sum(c.n_peaks for c in self.clusters)

    @property
    def has_converged(self) -> bool:
        """Check if all MCMC analyses converged."""
        if not self.mcmc_diagnostics:
            return True
        return all(d.converged for d in self.mcmc_diagnostics)


def build_fit_results(
    state: FittingState,
    spectra: Spectra,
    config: dict[str, Any],
    input_files: dict[str, Path],
) -> FitResults:
    """Build structured fit results for output writers."""
    metadata = _capture_metadata(config, input_files)
    noise = state.noise or 1.0
    z_values = spectra.z_values

    clusters: list[ClusterEstimates] = []
    statistics: list[FitStatistics] = []
    for cluster in state.clusters:
        cluster_estimate, cluster_stats = _build_cluster_output(
            cluster=cluster,
            params=state.scalar_params,
            noise=noise,
            z_values=z_values,
        )
        clusters.append(cluster_estimate)
        statistics.append(cluster_stats)

    if not clusters:
        msg = "No cluster estimates available for output."
        raise ValueError(msg)

    return FitResults(
        metadata=metadata,
        clusters=clusters,
        statistics=statistics,
        global_statistics=_build_global_statistics(statistics),
        z_values=z_values,
    )


def _capture_metadata(config: dict[str, Any], input_files: dict[str, Path]) -> RunMetadata:
    metadata = RunMetadata.capture(config)
    for name, path in input_files.items():
        if isinstance(path, Path) and path.exists():
            metadata.add_input_file(name, path)
    return metadata


def _build_cluster_output(
    cluster: Cluster,
    params: Parameters,
    noise: float,
    z_values: np.ndarray | None,
) -> tuple[ClusterEstimates, FitStatistics]:
    """Build parameter, amplitude, and fit-statistics output for one cluster."""
    shapes = cluster.evaluate(params)
    amplitudes, amplitude_errors, _covariance = calculate_amplitudes_with_uncertainty(
        shapes, cluster.corrected_data.real.astype(float), noise
    )

    cluster_stats = _build_cluster_statistics(cluster, params, noise)
    scale_factor = (
        np.sqrt(cluster_stats.reduced_chi_squared)
        if cluster_stats.reduced_chi_squared > 1.0
        else 1.0
    )
    series_z_values = z_values if z_values is not None else np.arange(amplitudes.shape[1])

    lineshape_params: list[ParameterEstimate] = []
    amplitudes_out: list[AmplitudeEstimate] = []
    for peak_index, peak in enumerate(cluster.peaks):
        lineshape_params.extend(_extract_peak_parameters(peak.name, params))

        for plane_index in range(amplitudes.shape[1]):
            z_value = (
                float(series_z_values[plane_index])
                if plane_index < len(series_z_values)
                else float(plane_index)
            )
            amplitudes_out.append(
                AmplitudeEstimate(
                    peak_name=peak.name,
                    plane_index=plane_index,
                    z_value=z_value,
                    value=float(amplitudes[peak_index, plane_index]),
                    std_error=float(amplitude_errors[peak_index]) * float(scale_factor),
                )
            )

    lineshape_params.extend(_extract_cluster_parameters(cluster.cluster_id, params))

    return (
        ClusterEstimates(
            cluster_id=cluster.cluster_id,
            peak_names=[p.name for p in cluster.peaks],
            lineshape_params=lineshape_params,
            amplitudes=amplitudes_out,
        ),
        cluster_stats,
    )


def _extract_peak_parameters(peak_name: str, params: Parameters) -> list[ParameterEstimate]:
    estimates: list[ParameterEstimate] = []
    for param_name, param in params.items():
        if not param_name.startswith(peak_name + ".") or _is_amplitude_parameter(param_name, param):
            continue
        estimates.append(_to_parameter_estimate(param_name, param))
    return estimates


def _extract_cluster_parameters(cluster_id: int, params: Parameters) -> list[ParameterEstimate]:
    estimates: list[ParameterEstimate] = []
    cluster_prefix = f"cluster_{cluster_id}."
    for param_name, param in params.items():
        if param_name.startswith(cluster_prefix):
            estimates.append(_to_parameter_estimate(param_name, param))
    return estimates


def _to_parameter_estimate(param_name: str, param: Parameter) -> ParameterEstimate:
    return ParameterEstimate(
        name=param_name,
        value=param.value,
        std_error=param.stderr,
        unit=param.unit,
        category=ParameterCategory.LINESHAPE,
        min_bound=param.min,
        max_bound=param.max,
        is_fixed=not param.vary,
        is_global=_is_global_parameter(param_name, param.param_id),
        param_id=param.param_id,
    )


def _is_amplitude_parameter(param_name: str, param: Parameter) -> bool:
    return (param.param_id is not None and param.param_id.label == "I") or bool(
        _AMPLITUDE_PARAM_PATTERN.search(param_name)
    )


def _is_global_parameter(param_name: str, param_id: Any | None) -> bool:
    if param_id is not None and getattr(param_id, "cluster_id", None) is not None:
        return True
    return param_name.startswith("cluster_")


def _build_cluster_statistics(
    cluster: Cluster,
    params: Parameters,
    noise: float,
) -> FitStatistics:
    """Build statistics for a cluster."""
    n_lineshape_params = _count_varying_lineshape_params(cluster, params)
    n_peaks = len(cluster.peaks)
    n_series = cluster.corrected_data.shape[0] if cluster.corrected_data.ndim > 1 else 1
    n_params = n_lineshape_params + (n_peaks * n_series)
    n_data = cluster.corrected_data.size

    normalized_residuals: np.ndarray | None = None
    try:
        normalized_residuals = residuals(params, cluster, noise)
    except (ValueError, KeyError, AttributeError):
        normalized_residuals = None

    chi_squared = (
        compute_chi_squared(normalized_residuals) if normalized_residuals is not None else 0.0
    )
    aic, bic, log_likelihood = _compute_information_criteria(
        chi_squared=chi_squared,
        n_data=n_data,
        n_params=n_params,
        noise=noise,
    )

    residual_stats = ResidualStatistics(
        raw_residuals=(normalized_residuals * noise) if normalized_residuals is not None else None,
        normalized_residuals=normalized_residuals,
        n_points=n_data,
        n_params=n_params,
        noise_level=noise,
    )

    return FitStatistics(
        chi_squared=chi_squared,
        reduced_chi_squared=compute_reduced_chi_squared(chi_squared, n_data, n_params),
        aic=aic,
        bic=bic,
        log_likelihood=log_likelihood,
        n_data=n_data,
        n_params=n_params,
        residuals=residual_stats,
        fit_converged=True,
        n_function_evals=0,
        fit_message="Statistics computed from fitted model",
    )


def _count_varying_lineshape_params(cluster: Cluster, params: Parameters) -> int:
    cluster_peak_names = {p.name for p in cluster.peaks}
    n_params = 0
    for param_name, param in params.items():
        if not param.vary or _is_amplitude_parameter(param_name, param):
            continue
        belongs_to_peak = any(param_name.startswith(f"{name}.") for name in cluster_peak_names)
        belongs_to_cluster = param_name.startswith(f"cluster_{cluster.cluster_id}.")
        if belongs_to_peak or belongs_to_cluster:
            n_params += 1
    return n_params


def _build_global_statistics(statistics: list[FitStatistics]) -> FitStatistics:
    total_chi_sq = sum(stats.chi_squared for stats in statistics)
    total_params = sum(stats.n_params for stats in statistics)
    total_data = sum(stats.n_data for stats in statistics)
    total_nfev = sum(stats.n_function_evals for stats in statistics)
    all_converged = all(stats.fit_converged for stats in statistics)

    total_log_likelihood: float | None = None
    total_aic: float | None = None
    total_bic: float | None = None
    if statistics and all(stats.log_likelihood is not None for stats in statistics):
        total_log_likelihood = float(
            sum(stats.log_likelihood for stats in statistics if stats.log_likelihood is not None)
        )
        total_aic = -2.0 * total_log_likelihood + 2.0 * total_params
        if total_data > 0:
            total_bic = -2.0 * total_log_likelihood + total_params * float(np.log(total_data))

    return FitStatistics(
        chi_squared=total_chi_sq,
        reduced_chi_squared=compute_reduced_chi_squared(total_chi_sq, total_data, total_params),
        aic=total_aic,
        bic=total_bic,
        log_likelihood=total_log_likelihood,
        n_data=total_data,
        n_params=total_params,
        fit_converged=all_converged,
        n_function_evals=total_nfev,
    )


def _compute_information_criteria(
    chi_squared: float,
    n_data: int,
    n_params: int,
    noise: float,
) -> tuple[float | None, float | None, float | None]:
    """Compute information criteria under Gaussian residual assumptions."""
    if n_data <= 0 or n_params < 0 or noise <= 0:
        return None, None, None

    log_likelihood = -0.5 * chi_squared - n_data * np.log(noise) - 0.5 * n_data * np.log(2 * np.pi)
    aic = -2.0 * log_likelihood + 2.0 * n_params
    bic = -2.0 * log_likelihood + n_params * np.log(n_data)
    return float(aic), float(bic), float(log_likelihood)


def _compute_file_checksum(path: Path, algorithm: str = "sha256") -> str:
    """Compute checksum of a file."""
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "AmplitudeEstimate",
    "ClusterEstimates",
    "ConvergenceStatus",
    "FitResults",
    "FitStatistics",
    "MCMCDiagnostics",
    "ParameterCategory",
    "ParameterDiagnostic",
    "ParameterEstimate",
    "ResidualStatistics",
    "RunMetadata",
    "build_fit_results",
]
