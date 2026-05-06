"""Builder for constructing FitResults from pipeline data.

This module provides the FitResultsBuilder class that constructs
FitResults objects from existing fitting pipeline outputs, bridging
the gap between the old and new output systems.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.engine.algorithms.linear_algebra import calculate_amplitudes_with_uncertainty
from peakfit.engine.fitting.computation import residuals
from peakfit.engine.results import (
    AmplitudeEstimate,
    ClusterEstimates,
    FitMethod,
    FitResults,
    FitStatistics,
    MCMCDiagnostics,
    ParameterCategory,
    ParameterDiagnostic,
    ParameterEstimate,
    ResidualStatistics,
    RunMetadata,
    compute_chi_squared,
    compute_reduced_chi_squared,
)

if TYPE_CHECKING:
    from peakfit.engine.domain.cluster import Cluster
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.engine.domain.state import FittingState
    from peakfit.engine.results import MCMCAnalysisResult

_AMPLITUDE_PARAM_PATTERN = re.compile(r"\.I\d+$")


@dataclass
class FitResultsBuilder:
    """Builder for constructing FitResults from pipeline data.

    This builder bridges between the existing fitting pipeline outputs
    and the new structured FitResults format.

    Example:
        >>> builder = FitResultsBuilder()
        >>> builder.set_metadata(config_dict, input_files)
        >>> builder.set_spectra(spectra)
        >>> for cluster in clusters:
        ...     builder.add_cluster(cluster, params, noise)
        >>> results = builder.build()
    """

    # Collected data
    _metadata: RunMetadata | None = None
    _spectra_z_values: np.ndarray | None = None
    _cluster_estimates: list[ClusterEstimates] = field(default_factory=list)
    _cluster_statistics: list[FitStatistics] = field(default_factory=list)
    _mcmc_diagnostics: list[MCMCDiagnostics] = field(default_factory=list)
    _fit_method: FitMethod = FitMethod.LEAST_SQUARES
    _config: dict[str, Any] = field(default_factory=dict)

    def set_metadata(
        self,
        config: dict[str, Any] | None = None,
        input_files: dict[str, Path] | None = None,
        command_line: str = "",
    ) -> FitResultsBuilder:
        """Set run metadata.

        Args:
            config: Configuration dictionary
            input_files: Dictionary mapping names to file paths
            command_line: Command line string

        Returns:
        -------
            Self for chaining
        """
        self._metadata = RunMetadata.capture(config)
        self._metadata.command_line = command_line
        self._config = config or {}

        if input_files:
            for name, path in input_files.items():
                if isinstance(path, Path) and path.exists():
                    self._metadata.add_input_file(name, path)

        return self

    def set_spectra(self, spectra: Spectra) -> FitResultsBuilder:
        """Set spectra information for z-values.

        Args:
            spectra: Spectra object with z_values

        Returns:
        -------
            Self for chaining
        """
        self._spectra_z_values = spectra.z_values
        return self

    def set_z_values(self, z_values: np.ndarray) -> FitResultsBuilder:
        """Set z-values directly.

        Args:
            z_values: Array of z-dimension values

        Returns:
        -------
            Self for chaining
        """
        self._spectra_z_values = z_values
        return self

    def set_fit_method(self, method: FitMethod) -> FitResultsBuilder:
        """Set the fitting method used.

        Args:
            method: Fitting method enum

        Returns:
        -------
            Self for chaining
        """
        self._fit_method = method
        return self

    def add_cluster(
        self,
        cluster: Cluster,
        params: Parameters,
        noise: float,
        scipy_result: Any | None = None,
    ) -> FitResultsBuilder:
        """Add a cluster's results.

        Args:
            cluster: Cluster object with peaks and data
            params: Fitted parameters
            noise: Noise level for amplitude uncertainty
            scipy_result: Optional scipy OptimizeResult for statistics

        Returns:
        -------
            Self for chaining
        """
        # Extract parameter estimates for this cluster's peaks
        all_lineshape_params: list[ParameterEstimate] = []
        all_amplitudes: list[AmplitudeEstimate] = []

        # Compute amplitudes with uncertainties
        shapes = cluster.evaluate(params)
        amplitudes, amplitudes_err, _covariance = calculate_amplitudes_with_uncertainty(
            shapes, cluster.corrected_data.real.astype(float), noise
        )

        z_values = self._spectra_z_values
        if z_values is None:
            z_values = np.arange(amplitudes.shape[1])

        # Build cluster statistics
        cluster_stats = self._build_cluster_statistics(cluster, params, scipy_result, noise)
        self._cluster_statistics.append(cluster_stats)

        # Scale amplitude uncertainties if reduced chi-squared > 1
        # This accounts for underestimated noise or lack of fit
        scale_factor = 1.0
        if cluster_stats.reduced_chi_squared > 1.0:
            scale_factor = np.sqrt(cluster_stats.reduced_chi_squared)

        for i, peak in enumerate(cluster.peaks):
            # Extract lineshape parameters for this peak
            peak_params = self._extract_peak_parameters(peak.name, params)
            all_lineshape_params.extend(peak_params)

            # Extract amplitudes
            n_series = amplitudes.shape[1]
            for j in range(n_series):
                amp = float(amplitudes[i, j])
                # amplitudes_err[i] is scalar per peak
                # Scale the error by sqrt(redchi)
                amp_err = float(amplitudes_err[i]) * scale_factor
                z_val = float(z_values[j]) if j < len(z_values) else float(j)
                all_amplitudes.append(
                    AmplitudeEstimate(
                        peak_name=peak.name,
                        plane_index=j,
                        z_value=z_val,
                        value=amp,
                        std_error=amp_err,
                    )
                )

        # Extract cluster-level shared parameters (e.g., phase terms).
        all_lineshape_params.extend(self._extract_cluster_parameters(cluster.cluster_id, params))

        # Build cluster estimates
        cluster_est = ClusterEstimates(
            cluster_id=cluster.cluster_id,
            peak_names=[p.name for p in cluster.peaks],
            lineshape_params=all_lineshape_params,
            amplitudes=all_amplitudes,
        )
        self._cluster_estimates.append(cluster_est)

        return self

    def add_cluster_from_state(
        self,
        state: FittingState,
        noise: float | None = None,
    ) -> FitResultsBuilder:
        """Add cluster results from a FittingState object.

        Args:
            state: FittingState containing fitted clusters
            noise: Noise level (uses state.noise if not provided)

        Returns:
        -------
            Self for chaining
        """
        noise_val = noise if noise is not None else (state.noise or 1.0)
        # Use rich scalar parameters if available, otherwise we'd need to reconstruct
        # But FittingState now guarantees scalar_params existence (default empty)
        # If it's empty, we might have issues, but assuming valid state.
        params = state.scalar_params
        for cluster in state.clusters:
            self.add_cluster(cluster, params, noise_val)
        return self

    def add_mcmc_results(
        self,
        mcmc_result: MCMCAnalysisResult,
    ) -> FitResultsBuilder:
        """Add MCMC analysis results.

        Updates the stored FitResults with MCMC-derived parameter estimates
        (medians), standard errors, and credible intervals.

        Args:
            mcmc_result: MCMCAnalysisResult from mcmc_analysis_service

        Returns:
        -------
            Self for chaining
        """
        self._fit_method = FitMethod.MCMC

        for i, cluster_result in enumerate(mcmc_result.cluster_results):
            # Ensure we have a matching existing cluster estimate
            if i >= len(self._cluster_estimates):
                continue

            cluster_est = self._cluster_estimates[i]
            uncertainty = cluster_result.result

            # 1. Update Parameters
            self._update_lineshape_params(cluster_est, uncertainty)
            self._update_amplitude_params(cluster_est, uncertainty)

            # 2. Extract Diagnostics
            cluster_diags = self._extract_cluster_diagnostics(uncertainty)
            if cluster_diags:
                mcmc_diag = self._build_cluster_mcmc_diagnostics(uncertainty, cluster_diags)
                self._mcmc_diagnostics.append(mcmc_diag)

        return self

    def _update_lineshape_params(self, cluster_est: ClusterEstimates, uncertainty: Any) -> None:
        """Update lineshape estimates with MCMC results."""
        for idx, name in enumerate(uncertainty.parameter_names):
            # Check if this is a lineshape parameter
            target_param = next((p for p in cluster_est.lineshape_params if p.name == name), None)

            if target_param:
                target_param.value = float(uncertainty.values[idx])
                target_param.std_error = float(uncertainty.std_errors[idx])
                target_param.ci_68_lower = float(uncertainty.confidence_intervals_68[idx, 0])
                target_param.ci_68_upper = float(uncertainty.confidence_intervals_68[idx, 1])
                target_param.ci_95_lower = float(uncertainty.confidence_intervals_95[idx, 0])
                target_param.ci_95_upper = float(uncertainty.confidence_intervals_95[idx, 1])

    def _update_amplitude_params(self, cluster_est: ClusterEstimates, uncertainty: Any) -> None:
        """Update amplitude estimates with MCMC results."""
        for amp in cluster_est.amplitudes:
            # Construct expected name in MCMC result
            amp_name = f"{amp.peak_name}.F1.I{amp.plane_index}"

            try:
                found_idx = uncertainty.parameter_names.index(amp_name)
            except ValueError:
                found_idx = -1

            if found_idx != -1:
                amp.value = float(uncertainty.values[found_idx])
                amp.std_error = float(uncertainty.std_errors[found_idx])
                amp.ci_68_lower = float(uncertainty.confidence_intervals_68[found_idx, 0])
                amp.ci_68_upper = float(uncertainty.confidence_intervals_68[found_idx, 1])

    def _extract_cluster_diagnostics(self, uncertainty: Any) -> list[ParameterDiagnostic]:
        """Extract diagnostics from uncertainty result."""
        param_diagnostics = []
        diag = uncertainty.mcmc_diagnostics

        for idx, param_name in enumerate(uncertainty.parameter_names):
            rhat = 1.0
            ess_bulk = 1000.0
            ess_tail = 1000.0

            if diag is not None:
                if idx < len(diag.rhat):
                    rhat = float(diag.rhat[idx])
                if idx < len(diag.ess_bulk):
                    ess_bulk = float(diag.ess_bulk[idx])
                if idx < len(diag.ess_tail):
                    ess_tail = float(diag.ess_tail[idx])

            param_diag = ParameterDiagnostic.from_values(
                name=param_name,
                rhat=rhat,
                ess_bulk=ess_bulk,
                ess_tail=ess_tail,
            )
            param_diagnostics.append(param_diag)

        return param_diagnostics

    def _build_cluster_mcmc_diagnostics(
        self,
        uncertainty: Any,
        param_diagnostics: list[ParameterDiagnostic],
    ) -> MCMCDiagnostics:
        """Build MCMC diagnostics for a single cluster."""
        burn_in = 500
        n_samples = 1000
        n_chains = 4

        if uncertainty.burn_in_info:
            burn_in = uncertainty.burn_in_info.get("burn_in", 500)
        if uncertainty.mcmc_samples is not None:
            n_samples = uncertainty.mcmc_samples.shape[0]
        if uncertainty.mcmc_diagnostics is not None:
            n_chains = uncertainty.mcmc_diagnostics.n_chains

        mcmc_diag = MCMCDiagnostics(
            n_chains=n_chains,
            n_samples=n_samples,
            burn_in=burn_in,
            parameter_diagnostics=param_diagnostics,
        )
        mcmc_diag.update_overall_status()
        return mcmc_diag

    def _extract_peak_parameters(
        self,
        peak_name: str,
        params: Parameters,
    ) -> list[ParameterEstimate]:
        """Extract parameter estimates for a specific peak.

        Args:
            peak_name: Peak identifier
            params: Parameters object

        Returns:
        -------
            List of ParameterEstimate objects
        """
        estimates: list[ParameterEstimate] = []

        for param_name, param in params.items():
            # Check if this parameter belongs to the peak (dot-notation: "peak_name.axis.type")
            if not param_name.startswith(peak_name + "."):
                continue
            # Keep amplitude series in the dedicated amplitudes output channel only.
            if (
                param.param_id is not None and param.param_id.label == "I"
            ) or _AMPLITUDE_PARAM_PATTERN.search(param_name):
                continue

            estimates.append(
                ParameterEstimate(
                    name=param_name,
                    value=param.value,
                    std_error=param.stderr,
                    unit=param.unit,
                    category=ParameterCategory.LINESHAPE,
                    min_bound=param.min,
                    max_bound=param.max,
                    is_fixed=not param.vary,
                    is_global=self._is_global_parameter(param_name, param.param_id),
                    param_id=param.param_id,
                )
            )

        return estimates

    def _extract_cluster_parameters(
        self,
        cluster_id: int,
        params: Parameters,
    ) -> list[ParameterEstimate]:
        """Extract cluster-level shared parameter estimates.

        Args:
            cluster_id: Cluster identifier
            params: Parameters object

        Returns:
        -------
            List of ParameterEstimate objects for cluster-scoped parameters.
        """
        estimates: list[ParameterEstimate] = []
        cluster_prefix = f"cluster_{cluster_id}."

        for param_name, param in params.items():
            if not param_name.startswith(cluster_prefix):
                continue

            estimates.append(
                ParameterEstimate(
                    name=param_name,
                    value=param.value,
                    std_error=param.stderr,
                    unit=param.unit,
                    category=ParameterCategory.LINESHAPE,
                    min_bound=param.min,
                    max_bound=param.max,
                    is_fixed=not param.vary,
                    is_global=self._is_global_parameter(param_name, param.param_id),
                    param_id=param.param_id,
                )
            )

        return estimates

    @staticmethod
    def _is_global_parameter(param_name: str, param_id: Any | None) -> bool:
        """Return whether a parameter is shared beyond a single peak.

        In current models, shared parameters are emitted as cluster-scoped names
        (e.g., ``cluster_12.F3.phase``) or carry ``ParameterId.cluster_id``.
        """
        if param_id is not None and getattr(param_id, "cluster_id", None) is not None:
            return True
        return param_name.startswith("cluster_")

    def _build_cluster_statistics(
        self,
        cluster: Cluster,
        params: Parameters,
        scipy_result: Any | None,
        noise: float,
    ) -> FitStatistics:
        """Build statistics for a cluster.

        Args:
            cluster: Cluster object
            params: Parameters
            scipy_result: Optional scipy result
            noise: Noise level

        Returns:
        -------
            FitStatistics object
        """
        # Count varying parameters for this cluster's peaks
        # Improved logic:
        n_lineshape_params = 0

        # We need to correctly count parameters that belong to this cluster
        # Using a set of peak names related to this cluster
        cluster_peak_names = {p.name for p in cluster.peaks}

        for param_name, param in params.items():
            if not param.vary:
                continue

            # Check if it belongs to one of the peaks (PeakName.Axis.Type) or cluster-wide
            if (
                param.param_id is not None and param.param_id.label == "I"
            ) or _AMPLITUDE_PARAM_PATTERN.search(param_name):
                continue
            if any(
                param_name.startswith(f"{pn}.") for pn in cluster_peak_names
            ) or param_name.startswith(f"cluster_{cluster.cluster_id}."):
                n_lineshape_params += 1

        # Add amplitude parameters to DOF calculation
        # Each peak has one amplitude per spectrum in the series
        n_peaks = len(cluster.peaks)
        n_series = cluster.corrected_data.shape[0] if cluster.corrected_data.ndim > 1 else 1
        n_amplitude_params = n_peaks * n_series

        n_params = n_lineshape_params + n_amplitude_params
        n_data = cluster.corrected_data.size

        # Compute residuals when available for residual statistics.
        normalized_residuals: np.ndarray | None = None
        try:
            normalized_residuals = residuals(params, cluster, noise)
        except (ValueError, KeyError, AttributeError):
            normalized_residuals = None

        # Extract metrics from scipy result if available
        if scipy_result is not None and hasattr(scipy_result, "cost"):
            cost = float(scipy_result.cost)
            nfev = int(getattr(scipy_result, "nfev", 0))
            success = bool(getattr(scipy_result, "success", True))
            message = str(getattr(scipy_result, "message", ""))
            chi_squared = cost * 2  # scipy uses 0.5 * sum(residuals**2)
        else:
            chi_squared = (
                compute_chi_squared(normalized_residuals)
                if normalized_residuals is not None
                else 0.0
            )
            nfev = 0
            success = True
            message = "Statistics computed from fitted model"

        aic, bic, log_likelihood = self._compute_information_criteria(
            chi_squared=chi_squared,
            n_data=n_data,
            n_params=n_params,
            noise=noise,
        )

        residual_stats = ResidualStatistics(
            raw_residuals=(normalized_residuals * noise)
            if normalized_residuals is not None
            else None,
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
            fit_converged=success,
            n_function_evals=nfev,
            fit_message=message,
        )

    def build(self) -> FitResults:
        """Build the final FitResults object.

        Returns:
        -------
            Constructed FitResults

        Raises:
        ------
            ValueError: If required data is missing
        """
        if not self._cluster_estimates:
            msg = "No cluster estimates added. Call add_cluster() first."
            raise ValueError(msg)

        # Build global statistics from cluster statistics
        global_stats = self._build_global_statistics()

        # Build metadata if not set
        if self._metadata is None:
            self._metadata = RunMetadata.capture(self._config)

        return FitResults(
            metadata=self._metadata,
            method=self._fit_method,
            clusters=self._cluster_estimates,
            statistics=self._cluster_statistics,
            global_statistics=global_stats,
            mcmc_diagnostics=self._mcmc_diagnostics,
            z_values=self._spectra_z_values,
        )

    def _build_global_statistics(self) -> FitStatistics:
        """Build global fit statistics from cluster statistics.

        Returns:
        -------
            FitStatistics aggregating all clusters
        """
        total_chi_sq = sum(cs.chi_squared for cs in self._cluster_statistics)
        total_params = sum(cs.n_params for cs in self._cluster_statistics)
        total_data = sum(cs.n_data for cs in self._cluster_statistics)
        total_nfev = sum(cs.n_function_evals for cs in self._cluster_statistics)
        all_converged = all(cs.fit_converged for cs in self._cluster_statistics)

        total_log_likelihood: float | None = None
        total_aic: float | None = None
        total_bic: float | None = None
        if self._cluster_statistics and all(
            stats.log_likelihood is not None for stats in self._cluster_statistics
        ):
            total_log_likelihood = float(
                sum(
                    stats.log_likelihood
                    for stats in self._cluster_statistics
                    if stats.log_likelihood is not None
                )
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

    @staticmethod
    def _compute_information_criteria(
        chi_squared: float,
        n_data: int,
        n_params: int,
        noise: float,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute information criteria under Gaussian residual assumptions."""
        if n_data <= 0 or n_params < 0 or noise <= 0:
            return None, None, None

        log_likelihood = (
            -0.5 * chi_squared - n_data * np.log(noise) - 0.5 * n_data * np.log(2 * np.pi)
        )
        aic = -2.0 * log_likelihood + 2.0 * n_params
        bic = -2.0 * log_likelihood + n_params * np.log(n_data)
        return float(aic), float(bic), float(log_likelihood)
