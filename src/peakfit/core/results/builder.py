"""Builder for constructing FitResults from pipeline outputs.

This module provides the FitResultsBuilder class that constructs
FitResults objects from existing fitting pipeline outputs, bridging
the gap between the old and new output systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.core.fitting.computation import calculate_amplitudes_with_uncertainty, calculate_shapes
from peakfit.core.results.diagnostics import MCMCDiagnostics, ParameterDiagnostic
from peakfit.core.results.estimates import (
    AmplitudeEstimate,
    ClusterEstimates,
    ParameterCategory,
    ParameterEstimate,
)
from peakfit.core.results.fit_results import FitMethod, FitResults, RunMetadata
from peakfit.core.results.statistics import (
    FitStatistics,
    compute_chi_squared,
    compute_reduced_chi_squared,
)

if TYPE_CHECKING:
    # Avoid importing from services layer to preserve architecture
    from typing import Any

    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.domain.spectrum import Spectra
    from peakfit.core.domain.state import FittingState
    from peakfit.core.fitting.parameters import Parameters

    MCMCAnalysisResult = Any


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
    _config: dict = field(default_factory=dict)
    _experiment_type: str = ""

    def __post_init__(self) -> None:
        """Initialize empty lists."""
        self._cluster_estimates = []
        self._cluster_statistics = []
        self._mcmc_diagnostics = []
        self._config = {}

    def set_metadata(
        self,
        config: dict | None = None,
        input_files: dict[str, Path] | None = None,
        command_line: str = "",
    ) -> FitResultsBuilder:
        """Set run metadata.

        Args:
            config: Configuration dictionary
            input_files: Dictionary mapping names to file paths
            command_line: Command line string

        Returns
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

        Returns
        -------
            Self for chaining
        """
        self._spectra_z_values = spectra.z_values
        return self

    def set_z_values(self, z_values: np.ndarray) -> FitResultsBuilder:
        """Set z-values directly.

        Args:
            z_values: Array of z-dimension values

        Returns
        -------
            Self for chaining
        """
        self._spectra_z_values = z_values
        return self

    def set_fit_method(self, method: FitMethod) -> FitResultsBuilder:
        """Set the fitting method used.

        Args:
            method: Fitting method enum

        Returns
        -------
            Self for chaining
        """
        self._fit_method = method
        return self

    def set_experiment_type(self, exp_type: str) -> FitResultsBuilder:
        """Set the experiment type.

        Args:
            exp_type: Experiment type string (e.g., "CPMG", "CEST")

        Returns
        -------
            Self for chaining
        """
        self._experiment_type = exp_type
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

        Returns
        -------
            Self for chaining
        """
        # Extract parameter estimates for this cluster's peaks
        all_lineshape_params: list[ParameterEstimate] = []
        all_amplitudes: list[AmplitudeEstimate] = []

        # Compute amplitudes with uncertainties
        shapes = calculate_shapes(params, cluster)
        amplitudes, amplitudes_err, _covariance = calculate_amplitudes_with_uncertainty(
            shapes, cluster.corrected_data, noise
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
            n_planes = amplitudes.shape[1]
            for j in range(n_planes):
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

        Returns
        -------
            Self for chaining
        """
        noise_val = noise if noise is not None else state.noise
        for cluster in state.clusters:
            self.add_cluster(cluster, state.params, noise_val)
        return self

    def add_mcmc_results(
        self,
        mcmc_result: MCMCAnalysisResult,
    ) -> FitResultsBuilder:
        """Add MCMC analysis results.

        Args:
            mcmc_result: MCMCAnalysisResult from mcmc_analysis_service

        Returns
        -------
            Self for chaining
        """
        self._fit_method = FitMethod.MCMC

        # Convert uncertainty results to diagnostics
        param_diagnostics: list[ParameterDiagnostic] = []

        for cluster_result in mcmc_result.cluster_results:
            # Extract diagnostic info from the UncertaintyResult
            uncertainty_result = cluster_result.result
            diag = uncertainty_result.mcmc_diagnostics

            for idx, param_name in enumerate(uncertainty_result.parameter_names):
                # Try to get diagnostics from mcmc_diagnostics if available
                rhat = 1.0
                ess_bulk = 1000.0
                ess_tail = 1000.0

                if diag is not None:
                    # Access by index since rhat/ess are arrays parallel to parameter_names
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

        if param_diagnostics:
            # Get burn-in info if available
            burn_in = 500
            n_samples = 1000
            n_chains = 4
            if mcmc_result.cluster_results:
                first_result = mcmc_result.cluster_results[0].result
                if first_result.burn_in_info:
                    burn_in = first_result.burn_in_info.get("burn_in", 500)
                if first_result.mcmc_samples is not None:
                    n_samples = first_result.mcmc_samples.shape[0]
                if first_result.mcmc_diagnostics is not None:
                    n_chains = first_result.mcmc_diagnostics.n_chains

            mcmc_diag = MCMCDiagnostics(
                n_chains=n_chains,
                n_samples=n_samples,
                burn_in=burn_in,
                parameter_diagnostics=param_diagnostics,
            )
            mcmc_diag.update_overall_status()
            self._mcmc_diagnostics.append(mcmc_diag)

        return self

    def _extract_peak_parameters(
        self,
        peak_name: str,
        params: Parameters,
    ) -> list[ParameterEstimate]:
        """Extract parameter estimates for a specific peak.

        Args:
            peak_name: Peak identifier
            params: Parameters object

        Returns
        -------
            List of ParameterEstimate objects
        """
        estimates: list[ParameterEstimate] = []

        for param_name, param in params.items():
            # Check if this parameter belongs to the peak (dot-notation: "peak_name.axis.type")
            if not param_name.startswith(peak_name + "."):
                continue

            # Skip phase parameters (they use cluster_id prefix: "cluster_N.axis.phase")
            if param_name.startswith("cluster_"):
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
                    param_id=param.param_id,
                )
            )

        return estimates

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

        Returns
        -------
            FitStatistics object
        """
        # Count varying parameters for this cluster's peaks
        n_lineshape_params = 0
        for peak in cluster.peaks:
            import re

            safe_prefix = re.sub(r"\W+|^(?=\d)", "_", peak.name)
            for param_name, param in params.items():
                if param_name.startswith(safe_prefix + "_") and param.vary:
                    n_lineshape_params += 1

        # Add amplitude parameters to DOF calculation
        # Each peak has one amplitude per plane, computed via linear least-squares
        n_peaks = len(cluster.peaks)
        n_planes = cluster.corrected_data.shape[0] if cluster.corrected_data.ndim > 1 else 1
        n_amplitude_params = n_peaks * n_planes

        n_params = n_lineshape_params + n_amplitude_params
        n_data = cluster.corrected_data.size

        # Extract metrics from scipy result if available
        if scipy_result is not None and hasattr(scipy_result, "cost"):
            cost = float(scipy_result.cost)
            nfev = int(getattr(scipy_result, "nfev", 0))
            success = bool(getattr(scipy_result, "success", True))
            message = str(getattr(scipy_result, "message", ""))
            chi_squared = cost * 2  # scipy uses 0.5 * sum(residuals**2)
        else:
            # Compute chi-squared directly from residuals
            from peakfit.core.fitting.computation import residuals

            try:
                resid = residuals(params, cluster, noise)
                chi_squared = compute_chi_squared(resid)
            except (ValueError, KeyError, AttributeError):
                chi_squared = 0.0
            nfev = 0
            success = True
            message = "Statistics computed from fitted model"

        return FitStatistics(
            chi_squared=chi_squared,
            reduced_chi_squared=compute_reduced_chi_squared(chi_squared, n_data, n_params),
            n_data=n_data,
            n_params=n_params,
            fit_converged=success,
            n_function_evals=nfev,
            fit_message=message,
        )

    def build(self) -> FitResults:
        """Build the final FitResults object.

        Returns
        -------
            Constructed FitResults

        Raises
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
            experiment_type=self._experiment_type,
        )

    def _build_global_statistics(self) -> FitStatistics:
        """Build global fit statistics from cluster statistics.

        Returns
        -------
            FitStatistics aggregating all clusters
        """
        total_chi_sq = sum(cs.chi_squared for cs in self._cluster_statistics)
        total_params = sum(cs.n_params for cs in self._cluster_statistics)
        total_data = sum(cs.n_data for cs in self._cluster_statistics)
        total_nfev = sum(cs.n_function_evals for cs in self._cluster_statistics)
        all_converged = all(cs.fit_converged for cs in self._cluster_statistics)

        return FitStatistics(
            chi_squared=total_chi_sq,
            reduced_chi_squared=compute_reduced_chi_squared(total_chi_sq, total_data, total_params),
            n_data=total_data,
            n_params=total_params,
            fit_converged=all_converged,
            n_function_evals=total_nfev,
        )


__all__ = ["FitResultsBuilder"]
