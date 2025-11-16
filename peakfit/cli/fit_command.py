"""Implementation of the fit command."""

from pathlib import Path

import lmfit as lf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from peakfit.clustering import create_clusters
from peakfit.computing import simulate_data, update_cluster_corrections
from peakfit.core.models import PeakFitConfig
from peakfit.messages import (
    export_html,
    print_fit_report,
    print_fitting,
    print_logo,
    print_peaks,
    print_refining,
    print_writing_spectra,
)
from peakfit.noise import prepare_noise_level
from peakfit.peak import create_params
from peakfit.peaklist import read_list
from peakfit.spectra import get_shape_names, read_spectra
from peakfit.writing import write_profiles, write_shifts

# Import legacy cli module for backwards compatibility
from peakfit.cli_legacy import Arguments as LegacyArguments

console = Console()


def config_to_legacy_args(
    config: PeakFitConfig,
    spectrum_path: Path,
    peaklist_path: Path,
    z_values_path: Path | None,
) -> LegacyArguments:
    """Convert modern config to legacy Arguments for backwards compatibility."""
    args = LegacyArguments()
    args.path_spectra = spectrum_path
    args.path_list = peaklist_path
    args.path_z_values = z_values_path
    args.contour_level = config.clustering.contour_level
    args.noise = config.noise_level
    args.path_output = config.output.directory
    args.refine_nb = config.fitting.refine_iterations
    args.fixed = config.fitting.fix_positions
    args.jx = config.fitting.fit_j_coupling
    args.phx = config.fitting.fit_phase_x
    args.phy = config.fitting.fit_phase_y
    args.exclude = config.exclude_planes

    # Map lineshape to flags
    args.pvoigt = config.fitting.lineshape == "pvoigt"
    args.lorentzian = config.fitting.lineshape == "lorentzian"
    args.gaussian = config.fitting.lineshape == "gaussian"

    return args


def run_fit(
    spectrum_path: Path,
    peaklist_path: Path,
    z_values_path: Path | None,
    config: PeakFitConfig,
    parallel: bool = False,  # noqa: ARG001
) -> None:
    """Run the fitting process.

    Args:
        spectrum_path: Path to NMRPipe spectrum file.
        peaklist_path: Path to peak list file.
        z_values_path: Optional path to Z-values file.
        config: Configuration object.
        parallel: Whether to use parallel processing (future feature).
    """
    print_logo()

    # Convert to legacy args for compatibility with existing modules
    clargs = config_to_legacy_args(config, spectrum_path, peaklist_path, z_values_path)

    # Load data
    with console.status("[bold yellow]Loading spectrum..."):
        spectra = read_spectra(clargs.path_spectra, clargs.path_z_values, clargs.exclude)

    console.print(f"[green]Loaded spectrum:[/green] {spectrum_path.name}")
    console.print(f"  Shape: {spectra.data.shape}")
    console.print(f"  Z-values: {len(spectra.z_values)} planes")

    # Estimate noise
    clargs.noise = prepare_noise_level(clargs, spectra)
    console.print(f"[green]Noise level:[/green] {clargs.noise:.2f}")

    # Determine lineshape
    shape_names = get_shape_names(clargs, spectra)
    console.print(f"[green]Lineshapes:[/green] {shape_names}")

    # Read peak list
    peaks = read_list(spectra, shape_names, clargs)
    console.print(f"[green]Loaded peaks:[/green] {len(peaks)} peaks")

    # Cluster peaks
    clargs.contour_level = clargs.contour_level or 5.0 * clargs.noise
    console.print(f"[green]Contour level:[/green] {clargs.contour_level:.2f}")

    clusters = create_clusters(spectra, peaks, clargs.contour_level)
    console.print(f"[green]Created clusters:[/green] {len(clusters)} clusters")

    # Fit clusters
    params = _fit_clusters(clargs, clusters)

    # Write outputs
    config.output.directory.mkdir(parents=True, exist_ok=True)

    write_profiles(config.output.directory, spectra.z_values, clusters, params, clargs)
    console.print(f"[green]Written profiles to:[/green] {config.output.directory}")

    if config.output.save_html_report:
        export_html(config.output.directory / "logs.html")
        console.print(f"[green]Written HTML report:[/green] {config.output.directory / 'logs.html'}")

    write_shifts(peaks, params, config.output.directory / "shifts.list")
    console.print(f"[green]Written shifts:[/green] {config.output.directory / 'shifts.list'}")

    if config.output.save_simulated:
        _write_spectra(config.output.directory, spectra, clusters, params)
        console.print(f"[green]Written simulated spectrum[/green]")

    console.print("\n[bold green]Fitting complete![/bold green]")


def _fit_clusters(clargs: LegacyArguments, clusters: list) -> lf.Parameters:
    """Fit all clusters and return parameters."""
    print_fitting()
    params_all = lf.Parameters()

    for index in range(clargs.refine_nb + 1):
        if index > 0:
            print_refining(index, clargs.refine_nb)
            update_cluster_corrections(params_all, clusters)
        for cluster in clusters:
            print_peaks(cluster.peaks)
            params = create_params(cluster.peaks, fixed=clargs.fixed)
            params = _update_params(params, params_all)
            mini = lf.Minimizer(
                lambda p, c, n: _residuals(p, c, n),  # noqa: E731
                params,
                fcn_args=(cluster, clargs.noise),
            )
            out = mini.least_squares(verbose=2)
            print_fit_report(out)
            params_all.update(getattr(out, "params", lf.Parameters()))

    return params_all


def _update_params(params: lf.Parameters, params_all: lf.Parameters) -> lf.Parameters:
    """Update parameters with global parameters."""
    for key in params:
        if key in params_all:
            params[key] = params_all[key]
    return params


def _residuals(params: lf.Parameters, cluster, noise: float):
    """Compute residuals for fitting."""
    from peakfit.computing import residuals

    return residuals(params, cluster, noise)


def _write_spectra(path: Path, spectra, clusters, params: lf.Parameters) -> None:
    """Write simulated spectra to file."""
    import nmrglue as ng
    import numpy as np

    print_writing_spectra()

    data_simulated = simulate_data(params, clusters, spectra.data)

    if spectra.pseudo_dim_added:
        data_simulated = np.squeeze(data_simulated, axis=0)

    ng.pipe.write(
        str(path / f"simulated.ft{data_simulated.ndim}"),
        spectra.dic,
        data_simulated.astype(np.float32),
        overwrite=True,
    )
