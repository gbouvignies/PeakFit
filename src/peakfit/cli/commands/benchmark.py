"""Benchmark command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from peakfit.ui import console


def benchmark_command(
    spectrum: Annotated[
        Path,
        typer.Argument(
            help="Path to NMRPipe spectrum file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    peaklist: Annotated[
        Path,
        typer.Argument(
            help="Path to peak list file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    z_values: Annotated[
        Path | None,
        typer.Option(
            "--z-values",
            "-z",
            help="Path to Z-dimension values file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    iterations: Annotated[
        int,
        typer.Option(
            "--iterations",
            "-n",
            help="Number of benchmark iterations",
            min=1,
            max=10,
        ),
    ] = 1,
) -> None:
    """Benchmark fitting performance with different methods.

    Compare standard lmfit and fast scipy approaches
    to determine the optimal method for your data.

    Example:
        peakfit benchmark spectrum.ft2 peaks.list --iterations 3
    """
    import time

    from peakfit.cli.models import SpectraInput
    from peakfit.core.algorithms.clustering import create_clusters
    from peakfit.core.algorithms.noise import prepare_noise_level
    from peakfit.core.domain.peaks_io import read_list
    from peakfit.core.domain.spectrum import get_shape_names
    from peakfit.core.fitting.optimizer import fit_clusters
    from peakfit.services.fit import FitArguments

    console.print("[bold]PeakFit Performance Benchmark[/bold]\n")

    # Load data
    with console.status("[yellow]Loading spectrum..."):
        # Create args for loading
        clargs = FitArguments(
            path_spectra=spectrum,
            path_z_values=z_values,
            path_list=peaklist,
            exclude=[],
            noise=0.0,
            contour_level=None,
            fixed=False,
            jx=False,
            phx=False,
            phy=False,
            pvoigt=False,
            lorentzian=False,
            gaussian=False,
        )

        spectra = SpectraInput(path=spectrum, z_values_path=z_values, exclude_list=[]).load()

    console.print(f"[green]Loaded spectrum:[/green] {spectrum.name}")
    console.print(f"  Shape: {spectra.data.shape}")

    # Prepare fitting
    clargs.noise = prepare_noise_level(clargs, spectra)
    shape_names = get_shape_names(clargs, spectra)
    peaks = read_list(spectra, shape_names, clargs)
    clargs.contour_level = 5.0 * clargs.noise
    clusters = create_clusters(spectra, peaks, clargs.contour_level)

    console.print(f"[green]Noise level:[/green] {clargs.noise:.2f}")
    console.print(f"[green]Clusters:[/green] {len(clusters)}")
    console.print(f"[green]Total peaks:[/green] {len(peaks)}")

    console.print(f"\n[bold]Running benchmark ({iterations} iteration(s))...[/bold]\n")

    # Benchmark fast sequential
    times_fast = []
    for _i in range(iterations):
        start = time.perf_counter()
        fit_clusters(
            clusters=list(clusters),
            noise=clargs.noise,
            refine_iterations=0,  # No refinement for speed comparison
            fixed=False,
            verbose=False,
        )
        times_fast.append(time.perf_counter() - start)

    avg_fast = sum(times_fast) / len(times_fast)
    console.print("[cyan]Fast Sequential:[/cyan]")
    console.print(f"  Average time: {avg_fast:.3f}s")
    if len(times_fast) > 1:
        console.print(f"  Min: {min(times_fast):.3f}s, Max: {max(times_fast):.3f}s")
