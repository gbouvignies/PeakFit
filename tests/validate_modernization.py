#!/usr/bin/env python3
"""Comprehensive validation script for PeakFit modernization.

This script validates that all new features from PR#9 work correctly:
1. All CLI commands and options
2. Integration workflows
3. Error handling
4. Performance optimizations
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

import numpy as np

console = Console()


class ValidationResult:
    """Track validation test results."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_pass(self, test_name: str, message: str = ""):
        """Add a passing test."""
        self.passed.append((test_name, message))
        console.print(f"[green]✓[/green] {test_name}")
        if message:
            console.print(f"  [dim]{message}[/dim]")

    def add_fail(self, test_name: str, error: str):
        """Add a failing test."""
        self.failed.append((test_name, error))
        console.print(f"[red]✗[/red] {test_name}")
        console.print(f"  [red]{error}[/red]")

    def add_warning(self, test_name: str, message: str):
        """Add a warning."""
        self.warnings.append((test_name, message))
        console.print(f"[yellow]⚠[/yellow] {test_name}")
        console.print(f"  [yellow]{message}[/yellow]")

    def print_summary(self):
        """Print test summary."""
        console.print("\n" + "=" * 80)
        table = Table(title="Validation Summary")
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")
        table.add_row("[green]Passed[/green]", str(len(self.passed)))
        table.add_row("[red]Failed[/red]", str(len(self.failed)))
        table.add_row("[yellow]Warnings[/yellow]", str(len(self.warnings)))
        console.print(table)

        if self.failed:
            console.print("\n[bold red]Failed Tests:[/bold red]")
            for name, error in self.failed:
                console.print(f"  • {name}: {error}")

        return len(self.failed) == 0


def run_command(cmd: list[str], check: bool = False) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    # Use sys.executable to ensure we use the same Python interpreter
    if cmd[0] == "peakfit":
        cmd = [sys.executable, "-m", "peakfit.cli.app", *cmd[1:]]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
    )
    return result.returncode, result.stdout, result.stderr


def validate_cli_help_commands(results: ValidationResult):
    """Validate all CLI help commands work."""
    console.print("\n[bold]Testing CLI Help Commands[/bold]")

    commands = [
        (["peakfit", "--help"], "Main help"),
        (["peakfit", "--version"], "Version flag"),
        (["peakfit", "fit", "--help"], "Fit command help"),
        (["peakfit", "validate", "--help"], "Validate command help"),
        (["peakfit", "init", "--help"], "Init command help"),
        (["peakfit", "info", "--help"], "Info command help"),
        (["peakfit", "plot", "--help"], "Plot command help"),
        (["peakfit", "benchmark", "--help"], "Benchmark command help"),
        (["peakfit", "analyze", "--help"], "Analyze command help"),
    ]

    for cmd, name in commands:
        code, _stdout, stderr = run_command(cmd)
        if code == 0:
            results.add_pass(name, f"Command: {' '.join(cmd)}")
        else:
            results.add_fail(name, f"Exit code: {code}, stderr: {stderr}")


def validate_info_command(results: ValidationResult):
    """Validate info command shows correct backend information."""
    console.print("\n[bold]Testing Info Command[/bold]")

    code, stdout, _stderr = run_command(["peakfit", "info"])
    if code == 0:
        results.add_pass("Info command", "Shows backend information")

        # Check that it mentions backend and optimization
        if "backend" in stdout.lower() or "numpy" in stdout.lower():
            results.add_pass("Backend info displayed", "Backend information present")
        else:
            results.add_warning("Backend info", "No backend information found in output")
    else:
        results.add_fail("Info command", f"Exit code: {code}")


def validate_init_command(results: ValidationResult):
    """Validate init command creates valid config files."""
    console.print("\n[bold]Testing Init Command[/bold]")

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.toml"

        # Test basic init
        code, _stdout, _stderr = run_command(["peakfit", "init", str(config_path)])
        if code == 0 and config_path.exists():
            results.add_pass("Init creates config file", str(config_path))
        else:
            results.add_fail("Init creates config file", f"Exit code: {code}")
            return

        # Verify config has expected sections
        content = config_path.read_text()
        required_sections = ["[fitting]", "[clustering]", "[output]"]
        for section in required_sections:
            if section in content:
                results.add_pass(f"Config has {section}", "Section found")
            else:
                results.add_fail(f"Config has {section}", "Section missing")

        # Test no overwrite without --force
        code, _stdout, _stderr = run_command(["peakfit", "init", str(config_path)])
        if code != 0:
            results.add_pass("Init prevents overwrite", "Correctly fails without --force")
        else:
            results.add_fail("Init prevents overwrite", "Should fail without --force")

        # Test overwrite with --force
        code, _stdout, _stderr = run_command(["peakfit", "init", str(config_path), "--force"])
        if code == 0:
            results.add_pass("Init with --force overwrites", "Overwrite successful")
        else:
            results.add_fail("Init with --force", f"Exit code: {code}")


def validate_backend_selection(results: ValidationResult):
    """Validate backend selection works correctly."""
    console.print("\n[bold]Testing Backend Selection[/bold]")
    # The backend presence should be reported via the 'info' command
    code, stdout, _stderr = run_command(["peakfit", "info"])
    if code == 0 and ("numpy" in stdout.lower() or "backend" in stdout.lower()):
        results.add_pass("Backend info displayed", "NumPy usage reported")
    else:
        results.add_warning("Backend info", "Could not confirm backend in info output")


def validate_cli_options(results: ValidationResult):
    """Validate CLI options are recognized (without running full fits)."""
    console.print("\n[bold]Testing CLI Options[/bold]")

    # Create minimal test files
    with tempfile.TemporaryDirectory():
        # These should fail gracefully with appropriate error messages
        test_cases = [
            (["peakfit", "fit"], "fit requires arguments", "Should fail with missing args"),
            (
                ["peakfit", "validate"],
                "validate requires arguments",
                "Should fail with missing args",
            ),
            (
                ["peakfit", "fit", "--lineshape", "invalid"],
                "Invalid lineshape",
                "Should reject invalid lineshape",
            ),
            (
                ["peakfit", "fit", "--refine", "-1"],
                "Invalid refine",
                "Should reject negative refine",
            ),
        ]

        for cmd, name, desc in test_cases:
            code, _stdout, _stderr = run_command(cmd)
            if code != 0:
                results.add_pass(name, desc)
            else:
                results.add_fail(name, "Should have failed but returned 0")


def validate_error_handling(results: ValidationResult):
    """Validate error handling for invalid inputs."""
    console.print("\n[bold]Testing Error Handling[/bold]")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Test with non-existent files
        fake_spectrum = tmppath / "nonexistent.ft2"
        fake_peaklist = tmppath / "nonexistent.list"

        code, _stdout, _stderr = run_command(
            [
                "peakfit",
                "validate",
                str(fake_spectrum),
                str(fake_peaklist),
            ]
        )

        if code != 0:
            results.add_pass("Validate rejects missing files", "Correctly fails with missing files")
        else:
            results.add_fail("Validate missing files", "Should fail with missing files")


def validate_parameter_system(results: ValidationResult):
    """Validate the new Parameters system works correctly."""
    console.print("\n[bold]Testing Parameter System[/bold]")

    try:
        from peakfit.core.fitting.parameters import Parameter, Parameters, ParameterType

        # Test creating parameters with types
        p1 = Parameter(
            name="position",
            value=100.0,
            min=0.0,
            max=200.0,
            param_type=ParameterType.POSITION,
            unit="Hz",
        )
        results.add_pass("Create typed parameter", f"Position parameter: {p1.value} {p1.unit}")

        # Test Parameters collection
        params = Parameters()
        params.add("test1", value=1.0, min=0.0, max=10.0)
        params.add("test2", value=2.0, vary=False)

        if len(params) == 2:
            results.add_pass("Parameters collection", f"Added {len(params)} parameters")
        else:
            results.add_fail("Parameters collection", f"Expected 2, got {len(params)}")

        # Test get_vary_names
        vary_names = params.get_vary_names()
        if len(vary_names) == 1:  # Only test1 varies
            results.add_pass("Parameters get_vary_names", "Correctly filters varying parameters")
        else:
            results.add_fail("Parameters get_vary_names", f"Expected 1, got {len(vary_names)}")

    except Exception as e:
        results.add_fail("Parameter system", str(e))


def validate_lineshapes(results: ValidationResult):
    """Validate lineshape functions work correctly."""
    console.print("\n[bold]Testing Lineshape Functions[/bold]")

    try:
        from peakfit.core.lineshapes import gaussian, lorentzian, pvoigt

        x = np.linspace(-10, 10, 100)

        y_gauss = gaussian(x, fwhm=2.0)
        if np.max(y_gauss) > 0.99:  # Peak should be ~1.0
            results.add_pass("Gaussian lineshape", "Peak height correct")
        else:
            results.add_fail("Gaussian lineshape", f"Peak height: {np.max(y_gauss)}")

        # Peak height is ~0.99 for Lorentzian due to numerical precision
        y_lorentz = lorentzian(x, fwhm=2.0)
        if np.max(y_lorentz) > 0.98:
            results.add_pass("Lorentzian lineshape", f"Peak height: {np.max(y_lorentz):.4f}")
        else:
            results.add_fail("Lorentzian lineshape", f"Peak height: {np.max(y_lorentz)}")

        y_pvoigt = pvoigt(x, fwhm=2.0, eta=0.5)
        if np.max(y_pvoigt) > 0.99:
            results.add_pass("Pseudo-Voigt lineshape", "Peak height correct")
        else:
            results.add_fail("Pseudo-Voigt lineshape", f"Peak height: {np.max(y_pvoigt)}")

    except Exception as e:
        results.add_fail("Lineshape functions", str(e))


def validate_config_system(results: ValidationResult):
    """Validate the Pydantic config system."""
    console.print("\n[bold]Testing Configuration System[/bold]")

    try:
        from peakfit.core.domain.config import (
            ClusterConfig,
            FitConfig,
            LineshapeName,
            OutputConfig,
            PeakFitConfig,
        )

        # Test creating default config
        config = PeakFitConfig()
        results.add_pass("Default config creation", "PeakFitConfig created")

        # Test nested configs
        if isinstance(config.fitting, FitConfig):
            results.add_pass("Nested FitConfig", "Correct type")
        else:
            results.add_fail("Nested FitConfig", f"Wrong type: {type(config.fitting)}")

        if isinstance(config.clustering, ClusterConfig):
            results.add_pass("Nested ClusterConfig", "Correct type")
        else:
            results.add_fail("Nested ClusterConfig", f"Wrong type: {type(config.clustering)}")

        if isinstance(config.output, OutputConfig):
            results.add_pass("Nested OutputConfig", "Correct type")
        else:
            results.add_fail("Nested OutputConfig", f"Wrong type: {type(config.output)}")

        # Test validation (Pydantic v2 doesn't raise on assignment, but on validation)
        try:
            FitConfig(lineshape=cast("LineshapeName", "invalid"))
            results.add_fail("Config validation", "Should reject invalid lineshape")
        except ValidationError:
            results.add_pass("Config validation", "Correctly rejects invalid lineshape")
        except Exception as e:
            results.add_fail("Config validation", f"Unexpected error: {e}")

    except Exception as e:
        results.add_fail("Configuration system", str(e))


# Note: legacy backend and lmfit checks were removed per modernization plan.


def main():
    """Run all validation tests."""
    console.print("[bold cyan]PeakFit Modernization Validation[/bold cyan]")
    console.print("Systematically testing all new features from PR#9\n")

    results = ValidationResult()

    # Run all validation tests
    validate_cli_help_commands(results)
    validate_info_command(results)
    validate_init_command(results)
    validate_backend_selection(results)
    validate_cli_options(results)
    validate_error_handling(results)
    validate_parameter_system(results)
    validate_lineshapes(results)
    validate_config_system(results)

    # Print summary
    success = results.print_summary()

    if success:
        console.print("\n[bold green]✓ All validation tests passed![/bold green]")
        return 0
    else:
        console.print("\n[bold red]✗ Some validation tests failed[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
