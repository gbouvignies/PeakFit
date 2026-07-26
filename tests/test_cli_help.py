from typer.testing import CliRunner

from peakfit.cli.app import app


def test_root_help_uses_pseudo_nd_terminology() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "pseudo-ND NMR spectra" in result.output
    assert "pseudo-3D" not in result.output


def test_fit_help_describes_plane_values_without_renaming_the_option() -> None:
    result = CliRunner().invoke(app, ["fit", "--help"])

    assert result.exit_code == 0, result.output
    assert "pseudo-ND NMR spectrum" in result.output
    assert "--z-values" in result.output
    assert "Plane values file" in result.output
    assert "Z-dimension" not in result.output
