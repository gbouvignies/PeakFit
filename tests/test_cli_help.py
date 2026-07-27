from click.utils import strip_ansi
from typer.testing import CliRunner

from peakfit.cli.app import app


def rendered_help(*args: str) -> str:
    result = CliRunner().invoke(app, [*args, "--help"])

    assert result.exit_code == 0, result.output
    return strip_ansi(result.output)


def test_root_help_uses_pseudo_nd_terminology() -> None:
    help_text = rendered_help()

    assert "pseudo-ND NMR spectra" in help_text
    assert "pseudo-3D" not in help_text


def test_fit_help_describes_plane_values_without_renaming_the_option() -> None:
    help_text = rendered_help("fit")

    assert "pseudo-ND NMR spectrum" in help_text
    assert "--z-values" in help_text
    assert "Plane values file" in help_text
    assert "Z-dimension" not in help_text
