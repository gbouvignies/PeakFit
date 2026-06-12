import tomllib

import pytest
import typer
from pydantic import ValidationError

from peakfit.cli.commands.fit import _normalize_output_formats
from peakfit.engine.domain.config import OutputConfig, PeakFitConfig
from peakfit.io.config import generate_default_config
from peakfit.io.writers.config import WriterConfig


def test_output_config_defaults_to_core_formats() -> None:
    config = OutputConfig()

    assert config.formats == ["json", "csv"]
    assert not config.save_simulated


def test_writer_config_defaults_to_core_formats() -> None:
    config = WriterConfig()

    assert config.formats == ("json", "csv")


def test_generated_default_config_is_valid() -> None:
    payload = tomllib.loads(generate_default_config())

    config = PeakFitConfig.model_validate(payload)

    assert config.output.formats == ["json", "csv"]
    assert not config.output.save_simulated


def test_cli_output_formats_are_normalized_and_deduped() -> None:
    assert _normalize_output_formats(["JSON", "csv", "json"]) == ["json", "csv"]


def test_invalid_cli_output_format_is_actionable() -> None:
    with pytest.raises(typer.BadParameter, match="Unknown output format"):
        _normalize_output_formats(["json", "html"])


@pytest.mark.parametrize(
    "removed_key",
    [
        "verbosity",
        "save_html_report",
        "save_figures",
        "save_chains",
        "log_format",
        "include_legacy",
    ],
)
def test_removed_output_options_are_rejected(removed_key: str) -> None:
    payload = {"output": {removed_key: True}}

    with pytest.raises(ValidationError):
        PeakFitConfig.model_validate(payload)
