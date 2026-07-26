import pytest

from peakfit.io.schemas import OUTPUT_SCHEMA_VERSION, FitSummarySchema


def test_fit_summary_declares_authoritative_final_outcome_schema_version() -> None:
    assert OUTPUT_SCHEMA_VERSION == "4.0.0"


def test_fit_summary_rejects_legacy_development_schema_with_explicit_versions() -> None:
    with pytest.raises(ValueError, match=r"3.0.0.*4.0.0"):
        FitSummarySchema.model_validate({"schema_version": "3.0.0"})
