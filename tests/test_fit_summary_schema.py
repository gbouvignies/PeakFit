from peakfit.io.schemas import FitSummarySchema


def _build_summary_payload(std_error: object) -> dict[str, object]:
    return {
        "metadata": {
            "timestamp": "2026-02-16T00:00:00",
            "software_version": "test",
            "python_version": "3.14.0",
            "platform": "darwin",
        },
        "method": "profile",
        "n_clusters": 1,
        "n_peaks": 1,
        "clusters": [
            {
                "cluster_id": 101,
                "peak_names": ["A1"],
                "parameters": [
                    {
                        "name": "A1.F2.cs",
                        "value": 8.42,
                        "std_error": std_error,
                        "unit": "ppm",
                    }
                ],
                "amplitudes": [
                    {
                        "peak_name": "A1",
                        "plane_index": 0,
                        "value": 1000.0,
                        "std_error": "nan",
                    }
                ],
            }
        ],
    }


def test_fit_summary_accepts_null_parameter_std_error() -> None:
    summary = FitSummarySchema.model_validate(_build_summary_payload(None))
    assert summary.clusters[0].lineshape_parameters[0].std_error is None


def test_fit_summary_normalizes_nonfinite_std_error() -> None:
    summary = FitSummarySchema.model_validate(_build_summary_payload("nan"))
    assert summary.clusters[0].lineshape_parameters[0].std_error is None
    assert summary.clusters[0].amplitudes[0].std_error is None
