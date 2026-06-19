from peakfit.io.schemas import OUTPUT_SCHEMA_VERSION, FitSummarySchema


def _build_summary_payload(std_error: object) -> dict[str, object]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
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
            }
        ],
    }


def test_fit_summary_accepts_null_parameter_std_error() -> None:
    summary = FitSummarySchema.model_validate(_build_summary_payload(None))
    assert summary.schema_version == OUTPUT_SCHEMA_VERSION
    assert summary.clusters[0].lineshape_parameters[0].std_error is None


def test_fit_summary_normalizes_nonfinite_std_error() -> None:
    summary = FitSummarySchema.model_validate(_build_summary_payload("nan"))
    assert summary.clusters[0].lineshape_parameters[0].std_error is None


def test_fit_summary_declares_mcmc_warnings() -> None:
    payload = _build_summary_payload(0.01)
    payload["mcmc_diagnostics"] = [
        {
            "n_chains": 4,
            "n_samples": 100,
            "burn_in": 20,
            "total_samples": 400,
            "overall_status": "marginal",
            "converged": False,
            "warnings": ["ESS_bulk is low."],
            "parameters": [
                {
                    "name": "A1.F2.cs",
                    "rhat": 1.02,
                    "ess_bulk": 250.0,
                    "ess_tail": 200.0,
                    "status": "marginal",
                }
            ],
        }
    ]

    summary = FitSummarySchema.model_validate(payload)

    assert summary.mcmc_diagnostics[0].warnings == ["ESS_bulk is low."]
