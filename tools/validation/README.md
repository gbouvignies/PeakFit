Developer validation utilities

This folder contains ad-hoc validation and reproduction scripts used during
numerical development and debugging. They are not part of the public API and
are excluded from packaging.

How to run (examples):
- Verify Jacobian consistency (analytical vs finite difference): run `python -m tools.validation.verify_jacobian`
- Reproduce Jacobian study on a toy Lorentzian model: run `python -m tools.validation.reproduce_jacobian`
- Check numerical stability of amplitude solve under NaNs/overflows: run `python -m tools.validation.verify_computation` and `python -m tools.validation.verify_overflow`
- Analyze residuals vs simulated spectra for the advanced example dataset (requires example outputs): run `python -m tools.validation.analyze_residuals`

Notes:
- These scripts may rely on optional datasets in examples/ and require development dependencies (see pyproject.toml).
- Results are informational; they do not constitute unit tests and should not be used to gate CI.