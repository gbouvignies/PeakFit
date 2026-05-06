"""Writer modules for PeakFit results output.

This package provides output writing functionality:
- CSV output (long format, suitable for pandas/R/Excel)
- JSON output (machine-readable, structured)
- Markdown output (human-readable reports)
- Legacy output (backward compatibility)
- Simulation output (simulated spectra)

Main entry points (import from concrete modules):
    from peakfit.io.writers.orchestrator import write_fit_outputs, ResultsWriter
    from peakfit.io.writers.config import Verbosity, WriterConfig
"""
