"""Writer modules for PeakFit result output.

This package contains small functions for canonical fit artifacts and
run-level files:
- JSON summaries
- CSV tables
- optional Markdown reports
- run README/state companions
- optional simulated spectra

Main entry points (import from concrete modules):
    from peakfit.io.writers.orchestrator import write_fit_outputs
    from peakfit.io.writers.config import WriterConfig
"""
