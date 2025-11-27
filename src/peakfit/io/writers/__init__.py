"""Output writers for PeakFit results.

This module provides the writer abstraction and concrete implementations
for different output formats (JSON, CSV, Markdown).
"""

from peakfit.io.writers.base import OutputWriter, Verbosity, WriterConfig
from peakfit.io.writers.chain_writer import ChainWriter, load_chain_metadata, load_chains
from peakfit.io.writers.csv_writer import CSVWriter
from peakfit.io.writers.figure_registry import FigureCategory, FigureInfo, FigureRegistry
from peakfit.io.writers.json_writer import JSONWriter
from peakfit.io.writers.legacy_writer import LegacyWriter, write_legacy_output
from peakfit.io.writers.markdown_writer import MarkdownReportGenerator
from peakfit.io.writers.results_writer import ResultsWriter

__all__ = [
    "CSVWriter",
    "ChainWriter",
    "FigureCategory",
    "FigureInfo",
    "FigureRegistry",
    "JSONWriter",
    "LegacyWriter",
    "MarkdownReportGenerator",
    "OutputWriter",
    "ResultsWriter",
    "Verbosity",
    "WriterConfig",
    "load_chain_metadata",
    "load_chains",
    "write_legacy_output",
]
