# Output System Architecture

This document describes the architecture of PeakFit's output system for developers who need to extend or modify it.

## Overview

The output system follows a layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                       FitPipeline                               │
│                    (Orchestration Layer)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FitResultsBuilder                            │
│               (Domain Model Construction)                       │
│  - Collects data from optimizer, MCMC, noise estimator         │
│  - Constructs FitResults dataclass hierarchy                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ResultsWriter                              │
│                  (Orchestration Layer)                          │
│  - Coordinates all format-specific writers                     │
│  - Manages output directory structure                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┬────────────────┐
              ▼               ▼               ▼                ▼
        ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌─────────────┐
        │JSONWriter│   │CSVWriter │   │ Markdown   │   │LegacyWriter │
        └──────────┘   └──────────┘   │ Generator  │   └─────────────┘
                                      └────────────┘
```

## Core Components

### Result Dataclasses (`core/results/`)

The result model is organized into focused modules:

```
core/results/
├── __init__.py          # Public exports
├── estimates.py         # ParameterEstimate, AmplitudeEstimate, ClusterEstimates
├── statistics.py        # FitStatistics, ResidualStatistics, ModelComparison
├── diagnostics.py       # MCMCDiagnostics, ConvergenceStatus
├── fit_results.py       # FitResults, RunMetadata, FitMethod
└── builder.py           # FitResultsBuilder
```

#### Key Dataclasses

```python
@dataclass
class ParameterEstimate:
    """Single parameter estimate with uncertainty."""
    name: str
    value: float
    uncertainty: float | None = None
    ci_lower: float | None = None  # 95% CI lower bound
    ci_upper: float | None = None  # 95% CI upper bound
    units: str | None = None

@dataclass
class ClusterEstimates:
    """All estimates for a single cluster."""
    cluster_id: int
    peak_names: list[str]
    parameters: list[ParameterEstimate]
    amplitudes: list[AmplitudeEstimate]
    statistics: FitStatistics
    mcmc_diagnostics: MCMCDiagnostics | None = None

@dataclass
class FitResults:
    """Complete results from a fitting run."""
    clusters: list[ClusterEstimates]
    metadata: RunMetadata
    global_statistics: FitStatistics | None = None
```

### FitResultsBuilder (`core/results/builder.py`)

The builder pattern allows incremental construction of results:

```python
from peakfit.core.results import FitResultsBuilder

# Create builder
builder = FitResultsBuilder(method=FitMethod.MCMC)

# Add metadata
builder.set_input_file("spectrum.ft2", input_dir / "spectrum.ft2")
builder.set_elapsed_time(45.2)

# Add cluster results
for cluster_id, peak_names, optimizer_result, mcmc_result in results:
    builder.add_cluster(
        cluster_id=cluster_id,
        peak_names=peak_names,
        optimizer_result=optimizer_result,
        noise_level=noise_estimate,
        mcmc_result=mcmc_result,  # Optional
    )

# Build final results
fit_results = builder.build()
```

### Writers (`io/writers/`)

```
io/writers/
├── __init__.py           # Public exports
├── base.py               # Verbosity, WriterConfig, OutputWriter protocol
├── json_writer.py        # JSON format output
├── csv_writer.py         # CSV tabular output
├── markdown_writer.py    # Human-readable Markdown reports
├── legacy_writer.py      # Backward-compatible .out format
├── results_writer.py     # Orchestrates all writers
├── chain_writer.py       # MCMC chain storage (NPZ)
└── figure_registry.py    # Figure tracking and manifests
```

#### Writer Protocol

All writers follow a common interface:

```python
from typing import Protocol
from pathlib import Path
from peakfit.core.results import FitResults

class OutputWriter(Protocol):
    """Protocol for output writers."""

    def write(self, results: FitResults, output_dir: Path) -> Path:
        """Write results to the specified directory.

        Args:
            results: Complete fit results to write
            output_dir: Directory to write output files

        Returns:
            Path to the primary output file created
        """
        ...
```

#### WriterConfig

```python
@dataclass
class WriterConfig:
    """Configuration for output writers."""
    verbosity: Verbosity = Verbosity.STANDARD
    include_timestamp: bool = True
    pretty_print: bool = True  # For JSON
```

#### Verbosity Levels

```python
class Verbosity(Enum):
    """Output verbosity levels."""
    MINIMAL = "minimal"     # Essential results only
    STANDARD = "standard"   # Default: includes statistics
    FULL = "full"           # Everything including all diagnostics
```

### ResultsWriter (Orchestrator)

The `ResultsWriter` coordinates all format-specific writers:

```python
from peakfit.io.writers import ResultsWriter, WriterConfig, Verbosity

config = WriterConfig(verbosity=Verbosity.FULL)
writer = ResultsWriter(config)

# Write all formats
outputs = writer.write_all(
    results=fit_results,
    output_dir=Path("results"),
    formats=["json", "csv", "markdown"],
    include_legacy=True,
)

# Returns dict of format -> output path
print(outputs["json"])  # Path("results/results.json")
```

## Adding a New Output Format

### Step 1: Create Writer Class

```python
# io/writers/xml_writer.py
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from peakfit.core.results import FitResults
from peakfit.io.writers.base import WriterConfig, Verbosity


@dataclass
class XMLWriter:
    """XML format output writer."""

    config: WriterConfig

    def write(self, results: FitResults, output_dir: Path) -> Path:
        """Write results as XML."""
        output_path = output_dir / "results.xml"

        root = ET.Element("peakfit_results")
        root.set("version", "1.0")

        # Add metadata
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "method").text = results.metadata.method.value

        # Add clusters based on verbosity
        clusters = ET.SubElement(root, "clusters")
        for cluster in results.clusters:
            self._add_cluster(clusters, cluster)

        tree = ET.ElementTree(root)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        return output_path

    def _add_cluster(self, parent: ET.Element, cluster: ClusterEstimates) -> None:
        """Add cluster element."""
        elem = ET.SubElement(parent, "cluster")
        elem.set("id", str(cluster.cluster_id))
        # ... add parameters, amplitudes, etc.
```

### Step 2: Register in ResultsWriter

```python
# io/writers/results_writer.py

class ResultsWriter:
    def __init__(self, config: WriterConfig):
        self.config = config
        self._writers = {
            "json": JSONWriter(config),
            "csv": CSVWriter(config),
            "markdown": MarkdownReportGenerator(config),
            "xml": XMLWriter(config),  # Add new format
        }
```

### Step 3: Export from Package

```python
# io/writers/__init__.py

from peakfit.io.writers.xml_writer import XMLWriter

__all__ = [
    # ... existing exports
    "XMLWriter",
]
```

### Step 4: Add Configuration Support

```python
# core/domain/config.py

class OutputConfig:
    formats: list[str] = field(default_factory=lambda: ["json", "csv", "markdown", "xml"])
```

## Extending Existing Writers

### Adding Fields to JSON Output

1. Add field to appropriate dataclass in `core/results/`
2. The JSON writer automatically serializes dataclass fields
3. Update schema documentation

### Customizing CSV Columns

Modify `CSVWriter._build_rows()`:

```python
def _build_rows(self, results: FitResults) -> list[dict[str, Any]]:
    rows = []
    for cluster in results.clusters:
        for param in cluster.parameters:
            row = {
                "cluster_id": cluster.cluster_id,
                "parameter": param.name,
                "value": param.value,
                "uncertainty": param.uncertainty,
                # Add custom column
                "relative_error": param.uncertainty / param.value if param.value else None,
            }
            rows.append(row)
    return rows
```

### Adding Markdown Sections

Extend `MarkdownReportGenerator`:

```python
def _generate_sections(self, results: FitResults) -> list[str]:
    sections = [
        self._header_section(results),
        self._summary_section(results),
        self._clusters_section(results),
        self._custom_analysis_section(results),  # Add new section
    ]
    return sections

def _custom_analysis_section(self, results: FitResults) -> str:
    """Generate custom analysis section."""
    lines = ["## Custom Analysis", ""]
    # Add content
    return "\n".join(lines)
```

## Schema Management

### JSON Schema (`io/schemas.py`)

The JSON output follows a defined schema:

```python
RESULTS_SCHEMA = {
    "version": "1.0",
    "properties": {
        "metadata": {...},
        "clusters": {...},
        "global_summary": {...},
    }
}
```

### Schema Versioning

When making breaking changes:

1. Increment schema version in `RESULTS_SCHEMA`
2. Include version in output: `{"schema_version": "1.1", ...}`
3. Document migration path
4. Consider backward compatibility in readers

## Testing Writers

### Unit Tests

```python
# tests/unit/test_xml_writer.py

def test_xml_writer_creates_file(tmp_path, sample_fit_results):
    """Test XML writer creates output file."""
    config = WriterConfig(verbosity=Verbosity.STANDARD)
    writer = XMLWriter(config)

    output_path = writer.write(sample_fit_results, tmp_path)

    assert output_path.exists()
    assert output_path.suffix == ".xml"

def test_xml_writer_includes_all_clusters(tmp_path, sample_fit_results):
    """Test all clusters are included in output."""
    config = WriterConfig(verbosity=Verbosity.FULL)
    writer = XMLWriter(config)

    output_path = writer.write(sample_fit_results, tmp_path)

    tree = ET.parse(output_path)
    clusters = tree.findall(".//cluster")
    assert len(clusters) == len(sample_fit_results.clusters)
```

### Integration Tests

```python
def test_results_writer_includes_xml(tmp_path, sample_fit_results):
    """Test ResultsWriter can produce XML output."""
    config = WriterConfig()
    writer = ResultsWriter(config)

    outputs = writer.write_all(
        sample_fit_results,
        tmp_path,
        formats=["json", "xml"],
    )

    assert "xml" in outputs
    assert outputs["xml"].exists()
```

## Configuration Integration

### Pipeline Integration

The output system is integrated into `FitPipeline`:

```python
# services/fit/pipeline.py

def _run_fit(self, clusters, ...) -> FitResults:
    # ... fitting logic ...

    # Build results
    results = self._build_fit_results(cluster_results)

    # Write outputs
    if self.config.output:
        self._write_outputs(results)

    return results

def _write_outputs(self, results: FitResults) -> None:
    """Write all configured output formats."""
    from peakfit.services.fit.writer import write_new_format_outputs

    write_new_format_outputs(
        results=results,
        output_dir=self.output_dir,
        config=self.config.output,
    )
```

### OutputConfig Fields

```python
@dataclass
class OutputConfig:
    """Configuration for output generation."""
    directory: str | None = None
    verbosity: OutputVerbosity = OutputVerbosity.STANDARD
    include_legacy: bool = False
    save_chains: bool = True
    save_figures: bool = True
    include_timestamp: bool = False
    formats: list[str] = field(default_factory=lambda: ["json", "csv", "markdown"])
```

## Best Practices

### 1. Respect Verbosity

Writers should adjust output based on verbosity level:

```python
def write(self, results: FitResults, output_dir: Path) -> Path:
    data = {"clusters": self._serialize_clusters(results.clusters)}

    if self.config.verbosity >= Verbosity.STANDARD:
        data["statistics"] = self._serialize_statistics(results)

    if self.config.verbosity >= Verbosity.FULL:
        data["diagnostics"] = self._serialize_diagnostics(results)
        data["raw_data"] = self._serialize_raw_data(results)

    return self._write_file(data, output_dir)
```

### 2. Handle Missing Data Gracefully

```python
def _serialize_mcmc(self, cluster: ClusterEstimates) -> dict | None:
    if cluster.mcmc_diagnostics is None:
        return None
    return {
        "status": cluster.mcmc_diagnostics.status.value,
        "n_samples": cluster.mcmc_diagnostics.n_samples,
    }
```

### 3. Use Type Hints

```python
def write(self, results: FitResults, output_dir: Path) -> Path:
    """Write results to output directory.

    Args:
        results: Complete fit results
        output_dir: Target directory (must exist)

    Returns:
        Path to primary output file

    Raises:
        IOError: If output cannot be written
    """
```

### 4. Document Output Format

Include format documentation in docstrings or README.

---

**Last Updated**: 2025-01-15
