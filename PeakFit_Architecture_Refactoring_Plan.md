# PeakFit Architecture Refactoring Plan

## Executive Summary

### Current State Assessment

PeakFit is a Python-based NMR spectroscopy lineshape fitting tool at approximately **70% architectural completion**. The codebase (~11,300 LOC) has made significant progress in transitioning from a monolithic structure to a layered architecture, with clear separation emerging between core domain logic, services, and CLI layers.

**Strengths:**
- Well-defined `core/` package with domain models, algorithms, and fitting logic
- Emerging service layer pattern in `services/analyze/`
- Type hints used consistently throughout
- Registry pattern for lineshapes already implemented
- 309 tests providing good coverage

**Critical Issues:**
1. **Layering violations**: `io/output.py` imports `ui` module (domain → presentation)
2. **Monolithic CLI**: `cli/app.py` (993 LOC) with 16+ imports, high fan-out
3. **Mixed concerns**: `core/diagnostics/plots.py` (865 LOC) combines computation with visualization
4. **Incomplete service abstraction**: CLI imports directly from `core` in many places
5. **Underutilized packages**: `data/` package is merely a re-export facade

**Architectural Maturity Rating: 6.5/10**

### Vision for Final Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                          │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌─────────────────┐  │
│  │   CLI   │  │ Plotting │  │   GUI   │  │ Future: REST API│  │
│  └────┬────┘  └────┬─────┘  └────┬────┘  └────────┬────────┘  │
└───────┼────────────┼────────────┼─────────────────┼───────────┘
        │            │            │                 │
        ▼            ▼            ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Service Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ FitService   │  │AnalyzeService│  │ ReportingService     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼─────────────────┼─────────────────────┼───────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Core Layer                                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐ │
│  │  Domain    │  │ Algorithms │  │  Fitting   │  │Diagnostics│ │
│  │  Models    │  │            │  │            │  │(compute)  │ │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │ StateRepo    │  │ FileAdapters │  │ ProgressReporter       ││
│  │ (Persistence)│  │ (I/O)        │  │ (abstraction)          ││
│  └──────────────┘  └──────────────┘  └────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Key Improvements

1. **Complete Decoupling**: No presentation layer dependencies in core/domain/infra
2. **Service Facade**: Single entry points for all major workflows
3. **Strategy Pattern**: Pluggable optimizers and lineshapes
4. **Observer Pattern**: Progress reporting via events, not direct console calls
5. **Repository Pattern**: Already started, to be completed with adapters

### Estimated Total Effort

| Phase | Duration | Effort (person-days) |
|-------|----------|---------------------|
| Phase 1: Reporter Abstraction | 1 week | 3-4 |
| Phase 2: Diagnostic Separation | 1 week | 4-5 |
| Phase 3: CLI Decomposition | 1.5 weeks | 5-6 |
| Phase 4: Service Completion | 1.5 weeks | 5-7 |
| Phase 5: Advanced Patterns | 1 week | 3-4 |
| Phase 6: Package Cleanup | 0.5 weeks | 2-3 |
| Phase 7: Documentation & Polish | 0.5 weeks | 2-3 |
| **Total** | **7 weeks** | **24-32** |

---

## Architectural Analysis

### 1. Current Architecture Assessment

#### SOLID Principles Evaluation

| Principle | Current State | Score |
|-----------|--------------|-------|
| **S**ingle Responsibility | Mixed - `cli/app.py` handles too many commands; `diagnostics/plots.py` mixes computation and viz | 5/10 |
| **O**pen/Closed | Good - lineshape registry allows extension without modification | 7/10 |
| **L**iskov Substitution | Good - Shape protocol properly defines contract | 8/10 |
| **I**nterface Segregation | Moderate - `FittingOptions` Protocol could be split | 6/10 |
| **D**ependency Inversion | Weak - Lower layers depend on concrete `ui` module | 4/10 |

**Overall SOLID Score: 6/10**

#### Anti-Patterns Identified

1. **God Object**: `cli/app.py` orchestrates everything
2. **Feature Envy**: `io/output.py` reaches into `ui` for status messages
3. **Inappropriate Intimacy**: CLI commands import deeply from `core`
4. **Shotgun Surgery**: Adding a new analysis method requires changes in CLI, services, and core
5. **Speculative Generality**: `data/` package exists but adds no value

#### Module Boundary Coherence

| Package | Purpose | Coherence | Issues |
|---------|---------|-----------|--------|
| `core/domain` | Domain models | High | Well-defined DTOs |
| `core/fitting` | Optimization | High | Clear responsibility |
| `core/algorithms` | Algorithms | High | Focused utilities |
| `core/diagnostics` | MCMC analysis | Low | Plotting mixed in |
| `core/lineshapes` | Shape models | High | Good registry pattern |
| `services/analyze` | Analysis orchestration | High | Good abstraction |
| `services/fit` | Fit orchestration | Medium | Still imports ui |
| `cli` | CLI interface | Low | Monolithic, high fan-out |
| `io` | File I/O | Medium | ui dependency |
| `ui` | Console output | High | Good styling utilities |
| `data` | (facade) | N/A | Should be removed |
| `plotting` | Visualization | Medium | Needs service layer |
| `analysis` | Benchmarking | Low | Should be in services or tools |

### 2. Dependency Graph Analysis

#### Current Problematic Dependencies

```
                    ┌────────────┐
                    │    CLI     │
                    │  (app.py)  │
                    └─────┬──────┘
                          │
    ┌──────────┬──────────┼──────────┬──────────┐
    ▼          ▼          ▼          ▼          ▼
┌──────┐  ┌───────┐  ┌────────┐  ┌─────┐  ┌────────┐
│ core │  │  io   │  │plotting│  │ ui  │  │services│
│(17)  │  │  (2)  │  │   (1)  │  │ (6) │  │  (via  │
└──┬───┘  └───┬───┘  └───┬────┘  └─────┘  │submod) │
   │          │          │                 └────────┘
   │          │          │
   │    ┌─────┴──────────┘
   │    │  ⚠️ VIOLATION
   │    ▼
   │  ┌─────┐
   └──│ ui  │◀────────── io/output.py imports ui
      └─────┘            (presentation in domain layer)
```

#### Optimal Dependency Flow (Target)

```
┌─────────────────────────────────────────────────────────────┐
│  Presentation: cli, plotting                                │
│  Can import: services, ui                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Services: services/*                                       │
│  Can import: core, infra                                    │
│  Receives: Reporter (injected)                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Core: core/*                                               │
│  Can import: core/shared only                               │
│  No external dependencies except numpy/scipy                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Infrastructure: infra/*                                    │
│  Can import: core/domain (for serialization contracts)      │
└─────────────────────────────────────────────────────────────┘
```

#### Circular Dependency Risks

Currently **no circular dependencies** detected at package level. However, the following risk exists:

- If `services/fit` continues to grow without abstraction, it may start importing from `cli` for shared models

### 3. Cohesion and Coupling Metrics

#### Module Cohesion (LCOM - Lack of Cohesion of Methods)

| Module | Methods | Shared State | LCOM Score | Assessment |
|--------|---------|--------------|------------|------------|
| `cli/app.py` | 12 commands | Minimal | High (bad) | Low cohesion |
| `ui/style.py` | 50+ methods | `console` singleton | Medium | Acceptable |
| `core/fitting/optimizer.py` | 15 | `Parameters` | Low (good) | High cohesion |
| `core/diagnostics/plots.py` | 8 | None | Medium | Split needed |
| `services/fit/pipeline.py` | 20 | Multiple | High (bad) | Needs decomposition |

#### Coupling Analysis

**Afferent Coupling (Ca)** - Who depends on this module:
| Module | Ca | Interpretation |
|--------|-----|----------------|
| `core/domain/*` | 25+ | Highly stable, many dependents |
| `ui/style.py` | 15+ | Too many layers depend on it |
| `core/fitting/parameters.py` | 20+ | Central abstraction |

**Efferent Coupling (Ce)** - What this module depends on:
| Module | Ce | Interpretation |
|--------|-----|----------------|
| `cli/app.py` | 16 | Too high, needs reduction |
| `services/fit/pipeline.py` | 16 | Acceptable for orchestrator |
| `core/diagnostics/plots.py` | 5 | Split will reduce |

**Instability (I = Ce / (Ca + Ce))**:
- `cli/app.py`: I = 0.94 (very unstable - good for adapter)
- `core/domain/*`: I = 0.15 (very stable - correct)
- `ui/style.py`: I = 0.25 (should be more stable if lower layers depend on it)

---

## Phased Refactoring Plan

### Phase 1: Reporter Abstraction (Week 1)

**Objective**: Eliminate UI dependencies in lower layers by introducing a Reporter protocol.

**Anti-patterns eliminated**: Feature Envy, Inappropriate Intimacy

**New capabilities**: Testable I/O layer, silent mode for batch processing

#### 1.1 Detailed Steps

##### Step 1: Create Reporter Protocol

```python
# File: peakfit/core/shared/reporter.py
"""Progress and status reporting abstraction."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reporter(Protocol):
    """Protocol for progress and status reporting.
    
    This abstraction allows core and infrastructure layers to report
    progress without depending on specific UI implementations.
    """
    
    def action(self, message: str) -> None:
        """Report an action being performed."""
        ...
    
    def info(self, message: str) -> None:
        """Report informational message."""
        ...
    
    def warning(self, message: str) -> None:
        """Report a warning."""
        ...
    
    def error(self, message: str) -> None:
        """Report an error."""
        ...
    
    def success(self, message: str) -> None:
        """Report successful completion."""
        ...


class NullReporter:
    """Silent reporter that discards all messages.
    
    Useful for testing or batch processing where output is not needed.
    """
    
    def action(self, message: str) -> None:
        pass
    
    def info(self, message: str) -> None:
        pass
    
    def warning(self, message: str) -> None:
        pass
    
    def error(self, message: str) -> None:
        pass
    
    def success(self, message: str) -> None:
        pass


class LoggingReporter:
    """Reporter that writes to Python logging.
    
    Useful for background processing or when console output is not available.
    """
    
    def __init__(self, logger_name: str = "peakfit") -> None:
        import logging
        self._logger = logging.getLogger(logger_name)
    
    def action(self, message: str) -> None:
        self._logger.info(f"[ACTION] {message}")
    
    def info(self, message: str) -> None:
        self._logger.info(message)
    
    def warning(self, message: str) -> None:
        self._logger.warning(message)
    
    def error(self, message: str) -> None:
        self._logger.error(message)
    
    def success(self, message: str) -> None:
        self._logger.info(f"[SUCCESS] {message}")
```

##### Step 2: Create Console Reporter Adapter

```python
# File: peakfit/ui/reporter.py
"""Console-based reporter implementation using Rich."""

from peakfit.core.shared.reporter import Reporter
from peakfit.ui.style import PeakFitUI


class ConsoleReporter:
    """Reporter implementation using Rich console output.
    
    Adapts the Reporter protocol to use PeakFitUI styling.
    """
    
    def action(self, message: str) -> None:
        PeakFitUI.action(message)
    
    def info(self, message: str) -> None:
        PeakFitUI.info(message)
    
    def warning(self, message: str) -> None:
        PeakFitUI.warning(message)
    
    def error(self, message: str) -> None:
        PeakFitUI.error(message)
    
    def success(self, message: str) -> None:
        PeakFitUI.success(message)


# Default reporter instance for CLI usage
default_reporter = ConsoleReporter()
```

##### Step 3: Refactor io/output.py

```python
# File: peakfit/io/output.py (refactored)
"""Output file writers for peak fitting results."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.fitting.computation import calculate_shape_heights
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.shared.reporter import NullReporter, Reporter
from peakfit.core.shared.typing import FittingOptions, FloatArray

if TYPE_CHECKING:
    pass


def write_profiles(
    path: Path,
    z_values: np.ndarray,
    clusters: list[Cluster],
    params: Parameters,
    args: FittingOptions,
    reporter: Reporter | None = None,
) -> None:
    """Write profile information to output files.
    
    Args:
        path: Output directory path
        z_values: Z-dimension values array
        clusters: List of clusters to write
        params: Fitting parameters
        args: Fitting options containing noise level
        reporter: Optional reporter for status messages (default: silent)
    """
    if reporter is None:
        reporter = NullReporter()
    
    reporter.action("Writing profiles...")
    for cluster in clusters:
        _shapes, amplitudes = calculate_shape_heights(params, cluster)
        amplitudes_err = np.full_like(amplitudes, args.noise)
        for i, peak in enumerate(cluster.peaks):
            write_profile(
                path,
                peak,
                params,
                z_values,
                amplitudes[i],
                amplitudes_err[i],
            )


def write_shifts(
    peaks: list[Peak],
    params: Parameters,
    file_shifts: Path,
    reporter: Reporter | None = None,
) -> None:
    """Write the shifts to the output file.
    
    Args:
        peaks: List of peaks
        params: Fitting parameters
        file_shifts: Output file path
        reporter: Optional reporter for status messages
    """
    if reporter is None:
        reporter = NullReporter()
    
    reporter.action("Writing shifts...")
    with file_shifts.open("w") as f:
        for peak in peaks:
            peak.update_positions(params)
            name = peak.name
            positions_str = " ".join(f"{position:10.5f}" for position in peak.positions)
            f.write(f"{name:>15s} {positions_str}\n")
```

##### Step 4: Update service layer to inject reporter

```python
# In services/fit/pipeline.py - add reporter parameter
from peakfit.core.shared.reporter import Reporter, NullReporter
from peakfit.ui.reporter import ConsoleReporter

class FitPipeline:
    @staticmethod
    def run(
        spectrum_path: Path,
        peaklist_path: Path,
        z_values_path: Path | None,
        config: PeakFitConfig,
        *,
        optimizer: str = "leastsq",
        save_state: bool = True,
        verbose: bool = False,
        reporter: Reporter | None = None,  # NEW PARAMETER
    ) -> None:
        if reporter is None:
            reporter = ConsoleReporter() if verbose else NullReporter()
        
        # Pass reporter to io functions
        write_profiles(..., reporter=reporter)
```

#### 1.2 Testing Strategy

```python
# File: tests/unit/test_reporter.py
"""Tests for reporter abstraction."""

import pytest
from peakfit.core.shared.reporter import NullReporter, LoggingReporter, Reporter


class MockReporter:
    """Test double for capturing reporter calls."""
    
    def __init__(self):
        self.messages: list[tuple[str, str]] = []
    
    def action(self, message: str) -> None:
        self.messages.append(("action", message))
    
    def info(self, message: str) -> None:
        self.messages.append(("info", message))
    
    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))
    
    def error(self, message: str) -> None:
        self.messages.append(("error", message))
    
    def success(self, message: str) -> None:
        self.messages.append(("success", message))


def test_null_reporter_is_silent():
    """NullReporter should not raise or produce output."""
    reporter = NullReporter()
    reporter.action("test")
    reporter.info("test")
    reporter.warning("test")
    reporter.error("test")
    reporter.success("test")
    # No assertions - just verify no exceptions


def test_mock_reporter_captures_messages():
    """MockReporter should capture all messages for testing."""
    reporter = MockReporter()
    reporter.action("doing something")
    reporter.info("info message")
    
    assert len(reporter.messages) == 2
    assert reporter.messages[0] == ("action", "doing something")
    assert reporter.messages[1] == ("info", "info message")


def test_reporter_protocol_compliance():
    """All reporters should satisfy the Reporter protocol."""
    assert isinstance(NullReporter(), Reporter)
    assert isinstance(MockReporter(), Reporter)
```

#### 1.3 Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking existing CLI output | Medium | Low | Keep ConsoleReporter as default |
| Performance overhead | Low | Low | Reporter calls are lightweight |
| Missing reporter calls | Medium | Medium | Grep for `ui.` to find all usages |

#### 1.4 Success Criteria

- [ ] `io/output.py` has no `from peakfit.ui` import
- [ ] All tests pass (309 tests)
- [ ] CLI output unchanged when using ConsoleReporter
- [ ] `grep -r "from peakfit.ui" src/peakfit/io/` returns empty

---

### Phase 2: Diagnostic Separation (Week 2)

**Objective**: Separate pure computation from visualization in `core/diagnostics/`.

**Anti-patterns eliminated**: Mixed Concerns, God Module

**New capabilities**: Headless MCMC diagnostics, reusable metrics

#### 2.1 Detailed Steps

##### Step 1: Create metrics module for pure computation

```python
# File: peakfit/core/diagnostics/metrics.py
"""Pure computation functions for MCMC diagnostics.

This module contains only stateless, pure functions that compute
diagnostic metrics from MCMC chains. No plotting or I/O.
"""

from dataclasses import dataclass

import numpy as np

from peakfit.core.shared.typing import FloatArray


@dataclass(frozen=True)
class TraceMetrics:
    """Computed metrics for a single parameter's trace."""
    
    mean: float
    std: float
    median: float
    q05: float  # 5th percentile
    q95: float  # 95th percentile
    ess: float  # Effective sample size
    rhat: float  # Gelman-Rubin statistic
    
    @property
    def is_converged(self) -> bool:
        """Check if R-hat indicates convergence (< 1.01)."""
        return self.rhat <= 1.01


@dataclass(frozen=True)
class AutocorrelationResult:
    """Autocorrelation analysis results."""
    
    lags: FloatArray
    autocorr: FloatArray
    integrated_autocorr_time: float


def compute_trace_metrics(
    chains: FloatArray,
    param_index: int,
) -> TraceMetrics:
    """Compute summary metrics for a single parameter across all chains.
    
    Args:
        chains: Array of shape (n_chains, n_samples, n_params)
        param_index: Index of parameter to analyze
        
    Returns:
        TraceMetrics with computed statistics
    """
    # Extract parameter samples from all chains, post burn-in
    samples = chains[:, :, param_index].flatten()
    
    # Basic statistics
    mean = float(np.mean(samples))
    std = float(np.std(samples))
    median = float(np.median(samples))
    q05 = float(np.percentile(samples, 5))
    q95 = float(np.percentile(samples, 95))
    
    # Effective sample size (simplified)
    ess = _compute_ess(chains[:, :, param_index])
    
    # R-hat (Gelman-Rubin)
    rhat = _compute_rhat(chains[:, :, param_index])
    
    return TraceMetrics(
        mean=mean,
        std=std,
        median=median,
        q05=q05,
        q95=q95,
        ess=ess,
        rhat=rhat,
    )


def compute_autocorrelation(
    chain: FloatArray,
    max_lag: int | None = None,
) -> AutocorrelationResult:
    """Compute autocorrelation function for a single chain.
    
    Args:
        chain: 1D array of samples from single chain
        max_lag: Maximum lag to compute (default: len(chain) // 2)
        
    Returns:
        AutocorrelationResult with lags and autocorrelation values
    """
    n = len(chain)
    if max_lag is None:
        max_lag = n // 2
    
    # Compute autocorrelation using FFT for efficiency
    chain_centered = chain - np.mean(chain)
    fft = np.fft.fft(chain_centered, n=2 * n)
    acf = np.fft.ifft(fft * np.conj(fft))[:n].real
    acf = acf / acf[0]  # Normalize
    
    lags = np.arange(max_lag)
    autocorr = acf[:max_lag]
    
    # Integrated autocorrelation time
    # Sum until autocorrelation goes negative
    cumsum = np.cumsum(autocorr)
    first_negative = np.argmax(autocorr < 0)
    if first_negative == 0:
        first_negative = max_lag
    iat = 1 + 2 * cumsum[first_negative - 1]
    
    return AutocorrelationResult(
        lags=lags.astype(np.float64),
        autocorr=autocorr,
        integrated_autocorr_time=float(iat),
    )


def _compute_ess(chains: FloatArray) -> float:
    """Compute effective sample size across chains."""
    n_chains, n_samples = chains.shape
    
    # Simple ESS estimation using variance ratio
    within_chain_var = np.mean(np.var(chains, axis=1))
    total_var = np.var(chains)
    
    if total_var == 0:
        return float(n_chains * n_samples)
    
    # Approximate ESS
    ess = n_chains * n_samples * within_chain_var / total_var
    return float(min(ess, n_chains * n_samples))


def _compute_rhat(chains: FloatArray) -> float:
    """Compute Gelman-Rubin R-hat statistic."""
    n_chains, n_samples = chains.shape
    
    if n_chains < 2:
        return float("nan")
    
    # Chain means
    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)
    
    # Between-chain variance
    B = n_samples * np.var(chain_means, ddof=1)
    
    # Within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1))
    
    if W == 0:
        return float("nan")
    
    # Pooled variance estimate
    var_hat = ((n_samples - 1) * W + B) / n_samples
    
    # R-hat
    rhat = np.sqrt(var_hat / W)
    return float(rhat)
```

##### Step 2: Create visualization module

```python
# File: peakfit/plotting/diagnostics.py
"""Visualization functions for MCMC diagnostics.

This module contains only plotting functions that consume
pre-computed metrics from core/diagnostics/metrics.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from peakfit.core.diagnostics.convergence import ConvergenceDiagnostics
from peakfit.core.diagnostics.metrics import (
    AutocorrelationResult,
    TraceMetrics,
    compute_autocorrelation,
    compute_trace_metrics,
)
from peakfit.core.shared.typing import FloatArray


def plot_trace(
    chains: FloatArray,
    parameter_names: list[str],
    burn_in: int = 0,
    metrics: list[TraceMetrics] | None = None,
    max_params: int = 20,
) -> Figure:
    """Create trace plots showing MCMC chain evolution.
    
    Args:
        chains: Array of shape (n_chains, n_samples, n_params)
        parameter_names: List of parameter names
        burn_in: Number of burn-in samples to mark
        metrics: Pre-computed metrics (computed if not provided)
        max_params: Maximum number of parameters to plot
        
    Returns:
        Matplotlib Figure object
    """
    n_chains, n_samples, n_params = chains.shape
    n_params_plot = min(n_params, max_params)
    
    # Compute metrics if not provided
    if metrics is None:
        metrics = [
            compute_trace_metrics(chains[:, burn_in:, :], i)
            for i in range(n_params_plot)
        ]
    
    # Create figure
    n_cols = min(3, n_params_plot)
    n_rows = int(np.ceil(n_params_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    
    if n_params_plot == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    color_map = plt.get_cmap("tab10")
    colors = color_map(np.linspace(0, 1, min(n_chains, 10)))
    
    for i in range(n_params_plot):
        ax = axes[i]
        param_name = parameter_names[i]
        metric = metrics[i]
        
        # Plot chains
        for chain_idx in range(n_chains):
            chain_data = chains[chain_idx, :, i]
            
            if burn_in > 0:
                ax.plot(range(burn_in), chain_data[:burn_in],
                       color="gray", alpha=0.3, linewidth=0.5)
                ax.plot(range(burn_in, n_samples), chain_data[burn_in:],
                       color=colors[chain_idx % len(colors)],
                       alpha=0.7, linewidth=0.5)
            else:
                ax.plot(chain_data, color=colors[chain_idx % len(colors)],
                       alpha=0.7, linewidth=0.5)
        
        if burn_in > 0:
            ax.axvline(burn_in, color="red", linestyle="--", alpha=0.5)
        
        # Annotate with R-hat
        status = "✓" if metric.is_converged else "⚠" if metric.rhat <= 1.05 else "✗"
        title = f"{param_name} ({status} R̂={metric.rhat:.3f})"
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params_plot, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(f"MCMC Trace Plots ({n_chains} chains, {n_samples} samples)",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    return fig


def save_diagnostic_plots(
    chains: FloatArray,
    parameter_names: list[str],
    output_path: Path,
    burn_in: int = 0,
) -> None:
    """Save all diagnostic plots to PDF.
    
    Args:
        chains: MCMC chains array
        parameter_names: Parameter names
        output_path: Output PDF path
        burn_in: Burn-in samples
    """
    with PdfPages(output_path) as pdf:
        # Trace plots
        fig = plot_trace(chains, parameter_names, burn_in)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Autocorrelation plots
        fig = plot_autocorrelation(chains, parameter_names)
        pdf.savefig(fig)
        plt.close(fig)
```

##### Step 3: Move existing plots.py functions

1. Keep only matplotlib figure creation in new `plotting/diagnostics.py`
2. Extract pure computation to `core/diagnostics/metrics.py`
3. Update imports in CLI commands

#### 2.2 File Organization Changes

**Before:**
```
core/diagnostics/
├── __init__.py
├── burnin.py
├── convergence.py
└── plots.py (865 LOC - MIXED)
```

**After:**
```
core/diagnostics/
├── __init__.py
├── burnin.py
├── convergence.py
└── metrics.py (NEW - pure computation)

plotting/
├── __init__.py
├── common.py
├── diagnostics.py (NEW - visualization only)
└── plots/
    └── spectra.py
```

#### 2.3 Migration Path

1. Create `metrics.py` with extracted pure functions
2. Create `plotting/diagnostics.py` with visualization
3. Update `plots.py` to import from new modules (deprecation period)
4. Update CLI to use new structure
5. Remove old `plots.py` after verification

#### 2.4 Success Criteria

- [ ] `core/diagnostics/` contains no matplotlib imports
- [ ] All diagnostic tests pass
- [ ] Plot output visually identical
- [ ] `core/diagnostics/plots.py` removed or deprecated

---

### Phase 3: CLI Decomposition (Weeks 3-4)

**Objective**: Break up monolithic `cli/app.py` into focused command modules.

**Anti-patterns eliminated**: God Object, High Fan-out

**New capabilities**: Easier testing, command-specific configuration

#### 3.1 Target File Structure

```
cli/
├── __init__.py
├── app.py              # Main Typer app, command registration only (~100 LOC)
├── common.py           # Shared CLI utilities
├── commands/
│   ├── __init__.py
│   ├── fit.py          # fit command
│   ├── validate.py     # validate command
│   ├── init.py         # init command
│   ├── info.py         # info command
│   ├── analyze.py      # analyze subcommand group
│   ├── plot.py         # plot subcommand group
│   └── benchmark.py    # benchmark command
└── models.py           # CLI-specific data models
```

#### 3.2 Refactored app.py

```python
# File: peakfit/cli/app.py (refactored - ~100 LOC target)
"""Main Typer application for PeakFit.

This module defines the CLI structure and imports commands from
dedicated modules. Each command is implemented in cli/commands/.
"""

import typer

from peakfit.cli.callbacks import version_callback
from peakfit.cli.commands import (
    analyze,
    benchmark,
    fit,
    info,
    init,
    plot,
    validate,
)

app = typer.Typer(
    name="peakfit",
    help="PeakFit - Lineshape fitting for pseudo-3D NMR spectra",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """PeakFit - Modern lineshape fitting for pseudo-3D NMR spectra."""


# Register commands
app.command()(fit.fit_command)
app.command(name="validate")(validate.validate_command)
app.command(name="init")(init.init_command)
app.command(name="info")(info.info_command)
app.command(name="analyze")(analyze.analyze_command)
app.command(name="benchmark")(benchmark.benchmark_command)

# Register plot subapp
app.add_typer(plot.plot_app, name="plot")


if __name__ == "__main__":
    app()
```

#### 3.3 Example Command Module

```python
# File: peakfit/cli/commands/fit.py
"""Implementation of the fit command."""

from pathlib import Path
from typing import Annotated

import typer

from peakfit.core.domain.config import (
    ClusterConfig,
    FitConfig,
    LineshapeName,
    OutputConfig,
    PeakFitConfig,
)
from peakfit.io.config import load_config
from peakfit.services.fit import FitPipeline


def fit_command(
    spectrum: Annotated[
        Path,
        typer.Argument(
            help="Path to NMRPipe spectrum file (.ft2, .ft3)",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    peaklist: Annotated[
        Path,
        typer.Argument(
            help="Path to peak list file (.list, .csv, .json, .xlsx)",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    z_values: Annotated[
        Path | None,
        typer.Option(
            "--z-values", "-z",
            help="Path to Z-dimension values file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o",
            help="Output directory for results",
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path("Fits"),
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c",
            help="Path to TOML configuration file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    lineshape: Annotated[
        LineshapeName,
        typer.Option("--lineshape", "-l", help="Lineshape model"),
    ] = "auto",
    refine: Annotated[
        int,
        typer.Option("--refine", "-r", help="Refinement iterations", min=0, max=20),
    ] = 1,
    optimizer: Annotated[
        str,
        typer.Option("--optimizer", help="Optimization algorithm"),
    ] = "leastsq",
    save_state: Annotated[
        bool,
        typer.Option("--save-state/--no-save-state", help="Save fitting state"),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Show verbose output"),
    ] = False,
    # ... other options
) -> None:
    """Fit lineshapes to peaks in pseudo-3D NMR spectrum."""
    # Build config
    if config is not None:
        fit_config = load_config(config)
        fit_config.output.directory = output
    else:
        fit_config = PeakFitConfig(
            fitting=FitConfig(
                lineshape=lineshape,
                refine_iterations=refine,
            ),
            output=OutputConfig(directory=output),
        )
    
    # Delegate to service layer
    FitPipeline.run(
        spectrum_path=spectrum,
        peaklist_path=peaklist,
        z_values_path=z_values,
        config=fit_config,
        optimizer=optimizer,
        save_state=save_state,
        verbose=verbose,
    )
```

#### 3.4 Testing Strategy

```python
# File: tests/unit/test_cli_commands.py
"""Unit tests for individual CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from peakfit.cli.app import app


runner = CliRunner()


class TestFitCommand:
    """Tests for the fit command."""
    
    @patch("peakfit.cli.commands.fit.FitPipeline")
    def test_fit_delegates_to_pipeline(self, mock_pipeline, tmp_path):
        """Fit command should delegate to FitPipeline."""
        # Create dummy files
        spectrum = tmp_path / "test.ft2"
        spectrum.write_bytes(b"dummy")
        peaklist = tmp_path / "peaks.list"
        peaklist.write_text("# dummy")
        
        result = runner.invoke(app, [
            "fit", str(spectrum), str(peaklist),
            "--output", str(tmp_path / "output"),
        ])
        
        # Should not raise
        mock_pipeline.run.assert_called_once()
    
    def test_fit_validates_spectrum_exists(self, tmp_path):
        """Fit command should fail for non-existent spectrum."""
        result = runner.invoke(app, [
            "fit", "/nonexistent/spectrum.ft2", "/nonexistent/peaks.list",
        ])
        
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or result.exit_code == 2
```

#### 3.5 Success Criteria

- [ ] `cli/app.py` < 150 LOC
- [ ] Each command in separate module
- [ ] CLI behavior unchanged (integration tests pass)
- [ ] Import fan-out of `app.py` < 5 packages

---

### Phase 4: Service Layer Completion (Weeks 4-5)

**Objective**: Complete the service abstraction so CLI only imports from services.

**Anti-patterns eliminated**: Shotgun Surgery, Leaky Abstractions

**New capabilities**: Reusable API for future GUIs

#### 4.1 Target Service Architecture

```
services/
├── __init__.py
├── fit/
│   ├── __init__.py
│   ├── service.py       # FitService facade
│   ├── pipeline.py      # Orchestration (internal)
│   └── config.py        # Service-specific config adapters
├── analyze/
│   ├── __init__.py      # Exports all services
│   ├── mcmc_service.py
│   ├── profile_service.py
│   ├── correlation_service.py
│   ├── uncertainty_service.py
│   └── state_service.py
└── plot/                 # NEW
    ├── __init__.py
    ├── service.py        # PlotService facade
    └── generators.py     # Plot generation logic
```

#### 4.2 FitService Facade

```python
# File: peakfit/services/fit/service.py
"""High-level fitting service facade.

This service provides the primary API for fitting operations.
CLI and other adapters should import only from this module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from peakfit.core.domain.config import PeakFitConfig
from peakfit.core.domain.state import FittingState
from peakfit.core.shared.reporter import NullReporter, Reporter


@dataclass(frozen=True)
class FitResult:
    """Result of a fitting operation.
    
    Attributes:
        state: The final fitting state
        output_dir: Directory where results were written
        success: Whether fitting completed successfully
        summary: Human-readable summary of results
    """
    state: FittingState
    output_dir: Path
    success: bool
    summary: dict[str, Any]


class FitService:
    """Service for NMR peak fitting operations.
    
    This is the primary entry point for fitting workflows.
    
    Example:
        service = FitService()
        result = service.fit(
            spectrum_path=Path("spectrum.ft2"),
            peaklist_path=Path("peaks.list"),
        )
        print(f"Fitted {len(result.state.peaks)} peaks")
    """
    
    def __init__(self, reporter: Reporter | None = None) -> None:
        """Initialize the fit service.
        
        Args:
            reporter: Reporter for status messages (default: silent)
        """
        self._reporter = reporter or NullReporter()
    
    def fit(
        self,
        spectrum_path: Path,
        peaklist_path: Path,
        z_values_path: Path | None = None,
        config: PeakFitConfig | None = None,
        *,
        optimizer: str = "leastsq",
        save_state: bool = True,
    ) -> FitResult:
        """Perform peak fitting on a spectrum.
        
        Args:
            spectrum_path: Path to NMRPipe spectrum file
            peaklist_path: Path to peak list file
            z_values_path: Optional path to Z-values file
            config: Fitting configuration (uses defaults if not provided)
            optimizer: Optimization algorithm
            save_state: Whether to save state for later analysis
            
        Returns:
            FitResult with fitting state and summary
            
        Raises:
            FileNotFoundError: If spectrum or peaklist doesn't exist
            ValueError: If configuration is invalid
        """
        from peakfit.services.fit.pipeline import FitPipeline
        
        if config is None:
            config = PeakFitConfig()
        
        # Run the pipeline
        state = FitPipeline.run_internal(
            spectrum_path=spectrum_path,
            peaklist_path=peaklist_path,
            z_values_path=z_values_path,
            config=config,
            optimizer=optimizer,
            save_state=save_state,
            reporter=self._reporter,
        )
        
        return FitResult(
            state=state,
            output_dir=config.output.directory,
            success=True,
            summary={
                "n_peaks": len(state.peaks),
                "n_clusters": len(state.clusters),
                "n_parameters": len(state.params),
            },
        )
    
    def validate_inputs(
        self,
        spectrum_path: Path,
        peaklist_path: Path,
    ) -> dict[str, Any]:
        """Validate input files without performing fitting.
        
        Args:
            spectrum_path: Path to spectrum file
            peaklist_path: Path to peak list file
            
        Returns:
            Validation report with details about each file
        """
        from peakfit.cli.validate_command import validate_spectrum, validate_peaklist
        
        return {
            "spectrum": validate_spectrum(spectrum_path),
            "peaklist": validate_peaklist(peaklist_path),
        }
```

#### 4.3 PlotService (New)

```python
# File: peakfit/services/plot/service.py
"""Plotting service for generating visualizations.

This service abstracts plot generation so adapters don't need
to know about matplotlib internals.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from peakfit.core.domain.state import FittingState
from peakfit.core.shared.reporter import NullReporter, Reporter


@dataclass(frozen=True)
class PlotOutput:
    """Result of plot generation."""
    
    path: Path
    plot_type: str
    n_plots: int


class PlotService:
    """Service for generating plots from fitting results."""
    
    def __init__(self, reporter: Reporter | None = None) -> None:
        self._reporter = reporter or NullReporter()
    
    def generate_intensity_plots(
        self,
        state: FittingState,
        output_path: Path | None = None,
        show: bool = False,
    ) -> PlotOutput:
        """Generate intensity profile plots.
        
        Args:
            state: Fitting state with results
            output_path: Output PDF path (auto-generated if None)
            show: Whether to display interactively
            
        Returns:
            PlotOutput with path and count
        """
        from peakfit.plotting.plots.spectra import plot_intensity_profiles
        
        if output_path is None:
            output_path = Path("intensity_profiles.pdf")
        
        n_plots = plot_intensity_profiles(
            state=state,
            output_path=output_path,
            show=show,
        )
        
        return PlotOutput(
            path=output_path,
            plot_type="intensity",
            n_plots=n_plots,
        )
    
    def generate_cest_plots(
        self,
        state: FittingState,
        output_path: Path | None = None,
        reference_indices: list[int] | None = None,
        show: bool = False,
    ) -> PlotOutput:
        """Generate CEST profile plots."""
        ...
    
    def generate_mcmc_diagnostics(
        self,
        chains: Any,  # FloatArray
        parameter_names: list[str],
        output_path: Path | None = None,
        burn_in: int = 0,
    ) -> PlotOutput:
        """Generate MCMC diagnostic plots."""
        from peakfit.plotting.diagnostics import save_diagnostic_plots
        
        if output_path is None:
            output_path = Path("mcmc_diagnostics.pdf")
        
        save_diagnostic_plots(
            chains=chains,
            parameter_names=parameter_names,
            output_path=output_path,
            burn_in=burn_in,
        )
        
        return PlotOutput(
            path=output_path,
            plot_type="mcmc_diagnostics",
            n_plots=len(parameter_names),
        )
```

#### 4.4 Updated CLI Using Services Only

```python
# File: peakfit/cli/commands/fit.py (final version)
"""Fit command using service layer only."""

from pathlib import Path
from typing import Annotated

import typer

from peakfit.core.domain.config import PeakFitConfig
from peakfit.services.fit import FitService
from peakfit.ui.reporter import ConsoleReporter


def fit_command(
    spectrum: Annotated[Path, typer.Argument(...)],
    peaklist: Annotated[Path, typer.Argument(...)],
    # ... options
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Fit lineshapes to peaks in pseudo-3D NMR spectrum."""
    # Create service with appropriate reporter
    reporter = ConsoleReporter() if verbose else None
    service = FitService(reporter=reporter)
    
    # Build config from CLI options
    config = _build_config_from_options(...)
    
    # Delegate to service - CLI has no knowledge of core internals
    try:
        result = service.fit(
            spectrum_path=spectrum,
            peaklist_path=peaklist,
            config=config,
        )
        
        if verbose:
            _print_summary(result.summary)
            
    except FileNotFoundError as e:
        raise typer.Exit(1) from e
```

#### 4.5 Success Criteria

- [ ] CLI imports only from `services/*` and `ui/*`
- [ ] No `from peakfit.core` in CLI modules
- [ ] Services provide complete API coverage
- [ ] Integration tests pass

---

### Phase 5: Advanced Patterns (Week 6)

**Objective**: Implement Strategy and Factory patterns for extensibility.

#### 5.1 Strategy Pattern for Optimizers

```python
# File: peakfit/core/fitting/strategies.py
"""Optimization strategies using Strategy pattern."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.optimize import least_squares, differential_evolution, basinhopping

from peakfit.core.fitting.parameters import Parameters
from peakfit.core.shared.typing import FloatArray


class OptimizationStrategy(Protocol):
    """Protocol for optimization strategies."""
    
    def optimize(
        self,
        objective: callable,
        x0: FloatArray,
        bounds: tuple[FloatArray, FloatArray],
        **kwargs: Any,
    ) -> "OptimizationResult":
        """Run optimization.
        
        Args:
            objective: Function to minimize
            x0: Initial parameter values
            bounds: (lower, upper) bounds
            **kwargs: Strategy-specific options
            
        Returns:
            OptimizationResult with solution and metadata
        """
        ...


@dataclass
class OptimizationResult:
    """Result from optimization."""
    
    x: FloatArray  # Optimal parameters
    cost: float    # Final cost value
    success: bool  # Whether optimization converged
    message: str   # Status message
    n_evaluations: int  # Number of function evaluations


class LeastSquaresStrategy:
    """Fast local optimization using Levenberg-Marquardt."""
    
    def __init__(
        self,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        max_nfev: int | None = None,
    ) -> None:
        self.ftol = ftol
        self.xtol = xtol
        self.max_nfev = max_nfev
    
    def optimize(
        self,
        objective: callable,
        x0: FloatArray,
        bounds: tuple[FloatArray, FloatArray],
        **kwargs: Any,
    ) -> OptimizationResult:
        result = least_squares(
            objective,
            x0,
            bounds=bounds,
            ftol=self.ftol,
            xtol=self.xtol,
            max_nfev=self.max_nfev,
            **kwargs,
        )
        
        return OptimizationResult(
            x=result.x,
            cost=float(result.cost),
            success=result.success,
            message=result.message,
            n_evaluations=result.nfev,
        )


class BasinHoppingStrategy:
    """Global optimization using basin hopping."""
    
    def __init__(
        self,
        n_iterations: int = 100,
        temperature: float = 1.0,
    ) -> None:
        self.n_iterations = n_iterations
        self.temperature = temperature
    
    def optimize(
        self,
        objective: callable,
        x0: FloatArray,
        bounds: tuple[FloatArray, FloatArray],
        **kwargs: Any,
    ) -> OptimizationResult:
        # Wrap for scipy interface
        def wrapper(x):
            return np.sum(objective(x) ** 2)
        
        result = basinhopping(
            wrapper,
            x0,
            niter=self.n_iterations,
            T=self.temperature,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": list(zip(bounds[0], bounds[1])),
            },
        )
        
        return OptimizationResult(
            x=result.x,
            cost=float(result.fun),
            success=result.lowest_optimization_result.success,
            message=str(result.message),
            n_evaluations=result.nfev,
        )


class DifferentialEvolutionStrategy:
    """Global optimization using differential evolution."""
    
    def __init__(
        self,
        max_iterations: int = 1000,
        population_size: int = 15,
        mutation: tuple[float, float] = (0.5, 1.0),
    ) -> None:
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.mutation = mutation
    
    def optimize(
        self,
        objective: callable,
        x0: FloatArray,
        bounds: tuple[FloatArray, FloatArray],
        **kwargs: Any,
    ) -> OptimizationResult:
        def wrapper(x):
            return np.sum(objective(x) ** 2)
        
        result = differential_evolution(
            wrapper,
            bounds=list(zip(bounds[0], bounds[1])),
            x0=x0,
            maxiter=self.max_iterations,
            popsize=self.population_size,
            mutation=self.mutation,
        )
        
        return OptimizationResult(
            x=result.x,
            cost=float(result.fun),
            success=result.success,
            message=result.message,
            n_evaluations=result.nfev,
        )


# Strategy factory
STRATEGIES: dict[str, type[OptimizationStrategy]] = {
    "leastsq": LeastSquaresStrategy,
    "basin-hopping": BasinHoppingStrategy,
    "differential-evolution": DifferentialEvolutionStrategy,
}


def get_strategy(name: str, **kwargs: Any) -> OptimizationStrategy:
    """Get an optimization strategy by name.
    
    Args:
        name: Strategy name
        **kwargs: Strategy-specific configuration
        
    Returns:
        Configured strategy instance
        
    Raises:
        KeyError: If strategy not found
    """
    strategy_class = STRATEGIES[name]
    return strategy_class(**kwargs)


def register_strategy(name: str, strategy_class: type[OptimizationStrategy]) -> None:
    """Register a custom optimization strategy.
    
    Args:
        name: Name to register under
        strategy_class: Strategy class to register
    """
    STRATEGIES[name] = strategy_class
```

#### 5.2 Factory Pattern for Lineshapes (Enhanced)

```python
# File: peakfit/core/lineshapes/factory.py
"""Factory for creating lineshape instances."""

from typing import Any

from peakfit.core.domain.spectrum import SpectralParameters
from peakfit.core.lineshapes.registry import SHAPES, Shape


class LineshapeFactory:
    """Factory for creating lineshape model instances.
    
    Provides a clean API for creating lineshapes without
    needing to know the concrete class names.
    
    Example:
        factory = LineshapeFactory()
        shape = factory.create(
            "gaussian",
            axis="x",
            cluster_id=0,
            center=100.0,
            size=16,
            spec_params=params,
        )
    """
    
    @staticmethod
    def create(
        shape_type: str,
        axis: str,
        cluster_id: int,
        center: float,
        size: int,
        spec_params: SpectralParameters,
        **kwargs: Any,
    ) -> Shape:
        """Create a lineshape instance.
        
        Args:
            shape_type: Type of lineshape (gaussian, lorentzian, pvoigt, etc.)
            axis: Axis label ('x' or 'y')
            cluster_id: Cluster identifier
            center: Peak center position
            size: Number of points
            spec_params: Spectral parameters
            **kwargs: Additional shape-specific parameters
            
        Returns:
            Configured Shape instance
            
        Raises:
            KeyError: If shape_type is not registered
        """
        shape_class = SHAPES[shape_type]
        return shape_class(
            axis=axis,
            cluster_id=cluster_id,
            center=center,
            size=size,
            spec_params=spec_params,
            **kwargs,
        )
    
    @staticmethod
    def available_shapes() -> list[str]:
        """Get list of available shape types."""
        return list(SHAPES.keys())
    
    @staticmethod
    def create_auto(
        axis: str,
        cluster_id: int,
        center: float,
        size: int,
        spec_params: SpectralParameters,
        hints: dict[str, Any] | None = None,
    ) -> Shape:
        """Auto-detect appropriate lineshape from spectral parameters.
        
        Args:
            axis: Axis label
            cluster_id: Cluster identifier
            center: Peak center position
            size: Number of points
            spec_params: Spectral parameters (used for auto-detection)
            hints: Optional hints for selection
            
        Returns:
            Automatically selected Shape instance
        """
        # Auto-detection logic based on apodization, etc.
        shape_type = _detect_shape_type(spec_params, hints)
        return LineshapeFactory.create(
            shape_type=shape_type,
            axis=axis,
            cluster_id=cluster_id,
            center=center,
            size=size,
            spec_params=spec_params,
        )


def _detect_shape_type(
    spec_params: SpectralParameters,
    hints: dict[str, Any] | None = None,
) -> str:
    """Detect appropriate lineshape based on spectral parameters."""
    # Implementation based on apodization windows
    if hints and "lineshape" in hints:
        return hints["lineshape"]
    
    # Default detection logic
    # ... (existing auto-detection code)
    return "pvoigt"  # Safe default
```

#### 5.3 Observer Pattern for Progress

```python
# File: peakfit/core/shared/events.py
"""Event system for progress reporting."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable


class EventType(Enum):
    """Types of events in PeakFit."""
    
    FIT_STARTED = auto()
    FIT_PROGRESS = auto()
    FIT_COMPLETED = auto()
    CLUSTER_STARTED = auto()
    CLUSTER_COMPLETED = auto()
    MCMC_PROGRESS = auto()
    ERROR = auto()


@dataclass
class Event:
    """Base event class."""
    
    event_type: EventType
    data: dict[str, Any]


@dataclass
class FitProgressEvent(Event):
    """Event for fitting progress."""
    
    current_cluster: int
    total_clusters: int
    current_iteration: int
    total_iterations: int


class EventHandler(ABC):
    """Abstract base for event handlers."""
    
    @abstractmethod
    def handle(self, event: Event) -> None:
        """Handle an event."""
        ...


class EventDispatcher:
    """Central event dispatcher."""
    
    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}
    
    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler | Callable[[Event], None],
    ) -> None:
        """Subscribe to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        if callable(handler) and not isinstance(handler, EventHandler):
            # Wrap callable in handler
            handler = _CallableHandler(handler)
        
        self._handlers[event_type].append(handler)
    
    def dispatch(self, event: Event) -> None:
        """Dispatch an event to all subscribers."""
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            handler.handle(event)


class _CallableHandler(EventHandler):
    """Adapter for callable handlers."""
    
    def __init__(self, func: Callable[[Event], None]) -> None:
        self._func = func
    
    def handle(self, event: Event) -> None:
        self._func(event)


# Usage in services:
class FitPipeline:
    def __init__(self, dispatcher: EventDispatcher | None = None):
        self._dispatcher = dispatcher or EventDispatcher()
    
    def _notify_progress(self, cluster: int, total: int) -> None:
        if self._dispatcher:
            self._dispatcher.dispatch(FitProgressEvent(
                event_type=EventType.FIT_PROGRESS,
                data={},
                current_cluster=cluster,
                total_clusters=total,
                current_iteration=0,
                total_iterations=1,
            ))
```

---

### Phase 6: Package Cleanup (Week 6.5)

**Objective**: Remove unused packages, clarify roles, establish public APIs.

#### 6.1 Package Reorganization

**Remove:**
- `data/` package (it's just re-exports from `core/domain`)

**Rename/Clarify:**
- `infra/` → Keep as-is (clear infrastructure role)
- `analysis/` → Move to `tools/benchmarks/` (development tooling, not production)

**Final Structure:**
```
peakfit/
├── __init__.py
├── __main__.py
├── cli/
│   ├── __init__.py
│   ├── app.py
│   ├── callbacks.py
│   ├── common.py
│   └── commands/
│       ├── __init__.py
│       ├── analyze.py
│       ├── benchmark.py
│       ├── fit.py
│       ├── info.py
│       ├── init.py
│       ├── plot.py
│       └── validate.py
├── core/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── clustering.py
│   │   └── noise.py
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── burnin.py
│   │   ├── convergence.py
│   │   └── metrics.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── cluster.py
│   │   ├── config.py
│   │   ├── peaks.py
│   │   ├── peaks_io.py
│   │   ├── spectrum.py
│   │   └── state.py
│   ├── fitting/
│   │   ├── __init__.py
│   │   ├── advanced.py
│   │   ├── computation.py
│   │   ├── optimizer.py
│   │   ├── parameters.py
│   │   ├── results.py
│   │   ├── simulation.py
│   │   └── strategies.py
│   ├── lineshapes/
│   │   ├── __init__.py
│   │   ├── factory.py
│   │   ├── functions.py
│   │   ├── models.py
│   │   └── registry.py
│   └── shared/
│       ├── __init__.py
│       ├── constants.py
│       ├── events.py
│       ├── reporter.py
│       └── typing.py
├── infra/
│   ├── __init__.py
│   └── state.py
├── io/
│   ├── __init__.py
│   ├── config.py
│   └── output.py
├── plotting/
│   ├── __init__.py
│   ├── common.py
│   ├── diagnostics.py
│   └── plots/
│       ├── __init__.py
│       └── spectra.py
├── services/
│   ├── __init__.py
│   ├── analyze/
│   │   ├── __init__.py
│   │   ├── correlation_service.py
│   │   ├── mcmc_service.py
│   │   ├── profile_service.py
│   │   ├── state_service.py
│   │   └── uncertainty_service.py
│   ├── fit/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── service.py
│   └── plot/
│       ├── __init__.py
│       └── service.py
└── ui/
    ├── __init__.py
    ├── reporter.py
    └── style.py
```

#### 6.2 Public API Definition

```python
# File: peakfit/__init__.py
"""PeakFit - Lineshape fitting for pseudo-3D NMR spectra.

Public API:
    - FitService: Main fitting service
    - AnalyzeService: Uncertainty analysis
    - PlotService: Visualization generation
    
Configuration:
    - PeakFitConfig: Main configuration object
    - FitConfig, ClusterConfig, OutputConfig: Sub-configurations

Domain Objects:
    - FittingState: Complete fitting state
    - Peak, Cluster, Parameters: Core domain objects
"""

from peakfit._version import __version__

# Services (primary API)
from peakfit.services.fit import FitService
from peakfit.services.analyze import (
    MCMCAnalysisService,
    ProfileLikelihoodService,
    ParameterCorrelationService,
    FittingStateService,
)
from peakfit.services.plot import PlotService

# Configuration
from peakfit.core.domain.config import (
    PeakFitConfig,
    FitConfig,
    ClusterConfig,
    OutputConfig,
)

# Domain objects (read-only access)
from peakfit.core.domain.state import FittingState

__all__ = [
    # Version
    "__version__",
    # Services
    "FitService",
    "MCMCAnalysisService",
    "ProfileLikelihoodService",
    "ParameterCorrelationService",
    "FittingStateService",
    "PlotService",
    # Configuration
    "PeakFitConfig",
    "FitConfig",
    "ClusterConfig",
    "OutputConfig",
    # Domain
    "FittingState",
]
```

#### 6.3 Import Validation

```python
# File: tools/validate_imports.py
"""Validate import structure follows architectural rules."""

import ast
import sys
from pathlib import Path


LAYER_ORDER = {
    "cli": {"services", "ui", "core.domain.config"},
    "services": {"core", "infra"},
    "plotting": {"core", "ui"},
    "io": {"core", "infra"},  # NOT ui!
    "core": {"core"},  # Only within core
    "infra": {"core.domain"},
    "ui": set(),  # Leaf node
}

FORBIDDEN_PATTERNS = [
    ("io", "ui"),  # io cannot import ui
    ("core", "ui"),  # core cannot import ui
    ("core", "cli"),  # core cannot import cli
    ("core", "services"),  # core cannot import services
    ("infra", "ui"),  # infra cannot import ui
]


def validate_imports(src_dir: Path) -> list[str]:
    """Validate all imports in source directory."""
    violations = []
    
    for py_file in src_dir.rglob("*.py"):
        rel_path = py_file.relative_to(src_dir)
        source_pkg = str(rel_path.parts[0]) if rel_path.parts else ""
        
        with open(py_file) as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                target = _get_import_target(node)
                if target and target.startswith("peakfit."):
                    target_pkg = target.replace("peakfit.", "").split(".")[0]
                    
                    for src, tgt in FORBIDDEN_PATTERNS:
                        if source_pkg == src and target_pkg == tgt:
                            violations.append(
                                f"{py_file}: {source_pkg} -> {target_pkg}"
                            )
    
    return violations


def _get_import_target(node: ast.Import | ast.ImportFrom) -> str | None:
    """Extract import target from AST node."""
    if isinstance(node, ast.Import):
        return node.names[0].name if node.names else None
    elif isinstance(node, ast.ImportFrom):
        return node.module
    return None


if __name__ == "__main__":
    violations = validate_imports(Path("src/peakfit"))
    if violations:
        print("Import violations found:")
        for v in violations:
            print(f"  ✗ {v}")
        sys.exit(1)
    else:
        print("✓ All imports follow architectural rules")
```

---

### Phase 7: Documentation & Polish (Week 7)

#### 7.1 Architecture Documentation

Create `docs/architecture/overview.md`:

```markdown
# PeakFit Architecture

## Layer Overview

PeakFit follows a layered architecture with clear dependency rules:

```
┌─────────────────┐
│  Presentation   │  cli/, plotting/, ui/
│  (Adapters)     │  - Depends on: Services, UI utilities
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Services     │  services/
│  (Orchestration)│  - Depends on: Core, Infrastructure
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      Core       │  core/
│  (Domain Logic) │  - No external dependencies (except numpy/scipy)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Infrastructure  │  infra/, io/
│  (Persistence)  │  - Depends on: Core domain contracts
└─────────────────┘
```

## Key Design Decisions

### ADR-001: Reporter Protocol for Decoupling UI
- **Decision**: Introduce `Reporter` protocol to abstract status messages
- **Rationale**: Allows core/io layers to report progress without UI dependency
- **Trade-off**: Slight increase in complexity for better testability

### ADR-002: Service Layer Pattern
- **Decision**: All CLI commands delegate to service classes
- **Rationale**: Enables reuse for GUI, API, and testing
- **Trade-off**: Additional indirection for simple commands

### ADR-003: Strategy Pattern for Optimizers
- **Decision**: Use Strategy pattern for optimization algorithms
- **Rationale**: Easy extension without modifying core fitting code
- **Trade-off**: More classes for simple use cases

## Extending PeakFit

### Adding a New Lineshape

1. Create class in `core/lineshapes/models.py`
2. Implement `Shape` protocol
3. Register with `@register_shape("name")` decorator
4. Add tests in `tests/unit/test_lineshapes.py`

### Adding a New Analysis Method

1. Create service in `services/analyze/`
2. Export from `services/analyze/__init__.py`
3. Create CLI command in `cli/commands/analyze.py`
4. Add integration tests
```

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Max file size | 993 LOC | <500 LOC | `wc -l` on largest file |
| CLI import fan-out | 16 packages | <5 packages | AST analysis |
| Layering violations | 2 | 0 | `validate_imports.py` |
| Test count | 309 | 320+ | `pytest --collect-only` |
| Test coverage | ~70% | >80% | `pytest --cov` |
| Service API coverage | 60% | 100% | Manual audit |

---

## Visual Diagrams

### Current Architecture (As-Is)

```
┌──────────────────────────────────────────────────────────────┐
│                        CLI (app.py)                          │
│                        993 LOC                               │
│         ┌───────────────────────────────────────┐            │
│         │ Imports: core(17), ui(6), io(2),      │            │
│         │         data(6), plotting(1), services│            │
│         └───────────────────────────────────────┘            │
└──────────────────────────┬───────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   services  │     │    core     │     │     ui      │
│  (partial)  │     │   domain    │     │   style     │
└─────────────┘     │  algorithms │     └──────▲──────┘
                    │  fitting    │            │
                    │  diagnostics│            │
                    │  lineshapes │            │
                    └─────────────┘            │
                           ▲                   │
                           │        ┌──────────┘
                    ┌──────┴──────┐ │ ⚠️ VIOLATION
                    │     io      │─┘
                    │   output    │
                    └─────────────┘
```

### Target Architecture (To-Be)

```
┌──────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  ┌────────┐  ┌──────────┐  ┌────────┐  ┌─────────────────┐  │
│  │CLI ~150│  │ Plotting │  │   UI   │  │ Future: GUI/API │  │
│  │  LOC   │  │   only   │  │  only  │  │                 │  │
│  └───┬────┘  └────┬─────┘  └────────┘  └────────┬────────┘  │
└──────┼────────────┼────────────────────────────┼────────────┘
       │            │                            │
       ▼            ▼                            ▼
┌──────────────────────────────────────────────────────────────┐
│                      Service Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  FitService  │  │AnalyzeService│  │   PlotService    │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────┘   │
└─────────┼─────────────────┼─────────────────────┼───────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌──────────────────────────────────────────────────────────────┐
│                        Core Layer                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────┐ │
│  │   Domain   │  │ Algorithms │  │  Fitting   │  │Diagnos-│ │
│  │   Models   │  │            │  │ +Strategies│  │  tics  │ │
│  └────────────┘  └────────────┘  └────────────┘  │(metrics│ │
│                                                   │  only) │ │
│                                                   └────────┘ │
└──────────────────────────────────────────────────────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌──────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ StateRepo    │  │ FileWriters  │  │     Reporter       │ │
│  │ (Persistence)│  │ (I/O)        │  │   (abstraction)    │ │
│  └──────────────┘  └──────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## Answers to Architectural Questions

### Q1: Should `analysis/` be separate from `services/`?

**Recommendation**: Merge into services or move to `tools/`.

The current `analysis/` package contains benchmarking and profiling utilities that are development tools, not production features. Options:
1. Move to `tools/benchmarks/` (recommended) - keeps production code clean
2. If needed in production, create `services/benchmark/` service

### Q2: What is the right abstraction for progress reporting?

**Recommendation**: Protocol-based Reporter with dependency injection.

```python
class Reporter(Protocol):
    def action(self, message: str) -> None: ...
    def progress(self, current: int, total: int) -> None: ...
```

This allows:
- `NullReporter` for testing/silent mode
- `ConsoleReporter` for CLI
- `LoggingReporter` for batch processing
- Future: `ProgressBarReporter`, `WebSocketReporter`

### Q3: How should configuration be handled across the service layer?

**Recommendation**: Pass config objects explicitly, don't use globals.

```python
class FitService:
    def fit(self, ..., config: PeakFitConfig) -> FitResult:
        # Config is explicit dependency
```

Benefits:
- Testable (easy to pass test configs)
- Thread-safe (no global state)
- Self-documenting API

### Q4: What's the best way to handle error propagation?

**Recommendation**: Domain-specific exceptions raised by services, translated by adapters.

```python
# In services/
class PeakFitError(Exception): ...
class InvalidSpectrumError(PeakFitError): ...
class FittingConvergenceError(PeakFitError): ...

# In CLI
try:
    result = service.fit(...)
except InvalidSpectrumError as e:
    ui.error(str(e))
    raise typer.Exit(1)
```

### Q5: Should there be a dedicated `adapters/` package?

**Recommendation**: No, keep current structure.

The current `cli/` package at top level is clearer than `adapters/cli/`. If we add more adapters (GUI, REST), consider:
```
adapters/
├── cli/
├── gui/
└── api/
```

But for now, `cli/` at top level is fine.

### Q6: How to handle cross-cutting concern of logging/diagnostics?

**Recommendation**: Use the Reporter abstraction (Phase 1) plus Python's standard `logging` module.

- Reporter: For user-facing status messages
- Logging: For debugging and audit trails
- Keep them separate - don't mix console output with log files

### Q7: What's the role of `infra/` vs `io/`?

**Recommendation**: Clarify and keep separate.

- `infra/`: Technical infrastructure (state persistence, caching, etc.)
- `io/`: File format I/O (reading spectra, writing results)

They serve different purposes:
- `infra/state.py` handles Python object serialization
- `io/output.py` handles NMR-specific file formats

### Q8: How to make lineshape registry more extensible?

**Recommendation**: Already well-designed, add plugin mechanism.

Current `@register_shape` decorator is good. Enhance with:
1. User plugins directory scanned at startup
2. Entry point mechanism for pip-installable extensions

```python
# In pyproject.toml
[project.entry-points."peakfit.lineshapes"]
custom = "my_package:CustomShape"
```

---

## Implementation Checklist

### Phase 1: Reporter Abstraction
- [ ] Create `core/shared/reporter.py` with Protocol
- [ ] Create `ui/reporter.py` with ConsoleReporter
- [ ] Refactor `io/output.py` to accept Reporter
- [ ] Update `services/fit/pipeline.py` to inject Reporter
- [ ] Add tests for Reporter implementations
- [ ] Verify no `ui` imports in `io/`

### Phase 2: Diagnostic Separation
- [ ] Create `core/diagnostics/metrics.py`
- [ ] Create `plotting/diagnostics.py`
- [ ] Migrate pure functions from `plots.py`
- [ ] Update CLI imports
- [ ] Deprecate/remove old `plots.py`
- [ ] Verify no matplotlib in `core/`

### Phase 3: CLI Decomposition
- [ ] Create `cli/commands/` directory
- [ ] Extract fit command
- [ ] Extract validate command
- [ ] Extract analyze commands
- [ ] Extract plot commands
- [ ] Reduce `app.py` to <150 LOC
- [ ] Update all CLI tests

### Phase 4: Service Completion
- [ ] Create `FitService` facade
- [ ] Create `PlotService`
- [ ] Update CLI to use services only
- [ ] Remove direct core imports from CLI
- [ ] Add service-level tests

### Phase 5: Advanced Patterns
- [ ] Implement Strategy pattern for optimizers
- [ ] Enhance Factory for lineshapes
- [ ] Add event dispatcher (optional)
- [ ] Document extension points

### Phase 6: Package Cleanup
- [ ] Remove `data/` package
- [ ] Move `analysis/` to tools
- [ ] Define `__all__` in all packages
- [ ] Create import validator
- [ ] Run validation in CI

### Phase 7: Documentation
- [ ] Update architecture docs
- [ ] Create ADRs for key decisions
- [ ] Update README with new structure
- [ ] Create contributor guide

---

*Document prepared for Guillaume Bouvignies, PeakFit maintainer*
*Analysis based on PeakFit codebase snapshot from November 2025*
