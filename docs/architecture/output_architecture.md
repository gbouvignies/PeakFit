# Output System Architecture

This document describes how PeakFit builds and writes structured results in the
vertical‑slice architecture.

## Overview

```
Fit pipeline → Result models → Writers → JSON/CSV/Markdown/PDF
```

## Core Components

### Result Models

Result dataclasses live in the **engine** so they can be used by `fit`, `plot`, and `mcmc`
without cross‑slice imports.

### Results Builder

A results builder in the **fit** slice assembles engine results, diagnostics, and metadata
into a single structured object.

### Writers

Writers live in `peakfit.io` and are coordinated by the **fit** slice. Each writer is
format‑specific and handles serialization only.

Supported outputs:
- JSON
- CSV
- Markdown
- PDF (Matplotlib‑generated pages; no ReportLab/PyPDF dependency)

## Output Directory Structure

The concrete file layout is documented in [docs/output_system.md](docs/output_system.md).
