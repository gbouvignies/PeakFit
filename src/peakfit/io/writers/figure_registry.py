"""Figure registration and management for output.

This module provides functionality for registering, tracking, and
organizing figures generated during fitting and analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class FigureCategory(str, Enum):
    """Categories of generated figures."""

    RESIDUALS = "residuals"
    FIT_QUALITY = "fit_quality"
    PARAMETER_CORRELATION = "parameter_correlation"
    MCMC_TRACE = "mcmc_trace"
    MCMC_POSTERIOR = "mcmc_posterior"
    MCMC_CORNER = "mcmc_corner"
    AMPLITUDE_PROFILE = "amplitude_profile"
    SUMMARY = "summary"
    DIAGNOSTIC = "diagnostic"
    CUSTOM = "custom"


@dataclass
class FigureInfo:
    """Information about a generated figure.

    Attributes:
        filename: Name of the figure file
        category: Figure category for organization
        title: Human-readable title
        description: Description of what the figure shows
        cluster_id: Associated cluster ID (if applicable)
        peak_name: Associated peak name (if applicable)
        format: File format (png, pdf, svg)
        width_px: Figure width in pixels
        height_px: Figure height in pixels
        dpi: Resolution in dots per inch
    """

    filename: str
    category: FigureCategory
    title: str = ""
    description: str = ""
    cluster_id: int | None = None
    peak_name: str | None = None
    format: str = "png"
    width_px: int = 800
    height_px: int = 600
    dpi: int = 150

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, object] = {
            "filename": self.filename,
            "category": self.category.value,
            "format": self.format,
        }
        if self.title:
            result["title"] = self.title
        if self.description:
            result["description"] = self.description
        if self.cluster_id is not None:
            result["cluster_id"] = self.cluster_id
        if self.peak_name:
            result["peak_name"] = self.peak_name
        result["dimensions"] = {
            "width_px": self.width_px,
            "height_px": self.height_px,
            "dpi": self.dpi,
        }
        return result


@dataclass
class FigureRegistry:
    """Registry for tracking all generated figures.

    This class maintains a catalog of all figures generated during
    a fitting run, with metadata for organization and retrieval.

    Example:
        >>> registry = FigureRegistry()
        >>> registry.register(
        ...     "residuals_cluster_0.png",
        ...     FigureCategory.RESIDUALS,
        ...     title="Residual Plot",
        ...     cluster_id=0,
        ... )
        >>> registry.save(output_dir / "figures")
    """

    figures: list[FigureInfo] = field(default_factory=list)
    base_dir: Path | None = None

    def register(
        self,
        filename: str,
        category: FigureCategory,
        *,
        title: str = "",
        description: str = "",
        cluster_id: int | None = None,
        peak_name: str | None = None,
        format: str = "png",
        width_px: int = 800,
        height_px: int = 600,
        dpi: int = 150,
    ) -> FigureInfo:
        """Register a new figure.

        Args:
            filename: Name of the figure file
            category: Figure category
            title: Human-readable title
            description: Description of the figure
            cluster_id: Associated cluster ID
            peak_name: Associated peak name
            format: File format
            width_px: Width in pixels
            height_px: Height in pixels
            dpi: Resolution

        Returns:
            FigureInfo object for the registered figure
        """
        info = FigureInfo(
            filename=filename,
            category=category,
            title=title,
            description=description,
            cluster_id=cluster_id,
            peak_name=peak_name,
            format=format,
            width_px=width_px,
            height_px=height_px,
            dpi=dpi,
        )
        self.figures.append(info)
        return info

    def get_by_category(self, category: FigureCategory) -> list[FigureInfo]:
        """Get all figures in a category.

        Args:
            category: Figure category to filter by

        Returns:
            List of FigureInfo objects
        """
        return [f for f in self.figures if f.category == category]

    def get_by_cluster(self, cluster_id: int) -> list[FigureInfo]:
        """Get all figures for a specific cluster.

        Args:
            cluster_id: Cluster ID to filter by

        Returns:
            List of FigureInfo objects
        """
        return [f for f in self.figures if f.cluster_id == cluster_id]

    def get_by_peak(self, peak_name: str) -> list[FigureInfo]:
        """Get all figures for a specific peak.

        Args:
            peak_name: Peak name to filter by

        Returns:
            List of FigureInfo objects
        """
        return [f for f in self.figures if f.peak_name == peak_name]

    def save(self, figures_dir: Path) -> Path:
        """Save figure registry to JSON file.

        Args:
            figures_dir: Directory to save registry

        Returns:
            Path to saved registry file
        """
        figures_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = figures_dir

        registry_path = figures_dir / "figure_registry.json"
        data = {
            "n_figures": len(self.figures),
            "categories": self._get_category_summary(),
            "figures": [f.to_dict() for f in self.figures],
        }

        with registry_path.open("w") as f:
            json.dump(data, f, indent=2)

        return registry_path

    def _get_category_summary(self) -> dict[str, int]:
        """Get count of figures per category."""
        summary: dict[str, int] = {}
        for fig in self.figures:
            cat = fig.category.value
            summary[cat] = summary.get(cat, 0) + 1
        return summary

    @classmethod
    def load(cls, figures_dir: Path) -> FigureRegistry:
        """Load figure registry from JSON file.

        Args:
            figures_dir: Directory containing registry

        Returns:
            FigureRegistry object

        Raises:
            FileNotFoundError: If registry file doesn't exist
        """
        registry_path = figures_dir / "figure_registry.json"
        if not registry_path.exists():
            msg = f"Figure registry not found at {registry_path}"
            raise FileNotFoundError(msg)

        with registry_path.open() as f:
            data = json.load(f)

        registry = cls(base_dir=figures_dir)
        for fig_data in data.get("figures", []):
            registry.figures.append(
                FigureInfo(
                    filename=fig_data["filename"],
                    category=FigureCategory(fig_data["category"]),
                    title=fig_data.get("title", ""),
                    description=fig_data.get("description", ""),
                    cluster_id=fig_data.get("cluster_id"),
                    peak_name=fig_data.get("peak_name"),
                    format=fig_data.get("format", "png"),
                    width_px=fig_data.get("dimensions", {}).get("width_px", 800),
                    height_px=fig_data.get("dimensions", {}).get("height_px", 600),
                    dpi=fig_data.get("dimensions", {}).get("dpi", 150),
                )
            )

        return registry

    def to_dict(self) -> dict:
        """Convert registry to dictionary."""
        return {
            "n_figures": len(self.figures),
            "categories": self._get_category_summary(),
            "figures": [f.to_dict() for f in self.figures],
        }


__all__ = [
    "FigureCategory",
    "FigureInfo",
    "FigureRegistry",
]
