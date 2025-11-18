"""Tests for writing output files."""

import numpy as np
import pytest

from peakfit.writing import print_heights


class TestPrintHeights:
    """Tests for print_heights function."""

    def test_print_heights_single_value(self):
        """Test printing single height value."""
        z_values = np.array([0.0])
        heights = np.array([100.5])
        height_err = np.array([5.2])

        result = print_heights(z_values, heights, height_err)

        assert "0.0" in result
        assert "1.005000e+02" in result
        assert "5.200000e+00" in result

    def test_print_heights_multiple_values(self):
        """Test printing multiple height values."""
        z_values = np.array([0.0, 1.0, 2.0])
        heights = np.array([100.0, 90.0, 80.0])
        height_err = np.array([5.0, 4.5, 4.0])

        result = print_heights(z_values, heights, height_err)

        lines = result.strip().split("\n")
        # Header + 3 data lines
        assert len(lines) == 4
        assert "Z" in lines[0]
        assert "I" in lines[0]
        assert "I_err" in lines[0]

    def test_print_heights_header_format(self):
        """Test that header is formatted correctly."""
        z_values = np.array([0.0])
        heights = np.array([100.0])
        height_err = np.array([5.0])

        result = print_heights(z_values, heights, height_err)

        lines = result.split("\n")
        assert lines[0].startswith("#")
        assert "Z" in lines[0]
        assert "I" in lines[0]
        assert "I_err" in lines[0]

    def test_print_heights_scientific_notation(self):
        """Test that values are in scientific notation."""
        z_values = np.array([0.0])
        heights = np.array([1234567.89])
        height_err = np.array([0.000123])

        result = print_heights(z_values, heights, height_err)

        # Should contain 'e+' or 'e-' for scientific notation
        assert "e+" in result or "e-" in result

    def test_print_heights_mismatched_lengths(self):
        """Test that mismatched array lengths raise ValueError."""
        z_values = np.array([0.0, 1.0])
        heights = np.array([100.0])
        height_err = np.array([5.0])

        with pytest.raises(ValueError, match="Array length mismatch"):
            print_heights(z_values, heights, height_err)

    def test_print_heights_all_different_lengths(self):
        """Test error when all arrays have different lengths."""
        z_values = np.array([0.0])
        heights = np.array([100.0, 90.0])
        height_err = np.array([5.0, 4.5, 3.5])

        with pytest.raises(ValueError, match="Array length mismatch"):
            print_heights(z_values, heights, height_err)

    def test_print_heights_empty_arrays(self):
        """Test with empty arrays."""
        z_values = np.array([])
        heights = np.array([])
        height_err = np.array([])

        result = print_heights(z_values, heights, height_err)

        # Should have header only
        lines = result.strip().split("\n")
        assert len(lines) == 1
        assert "Z" in lines[0]

    def test_print_heights_negative_values(self):
        """Test with negative values."""
        z_values = np.array([-1.0, 0.0, 1.0])
        heights = np.array([-50.0, 0.0, 50.0])
        height_err = np.array([5.0, 5.0, 5.0])

        result = print_heights(z_values, heights, height_err)

        lines = result.strip().split("\n")
        assert len(lines) == 4  # Header + 3 data lines
        assert "-1.0" in result or "-1" in result
        assert "-5.000000e+01" in result

    def test_print_heights_large_values(self):
        """Test with large values."""
        z_values = np.array([1e6])
        heights = np.array([1e10])
        height_err = np.array([1e8])

        result = print_heights(z_values, heights, height_err)

        assert "e+" in result  # Should use scientific notation
