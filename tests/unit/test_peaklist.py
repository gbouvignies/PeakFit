"""Tests for peaklist reading functionality."""

import numpy as np

from peakfit.core.domain.peaks_io import _make_names, register_reader


class TestMakeNames:
    """Tests for _make_names function."""

    def test_make_names_valid_assignment(self):
        """Test creating name from valid assignment strings."""
        f1name = "A.51.LEU.HD1"
        f2name = "A.51.LEU.HG"
        peak_id = 1

        result = _make_names(f1name, f2name, peak_id)
        # _make_names is vectorized and returns numpy array/scalar
        result_str = str(result)

        # Should combine residue info: LEU51HD1-HG
        assert "LEU" in result_str
        assert "51" in result_str
        assert "HD1" in result_str
        assert "HG" in result_str

    def test_make_names_same_residue(self):
        """Test creating name when both assignments are same residue."""
        f1name = "A.51.LEU.HD1"
        f2name = "A.51.LEU.HG"
        peak_id = 1

        result = _make_names(f1name, f2name, peak_id)
        result_str = str(result)

        # Second assignment shouldn't repeat residue info
        assert result_str.count("LEU") == 1
        assert result_str.count("51") == 1

    def test_make_names_different_residues(self):
        """Test creating name when assignments are different residues."""
        f1name = "A.51.LEU.HD1"
        f2name = "A.52.VAL.HG"
        peak_id = 1

        result = _make_names(f1name, f2name, peak_id)
        result_str = str(result)

        # Should include both residues
        assert "LEU" in result_str
        assert "VAL" in result_str
        assert "51" in result_str
        assert "52" in result_str

    def test_make_names_invalid_format(self):
        """Test with invalid assignment format."""
        f1name = "InvalidFormat"
        f2name = "A.51.LEU.HG"
        peak_id = 42

        result = _make_names(f1name, f2name, peak_id)
        result_str = str(result)

        # Should fall back to peak_id
        assert result_str == "42"

    def test_make_names_wrong_number_of_items(self):
        """Test with wrong number of dot-separated items."""
        f1name = "A.51.LEU"  # Only 3 items instead of 4
        f2name = "A.51.LEU.HG"
        peak_id = 42

        result = _make_names(f1name, f2name, peak_id)
        result_str = str(result)

        # Should fall back to peak_id
        assert result_str == "42"

    def test_make_names_numeric_input(self):
        """Test with numeric input instead of strings."""
        f1name = 5.5
        f2name = 120.3
        peak_id = 42

        result = _make_names(f1name, f2name, peak_id)
        result_str = str(result)

        # Should fall back to peak_id
        assert result_str == "42"

    def test_make_names_vectorized(self):
        """Test that _make_names is vectorized and works with arrays."""
        f1names = np.array(["A.51.LEU.HD1", "A.52.VAL.HG1"])
        f2names = np.array(["A.51.LEU.HG", "A.52.VAL.HG2"])
        peak_ids = np.array([1, 2])

        results = _make_names(f1names, f2names, peak_ids)

        assert len(results) == 2
        assert isinstance(results, np.ndarray)


class TestRegisterReader:
    """Tests for register_reader decorator."""

    def test_register_single_extension(self):
        """Test registering reader for single extension."""
        from peakfit.core.domain.peaks_io import READERS

        initial_count = len(READERS)

        @register_reader("test_ext")
        def test_reader(path, spectra, shape_names, args_cli):
            return []

        assert "test_ext" in READERS
        assert READERS["test_ext"] == test_reader
        assert len(READERS) == initial_count + 1

    def test_register_multiple_extensions(self):
        """Test registering reader for multiple extensions."""
        from peakfit.core.domain.peaks_io import READERS

        @register_reader(["test_a", "test_b"])
        def test_reader_multi(path, spectra, shape_names, args_cli):
            return []

        assert "test_a" in READERS
        assert "test_b" in READERS
        assert READERS["test_a"] == test_reader_multi
        assert READERS["test_b"] == test_reader_multi

    def test_standard_readers_registered(self):
        """Test that standard readers are registered."""
        from peakfit.core.domain.peaks_io import READERS

        # These should be registered by default
        assert "list" in READERS
        assert "csv" in READERS
        assert "json" in READERS
        assert "xlsx" in READERS
        assert "xls" in READERS


class TestReadersRegistry:
    """Tests for readers registry."""

    def test_readers_dict_populated(self):
        """Test that READERS dictionary is populated with default readers."""
        from peakfit.core.domain.peaks_io import READERS

        # Should have at least the standard readers
        assert len(READERS) >= 5

    def test_all_registered_readers_callable(self):
        """Test that all registered readers are callable."""
        from peakfit.core.domain.peaks_io import READERS

        for extension, reader in READERS.items():
            assert callable(reader), f"Reader for {extension} is not callable"
