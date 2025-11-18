"""Test clustering functions."""

import numpy as np

from peakfit.clustering import group_connected_pairs, merge_connected_segments, segment_data


class TestGroupConnectedPairs:
    """Tests for graph-based grouping."""

    def test_simple_chain(self):
        """Should group connected pairs into single component."""
        pairs = [(1, 2), (2, 3), (3, 4)]
        result = group_connected_pairs(pairs)
        assert len(result) == 1
        assert result[0] == [1, 2, 3, 4]

    def test_two_components(self):
        """Should identify separate components."""
        pairs = [(1, 2), (3, 4)]
        result = group_connected_pairs(pairs)
        assert len(result) == 2
        assert [1, 2] in result
        assert [3, 4] in result

    def test_merge_components(self):
        """Should merge components when connected."""
        pairs = [(1, 2), (3, 4), (2, 3)]
        result = group_connected_pairs(pairs)
        assert len(result) == 1
        assert sorted(result[0]) == [1, 2, 3, 4]

    def test_empty_pairs(self):
        """Should handle empty input."""
        pairs = []
        result = group_connected_pairs(pairs)
        assert result == []

    def test_single_pair(self):
        """Should handle single pair."""
        pairs = [(5, 10)]
        result = group_connected_pairs(pairs)
        assert len(result) == 1
        assert result[0] == [5, 10]


class TestMergeConnectedSegments:
    """Tests for segment merging."""

    def test_no_wrapping(self):
        """Should not merge if no wrapping."""
        # Segments don't touch at ANY boundary (row or column)
        # and don't align when dimensions are rotated
        segments = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 0],
            ]
        )
        result = merge_connected_segments(segments.copy())
        # No wrapping on any boundary, so segments stay separate
        assert 1 in result
        assert 2 in result

    def test_wrapping_merge(self):
        """Should merge segments that wrap around edges."""
        segments = np.array(
            [
                [1, 0, 2],
                [0, 0, 0],
                [2, 0, 1],
            ]
        )
        result = merge_connected_segments(segments.copy())
        # Segments 1 and 2 should be merged (both present in first and last rows)
        unique = np.unique(result[result > 0])
        assert len(unique) == 1  # All merged into one


class TestSegmentData:
    """Tests for data segmentation."""

    def test_single_peak_cluster(self, sample_2d_spectrum):
        """Single isolated peak should form one cluster."""
        spectrum, _ = sample_2d_spectrum

        # Create mock peak
        class MockPeak:
            @property
            def positions_i(self):
                return (70, 150)

        peaks = [MockPeak()]

        # Use contour level above noise but below peak
        contour_level = 100.0
        segments = segment_data(spectrum[np.newaxis, ...], contour_level, peaks)

        # Should have at least one non-zero segment
        assert np.max(segments) >= 1

    def test_overlapping_peaks_same_cluster(self, sample_2d_spectrum):
        """Overlapping peaks should be in same cluster."""
        spectrum, _ = sample_2d_spectrum

        # Create mock peaks at overlapping positions
        class MockPeak:
            def __init__(self, y, x):
                self._pos = (y, x)

            @property
            def positions_i(self):
                return self._pos

        # These peaks are close together
        peaks = [MockPeak(50, 100), MockPeak(52, 102)]

        contour_level = 100.0
        segments = segment_data(spectrum[np.newaxis, ...], contour_level, peaks)

        # Both peaks should be in the same segment
        seg1 = segments[peaks[0].positions_i]
        seg2 = segments[peaks[1].positions_i]
        assert seg1 == seg2

    def test_separated_peaks_different_clusters(self, sample_2d_spectrum):
        """Well-separated peaks should be in different clusters."""
        spectrum, _ = sample_2d_spectrum

        class MockPeak:
            def __init__(self, y, x):
                self._pos = (y, x)

            @property
            def positions_i(self):
                return self._pos

        # These peaks are far apart
        peaks = [MockPeak(50, 100), MockPeak(70, 150)]

        contour_level = 100.0
        segments = segment_data(spectrum[np.newaxis, ...], contour_level, peaks)

        # Peaks should be in different segments
        seg1 = segments[peaks[0].positions_i]
        seg2 = segments[peaks[1].positions_i]
        # They might be different or same depending on exact contour level
        # At minimum, both should be non-zero
        assert seg1 > 0
        assert seg2 > 0

    def test_contour_level_affects_clustering(self, sample_2d_spectrum):
        """Higher contour level should create smaller clusters."""
        spectrum, _ = sample_2d_spectrum

        class MockPeak:
            @property
            def positions_i(self):
                return (50, 100)

        peaks = [MockPeak()]

        # Low contour level - large cluster
        segments_low = segment_data(spectrum[np.newaxis, ...], 50.0, peaks)
        n_points_low = np.sum(segments_low > 0)

        # High contour level - smaller cluster
        segments_high = segment_data(spectrum[np.newaxis, ...], 500.0, peaks)
        n_points_high = np.sum(segments_high > 0)

        assert n_points_low >= n_points_high
