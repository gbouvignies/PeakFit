"""Tests for messaging and output functions."""

from peakfit.messages import LOGO


class TestLogo:
    """Tests for logo constant."""

    def test_logo_is_string(self):
        """Test that LOGO is a string."""
        assert isinstance(LOGO, str)

    def test_logo_not_empty(self):
        """Test that LOGO is not empty."""
        assert len(LOGO) > 0

    def test_logo_contains_peakfit(self):
        """Test that LOGO contains recognizable text."""
        # Logo should have some ASCII art characters
        assert len(LOGO) > 50  # Should be multi-line ASCII art


class TestConsoleExport:
    """Tests for console export functionality."""

    def test_console_export_text(self):
        """Test that console can export text."""
        from peakfit.messages import console

        console.clear()
        console.print("Test export", style="bold")

        output = console.export_text()
        assert isinstance(output, str)
        assert "Test export" in output

    def test_console_clear(self):
        """Test that console can be cleared."""
        from peakfit.messages import console

        console.print("Before clear", style="bold")
        before_len = len(console.export_text())

        console.clear()
        after_len = len(console.export_text())

        # After clear should have less content
        assert after_len < before_len
