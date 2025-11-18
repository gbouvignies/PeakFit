"""Tests for messaging and output functions."""

from peakfit.messages import LOGO, print_logo, print_message


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


class TestPrintLogo:
    """Tests for print_logo function."""

    def test_print_logo_executes(self):
        """Test that print_logo executes without error."""
        # Should not raise any exception
        print_logo()

    def test_print_logo_outputs_to_console(self):
        """Test that print_logo produces console output."""
        from peakfit.messages import console

        # Clear console
        console.clear()
        initial_len = len(console.export_text())

        print_logo()

        # Should have produced output
        output = console.export_text()
        assert len(output) > initial_len


class TestPrintMessage:
    """Tests for print_message function."""

    def test_print_message_executes(self):
        """Test that print_message executes without error."""
        print_message("Test message", "bold")

    def test_print_message_with_different_styles(self):
        """Test print_message with various styles."""
        styles = ["bold", "italic", "bold red", "green", "yellow"]

        for style in styles:
            print_message(f"Test with {style}", style)

    def test_print_message_empty_string(self):
        """Test print_message with empty string."""
        print_message("", "bold")


class TestFittingMessages:
    """Tests for fitting-related message functions."""

    def test_print_fitting_executes(self):
        """Test that print_fitting executes without error."""
        from peakfit.messages import print_fitting

        print_fitting()

    def test_print_segmenting_executes(self):
        """Test that print_segmenting executes without error."""
        from peakfit.messages import print_segmenting

        print_segmenting()


class TestConsoleExport:
    """Tests for console export functionality."""

    def test_console_export_text(self):
        """Test that console can export text."""
        from peakfit.messages import console

        console.clear()
        print_message("Test export", "bold")

        output = console.export_text()
        assert isinstance(output, str)
        assert "Test export" in output

    def test_console_clear(self):
        """Test that console can be cleared."""
        from peakfit.messages import console

        print_message("Before clear", "bold")
        before_len = len(console.export_text())

        console.clear()
        after_len = len(console.export_text())

        # After clear should have less content
        assert after_len < before_len
