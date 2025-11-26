"""Tests for reporter abstraction."""

import logging
from io import StringIO

import pytest

from peakfit.core.shared.reporter import CompositeReporter, LoggingReporter, NullReporter, Reporter


class MockReporter:
    """Test double for capturing reporter calls."""

    def __init__(self) -> None:
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


class TestReporterProtocol:
    """Tests for Reporter protocol compliance."""

    def test_null_reporter_satisfies_protocol(self) -> None:
        """NullReporter should satisfy the Reporter protocol."""
        reporter = NullReporter()
        # runtime_checkable protocols can be checked with isinstance
        assert isinstance(reporter, Reporter)

    def test_logging_reporter_satisfies_protocol(self) -> None:
        """LoggingReporter should satisfy the Reporter protocol."""
        reporter = LoggingReporter()
        assert isinstance(reporter, Reporter)

    def test_mock_reporter_satisfies_protocol(self) -> None:
        """MockReporter should satisfy the Reporter protocol."""
        reporter = MockReporter()
        assert isinstance(reporter, Reporter)

    def test_composite_reporter_satisfies_protocol(self) -> None:
        """CompositeReporter should satisfy the Reporter protocol."""
        reporter = CompositeReporter([NullReporter()])
        assert isinstance(reporter, Reporter)


class TestNullReporter:
    """Tests for NullReporter."""

    def test_null_reporter_is_silent(self) -> None:
        """NullReporter should not raise or produce output."""
        reporter = NullReporter()
        # These should all complete without exception
        reporter.action("test action")
        reporter.info("test info")
        reporter.warning("test warning")
        reporter.error("test error")
        reporter.success("test success")
        # No assertions needed - just verify no exceptions

    def test_null_reporter_methods_return_none(self) -> None:
        """NullReporter methods should return None."""
        reporter = NullReporter()
        assert reporter.action("test") is None
        assert reporter.info("test") is None
        assert reporter.warning("test") is None
        assert reporter.error("test") is None
        assert reporter.success("test") is None


class TestLoggingReporter:
    """Tests for LoggingReporter."""

    @pytest.fixture
    def log_capture(self) -> tuple[logging.Logger, StringIO]:
        """Set up log capture."""
        logger = logging.getLogger("test_peakfit")
        logger.setLevel(logging.DEBUG)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
        logger.addHandler(handler)
        return logger, stream

    def test_logging_reporter_action(self, log_capture: tuple[logging.Logger, StringIO]) -> None:
        """LoggingReporter should log actions at INFO level."""
        _logger, stream = log_capture
        reporter = LoggingReporter("test_peakfit")
        reporter.action("doing something")
        assert "[ACTION] doing something" in stream.getvalue()
        assert "INFO" in stream.getvalue()

    def test_logging_reporter_info(self, log_capture: tuple[logging.Logger, StringIO]) -> None:
        """LoggingReporter should log info at INFO level."""
        _logger, stream = log_capture
        reporter = LoggingReporter("test_peakfit")
        reporter.info("some info")
        assert "some info" in stream.getvalue()
        assert "INFO" in stream.getvalue()

    def test_logging_reporter_warning(self, log_capture: tuple[logging.Logger, StringIO]) -> None:
        """LoggingReporter should log warnings at WARNING level."""
        _logger, stream = log_capture
        reporter = LoggingReporter("test_peakfit")
        reporter.warning("a warning")
        assert "a warning" in stream.getvalue()
        assert "WARNING" in stream.getvalue()

    def test_logging_reporter_error(self, log_capture: tuple[logging.Logger, StringIO]) -> None:
        """LoggingReporter should log errors at ERROR level."""
        _logger, stream = log_capture
        reporter = LoggingReporter("test_peakfit")
        reporter.error("an error")
        assert "an error" in stream.getvalue()
        assert "ERROR" in stream.getvalue()

    def test_logging_reporter_success(self, log_capture: tuple[logging.Logger, StringIO]) -> None:
        """LoggingReporter should log success at INFO level."""
        _logger, stream = log_capture
        reporter = LoggingReporter("test_peakfit")
        reporter.success("completed")
        assert "[SUCCESS] completed" in stream.getvalue()
        assert "INFO" in stream.getvalue()


class TestMockReporter:
    """Tests for MockReporter (test double)."""

    def test_mock_reporter_captures_messages(self) -> None:
        """MockReporter should capture all messages for testing."""
        reporter = MockReporter()
        reporter.action("doing something")
        reporter.info("info message")

        assert len(reporter.messages) == 2
        assert reporter.messages[0] == ("action", "doing something")
        assert reporter.messages[1] == ("info", "info message")

    def test_mock_reporter_captures_all_types(self) -> None:
        """MockReporter should capture all message types."""
        reporter = MockReporter()
        reporter.action("action")
        reporter.info("info")
        reporter.warning("warning")
        reporter.error("error")
        reporter.success("success")

        assert len(reporter.messages) == 5
        types = [msg[0] for msg in reporter.messages]
        assert types == ["action", "info", "warning", "error", "success"]


class TestCompositeReporter:
    """Tests for CompositeReporter."""

    def test_composite_delegates_to_all(self) -> None:
        """CompositeReporter should delegate to all reporters."""
        mock1 = MockReporter()
        mock2 = MockReporter()
        composite = CompositeReporter([mock1, mock2])

        composite.action("test")
        composite.info("info")

        assert len(mock1.messages) == 2
        assert len(mock2.messages) == 2
        assert mock1.messages == mock2.messages

    def test_composite_with_single_reporter(self) -> None:
        """CompositeReporter should work with single reporter."""
        mock = MockReporter()
        composite = CompositeReporter([mock])

        composite.success("done")

        assert len(mock.messages) == 1
        assert mock.messages[0] == ("success", "done")

    def test_composite_with_empty_list(self) -> None:
        """CompositeReporter should work with empty reporter list."""
        composite = CompositeReporter([])

        # Should not raise
        composite.action("test")
        composite.info("test")
        composite.warning("test")
        composite.error("test")
        composite.success("test")

    def test_composite_with_mixed_reporters(self) -> None:
        """CompositeReporter should work with mixed reporter types."""
        mock = MockReporter()
        null = NullReporter()
        composite = CompositeReporter([mock, null])

        composite.warning("warning message")

        # MockReporter should capture, NullReporter should not fail
        assert len(mock.messages) == 1
        assert mock.messages[0] == ("warning", "warning message")
