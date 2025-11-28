#!/usr/bin/env bash
# PeakFit Development with uv

set -e

echo "🔧 PeakFit Development Setup using uv"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "[BAD] uv is not installed. Install it with:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "[GOOD] uv is installed"
echo ""

# Sync all dependencies
echo "📦 Installing dependencies..."
uv sync --all-extras
echo ""

echo "✨ Setup complete! Available commands:"
echo ""
echo "  uv run peakfit --help          # Run PeakFit"
echo "  uv run pytest                  # Run tests"
echo "  uv run pytest --cov=peakfit    # Run tests with coverage"
echo "  uv run ruff check peakfit/     # Lint code"
echo "  uv run mypy peakfit/           # Type check"
echo "  uv build                       # Build distribution"
echo ""
