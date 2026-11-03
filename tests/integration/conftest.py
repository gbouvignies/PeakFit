"""One CLI fit of the bundled example, shared by the integration tests."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_REFERENCE = REPO_ROOT / "tests/data/golden/example-fit.json"


@pytest.fixture(scope="session")
def example_fit_dir(tmp_path_factory):
    """Fit the example with the golden reference's recorded arguments; return the run."""
    arguments = json.loads(GOLDEN_REFERENCE.read_text())["fit_arguments"]
    output = tmp_path_factory.mktemp("example_fit")
    command = [sys.executable, "-m", "peakfit", "fit", *arguments]
    command += ["--output", str(output), "--headless"]
    result = subprocess.run(command, check=False, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            f"Example fit failed ({result.returncode}).\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    (run,) = (path for path in output.iterdir() if path.is_dir())
    return run
