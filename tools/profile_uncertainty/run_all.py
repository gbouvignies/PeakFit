"""Run every study and record its output in docs/validation/profile-uncertainty/.

    uv run python tools/profile_uncertainty/run_all.py [name ...]

Each study is deterministic (fixed seeds). The realization counts are those of
docs/profile-uncertainty.md.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORDS = HERE.parents[1] / "docs" / "validation" / "profile-uncertainty"

STUDIES = {
    "estimators": ["2000"],
    "ratios": ["1500"],
    "downstream_decay": ["1500"],
    "global_cpmg": ["1500"],
    "formula_check": [],
    "linear_prediction": [],
    "cest_conditioning": [],
    "cest_first_order": [],
    "cest_refit": [],
    "cest_replicates": [],
    "curvature_term": [],
}


def main() -> None:
    """Run the named studies (all by default), writing one record each."""
    names = sys.argv[1:] or list(STUDIES)
    RECORDS.mkdir(parents=True, exist_ok=True)
    for name in names:
        command = [sys.executable, str(HERE / f"{name}.py"), *STUDIES[name]]
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=HERE)
        (RECORDS / f"{name}.txt").write_text(result.stdout, encoding="utf-8")
        print(f"{name}: {RECORDS / f'{name}.txt'}")


if __name__ == "__main__":
    main()
