"""Write ``nmrpipe_lp/``: the NMRPipe-processed data of the T7 study.

Every FID of ``lp_known_truth.py`` that LP touches goes through NMRPipe: 47 acquired
points, ``LP -fb -ord 16 -pred 45``, then the example's ¹⁵N chain (sine-bell² offset
0.4, zero fill to 256, FT, real part). The fixtures keep each case's 19-point window,
in NumPy's frequency order. They also record the noise-power signature of LP orders
12–20 (§2.2), from which the order is chosen, and the signal processed without LP,
which checks that the study's model reproduces NMRPipe's processing.

Only this script needs NMRPipe; the study reads the committed fixtures. Run:

    uv run python tools/profile_uncertainty/make_lp_fixtures.py --nmrpipe ~/path/to/NMRPipe

The fixtures were generated with NMRPipe 13.0 (2026-05-20).
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import nmrglue as ng
import numpy as np
from lp_known_truth import (
    ACQUIRED,
    CASES,
    COUNT,
    FIXTURES,
    FULL,
    LINESHAPES,
    LP_ORDER,
    SEED,
    power_profile,
    realization_noise,
    white_profile,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/
from make_nmrpipe_noise_fixtures import _environment, _run, add_nmrpipe_argument

ORDERS = (12, 14, 16, 18, 20)
SIGNATURE_DRAWS = 2000
LINEAR = ["SP -off 0.4 -end 0.98 -pow 2 -c 1.0", "ZF -size 256", "FT", "PS -p0 0 -p1 0 -di"]
# NMRPipe stores 256 points from +SW/2 down: its point j is NumPy's FFT index 128 − j.
NUMPY_ORDER = (128 - np.arange(256)) % 256


def nmrpipe(fids: np.ndarray, steps: list[str], env: dict[str, str]) -> np.ndarray:
    """Process each row of ``fids`` with NMRPipe; spectra (rows, 256) in NumPy order."""
    rows, n = fids.shape
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        raw = np.empty((rows, 2 * n), dtype="<f4")
        raw[:, 0::2], raw[:, 1::2] = fids.real, fids.imag
        (cwd / "lp.bin").write_bytes(raw.tobytes())
        axes = (
            f"-xN {2 * n} -xT {n} -xMODE Complex -xSW 2000 -xOBS 80 -xCAR 118 -xLAB N "
            f"-yN {rows} -yT {rows} -yMODE Real -ySW 1 -yOBS 1 -yCAR 0 -yLAB Z -ndim 2"
        )
        _run(f"bin2pipe -in lp.bin -noswap -ri {axes} -out lp.fid -ov", env, cwd)
        chain = " | ".join(f"nmrPipe -fn {step}" for step in steps)
        _run(f"nmrPipe -in lp.fid | {chain} -out lp.ft1 -ov", env, cwd)
        _, data = ng.pipe.read(str(cwd / "lp.ft1"))
    return np.asarray(data, dtype=float).reshape(rows, -1)[:, NUMPY_ORDER]


def lp_steps(order: int) -> list[str]:
    return [f"LP -fb -ord {order} -pred {FULL - ACQUIRED}", *LINEAR]


def _write(name: str, arrays: dict[str, Any]) -> None:
    """One compressed .npz; Any because savez's **kwds also cover allow_pickle."""
    path = FIXTURES / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    print(f"{path}: {path.stat().st_size / 1e3:.0f} kB")


def main() -> None:
    """Process every LP input with NMRPipe and write the fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_nmrpipe_argument(parser)
    env = _environment(parser.parse_args().nmrpipe.expanduser())
    FIXTURES.mkdir(exist_ok=True)
    rng = np.random.default_rng([SEED, 97])
    shape = (SIGNATURE_DRAWS, ACQUIRED)
    z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    white = white_profile()
    ratio = [power_profile(nmrpipe(z, lp_steps(order), env)) / white for order in ORDERS]
    _write(
        "signature",
        {
            "orders": np.array(ORDERS),
            "order": np.array(LP_ORDER),
            "count": np.array(COUNT),
            "power_ratio": np.array(ratio, dtype=np.float32),
        },
    )
    steps = lp_steps(LP_ORDER)
    for number, case in enumerate(CASES):
        idx = case.index
        noise = realization_noise(number)[:, :ACQUIRED]
        out: dict[str, Any] = {
            f"c{number}_lp_n": nmrpipe(noise, steps, env)[:, idx].astype(np.float32),
            f"c{number}_linear": nmrpipe(case.signal(0.0), LINEAR, env)[:, idx],
        }
        for j, gauss in enumerate(LINESHAPES.values()):
            signal = case.signal(gauss)[:, :ACQUIRED]
            data = np.tile(signal, (COUNT, 1)) + noise
            out[f"c{number}_g{j}_lp_s"] = nmrpipe(signal, steps, env)[:, idx]
            out[f"c{number}_g{j}_lp_sn"] = nmrpipe(data, steps, env)[:, idx].astype(np.float32)
        _write(f"case{number}", out)


if __name__ == "__main__":
    main()
