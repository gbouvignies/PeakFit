"""Generate the NMRPipe-processed noise-correlation fixtures in ``tests/data/nmrpipe``.

Each fixture is a pseudo-2D NMRPipe file whose rows are one processed time-domain
impulse each: the real unit impulse at every acquired point, then the imaginary one.
Processing is linear, so Σ_rows x xᵀ is the exact covariance of the processed real
noise for complex white noise in the FID, and its normalized form is the correlation
``read_noise_model`` must reproduce. The headers are NMRPipe's own.

Linear prediction is not linear in the data, so its fixtures (``NOISE_FIXTURES``) are
instead rows of seeded complex white noise, each processed as one FID: the noise-power
check (docs/correlated-noise.md §2.6) must detect it, and must not flag the same noise
processed without it.

Requires a local NMRPipe installation (bin2pipe, bruk2pipe, nmrPipe); the tests only
read the committed files. Run:

    uv run python tools/make_nmrpipe_noise_fixtures.py --nmrpipe ~/path/to/NMRPipe

The fixtures were generated with NMRPipe 13.0 (rev 2026.072).
"""

# ruff: noqa: T201
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "tests/data/nmrpipe"
# Bruker DMX parameters of a typical Avance acquisition (acqus DECIM, DSPFVS, GRPDLY).
BRUKER = "-AMX -decim 1600 -dspfvs 20 -grpdly 67.9842376708984"
SP_N = "SP -off 0.4 -end 0.98 -pow 2 -c 0.5"
SP_H = "SP -off 0.3 -end 0.98 -pow 2 -c 0.5"

# name: (acquired points, converter, processing chain)
FIXTURES: dict[str, tuple[int, str, list[str]]] = {
    # An indirect-like axis with the half-dwell first-order phase.
    "sine_bell_p1_180": (32, "bin", [SP_N, "ZF -size 128", "FT -auto", "PS -p0 -90 -p1 180 -di"]),
    # A Bruker direct axis: FT applies the digital-filter correction (FDDMXFLAG 1).
    "digital_filter": (
        96,
        "bruker",
        [SP_H, "ZF -size 256", "FT -auto", "PS -p0 0 -p1 0 -di", "EXT -x1 96 -xn 160 -sw"],
    ),
    # The same, with the correction switched off (FDDMXFLAG stays 0).
    "digital_filter_off": (
        96,
        "bruker",
        [SP_H, "ZF -size 256", "FT -auto -nodmx", "PS -p0 0 -p1 0 -di", "EXT -x1 96 -xn 160 -sw"],
    ),
    # Window terms NMRPipe records and the noise model does not support.
    "sine_bell_elb": (8, "bin", [SP_N + " -elb 10", "ZF -size 32", "FT -auto", "PS -di"]),
    "sine_bell_glb": (8, "bin", [SP_N + " -glb 10 -goff 0.2", "ZF -size 32", "FT -auto", "PS -di"]),
    "sine_bell_df": (8, "bruker", [SP_H + " -df", "ZF -size 32", "FT -auto", "PS -di"]),
    # Processing the final header cannot show (docs/correlated-noise.md §2.5): it
    # records the last sine bell over every acquired point, as if it were the only one.
    "earlier_window": (
        32,
        "bin",
        ["EM -lb 100", SP_N, "ZF -size 128", "FT -auto", "PS -p0 -90 -p1 180 -di"],
    ),
    "sp_size": (
        32,
        "bin",
        [SP_N + " -size 16", "ZF -size 128", "FT -auto", "PS -p0 -90 -p1 180 -di"],
    ),
}

# name: (acquired points, processing chain); rows of complex white noise.
NOISE_ROWS = 512
NOISE_FIXTURES: dict[str, tuple[int, list[str]]] = {
    # 32 acquired points extended to 64 by linear prediction, as on the CEST ¹⁵N axis.
    "linear_prediction": (
        32,
        ["LP -fb -ord 16 -pred 32", SP_N, "ZF -size 128", "FT -auto", "PS -p0 -90 -p1 180 -di"],
    ),
    # The control: 64 acquired points, the same processing without prediction.
    "noise_without_lp": (64, [SP_N, "ZF -size 128", "FT -auto", "PS -p0 -90 -p1 180 -di"]),
}


def _environment(base: Path) -> dict[str, str]:
    arch = "mac_arm64" if platform.system() == "Darwin" else "linux239_64"
    binaries = base / f"nmrbin.{arch}"
    return {
        **os.environ,
        "NMRBASE": str(base),
        "NMRTXT": str(base / "nmrtxt"),
        "PATH": f"{binaries}:{base / 'com'}:{os.environ['PATH']}",
        "LD_LIBRARY_PATH": str(binaries / "lib"),
        "DYLD_LIBRARY_PATH": str(binaries / "lib"),
        "NMR_IO_TIMEOUT": "0",
        "NMR_IO_SELECT": "0",
        "NMR_AUTOSWAP": "1",
        "NMR_FSCHECK": "NO",
        "LC_ALL": "C",
    }


def _run(command: str, env: dict[str, str], cwd: Path) -> None:
    run = subprocess.run(
        command, check=False, shell=True, env=env, cwd=cwd, capture_output=True, text=True
    )
    if run.returncode != 0 or "rror" in run.stderr:
        raise RuntimeError(f"{command}\n{run.stderr[-2000:]}")


def _impulses(n: int) -> np.ndarray:
    """2n complex rows: the real, then the imaginary unit impulse at each point."""
    eye = np.eye(n)
    return np.vstack([eye, 1j * eye])


def _noise(n: int, seed: int) -> np.ndarray:
    """NOISE_ROWS rows of unit complex white noise, from a fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((NOISE_ROWS, n)) + 1j * rng.standard_normal((NOISE_ROWS, n))


def _fid(name: str, rows: np.ndarray, converter: str, env: dict[str, str], cwd: Path) -> str:
    n = rows.shape[1]
    axes = (
        f"-xN {2 * n} -xT {n} -xSW 9615.38 -xOBS 800.0 -xCAR 4.7 -xLAB HN "
        f"-yN {len(rows)} -yT {len(rows)} -yMODE Real -ySW 1 -yOBS 1 -yCAR 0 -yLAB Z -ndim 2"
    )
    if converter == "bin":
        raw = np.empty((len(rows), 2 * n), dtype="<f4")
        raw[:, 0::2], raw[:, 1::2] = rows.real, rows.imag
        (cwd / f"{name}.bin").write_bytes(raw.tobytes())
        _run(
            f"bin2pipe -in {name}.bin -noswap -ri -xMODE Complex {axes} -out {name}.fid -ov",
            env,
            cwd,
        )
    else:
        raw = np.empty((len(rows), 2 * n), dtype="<i4")
        raw[:, 0::2], raw[:, 1::2] = 10000 * rows.real, 10000 * rows.imag
        (cwd / f"{name}.ser").write_bytes(raw.tobytes())
        _run(
            f"bruk2pipe -in {name}.ser {BRUKER} -bad 0.0 -noswap -xMODE DQD {axes} "
            f"-out {name}.fid -ov",
            env,
            cwd,
        )
    return f"{name}.fid"


def main() -> None:
    """Process every fixture with the given NMRPipe and copy it into the tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmrpipe", type=Path, required=True, help="NMRPipe installation")
    args = parser.parse_args()
    env = _environment(args.nmrpipe.expanduser())
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        jobs = [
            (name, _impulses(n), converter, chain)
            for name, (n, converter, chain) in FIXTURES.items()
        ]
        jobs += [
            (name, _noise(n, seed), "bin", chain)
            for seed, (name, (n, chain)) in enumerate(NOISE_FIXTURES.items())
        ]
        for name, rows, converter, chain in jobs:
            fid = _fid(name, rows, converter, env, cwd)
            steps = " | ".join(f"nmrPipe -fn {step}" for step in chain)
            _run(f"nmrPipe -in {fid} | {steps} -out {name}.ft1 -ov", env, cwd)
            shutil.copy(cwd / f"{name}.ft1", OUT / f"{name}.ft1")
            print(f"{name}: {' | '.join(chain)}")


if __name__ == "__main__":
    main()
