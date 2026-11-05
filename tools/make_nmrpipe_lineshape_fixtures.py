"""Generate the NMRPipe-processed lineshape fixtures in ``tests/data/nmrpipe/lineshapes``.

Each fixture is a noiseless spectrum NMRPipe processed from exact time-domain samples
of known decaying signals, written beside ``truth.json`` (positions, linewidths and
amplitudes). Together they cover sampling from t = 0 (first point ×½, P1 0) and with
a half-dwell delay (first point ×1, P1 180), sp1, sp2 and no apodization, direct and
indirect axes, and a Bruker direct axis whose digital-filter delay FT corrects (with
and without SP -df). The Bruker filter is emulated as an ideal band-limited delay.

Requires a local NMRPipe installation; the tests only read the committed files:

    uv run python tools/make_nmrpipe_lineshape_fixtures.py --nmrpipe ~/path/to/NMRPipe

The fixtures were generated with NMRPipe 13.0 (rev 2026.072).
"""

# ruff: noqa: T201
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from make_nmrpipe_noise_fixtures import _environment, _run

OUT = Path(__file__).resolve().parents[1] / "tests/data/nmrpipe/lineshapes"
GROUP_DELAY = 67.9842376708984  # a typical Bruker GRPDLY, points
OBS, CAR = 800.0, 4.7  # the axes' calibration, so ppm = CAR + Hz/OBS


def _samples(peaks, n, sw, delay, planes=2):
    """Exact complex samples at (j + delay)/sw; plane z scales every amplitude by ½^z."""
    t = (np.arange(n) + delay) / sw
    signal = sum(a * np.exp((2j * np.pi * nu - r2) * t) for nu, r2, a in peaks)
    return np.array([signal * 0.5**z for z in range(planes)])


def _filtered(peaks, n, sw, planes=2, oversample=32):
    """Samples through an ideal digital filter: band-limited to ±sw/2, delayed by g."""
    size = 1 << int(np.ceil(np.log2((n + int(GROUP_DELAY) + 64) * oversample * 2)))
    t = np.arange(size) / (sw * oversample)
    f = np.fft.fftfreq(size, 1 / (sw * oversample))
    signal = sum(a * np.exp((2j * np.pi * nu - r2) * t) for nu, r2, a in peaks)
    signal[0] *= 0.5  # the signal starts at t = 0
    spectrum = np.fft.fft(signal)
    spectrum[np.abs(f) > sw / 2] = 0.0
    delayed = np.fft.ifft(spectrum * np.exp(-2j * np.pi * f * GROUP_DELAY / sw))[::oversample][:n]
    return np.array([delayed * 0.5**z for z in range(planes)])


def _series(name, rows, sw, env, cwd, *, bruker=False):
    """A pseudo-2D FID: one plane per row, converted by bin2pipe or bruk2pipe."""
    planes, n = rows.shape
    axes = (
        f"-xN {2 * n} -xT {n} -xMODE Complex -xSW {sw} -xOBS {OBS} -xCAR {CAR} -xLAB H "
        f"-yN {planes} -yT {planes} -yMODE Real -ySW 1 -yOBS 1 -yCAR 0 -yLAB Z -ndim 2"
    )
    if bruker:
        raw = np.empty((planes, 2 * n), "<i4")
        scale = 2e8 / np.abs(rows).max()
        raw[:, 0::2], raw[:, 1::2] = np.round(scale * rows.real), np.round(scale * rows.imag)
        (cwd / f"{name}.ser").write_bytes(raw.tobytes())
        command = (
            f"bruk2pipe -in {name}.ser -AMX -decim 1600 -dspfvs 20 -grpdly {GROUP_DELAY} "
            f"-bad 0.0 -noswap {axes} -out {name}.fid -ov"
        )
    else:
        raw = np.empty((planes, 2 * n), "<f4")
        raw[:, 0::2], raw[:, 1::2] = rows.real, rows.imag
        (cwd / f"{name}.bin").write_bytes(raw.tobytes())
        command = f"bin2pipe -in {name}.bin -noswap -ri {axes} -out {name}.fid -ov"
    _run(command, env, cwd)


def _indirect(name, peaks, n1, delay, env, cwd, n2=64, sw1=2000.0, sw2=8000.0):
    """A 2D FID: States indirect axis sampled from (j + delay)/sw1, direct from t = 0."""
    t1 = (np.arange(n1) + delay) / sw1
    t2 = np.arange(n2) / sw2
    rows = np.zeros((n1, 2, n2), complex)
    for (nu1, nu2), (r1, r2), a in peaks:
        direct = a * np.exp((2j * np.pi * nu2 - r2) * t2)
        rows[:, 0] += (np.cos(2 * np.pi * nu1 * t1) * np.exp(-r1 * t1))[:, None] * direct
        rows[:, 1] += (np.sin(2 * np.pi * nu1 * t1) * np.exp(-r1 * t1))[:, None] * direct
    rows = rows.reshape(-1, n2)
    raw = np.empty((len(rows), 2 * n2), "<f4")
    raw[:, 0::2], raw[:, 1::2] = rows.real, rows.imag
    (cwd / f"{name}.bin").write_bytes(raw.tobytes())
    _run(
        f"bin2pipe -in {name}.bin -noswap -ri -ndim 2 -aq2D States "
        f"-xN {2 * n2} -xT {n2} -xMODE Complex -xSW {sw2} -xOBS {OBS} -xCAR {CAR} -xLAB H "
        f"-yN {2 * n1} -yT {n1} -yMODE Complex -ySW {sw1} -yOBS {OBS} -yCAR {CAR} -yLAB N "
        f"-out {name}.fid -ov",
        env,
        cwd,
    )


def _process(name, chain, env, cwd):
    steps = " | ".join(f"nmrPipe -fn {step}" for step in chain)
    _run(f"nmrPipe -in {name}.fid | {steps} -out {name}.ft -ov", env, cwd)
    return cwd / f"{name}.ft"


PEAKS_15N = [(-500.0, 5.0, 1.0), (100.0, 15.0, 0.7), (600.0, 40.0, 0.5)]  # Hz, R2 (s⁻¹), amplitude
PEAKS_1H = [(-2000.0, 20.0, 1.0), (500.0, 40.0, 0.7), (2500.0, 80.0, 0.5)]
PEAKS_2D = [((-300.0, -1500.0), (15.0, 40.0), 1.0), ((400.0, 1200.0), (30.0, 60.0), 0.6)]
SP2 = "SP -off 0.4 -end 0.98 -pow 2"
SP1 = "SP -off 0.4 -end 0.98 -pow 1"


def _fixtures(env, cwd):
    truth = {}

    def record(name, path, peaks, axes):
        shutil.copy(path, OUT / f"{name}.ft")
        truth[name] = {"axes": axes, "peaks": peaks}
        print(name)

    one_axis = lambda peaks: [  # noqa: E731
        {"ppm": [CAR + nu / OBS], "lw_hz": [r2 / np.pi], "amplitude": a} for nu, r2, a in peaks
    ]

    _series("sp2_t0", _samples(PEAKS_15N, 32, 2000.0, 0.0), 2000.0, env, cwd)
    path = _process(
        "sp2_t0", [f"{SP2} -c 0.5", "ZF -size 1024", "FT -auto", "PS -p0 0 -p1 0 -di"], env, cwd
    )
    record("sp2_t0", path, one_axis(PEAKS_15N), ["H"])

    _series("sp1_half_dwell", _samples(PEAKS_15N, 32, 2000.0, 0.5), 2000.0, env, cwd)
    chain = [f"{SP1} -c 1.0", "ZF -size 1024", "FT -auto", "PS -p0 -90 -p1 180 -di"]
    record(
        "sp1_half_dwell", _process("sp1_half_dwell", chain, env, cwd), one_axis(PEAKS_15N), ["H"]
    )

    rows = _filtered(PEAKS_1H, 512, 8000.0)
    for name, window in (
        ("sp2_digital_filter", f"{SP2} -c 0.5"),
        ("sp2_digital_filter_df", f"{SP2} -c 0.5 -df"),
        ("no_apod_digital_filter", "APOD -qName NONE -c 0.5"),
    ):
        _series(name, rows, 8000.0, env, cwd, bruker=True)
        path = _process(name, [window, "ZF -size 2048", "FT -auto", "PS -p0 0 -p1 0 -di"], env, cwd)
        record(name, path, one_axis(PEAKS_1H), ["H"])

    for name, delay, c, phase in (
        ("sp2_indirect_t0", 0.0, 0.5, "PS -p0 0 -p1 0 -di"),
        ("sp2_indirect_half_dwell", 0.5, 1.0, "PS -p0 -90 -p1 180 -di"),
    ):
        _indirect(name, PEAKS_2D, 32, delay, env, cwd)
        chain = [
            f"{SP2} -c 0.5", "ZF -size 128", "FT -auto", "PS -p0 0 -p1 0 -di", "TP",
            f"{SP2} -c {c}", "ZF -size 128", "FT -auto", phase, "TP",
        ]  # fmt: skip
        path = _process(name, chain, env, cwd)
        peaks = [
            {
                "ppm": [CAR + n1 / OBS, CAR + n2 / OBS],
                "lw_hz": [r1 / np.pi, r2 / np.pi],
                "amplitude": a,
            }
            for (n1, n2), (r1, r2), a in PEAKS_2D
        ]
        record(name, path, peaks, ["N", "H"])
    (OUT / "truth.json").write_text(json.dumps(truth, indent=1) + "\n")


def main() -> None:
    """Process every fixture with the given NMRPipe and copy it into the tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmrpipe", type=Path, required=True, help="NMRPipe installation")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        _fixtures(_environment(args.nmrpipe.expanduser()), Path(tmp))


if __name__ == "__main__":
    main()
