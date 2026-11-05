"""Validate the sp1/sp2/no_apod acquisition time against NMRPipe-processed signals.

Exact noiseless signals of known linewidth, position and amplitude are converted and
processed by a local NMRPipe, then fitted by PeakFit under two rules:
- ``recorded``: the earlier aq = APOD/SW with the recorded window (Q1, Q2);
- ``signal``: PeakFit's window over the acquired signal (io/readers/spectrum.py).

Cases: sampling from t = 0 (first point ×½), half-dwell (×1, P1 180) and a 0.1-dwell
delay (×0.6, P1 36) for 32, 64 and 256 points; sp1, sp2 and no apodization; a Bruker
direct axis whose digital-filter delay FT corrects (ideal filter emulation), with and
without SP -df; and a ¹⁵N States indirect axis (transposed processing). With ``--gb3``,
also the NMRPipe GB3 HSQC demo truncated at conversion, a real-data consistency check.

    uv run python tools/validate_acquisition_time.py --nmrpipe ~/path/to/NMRPipe [--gb3]

Prints one line per case and rule: linewidth error (%), position error (Hz), and
amplitude error relative to the first peak (%).
"""

# ruff: noqa: T201
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from make_nmrpipe_lineshape_fixtures import CAR, OBS, _filtered, _indirect, _samples, _series
from make_nmrpipe_noise_fixtures import _environment, _run

import peakfit.io.readers.spectrum as reader
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.fit.fitting import load_data
from peakfit.fit.pipeline import run_pipeline

FULL_1H, FULL_15N = 1676, 170  # GB3's acquired complex points
ISOLATION = (12, 25)  # no other maximum within this many 15N, 1H points


def _recorded(dic, fdf, _size, sw):
    """The earlier rule: aq = APOD/SW under the recorded window."""
    apod = float(dic.get(f"{fdf}APOD", 0.0)) / max(sw, 1e-6)
    return apod, float(dic.get(f"{fdf}APODQ1", 0.0)), float(dic.get(f"{fdf}APODQ2", 0.0))


RULES = {"recorded": _recorded, "signal": reader._signal_window}


def _fit(path: Path, peaks: list[dict], rule: str):
    listing = path.with_suffix(".list")
    listing.write_text(
        "".join(
            f"P{k} " + " ".join(f"{x:.6f}" for x in p["ppm"]) + "\n" for k, p in enumerate(peaks)
        )
    )
    config = PeakFitConfig()
    with mock.patch.object(reader, "_signal_window", RULES[rule]):
        data = load_data(path, listing, None, config)
    params = Parameters.from_peaks(data.peaks, fixed=False)
    completed = run_pipeline(
        config, data.clusters, data.noise, params, noise_model=data.noise_model
    ).completed_fit
    values = {p.name: p.value for p in completed.outcome.final_nonlinear_parameters}
    amplitudes = {
        n: float(np.asarray(c.analytical_evaluation.amplitudes[k])[0])
        for c in completed.outcome.clusters
        if c.analytical_evaluation is not None
        for k, n in enumerate(c.peak_names)
    }
    dims = [f"F{i + 2}" for i in range(len(peaks[0]["ppm"]))]
    lw = [
        [100 * (values[f"P{k}.{d}.lw"] / p["lw_hz"][i] - 1) for i, d in enumerate(dims)]
        for k, p in enumerate(peaks)
    ]
    pos = [
        [(values[f"P{k}.{d}.cs"] - p["ppm"][i]) * OBS for i, d in enumerate(dims)]
        for k, p in enumerate(peaks)
    ]
    rel = np.array([amplitudes[f"P{k}"] / p["amplitude"] for k, p in enumerate(peaks)])
    return (
        np.array(lw),
        np.array(pos),
        100 * (rel / rel[0] - 1),
        data.peaks[0].shapes[0].__class__.__name__,
    )


def _report(label, path, peaks):
    for rule in RULES:
        lw, pos, amp, shape = _fit(path, peaks, rule)
        print(
            f"{label:46} {shape:7} {rule:8} lw% {np.round(lw.ravel(), 1).tolist()} "
            f"pos Hz {np.round(pos.ravel(), 3).tolist()} amp% {np.round(amp, 1).tolist()}",
            flush=True,
        )


def _one_axis(peaks):
    return [{"ppm": [CAR + nu / OBS], "lw_hz": [r2 / np.pi], "amplitude": a} for nu, r2, a in peaks]


def conventional(env, cwd):
    """Sampling from t = 0, half-dwell and a 0.1-dwell delay; sp1, sp2, no window."""
    peaks = [(-500.0, 5.0, 1.0), (100.0, 15.0, 0.7), (600.0, 40.0, 0.5)]
    windows = {
        "sp2": "SP -off 0.4 -end 0.98 -pow 2",
        "sp1": "SP -off 0.4 -end 0.98 -pow 1",
        "none": "APOD -qName NONE",
    }
    for label, delay, scale, phase in (
        ("t0 = 0 (c 0.5, P1 0)", 0.0, 0.5, "PS -p0 0 -p1 0 -di"),
        ("half dwell (c 1, P1 180)", 0.5, 1.0, "PS -p0 -90 -p1 180 -di"),
        ("0.1-dwell delay (c 0.6, P1 36)", 0.1, 0.6, "PS -p0 -18 -p1 36 -di"),
    ):
        for n in (32, 64, 256):
            for name, window in windows.items():
                _series("c", _samples(peaks, n, 2000.0, delay), 2000.0, env, cwd)
                chain = " | ".join(
                    f"nmrPipe -fn {s}"
                    for s in (f"{window} -c {scale}", "ZF -size 2048", "FT -auto", phase)
                )
                _run(f"nmrPipe -in c.fid | {chain} -out c_{name}.ft -ov", env, cwd)
                _report(f"{label}, N {n}", cwd / f"c_{name}.ft", _one_axis(peaks))


def digital_filter(env, cwd):
    """A Bruker direct axis whose digital-filter delay FT corrects."""
    peaks = [(-2000.0, 20.0, 1.0), (500.0, 40.0, 0.7), (2500.0, 80.0, 0.5)]
    for n in (256, 512, 1024, 2048):
        for name, window in (
            ("sp2", "SP -off 0.4 -end 0.98 -pow 2 -c 0.5"),
            ("sp2 -df", "SP -off 0.4 -end 0.98 -pow 2 -c 0.5 -df"),
            ("none", "APOD -qName NONE -c 0.5"),
        ):
            _series("b", _filtered(peaks, n, 8000.0), 8000.0, env, cwd, bruker=True)
            chain = " | ".join(
                f"nmrPipe -fn {s}"
                for s in (window, "ZF -size 4096", "FT -auto", "PS -p0 0 -p1 0 -di")
            )
            _run(f"nmrPipe -in b.fid | {chain} -out b.ft -ov", env, cwd)
            _report(f"Bruker digital filter, {name}, N {n}", cwd / "b.ft", _one_axis(peaks))


def indirect(env, cwd):
    """A 15N States indirect axis, processed after a transpose."""
    pairs = [((-300.0, -1500.0), (15.0, 40.0), 1.0), ((400.0, 1200.0), (30.0, 60.0), 0.6)]
    peaks = [
        {"ppm": [CAR + a / OBS, CAR + b / OBS], "lw_hz": [r1 / np.pi, r2 / np.pi], "amplitude": amp}
        for (a, b), (r1, r2), amp in pairs
    ]
    for label, delay, scale, phase in (
        ("t0 = 0", 0.0, 0.5, "PS -p0 0 -p1 0 -di"),
        ("half dwell", 0.5, 1.0, "PS -p0 -90 -p1 180 -di"),
    ):
        for n1 in (32, 64):
            _indirect("i", pairs, n1, delay, env, cwd)
            chain = " | ".join(
                f"nmrPipe -fn {s}"
                for s in (
                    "SP -off 0.4 -end 0.98 -pow 2 -c 0.5",
                    "ZF -size 128",
                    "FT -auto",
                    "PS -p0 0 -p1 0 -di",
                    "TP",
                    f"SP -off 0.4 -end 0.98 -pow 2 -c {scale}",
                    "ZF -size 128",
                    "FT -auto",
                    phase,
                    "TP",
                )
            )
            _run(f"nmrPipe -in i.fid | {chain} -out i.ft -ov", env, cwd)
            _report(f"indirect 15N {label}, N1 {n1} (15N, 1H per peak)", cwd / "i.ft", peaks)


def gb3(env, cwd, nmrpipe: Path):
    """Real data: fitted linewidths should not depend on the acquisition length."""
    ser = nmrpipe / "demo/gb3/ser"
    full = (FULL_1H, FULL_15N)
    variants = [full, (512, FULL_15N), (256, FULL_15N), (FULL_1H, 64), (FULL_1H, 32)]
    paths = {}
    for xt, yt in variants:
        _run(
            f"bruk2pipe -in {ser} -bad 0.0 -ext -noaswap -AMX -decim 16 -dspfvs 12 -grpdly -1 "
            f"-xN 3584 -yN {2 * yt} -xT {xt} -yT {yt} -xMODE DQD -yMODE Complex "
            "-xSW 8389.262 -ySW 2128.625 -xOBS 600.133 -yOBS 60.818 -xCAR 4.773 -yCAR 118.562 "
            "-xLAB HN -yLAB 15N -ndim 2 -aq2D Complex | nmrPipe -fn MULT -c 1.5625e+01 "
            "| nmrPipe -fn SP -off 0.4 -end 0.98 -pow 2 -c 0.5 | nmrPipe -fn ZF -size 4096 "
            "| nmrPipe -fn FT | nmrPipe -fn PS -p0 -48 -p1 0 -di "
            "| nmrPipe -fn EXT -x1 10.8ppm -xn 5.8ppm -sw | nmrPipe -fn TP "
            "| nmrPipe -fn SP -off 0.4 -end 0.98 -pow 2 -c 0.5 | nmrPipe -fn ZF -size 1024 "
            "| nmrPipe -fn FT -alt | nmrPipe -fn PS -p0 0 -p1 0 -di "
            f"| nmrPipe -fn TP -out g{xt}_{yt}.ft -ov",
            env,
            cwd,
        )
        paths[xt, yt] = cwd / f"g{xt}_{yt}.ft"
    peaks = _gb3_peaks(paths[full])
    for rule in RULES:
        widths = {v: _gb3_widths(paths[v], peaks, rule) for v in variants}
        for v in variants[1:]:
            axis = 1 if v[0] != FULL_1H else 0  # the truncated axis: 1H or 15N
            ratio = widths[v][:, axis] / widths[full][:, axis]
            q = np.nanpercentile(ratio, [25, 50, 75])
            print(
                f"GB3 {rule:8} 1H {v[0]:4d} / 15N {v[1]:3d}: {'1H' if axis else '15N'} lw / "
                f"full-length lw median {q[1]:.2f} (IQR {q[0]:.2f}-{q[2]:.2f}), "
                f"{len(ratio)} isolated peaks",
                flush=True,
            )


def _gb3_peaks(path, count=25):
    from peakfit.engine.algorithms.noise import prepare_noise_level  # noqa: PLC0415

    spectra = reader.read_spectra(path)
    data = np.asarray(spectra.data[0], float)
    noise = prepare_noise_level(None, spectra)
    padded = np.pad(data, 1, constant_values=-np.inf)
    core = padded[1:-1, 1:-1]
    is_max = np.ones_like(core, bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                is_max &= (
                    core
                    > padded[1 + di : padded.shape[0] - 1 + di, 1 + dj : padded.shape[1] - 1 + dj]
                )
    found = np.argwhere(is_max & (core > 30 * noise))
    order = np.argsort(-core[tuple(found.T)])
    picked = []
    for i, j in found[order]:
        near = found[
            (np.abs(found[:, 0] - i) <= ISOLATION[0]) & (np.abs(found[:, 1] - j) <= ISOLATION[1])
        ]
        if len(near) == 1:
            picked.append(
                (
                    float(spectra.spectral_params[0].pts2ppm(i)),
                    float(spectra.spectral_params[1].pts2ppm(j)),
                )
            )
        if len(picked) == count:
            break
    return picked


def _gb3_widths(path, picked, rule):
    peaks = [{"ppm": list(p), "lw_hz": [1.0, 1.0], "amplitude": 1.0} for p in picked]
    lw, _, _, _ = _fit(path, peaks, rule)
    return lw / 100 + 1  # back to Hz (lw_hz of 1)


def main() -> None:
    """Run every case, and the GB3 check when asked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmrpipe", type=Path, required=True, help="NMRPipe installation")
    parser.add_argument("--gb3", action="store_true", help="also run the GB3 real-data check")
    args = parser.parse_args()
    nmrpipe = args.nmrpipe.expanduser()
    env = _environment(nmrpipe)
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        conventional(env, cwd)
        digital_filter(env, cwd)
        indirect(env, cwd)
        if args.gb3:
            gb3(env, cwd, nmrpipe)


if __name__ == "__main__":
    main()
