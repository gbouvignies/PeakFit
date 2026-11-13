"""Processed synthetic lines shared by the synthetic studies.

The processing chain is `tools/validate_correlated_noise.py`'s: a complex FID, sine-bell
apodization, first-point scaling, zero fill, Fourier transform, real part.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/
from validate_correlated_noise import Chain, _complex_noise, lag_matrix

__all__ = ["Chain", "_complex_noise", "lag_matrix", "shapes"]


def shapes(chain, index, theta):
    """Unit-amplitude processed lines and their θ derivatives; θ = (pos, decay) per line."""
    t = np.arange(chain.n)
    phi, dphi = [], []
    for p, d in theta.reshape(-1, 2):
        f = np.exp(2j * np.pi * p * t / chain.m - d * t / chain.n)
        phi.append(chain.process(f)[index])
        dphi.append(
            [
                chain.process(f * 2j * np.pi * t / chain.m)[index],
                chain.process(-f * t / chain.n)[index],
            ]
        )
    return np.array(phi).T, dphi  # (n, L)
