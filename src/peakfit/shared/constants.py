"""Centralized constants and default parameters for PeakFit."""

from typing import Final

# Basin-Hopping Optimization Defaults
BASIN_HOPPING_NITER: Final = 200
BASIN_HOPPING_TEMPERATURE: Final = 1.0
BASIN_HOPPING_STEPSIZE: Final = 0.5
BASIN_HOPPING_LOCAL_MAXITER: Final = 1000

# Uncertainty Estimation
MCMC_N_WALKERS: Final = 64
MCMC_N_STEPS: Final = 5000
