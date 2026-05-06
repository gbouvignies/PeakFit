"""Centralized constants and default parameters for PeakFit."""

from typing import Final

# Least-Squares Optimization Defaults
LEAST_SQUARES_FTOL: Final = 1e-7
LEAST_SQUARES_XTOL: Final = 1e-7
LEAST_SQUARES_MAX_NFEV: Final = 1000

# Basin-Hopping Optimization Defaults
BASIN_HOPPING_NITER: Final = 200
BASIN_HOPPING_TEMPERATURE: Final = 1.0
BASIN_HOPPING_STEPSIZE: Final = 0.5
BASIN_HOPPING_LOCAL_MAXITER: Final = 1000

DIFF_EVOLUTION_MAXITER: Final = 100000
DIFF_EVOLUTION_MUTATION: Final = (0.5, 1.0)
DIFF_EVOLUTION_RECOMBINATION: Final = 0.7
DIFF_EVOLUTION_STRATEGY: Final = "rand1bin"
DIFF_EVOLUTION_INIT: Final = "sobol"

# Convergence Criteria
CONVERGENCE_CHI2_THRESHOLD: Final = 1e-4
MAX_REFINEMENT_ITERATIONS: Final = 3

# Uncertainty Estimation
PROFILE_LIKELIHOOD_NPOINTS: Final = 20
PROFILE_LIKELIHOOD_DELTA_CHI2: Final = 3.84
MCMC_N_WALKERS: Final = 64
MCMC_N_STEPS: Final = 5000
MCMC_BURN_IN: Final = 1000
