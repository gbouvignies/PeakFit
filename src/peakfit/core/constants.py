"""Core constants for PeakFit optimization and fitting.

These constants define default parameters for various optimization
algorithms. They can be overridden via configuration files or CLI arguments.
"""

# =============================================================================
# Least-Squares Optimization Defaults
# =============================================================================

LEAST_SQUARES_FTOL = 1e-7  # Function tolerance for convergence
"""Default function tolerance for least-squares optimization.

The optimization terminates when the relative change in cost function
is less than this value.
"""

LEAST_SQUARES_XTOL = 1e-7  # Parameter tolerance for convergence
"""Default parameter tolerance for least-squares optimization.

The optimization terminates when the relative change in parameter
values is less than this value.
"""

LEAST_SQUARES_MAX_NFEV = 1000  # Maximum function evaluations
"""Maximum number of function evaluations for least-squares optimization.

Prevents runaway optimization for difficult fitting problems.
"""

# =============================================================================
# Basin-Hopping Optimization Defaults
# =============================================================================

BASIN_HOPPING_NITER = 100  # Number of basin-hopping iterations
"""Default number of iterations for basin-hopping global optimization.

More iterations increase the likelihood of finding the global minimum
but increase computation time. Typical range: 50-500.
"""

BASIN_HOPPING_TEMPERATURE = 1.0  # Temperature parameter for acceptance
"""Temperature parameter for basin-hopping acceptance criterion.

Higher temperatures allow acceptance of worse solutions (more exploration).
Lower temperatures make the algorithm more selective (more exploitation).
Typical range: 0.5-2.0.
"""

BASIN_HOPPING_STEPSIZE = 0.5  # Step size for random displacement
"""Step size for random displacement in basin-hopping.

Controls the size of random jumps when escaping local minima.
Should be scaled to typical parameter magnitudes.
Typical range: 0.1-1.0.
"""

BASIN_HOPPING_LOCAL_MAXITER = 1000  # Maximum iterations for local minimization
"""Maximum iterations for the local minimizer in each basin-hopping step.

Each basin-hopping iteration uses L-BFGS-B as the local minimizer.
"""

# =============================================================================
# Differential Evolution Defaults
# =============================================================================

DIFF_EVOLUTION_MAXITER = 1000  # Maximum generations
"""Maximum number of generations for differential evolution.

Each generation evaluates the objective function for the entire population.
Typical range: 500-2000 depending on problem complexity.
"""

DIFF_EVOLUTION_POPSIZE = 15  # Population size multiplier
"""Population size multiplier for differential evolution.

The actual population size is popsize * number_of_parameters.
Higher values provide better exploration but increase computation.
Typical range: 10-20.
"""

DIFF_EVOLUTION_MUTATION = (0.5, 1.0)  # Mutation constant range
"""Mutation constant range for differential evolution.

Controls the amplification of differential variation.
Uses dithering (random value in range) for better convergence.
"""

DIFF_EVOLUTION_RECOMBINATION = 0.7  # Recombination constant
"""Recombination (crossover) probability for differential evolution.

Probability that each parameter comes from the mutant rather than the target.
Typical range: 0.5-0.9.
"""

# =============================================================================
# Convergence Criteria
# =============================================================================

CONVERGENCE_CHI2_THRESHOLD = 1e-4  # Chi-squared change threshold
"""Threshold for chi-squared change to consider refinement converged.

If relative change in chi-squared is less than this value, the
refinement loop terminates. Lower values increase accuracy but
may require more iterations.
"""

MAX_REFINEMENT_ITERATIONS = 3  # Maximum refinement passes
"""Maximum number of refinement iterations in iterative fitting.

Refinement re-optimizes previously fitted parameters with updated
estimates. Usually converges within 2-3 iterations.
"""

# =============================================================================
# Uncertainty Estimation
# =============================================================================

PROFILE_LIKELIHOOD_NPOINTS = 20  # Number of profile likelihood points
"""Number of points to compute for profile likelihood confidence intervals.

More points provide smoother profiles but increase computation time.
Typical range: 15-30.
"""

PROFILE_LIKELIHOOD_DELTA_CHI2 = 3.84  # Chi-squared threshold for 95% CI
"""Chi-squared threshold for profile likelihood confidence intervals.

3.84 corresponds to 95% CI for 1 parameter (from chi-squared distribution).
For other confidence levels: 68% = 1.00, 99% = 6.63.
"""

MCMC_N_WALKERS = 32  # Number of MCMC walkers
"""Number of walkers for MCMC uncertainty estimation using emcee.

Should be at least 2 * number_of_parameters. More walkers provide
better sampling but increase computation. Typical range: 20-50.
"""

MCMC_N_STEPS = 1000  # Number of MCMC steps
"""Number of MCMC steps per walker.

More steps provide better convergence and more samples.
Typical range: 500-5000 depending on problem complexity.
"""

MCMC_BURN_IN = 200  # MCMC burn-in steps to discard
"""Number of initial MCMC steps to discard as burn-in.

These steps allow the chains to reach the posterior distribution.
Usually 10-30% of total steps. Typical range: 100-500.
"""

# =============================================================================
# Progress Display
# =============================================================================

PROGRESS_UPDATE_INTERVAL = 1  # Seconds between progress updates
"""Minimum time interval between progress bar updates.

Reduces console flickering while providing responsive feedback.
"""

# =============================================================================
# Benchmarking
# =============================================================================

BENCHMARK_MAX_NFEV = 500  # Function evaluations for benchmarks
"""Reduced max_nfev for benchmarking to ensure fast execution.

Benchmarks prioritize speed over convergence precision.
"""
