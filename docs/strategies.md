# Optimizers and Protocols

## Available strategies
- `varpro` (default): variable projection with analytical Jacobians where available
- `leastsq`: SciPy `least_squares`
- `basin_hopping`: global search warm‑start
- `differential_evolution`: global search warm‑start
- `mcmc`: posterior exploration and uncertainty estimation (requires `emcee`)

## Choosing a strategy
CLI examples:

```bash
peakfit fit --optimizer varpro  # default
peakfit fit --optimizer basin_hopping
peakfit fit --optimizer differential_evolution
peakfit fit --optimizer leastsq
peakfit fit --optimizer mcmc
```

YAML config snippet:

```yaml
fit:
  optimizer: basin_hopping
  optimizer_params:
    n_iterations: 50
    temperature: 1.0
    step_size: 0.5
    seed: 1234
```

## Recommended defaults (configurable)
- Basin hopping: `n_iterations=50`, `temperature=1.0`, `step_size=0.5`, `seed=1234`
- Differential evolution: `max_iterations=200`, `population_size=15`, `mutation=(0.5,1.0)`, `recombination=0.7`, `polish=True`, `seed=1234`
- MCMC: `chains=2`, `steps=1000`, `warmup=200`, `thin=1`, `seed=1234`

## Protocol: Global warm‑start → VarPro refine (optional)
- Stage 1: run BH/DE/MCMC to obtain an initial parameter set
- Stage 2: run `varpro` to refine to local optimum
- Benefits: robustness when initial guesses are poor; predictable convergence
- Caveats: apply time/iteration budgets; fix seeds for reproducibility
