"""Size and conditioning of R on the bundled CEST example's clusters (§2.4)."""

import numpy as np
from cest_example import load

from peakfit.engine.algorithms.noise_correlation import cluster_correlation


def main() -> None:
    """Print cluster sizes and the condition number of each cluster's R."""
    example = load()
    sizes, conditions, weak = [], [], []
    for cluster in example.clusters:
        r = cluster_correlation(example.model, cluster.grid_indices)
        lam = np.linalg.eigvalsh(r)
        sizes.append(r.shape[0])
        conditions.append(lam.max() / max(lam.min(), 1e-300))
        weak.append(np.mean(lam < 1e-3 * lam.max()))
    print(f"clusters {len(sizes)}, planes {example.spectra.data.shape[0]}")
    print(f"points per cluster: median {np.median(sizes):g}, max {max(sizes)}")
    print(f"cond(R): median {np.median(conditions):.2g}, max {max(conditions):.2g}")
    print(f"fraction of directions with λ < 1e-3 λmax: median {np.median(weak):.2f}")


if __name__ == "__main__":
    main()
