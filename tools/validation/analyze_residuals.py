import json
import numpy as np
import nmrglue as ng
from pathlib import Path
from peakfit.core.domain.spectrum import Spectra
from peakfit.core.domain.cluster import Cluster
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.fitting.simulation import simulate_data
from peakfit.core.fitting.computation import calculate_shapes, calculate_amplitudes


def analyze_residuals():
    # Paths
    base_dir = Path("examples/02-advanced-fitting")
    data_path = base_dir / "data/pseudo3d.ft2"
    results_path = base_dir / "Fits/fit_results.json"

    print(f"Loading data from {data_path}...")
    dic, data = ng.pipe.read(str(data_path))

    print(f"Loading results from {results_path}...")
    with open(results_path) as f:
        results = json.load(f)

    # Reconstruct parameters
    params = Parameters()
    # We need to reconstruct the Parameters object from the JSON
    # This is a bit complex because the JSON structure is flattened/processed
    # A simpler way might be to just simulate the peaks if we have their shapes and amplitudes

    # Let's try to use the peakfit library's simulation tools if possible
    # But we need to reconstruct the full state.

    # Alternative: The fit produced simulated spectra!
    # run.sh output said: "save_simulated": true
    # Let's check if simulated.ft2 exists in Fits/

    sim_path = base_dir / "Fits/simulated.ft3"
    if sim_path.exists():
        print(f"Found simulated spectrum at {sim_path}")
        dic_sim, data_sim = ng.pipe.read(str(sim_path))

        print(f"Data shape: {data.shape}")
        print(f"Simulated shape: {data_sim.shape}")

        # Ensure shapes match
        if data.shape != data_sim.shape:
            print("Shapes do not match. Attempting to reshape...")
            # If data is (N, Y, X) and sim is (N, Y, X), it's fine.
            # If one is flattened, we might need to fix.
            if data.ndim == 2 and data_sim.ndim == 3:
                # Maybe data was read as 2D but it's actually pseudo-3D
                # But ng.pipe.read usually respects the header.
                pass
            elif data.ndim == 3 and data_sim.ndim == 2:
                pass

        # Compute residuals
        residuals = data - data_sim

        # Analyze a few planes
        n_planes = residuals.shape[0]
        print(f"Analyzing residuals for {n_planes} planes...")

        autocorr_results = []

        for i in range(n_planes):
            plane = residuals[i]
            # Calculate 2D autocorrelation
            # Normalize plane
            plane_mean = np.mean(plane)
            plane_std = np.std(plane)
            if plane_std == 0:
                continue

            normalized_plane = (plane - plane_mean) / plane_std

            # FFT based autocorrelation
            f = np.fft.fft2(normalized_plane)
            acf = np.real(np.fft.ifft2(f * np.conj(f)))
            acf = acf / (plane.shape[0] * plane.shape[1])

            # Shift so zero lag is in center
            acf = np.fft.fftshift(acf)

            # Check nearest neighbors correlation (lag 1)
            # Center index
            cy, cx = acf.shape[0] // 2, acf.shape[1] // 2

            # Average correlation at lag 1 (up, down, left, right)
            corr_lag1 = (acf[cy + 1, cx] + acf[cy - 1, cx] + acf[cy, cx + 1] + acf[cy, cx - 1]) / 4
            autocorr_results.append(corr_lag1)

        avg_corr = np.mean(autocorr_results)
        print(f"Average Lag-1 Autocorrelation: {avg_corr:.4f}")

        if abs(avg_corr) > 0.1:
            print("WARNING: Significant non-white noise detected.")
        else:
            print("Noise appears mostly white.")

    else:
        print("Simulated spectrum not found. Cannot compute residuals easily.")


if __name__ == "__main__":
    analyze_residuals()
