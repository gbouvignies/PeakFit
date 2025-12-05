import numpy as np
from peakfit.core.fitting.optimizer import VarProOptimizer
from peakfit.core.domain.cluster import Cluster
from peakfit.core.lineshapes.gaussian import Gaussian
from peakfit.core.lineshapes.pvoigt import PseudoVoigt
from peakfit.core.shared.typing import FittingOptions


# Mock classes
class MockSpecParams:
    def __init__(self):
        self.size = 100
        self.direct = True
        self.p180 = False
        self.aq_time = 0.1
        self.sw = 1000.0
        self.frequency = 500.0  # MHz
        self.apodq1 = 0.0
        self.apodq2 = 0.0

    def ppm2pts(self, ppm):
        # 1 ppm = 500 Hz. 1 pt = 10 Hz. So 1 ppm = 50 pts.
        return 50 + ppm * 50

    def ppm2pt_i(self, ppm):
        return int(self.ppm2pts(ppm))

    def pts2hz_delta(self, pts):
        # 1 pt = 10 Hz
        return pts * 10.0

    def hz2ppm(self, hz):
        return hz / self.frequency

    def ppm2hz(self, ppm):
        return ppm * self.frequency


class MockSpectra:
    def __init__(self):
        self.params = {1: MockSpecParams()}


class MockArgs(FittingOptions):
    def __init__(self):
        self.jx = False
        self.phx = False
        self.phy = False


def verify():
    print("Verifying Jacobian Implementation...")
    # Setup
    spectra = MockSpectra()
    args = MockArgs()

    # 1. Test Gaussian
    print("\n--- Testing Gaussian ---")
    peak = Gaussian("test_peak", 0.0, spectra, 1, args)

    # Create parameters
    params = peak.create_params()
    print("Parameters:", list(params.keys()))
    # Relax bounds
    for p in params.values():
        p.min = -np.inf
        p.max = np.inf

    # Create Peak object
    from peakfit.core.domain.peaks import Peak

    peak_obj = Peak(name="test_peak", positions=np.array([0.0]), shapes=[peak])

    # Create cluster
    cluster = Cluster(
        cluster_id=1,
        peaks=[peak_obj],
        positions=[np.arange(100)],
        data=np.zeros(100),  # Placeholder
    )

    # Generate synthetic data
    # True params
    true_x0 = 0.5  # ppm -> 55 pts -> dx = 5 pts -> 50 Hz
    true_lw = 25.0  # Hz
    params["test_peak.F2.cs"].value = true_x0
    params["test_peak.F2.lw"].value = true_lw

    # Evaluate shape
    shape = peak_obj.evaluate(cluster.positions, params)
    # Add amplitude
    true_amp = 100.0
    data = shape * true_amp
    cluster.data = data
    cluster.initialize_corrections()

    # Define wrapper
    names = ["test_peak.F2.cs", "test_peak.F2.lw"]
    # Verify at optimum (where Kaufman approx is exact)
    x_test = np.array([true_x0, true_lw])
    noise = 1.0

    def create_optimizer(x, names, params, cluster, noise):
        opt = VarProOptimizer(cluster, names, params, noise)
        return opt

    def func(x):
        opt = create_optimizer(x, names, params, cluster, noise)
        return opt.compute_residuals(x)

    def jac(x):
        opt = create_optimizer(x, names, params, cluster, noise)
        return opt.compute_jacobian(x)

    # Verify values manually
    J_ana = jac(x_test)

    # Finite difference
    eps = 1e-6
    J_fd = np.zeros_like(J_ana)
    r0 = func(x_test)
    for i in range(len(x_test)):
        x_p = x_test.copy()
        x_p[i] += eps
        r_p = func(x_p)
        J_fd[:, i] = (r_p - r0) / eps

    max_diff = np.max(np.abs(J_ana - J_fd))
    max_val = np.max(np.abs(J_ana))
    rel_diff = max_diff / max_val if max_val > 0 else max_diff

    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Max relative difference: {rel_diff:.2e}")

    if rel_diff < 1e-4:
        print("Gaussian Jacobian Verified!")
    else:
        print("Gaussian Jacobian FAILED")
        print("Analytical (first 5 rows):\n", J_ana[:5])
        print("Finite Diff (first 5 rows):\n", J_fd[:5])

    # 2. Test PseudoVoigt
    print("\n--- Testing PseudoVoigt ---")
    peak_pv = PseudoVoigt("pv_peak", 0.0, spectra, 1, args)
    params_pv = peak_pv.create_params()
    # Relax bounds
    for p in params_pv.values():
        p.min = -np.inf
        p.max = np.inf
    # Create Peak object for PseudoVoigt
    from peakfit.core.domain.peaks import Peak

    peak_pv_obj = Peak(name="pv_peak", positions=np.array([0.0]), shapes=[peak_pv])

    cluster_pv = Cluster(
        cluster_id=2, peaks=[peak_pv_obj], positions=[np.arange(100)], data=np.zeros(100)
    )

    true_x0 = -0.5
    true_lw = 40.0
    true_eta = 0.5
    params_pv["pv_peak.F2.cs"].value = true_x0
    params_pv["pv_peak.F2.lw"].value = true_lw
    params_pv["pv_peak.F2.eta"].value = true_eta

    shape = peak_pv_obj.evaluate(cluster_pv.positions, params_pv)
    true_amp = 50.0
    cluster_pv.data = shape * true_amp
    cluster_pv.initialize_corrections()

    names_pv = ["pv_peak.F2.cs", "pv_peak.F2.lw", "pv_peak.F2.eta"]
    x_pv_test = np.array([true_x0, true_lw, true_eta])  # At optimum

    def func_pv(x):
        opt = create_optimizer(x, names_pv, params_pv, cluster_pv, noise)
        return opt.compute_residuals(x)

    def jac_pv(x):
        opt = create_optimizer(x, names_pv, params_pv, cluster_pv, noise)
        return opt.compute_jacobian(x)

    J_ana_pv = jac_pv(x_pv_test)
    J_fd_pv = np.zeros_like(J_ana_pv)
    r0_pv = func_pv(x_pv_test)
    for i in range(len(x_pv_test)):
        x_p = x_pv_test.copy()
        x_p[i] += eps
        r_p = func_pv(x_p)
        J_fd_pv[:, i] = (r_p - r0_pv) / eps

    max_diff_pv = np.max(np.abs(J_ana_pv - J_fd_pv))
    max_val_pv = np.max(np.abs(J_ana_pv))
    rel_diff_pv = max_diff_pv / max_val_pv if max_val_pv > 0 else max_diff_pv

    print(f"Max absolute difference: {max_diff_pv:.2e}")
    print(f"Max relative difference: {rel_diff_pv:.2e}")

    if rel_diff_pv < 1e-4:
        print("PseudoVoigt Jacobian Verified!")
    else:
        print("PseudoVoigt Jacobian FAILED")


if __name__ == "__main__":
    verify()
