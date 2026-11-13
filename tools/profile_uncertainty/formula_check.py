"""V1 preview: per-cluster factor formula vs full-Jacobian OLS sandwich, amplitude block."""

import numpy as np
from processed_lines import Chain, lag_matrix, shapes

chain = Chain(92, 256, 0.4)
index = np.arange(91, 110)
theta = np.array([98.5, 1.3, 101.5, 1.6])
rng = np.random.default_rng(1)
L, S = 2, 6
amps = rng.uniform(0.2, 1.0, (L, S))
r = lag_matrix(chain.rho(), index)
phi, dphi = shapes(chain, index, theta)  # (n, L)
n, P = index.size, theta.size
# full Jacobian, rows (plane-major), columns θ then amplitudes (l, s)
J = np.zeros((n * S, P + L * S))
E = [np.zeros((n, P)) for _ in range(L)]
for l in range(L):
    E[l][:, 2 * l], E[l][:, 2 * l + 1] = dphi[l][0], dphi[l][1]
for s in range(S):
    rows = slice(s * n, (s + 1) * n)
    J[rows, :P] = sum(amps[l, s] * E[l] for l in range(L))
    for l in range(L):
        J[rows, P + l * S + s] = phi[:, l]
B = np.linalg.inv(J.T @ J)
full = (B @ J.T @ np.kron(np.eye(S), r) @ J @ B)[P:, P:].reshape(L, S, L, S)
# factor formula: only per-cluster constants and amplitudes
Phi = phi.T
G = Phi @ Phi.T
M = np.linalg.solve(G, Phi)
Pp = np.eye(n) - Phi.T @ M
D = [sum(amps[l, s] * E[l] for l in range(L)) for s in range(S)]
H = sum(d.T @ Pp @ d for d in D)
Hi = np.linalg.inv(H)
T = [Hi @ d.T @ Pp for d in D]
Ct = sum(t @ r @ t.T for t in T)
MRM = M @ r @ M.T
Gs = [M @ d for d in D]
f = np.zeros((L, S, L, S))
for s in range(S):
    for t in range(S):
        c = -M @ r @ T[s].T @ Gs[t].T - Gs[s] @ T[t] @ r @ M.T + Gs[s] @ Ct @ Gs[t].T
        f[:, s, :, t] = c + (MRM if s == t else 0)
print("max |formula − full| / max|full| =", np.abs(f - full).max() / np.abs(full).max())
# R = I: cross terms vanish
print("cross term norm under R=I:", np.abs(M @ T[0].T).max())
