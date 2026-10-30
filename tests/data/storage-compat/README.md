# Older-format regression fixtures

These small fitted fixtures were written by the actual older implementations:

- `science-v1.npz` and `Legacy.peakfit`: `1051ceb` (scientific 1.0, project 2.0).
- `Milestone3.peakfit`: `06157dd` (project 1.1, JSON plus sampled-input NPZ).

They use the deterministic three-plane overlap/remote-peak test dataset and two
optimizer passes. Original source paths intentionally point to an unavailable
placeholder. Tests validate numerical-only reading, exact correction preservation,
and explicit verified-source migration to a self-contained project. The synthetic
source reader is supplied only for migration; the serialization/readers are real.
