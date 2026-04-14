"""
Eigenprim: Double-precision gravitational N-body pairwise interactions.

Each CUDA thread computes the gravitational potential and acceleration
between one pair of bodies using double-precision vectors. Demonstrates
that all *d types (Vector3d, Matrix3d) work identically to float.

Demonstrates: Vector3d, sub (-), squared_norm, norm, scale (*), sum, dot.

Run:  pixi run python examples/09_double_precision.py
"""

import numpy as np
from numba import cuda

from eigenprim import Vector3d, dot, norm, squared_norm, sum as eigsum, links


@cuda.jit(link=links())
def gravity_kernel(
    # Body i positions
    ix, iy, iz,
    # Body j positions
    jx, jy, jz,
    # Body j masses
    jm,
    # Outputs per pair
    potentials,      # gravitational potential: -G * m_j / r
    accel_magnitudes,  # |acceleration| = G * m_j / r^2
):
    idx = cuda.grid(1)
    if idx >= ix.shape[0]:
        return

    pi = Vector3d(ix[idx], iy[idx], iz[idx])
    pj = Vector3d(jx[idx], jy[idx], jz[idx])
    mj = jm[idx]

    # Displacement vector
    r_vec = pj - pi

    # Distance and squared distance (double precision matters here)
    r2 = squared_norm(r_vec)
    r = norm(r_vec)

    # Gravitational potential: phi = -G * m_j / r  (G = 1 for simplicity)
    potentials[idx] = -mj / r

    # Acceleration magnitude: |a| = G * m_j / r^2
    accel_magnitudes[idx] = mj / r2


# ── Host-side setup ──────────────────────────────────────────────

N = 1024
rng = np.random.default_rng(42)

# Random body pairs (double precision)
pos_i = rng.standard_normal((N, 3))  # float64
pos_j = rng.standard_normal((N, 3)) + 5.0  # offset to avoid near-zero distances
masses_j = rng.uniform(0.1, 10.0, N)

potentials = np.zeros(N, dtype=np.float64)
accel_mags = np.zeros(N, dtype=np.float64)

threads = 256
blocks = (N + threads - 1) // threads
gravity_kernel[blocks, threads](
    pos_i[:, 0].copy(), pos_i[:, 1].copy(), pos_i[:, 2].copy(),
    pos_j[:, 0].copy(), pos_j[:, 1].copy(), pos_j[:, 2].copy(),
    masses_j.copy(),
    potentials, accel_mags,
)

# ── Verify ───────────────────────────────────────────────────────

r_np = pos_j - pos_i
dist_np = np.linalg.norm(r_np, axis=1)
expected_potentials = -masses_j / dist_np
expected_accel_mags = masses_j / (dist_np ** 2)

print(f"Double-precision gravitational interactions ({N} pairs)")
pot_ok = np.allclose(potentials, expected_potentials, rtol=1e-10)
acc_ok = np.allclose(accel_mags, expected_accel_mags, rtol=1e-10)
print(f"  Potentials:     {'PASS' if pot_ok else 'FAIL'}  (max rel err: {np.max(np.abs((potentials - expected_potentials) / expected_potentials)):.2e})")
print(f"  Accelerations:  {'PASS' if acc_ok else 'FAIL'}  (max rel err: {np.max(np.abs((accel_mags - expected_accel_mags) / expected_accel_mags)):.2e})")

all_pass = pot_ok and acc_ok
print(f"\n{'All passed.' if all_pass else 'FAILURES!'}")
if not all_pass:
    raise AssertionError("Example failures")
