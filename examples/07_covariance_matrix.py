"""
Eigenprim: Compute per-point outer products for a covariance matrix.

Each CUDA thread centers one point (p - mean), computes the rank-1
outer product, and extracts the diagonal (variances). The host sums
the outer products to get the full covariance matrix.

Demonstrates: sub (-), outer, diagonal, trace, dot.

Run:  python examples/07_covariance_matrix.py
"""

import numpy as np
from numba import cuda

from eigenprim import Vector3f, Matrix3f, outer, diagonal, trace, dot, links


@cuda.jit(link=links())
def outer_products_kernel(
    px, py, pz,           # input points (SoA)
    mx, my, mz,           # mean (scalars)
    # Output: per-point outer product components (9 floats per point, column-major)
    out_c0r0, out_c0r1, out_c0r2,
    out_c1r0, out_c1r1, out_c1r2,
    out_c2r0, out_c2r1, out_c2r2,
    out_variances,         # diagonal sum per point (trace of outer product)
):
    i = cuda.grid(1)
    if i >= px.shape[0]:
        return

    p = Vector3f(px[i], py[i], pz[i])
    mean = Vector3f(mx, my, mz)

    # Center the point
    centered = p - mean

    # Rank-1 outer product: (p - mean) * (p - mean)^T
    P = outer(centered, centered)

    # Extract diagonal = per-axis variances for this point
    d = diagonal(P)

    # Trace = sum of variances = squared distance from mean
    out_variances[i] = trace(P)

    # Store outer product components (column-major)
    out_c0r0[i] = dot(d, Vector3f(1.0, 0.0, 0.0))  # P[0,0] via diagonal
    # For the full matrix, extract via matrix-vector multiply with basis vectors
    e0 = Vector3f(1.0, 0.0, 0.0)
    e1 = Vector3f(0.0, 1.0, 0.0)
    e2 = Vector3f(0.0, 0.0, 1.0)
    col0 = P @ e0
    col1 = P @ e1
    col2 = P @ e2
    out_c0r0[i] = dot(col0, e0)
    out_c0r1[i] = dot(col0, e1)
    out_c0r2[i] = dot(col0, e2)
    out_c1r0[i] = dot(col1, e0)
    out_c1r1[i] = dot(col1, e1)
    out_c1r2[i] = dot(col1, e2)
    out_c2r0[i] = dot(col2, e0)
    out_c2r1[i] = dot(col2, e1)
    out_c2r2[i] = dot(col2, e2)


# ── Host-side setup ──────────────────────────────────────────────

N = 2048
rng = np.random.default_rng(42)

# Anisotropic point cloud: more spread along X than Y, Z
points = rng.standard_normal((N, 3)).astype(np.float32)
points[:, 0] *= 3.0  # stretch X
points[:, 1] *= 1.0
points[:, 2] *= 0.5  # compress Z

mean = points.mean(axis=0)
px, py, pz = points[:, 0].copy(), points[:, 1].copy(), points[:, 2].copy()

# 9 output arrays for the outer product components
outer_arrays = [np.zeros(N, dtype=np.float32) for _ in range(9)]
variances = np.zeros(N, dtype=np.float32)

threads = 256
blocks = (N + threads - 1) // threads
outer_products_kernel[blocks, threads](
    px, py, pz,
    float(mean[0]), float(mean[1]), float(mean[2]),
    *outer_arrays,
    variances,
)

# Sum outer products on host to get covariance matrix
cov_gpu = np.zeros((3, 3), dtype=np.float32)
for idx, arr in enumerate(outer_arrays):
    col, row = divmod(idx, 3)
    cov_gpu[row, col] = arr.sum() / N

# ── Verify ───────────────────────────────────────────────────────

centered_np = points - mean
cov_np = (centered_np.T @ centered_np) / N

print(f"Covariance matrix ({N} points)")
print(f"  GPU:\n{cov_gpu}")
print(f"  Expected:\n{cov_np}")

cov_ok = np.allclose(cov_gpu, cov_np, rtol=1e-3)
print(f"\n  Match: {'PASS' if cov_ok else 'FAIL'}  (max err: {np.max(np.abs(cov_gpu - cov_np)):.2e})")

# Verify per-point variances = squared distance from mean
expected_variances = np.sum(centered_np ** 2, axis=1)
var_ok = np.allclose(variances, expected_variances, rtol=1e-4)
print(f"  Per-point variances: {'PASS' if var_ok else 'FAIL'}  (max err: {np.max(np.abs(variances - expected_variances)):.2e})")

all_pass = cov_ok and var_ok
print(f"\n{'All passed.' if all_pass else 'FAILURES!'}")
if not all_pass:
    raise AssertionError("Example failures")
