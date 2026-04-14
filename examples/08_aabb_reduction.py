"""
Eigenprim: Axis-aligned bounding box (AABB) of a 3D point cloud.

Each CUDA thread processes one point, writing component-wise min/max
into per-point output arrays. The host reduces to get the global AABB.

Demonstrates: cwise_min, cwise_max, min_coeff, max_coeff, sub (-), dot.

Run:  pixi run python examples/08_aabb_reduction.py
"""

import numpy as np
from numba import cuda

from eigenprim import (
    Vector3f,
    cwise_min, cwise_max, min_coeff, max_coeff, dot, norm,
    links,
)


@cuda.jit(link=links())
def aabb_kernel(
    px, py, pz,                   # input points (SoA)
    out_min_x, out_min_y, out_min_z,  # per-point running min
    out_max_x, out_max_y, out_max_z,  # per-point running max
    out_diag_length,               # per-point: diagonal length if AABB were just this point vs origin
):
    i = cuda.grid(1)
    if i >= px.shape[0]:
        return

    p = Vector3f(px[i], py[i], pz[i])

    # Component-wise min/max with the origin (clamp negative to 0)
    zero = Vector3f(0.0, 0.0, 0.0)
    lo = cwise_min(p, zero)
    hi = cwise_max(p, zero)

    # Extract components
    e0 = Vector3f(1.0, 0.0, 0.0)
    e1 = Vector3f(0.0, 1.0, 0.0)
    e2 = Vector3f(0.0, 0.0, 1.0)

    out_min_x[i] = dot(lo, e0)
    out_min_y[i] = dot(lo, e1)
    out_min_z[i] = dot(lo, e2)
    out_max_x[i] = dot(hi, e0)
    out_max_y[i] = dot(hi, e1)
    out_max_z[i] = dot(hi, e2)

    # Diagonal length of the AABB from lo to hi
    extent = hi - lo
    out_diag_length[i] = norm(extent)


# ── Host-side setup ──────────────────────────────────────────────

N = 4096
rng = np.random.default_rng(42)
points = rng.standard_normal((N, 3)).astype(np.float32) * 10.0
px, py, pz = points[:, 0].copy(), points[:, 1].copy(), points[:, 2].copy()

min_x = np.zeros(N, dtype=np.float32)
min_y = np.zeros(N, dtype=np.float32)
min_z = np.zeros(N, dtype=np.float32)
max_x = np.zeros(N, dtype=np.float32)
max_y = np.zeros(N, dtype=np.float32)
max_z = np.zeros(N, dtype=np.float32)
diag_length = np.zeros(N, dtype=np.float32)

threads = 256
blocks = (N + threads - 1) // threads
aabb_kernel[blocks, threads](
    px, py, pz,
    min_x, min_y, min_z,
    max_x, max_y, max_z,
    diag_length,
)

# Host reduction: global AABB = min of all per-point mins, max of all per-point maxes
gpu_lo = np.array([min_x.min(), min_y.min(), min_z.min()])
gpu_hi = np.array([max_x.max(), max_y.max(), max_z.max()])

# ── Verify ───────────────────────────────────────────────────────

expected_lo = np.minimum(points, 0).min(axis=0)
expected_hi = np.maximum(points, 0).max(axis=0)

# Per-point diagonal length
expected_lo_per = np.minimum(points, 0)
expected_hi_per = np.maximum(points, 0)
expected_diag = np.linalg.norm(expected_hi_per - expected_lo_per, axis=1)

print(f"AABB reduction ({N} points)")
print(f"  GPU  lo: [{gpu_lo[0]:.2f}, {gpu_lo[1]:.2f}, {gpu_lo[2]:.2f}]")
print(f"  GPU  hi: [{gpu_hi[0]:.2f}, {gpu_hi[1]:.2f}, {gpu_hi[2]:.2f}]")
print(f"  Expected lo: [{expected_lo[0]:.2f}, {expected_lo[1]:.2f}, {expected_lo[2]:.2f}]")
print(f"  Expected hi: [{expected_hi[0]:.2f}, {expected_hi[1]:.2f}, {expected_hi[2]:.2f}]")

lo_ok = np.allclose(gpu_lo, expected_lo, rtol=1e-5)
hi_ok = np.allclose(gpu_hi, expected_hi, rtol=1e-5)
diag_ok = np.allclose(diag_length, expected_diag, rtol=1e-4)
print(f"  Lo match: {'PASS' if lo_ok else 'FAIL'}")
print(f"  Hi match: {'PASS' if hi_ok else 'FAIL'}")
print(f"  Diag lengths: {'PASS' if diag_ok else 'FAIL'}  (max err: {np.max(np.abs(diag_length - expected_diag)):.2e})")

all_pass = lo_ok and hi_ok and diag_ok
print(f"\n{'All passed.' if all_pass else 'FAILURES!'}")
if not all_pass:
    raise AssertionError("Example failures")
