"""
Eigenprim: Rigid-body transform of a 3D point cloud.

Each CUDA thread transforms one point by a rotation matrix + translation,
then computes the distance from origin and projection onto a direction.
Also demonstrates per-element linear system solve via inverse.

Run:  pixi run python example_transform.py
"""

import numpy as np
from numba import cuda

from eigenprim import (
    Vector3f, Matrix3f,
    dot, norm, inverse, determinant,
    links,
)


# ── Kernel 1: Rigid-body transform ───────────────────────────────
#
# Given N 3D points, a rotation matrix R, and a translation t:
#   q = R @ p + t
# Compute distance from origin and projection onto a reference direction.

@cuda.jit(link=links())
def transform_kernel(
    px, py, pz,           # input points (SoA)
    distances, projections,  # scalar outputs
    # Rotation matrix (9 scalars, column-major)
    r00, r10, r20, r01, r11, r21, r02, r12, r22,
    # Translation
    tx, ty, tz,
    # Projection direction
    dx, dy, dz,
):
    i = cuda.grid(1)
    if i >= px.shape[0]:
        return

    p = Vector3f(px[i], py[i], pz[i])
    R = Matrix3f(r00, r10, r20, r01, r11, r21, r02, r12, r22)
    t = Vector3f(tx, ty, tz)

    # Rigid-body transform
    q = R @ p + t

    # Distance from origin
    distances[i] = norm(q)

    # Projection onto direction
    direction = Vector3f(dx, dy, dz)
    projections[i] = dot(q, direction)


# ── Kernel 2: Per-element linear solve ────────────────────────────
#
# Given N right-hand sides b_i and a single matrix A:
#   x_i = A^{-1} * b_i
# Computes the solution norm for each element.

@cuda.jit(link=links())
def solve_kernel(
    bx, by, bz,       # input RHS vectors (SoA)
    solution_norms,    # output: norm of each solution
    # Matrix A (9 scalars, column-major)
    a00, a10, a20, a01, a11, a21, a02, a12, a22,
):
    i = cuda.grid(1)
    if i >= bx.shape[0]:
        return

    A = Matrix3f(a00, a10, a20, a01, a11, a21, a02, a12, a22)
    b = Vector3f(bx[i], by[i], bz[i])

    # Solve Ax = b via inverse (efficient for small fixed-size matrices)
    x = inverse(A) @ b

    solution_norms[i] = norm(x)


# ── Host-side setup and verification ─────────────────────────────

N = 1024

# Random 3D point cloud
rng = np.random.default_rng(42)
points = rng.standard_normal((N, 3)).astype(np.float32)
px, py, pz = points[:, 0].copy(), points[:, 1].copy(), points[:, 2].copy()

# 45° rotation around Z axis
angle = np.pi / 4
c, s = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
R_np = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

# Translation
t_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# Projection direction (unit vector along [1,1,0])
direction = np.array([1, 1, 0], dtype=np.float32) / np.sqrt(2)

# Output arrays
distances = np.zeros(N, dtype=np.float32)
projections = np.zeros(N, dtype=np.float32)

# Launch kernel 1
threads = 256
blocks = (N + threads - 1) // threads
transform_kernel[blocks, threads](
    px, py, pz,
    distances, projections,
    # R column-major: col0=(c,s,0), col1=(-s,c,0), col2=(0,0,1)
    c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0,
    # t
    1.0, 2.0, 3.0,
    # direction
    direction[0], direction[1], direction[2],
)

# Verify against numpy
expected_q = (R_np @ points.T).T + t_np
expected_distances = np.linalg.norm(expected_q, axis=1)
expected_projections = expected_q @ direction

print(f"Kernel 1: Rigid-body transform ({N} points)")
dist_ok = np.allclose(distances, expected_distances, rtol=1e-4)
proj_ok = np.allclose(projections, expected_projections, rtol=1e-4)
print(f"  Distances:   {'PASS' if dist_ok else 'FAIL'}  (max err: {np.max(np.abs(distances - expected_distances)):.2e})")
print(f"  Projections: {'PASS' if proj_ok else 'FAIL'}  (max err: {np.max(np.abs(projections - expected_projections)):.2e})")

# ── Kernel 2: Linear solve ────────────────────────────────────────

# A = diag(2, 3, 4) — simple invertible matrix
A_np = np.diag([2.0, 3.0, 4.0]).astype(np.float32)

# Random RHS vectors
rhs = rng.standard_normal((N, 3)).astype(np.float32)
bx, by, bz = rhs[:, 0].copy(), rhs[:, 1].copy(), rhs[:, 2].copy()

solution_norms = np.zeros(N, dtype=np.float32)

solve_kernel[blocks, threads](
    bx, by, bz,
    solution_norms,
    # A column-major: col0=(2,0,0), col1=(0,3,0), col2=(0,0,4)
    2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0,
)

# Verify: x = A^{-1} b, norm(x)
expected_x = np.linalg.solve(A_np, rhs.T).T
expected_norms = np.linalg.norm(expected_x, axis=1)

print(f"\nKernel 2: Per-element linear solve ({N} systems)")
solve_ok = np.allclose(solution_norms, expected_norms, rtol=1e-4)
print(f"  Solution norms: {'PASS' if solve_ok else 'FAIL'}  (max err: {np.max(np.abs(solution_norms - expected_norms)):.2e})")

all_pass = dist_ok and proj_ok and solve_ok
print(f"\n{'All passed.' if all_pass else 'FAILURES!'}")
if not all_pass:
    raise AssertionError("Example failures")
