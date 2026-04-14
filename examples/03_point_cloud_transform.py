"""
Eigenprim: Rigid-body transform of a 3D point cloud.

Each CUDA thread applies a rotation matrix + translation to one point,
then computes the distance from origin and projection onto a direction.

Run:  pixi run python examples/03_point_cloud_transform.py
"""

import numpy as np
from numba import cuda

from eigenprim import Vector3f, Matrix3f, dot, norm, links


@cuda.jit(link=links())
def transform_kernel(
    px, py, pz,              # input points (SoA)
    distances, projections,   # scalar outputs
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

    # Rigid-body transform: q = R * p + t
    q = R @ p + t

    # Distance from origin
    distances[i] = norm(q)

    # Projection onto direction
    direction = Vector3f(dx, dy, dz)
    projections[i] = dot(q, direction)


# ── Host-side setup ──────────────────────────────────────────────

N = 1024
rng = np.random.default_rng(42)
points = rng.standard_normal((N, 3)).astype(np.float32)
px, py, pz = points[:, 0].copy(), points[:, 1].copy(), points[:, 2].copy()

# 45° rotation around Z axis
angle = np.pi / 4
c, s = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
R_np = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

t_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
direction = np.array([1, 1, 0], dtype=np.float32) / np.sqrt(2)

distances = np.zeros(N, dtype=np.float32)
projections = np.zeros(N, dtype=np.float32)

threads = 256
blocks = (N + threads - 1) // threads
transform_kernel[blocks, threads](
    px, py, pz,
    distances, projections,
    c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0,
    1.0, 2.0, 3.0,
    direction[0], direction[1], direction[2],
)

# ── Verify ───────────────────────────────────────────────────────

expected_q = (R_np @ points.T).T + t_np
expected_distances = np.linalg.norm(expected_q, axis=1)
expected_projections = expected_q @ direction

print(f"Rigid-body transform ({N} points)")
dist_ok = np.allclose(distances, expected_distances, rtol=1e-4)
proj_ok = np.allclose(projections, expected_projections, rtol=1e-4)
print(f"  Distances:   {'PASS' if dist_ok else 'FAIL'}  (max err: {np.max(np.abs(distances - expected_distances)):.2e})")
print(f"  Projections: {'PASS' if proj_ok else 'FAIL'}  (max err: {np.max(np.abs(projections - expected_projections)):.2e})")

all_pass = dist_ok and proj_ok
print(f"\n{'All passed.' if all_pass else 'FAILURES!'}")
if not all_pass:
    raise AssertionError("Example failures")
