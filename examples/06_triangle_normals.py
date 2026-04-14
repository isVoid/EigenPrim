"""
Eigenprim: Compute surface normals for a triangle mesh.

Each CUDA thread takes one triangle (3 vertices), computes two edge
vectors, takes the cross product, and normalizes to get the unit normal.
Also computes twice the triangle area as the norm of the cross product.

Demonstrates: sub (-), cross, normalized, squared_norm, norm.

Run:  pixi run python examples/06_triangle_normals.py
"""

import numpy as np
from numba import cuda

from eigenprim import Vector3f, cross, normalized, norm, squared_norm, dot, links


@cuda.jit(link=links())
def normals_kernel(
    v0x, v0y, v0z,   # vertex 0 (SoA)
    v1x, v1y, v1z,   # vertex 1
    v2x, v2y, v2z,   # vertex 2
    nx, ny, nz,       # output unit normals (SoA)
    areas,            # output triangle areas
):
    i = cuda.grid(1)
    if i >= v0x.shape[0]:
        return

    p0 = Vector3f(v0x[i], v0y[i], v0z[i])
    p1 = Vector3f(v1x[i], v1y[i], v1z[i])
    p2 = Vector3f(v2x[i], v2y[i], v2z[i])

    # Edge vectors
    e1 = p1 - p0
    e2 = p2 - p0

    # Cross product = normal direction, magnitude = 2 * area
    n = cross(e1, e2)

    # Triangle area = 0.5 * |cross product|
    areas[i] = norm(n) * 0.5

    # Unit normal
    unit_n = normalized(n)

    # Extract components via dot with basis vectors
    nx[i] = dot(unit_n, Vector3f(1.0, 0.0, 0.0))
    ny[i] = dot(unit_n, Vector3f(0.0, 1.0, 0.0))
    nz[i] = dot(unit_n, Vector3f(0.0, 0.0, 1.0))


# ── Host-side setup ──────────────────────────────────────────────

N = 1024
rng = np.random.default_rng(42)

# Random triangles
v0 = rng.standard_normal((N, 3)).astype(np.float32)
v1 = v0 + rng.standard_normal((N, 3)).astype(np.float32) * 0.5
v2 = v0 + rng.standard_normal((N, 3)).astype(np.float32) * 0.5

nx = np.zeros(N, dtype=np.float32)
ny = np.zeros(N, dtype=np.float32)
nz = np.zeros(N, dtype=np.float32)
areas = np.zeros(N, dtype=np.float32)

threads = 256
blocks = (N + threads - 1) // threads
normals_kernel[blocks, threads](
    v0[:, 0].copy(), v0[:, 1].copy(), v0[:, 2].copy(),
    v1[:, 0].copy(), v1[:, 1].copy(), v1[:, 2].copy(),
    v2[:, 0].copy(), v2[:, 1].copy(), v2[:, 2].copy(),
    nx, ny, nz, areas,
)

# ── Verify ───────────────────────────────────────────────────────

e1_np = v1 - v0
e2_np = v2 - v0
cross_np = np.cross(e1_np, e2_np)
expected_areas = np.linalg.norm(cross_np, axis=1) * 0.5
expected_normals = cross_np / np.linalg.norm(cross_np, axis=1, keepdims=True)

gpu_normals = np.stack([nx, ny, nz], axis=1)

print(f"Triangle normals ({N} triangles)")
normals_ok = np.allclose(gpu_normals, expected_normals, atol=1e-5)
areas_ok = np.allclose(areas, expected_areas, rtol=1e-4)
print(f"  Normals: {'PASS' if normals_ok else 'FAIL'}  (max err: {np.max(np.abs(gpu_normals - expected_normals)):.2e})")
print(f"  Areas:   {'PASS' if areas_ok else 'FAIL'}  (max err: {np.max(np.abs(areas - expected_areas)):.2e})")

all_pass = normals_ok and areas_ok
print(f"\n{'All passed.' if all_pass else 'FAILURES!'}")
if not all_pass:
    raise AssertionError("Example failures")
