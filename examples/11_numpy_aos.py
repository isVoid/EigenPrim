"""
eigenprim: Array-of-Structures (AoS) patterns with 2D/3D numpy arrays.

All previous examples use Structure-of-Arrays (SoA): a (N, 3) numpy array
is split into three separate (N,) arrays on the host before the kernel call,
then reassembled component-by-component inside the kernel::

    # SoA pattern (existing examples)
    px, py, pz = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()

    @cuda.jit(link=links())
    def kernel(px, py, pz, ...):
        p = Vector3f(px[i], py[i], pz[i])

This example shows the Array-of-Structures (AoS) pattern: the kernel
accepts the 2D numpy array directly and indexes into it with ``pts[i, j]``.
No host-side decomposition is needed. ``eigenprim.numpy_utils`` provides
helpers for converting between AoS and SoA layouts when required.

    # AoS pattern (this example)
    pts = rng.standard_normal((N, 3)).astype(np.float32)  # shape (N, 3)

    @cuda.jit(link=links())
    def kernel(pts, ...):
        p = Vector3f(pts[i, 0], pts[i, 1], pts[i, 2])

Three kernels are demonstrated:

  Kernel A — Batch dot products      (N, 3) x (N, 3)   → (N,)    [scalar output]
  Kernel B — Batch normalize         (N, 3)             → (N, 3)  [AoS input + output]
  Kernel C — Batch mat-vec multiply  (N, 3, 3) x (N, 3) → (N, 3)  [3-D AoS input]

Run:  pixi run python examples/11_numpy_aos.py
"""

import numpy as np
from numba import cuda

from eigenprim import Vector3f, Matrix3f, dot, normalized, links
from eigenprim.numpy_utils import vec_to_soa, soa_to_vec, mat_to_soa, soa_to_mat


# ── Kernel A: Batch dot products ──────────────────────────────────────────────
# Input : a, b — shape (N, 3) float32
# Output: out  — shape (N,)  float32  (scalar per pair)

@cuda.jit(link=links())
def dot_kernel(a, b, out):
    i = cuda.grid(1)
    if i >= a.shape[0]:
        return
    va = Vector3f(a[i, 0], a[i, 1], a[i, 2])
    vb = Vector3f(b[i, 0], b[i, 1], b[i, 2])
    out[i] = dot(va, vb)


# ── Kernel B: Batch normalize ─────────────────────────────────────────────────
# Input : pts — shape (N, 3) float32
# Output: out — shape (N, 3) float32  (unit vectors)
#
# Writing a vector result back to a 2-D output array uses the established
# dot-with-basis-vector pattern (see also examples/06_triangle_normals.py).

@cuda.jit(link=links())
def normalize_kernel(pts, out):
    i = cuda.grid(1)
    if i >= pts.shape[0]:
        return
    p = Vector3f(pts[i, 0], pts[i, 1], pts[i, 2])
    u = normalized(p)
    # Extract components via dot with standard basis vectors
    e0 = Vector3f(1.0, 0.0, 0.0)
    e1 = Vector3f(0.0, 1.0, 0.0)
    e2 = Vector3f(0.0, 0.0, 1.0)
    out[i, 0] = dot(u, e0)
    out[i, 1] = dot(u, e1)
    out[i, 2] = dot(u, e2)


# ── Kernel C: Batch matrix-vector multiply ────────────────────────────────────
# Input : mats — shape (N, 3, 3) float32  (one matrix per element)
#         vecs — shape (N, 3)    float32
# Output: out  — shape (N, 3)   float32
#
# numpy stores (N, 3, 3) in row-major order: mats[i, row, col].
# Eigen's Matrix3f constructor is column-major:
#   Matrix3f(col0row0, col0row1, col0row2,
#            col1row0, col1row1, col1row2,
#            col2row0, col2row1, col2row2)
# So to read column 0 we hold col=0 and vary row: mats[i, 0, 0], mats[i, 1, 0], mats[i, 2, 0].

@cuda.jit(link=links())
def matvec_kernel(mats, vecs, out):
    i = cuda.grid(1)
    if i >= vecs.shape[0]:
        return
    M = Matrix3f(
        mats[i, 0, 0], mats[i, 1, 0], mats[i, 2, 0],   # column 0
        mats[i, 0, 1], mats[i, 1, 1], mats[i, 2, 1],   # column 1
        mats[i, 0, 2], mats[i, 1, 2], mats[i, 2, 2],   # column 2
    )
    v      = Vector3f(vecs[i, 0], vecs[i, 1], vecs[i, 2])
    result = M @ v
    e0     = Vector3f(1.0, 0.0, 0.0)
    e1     = Vector3f(0.0, 1.0, 0.0)
    e2     = Vector3f(0.0, 0.0, 1.0)
    out[i, 0] = dot(result, e0)
    out[i, 1] = dot(result, e1)
    out[i, 2] = dot(result, e2)


# ── Host setup ────────────────────────────────────────────────────────────────

N   = 1024
rng = np.random.default_rng(42)
TPB = 256
blocks = (N + TPB - 1) // TPB

# Kernel A — AoS inputs passed directly
a_pts   = rng.standard_normal((N, 3)).astype(np.float32)
b_pts   = rng.standard_normal((N, 3)).astype(np.float32)
dot_out = np.zeros(N, dtype=np.float32)
dot_kernel[blocks, TPB](a_pts, b_pts, dot_out)

# Kernel B — AoS input, AoS output
raw_pts  = rng.standard_normal((N, 3)).astype(np.float32)
norm_out = np.zeros((N, 3), dtype=np.float32)
normalize_kernel[blocks, TPB](raw_pts, norm_out)

# Kernel C — 3-D AoS input
raw_mats = rng.standard_normal((N, 3, 3)).astype(np.float32)
mv_vecs  = rng.standard_normal((N, 3)).astype(np.float32)
mv_out   = np.zeros((N, 3), dtype=np.float32)
matvec_kernel[blocks, TPB](raw_mats, mv_vecs, mv_out)

# ── numpy_utils round-trip demo ───────────────────────────────────────────────
# vec_to_soa / soa_to_vec convert between AoS and SoA layouts on the host.
# Useful when you have AoS data but need to call an existing SoA kernel.

px, py, pz = vec_to_soa(raw_pts)              # (N,3) → three (N,) arrays
pts_roundtrip = soa_to_vec(px, py, pz)        # three (N,) arrays → (N,3)
assert np.allclose(raw_pts, pts_roundtrip), "vec_to_soa / soa_to_vec round-trip failed"

# mat_to_soa / soa_to_mat for 3-D arrays (column-major order)
cols = mat_to_soa(raw_mats)                   # (N,3,3) → nine (N,) arrays
mats_roundtrip = soa_to_mat(3, 3, *cols)      # nine (N,) arrays → (N,3,3)
assert np.allclose(raw_mats, mats_roundtrip), "mat_to_soa / soa_to_mat round-trip failed"

# ── Verify against numpy references ──────────────────────────────────────────

# Kernel A
expected_dots = np.einsum("ni,ni->n", a_pts, b_pts)
ok_a = np.allclose(dot_out, expected_dots, rtol=1e-5)

# Kernel B
inv_norms     = 1.0 / np.linalg.norm(raw_pts, axis=1, keepdims=True)
expected_norm = raw_pts * inv_norms
ok_b = np.allclose(norm_out, expected_norm, atol=1e-5)

# Kernel C
expected_mv = np.einsum("nij,nj->ni", raw_mats, mv_vecs)
ok_c = np.allclose(mv_out, expected_mv, rtol=1e-4)

print(f"Kernel A — Batch dot products ({N} pairs)")
print(f"  {'PASS' if ok_a else 'FAIL'}  max err: {np.max(np.abs(dot_out - expected_dots)):.2e}")

print(f"Kernel B — Batch normalize ({N} vectors)")
print(f"  {'PASS' if ok_b else 'FAIL'}  max err: {np.max(np.abs(norm_out - expected_norm)):.2e}")

print(f"Kernel C — Batch mat-vec multiply ({N} systems)")
print(f"  {'PASS' if ok_c else 'FAIL'}  max err: {np.max(np.abs(mv_out - expected_mv)):.2e}")

print(f"\nnumpy_utils round-trips (vec, mat): PASS")

all_pass = ok_a and ok_b and ok_c
print(f"\n{'All passed.' if all_pass else 'FAILURES detected.'}")
if not all_pass:
    raise AssertionError("One or more kernels produced incorrect results")
