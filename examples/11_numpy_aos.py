"""
eigenprim: numpy packed-struct arrays and element-wise kernels.

This example shows two patterns for integrating eigenprim with numpy at
a higher level than raw Structure-of-Arrays (SoA).

─────────────────────────────────────────────────────────────────────────────
Pattern A — Structured dtype + zero-copy view
─────────────────────────────────────────────────────────────────────────────
numpy's structured dtype creates a true array-of-structs in memory:

    vec3f_dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    pts = np.empty(N, dtype=vec3f_dtype)

Because the struct fields are all the same scalar type, ``as_floats`` can
reinterpret the packed memory as a C-contiguous ``(N, 3)`` float32 array
with **zero copies**:

    pts_f = as_floats(pts)   # same data pointer, shape (N, 3)

The kernel receives the 2-D float view and unpacks each row into a Vector3f:

    @cuda.jit(link=links())
    def _normalize_kernel(pts, out):   # (N, 3) float32 views
        i = cuda.grid(1)
        p = Vector3f(pts[i, 0], pts[i, 1], pts[i, 2])
        ...

The Python wrapper is three lines: allocate output struct, call kernel,
return struct.  The ``out`` struct array is written through automatically
because ``as_floats(out)`` is a view of the same memory.

─────────────────────────────────────────────────────────────────────────────
Pattern B — Element-wise device function (ufunc-style)
─────────────────────────────────────────────────────────────────────────────
Separate per-element logic from iteration scaffolding:

    @cuda.jit(device=True)             # element kernel — pure logic
    def _normalize_element(px, py, pz):
        p = Vector3f(px, py, pz)
        ...
        return ux, uy, uz

    @cuda.jit(link=links())            # batch kernel — iteration only
    def _normalize_batch(pts, out):
        i = cuda.grid(1)
        if i >= pts.shape[0]: return
        out[i, 0], out[i, 1], out[i, 2] = _normalize_element(
            pts[i, 0], pts[i, 1], pts[i, 2])

Run:  pixi run python examples/11_numpy_aos.py
"""

import numpy as np
from numba import cuda

from eigenprim import Vector3f, Matrix3f, dot, normalized, links
from eigenprim.numpy_utils import (
    vec3f_dtype, mat3f_dtype,
    as_floats,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern A — Structured dtype + zero-copy view
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(link=links())
def _normalize_kernel(pts, out):
    """Normalize each row of pts, write unit vector into out.

    pts, out : (N, 3) float32 — zero-copy views of vec3f_dtype struct arrays.
    """
    i = cuda.grid(1)
    if i >= pts.shape[0]:
        return
    p = Vector3f(pts[i, 0], pts[i, 1], pts[i, 2])
    u = normalized(p)
    e0 = Vector3f(1.0, 0.0, 0.0)
    e1 = Vector3f(0.0, 1.0, 0.0)
    e2 = Vector3f(0.0, 0.0, 1.0)
    out[i, 0] = dot(u, e0)
    out[i, 1] = dot(u, e1)
    out[i, 2] = dot(u, e2)


def normalize(pts):
    """Normalize a Vec3f struct array.  Returns a new Vec3f struct array."""
    assert pts.dtype == vec3f_dtype
    N = len(pts)
    out = np.empty(N, dtype=vec3f_dtype)
    TPB = 256
    # as_floats: zero-copy (N,3) view — writes go straight through to out
    _normalize_kernel[(N + TPB - 1) // TPB, TPB](as_floats(pts), as_floats(out))
    return out


@cuda.jit(link=links())
def _dot_kernel(a, b, out):
    """Dot product of paired rows.  a, b : (N, 3) float32."""
    i = cuda.grid(1)
    if i >= a.shape[0]:
        return
    va = Vector3f(a[i, 0], a[i, 1], a[i, 2])
    vb = Vector3f(b[i, 0], b[i, 1], b[i, 2])
    out[i] = dot(va, vb)


def pairwise_dot(a, b):
    """Dot product of each pair of Vec3f structs.  Returns float32 (N,) array."""
    assert a.dtype == vec3f_dtype and b.dtype == vec3f_dtype
    N = len(a)
    out = np.empty(N, dtype=np.float32)
    TPB = 256
    _dot_kernel[(N + TPB - 1) // TPB, TPB](as_floats(a), as_floats(b), out)
    return out


@cuda.jit(link=links())
def _matvec_kernel(mats, vecs, out):
    """Mat-vec multiply.  mats: (N,9), vecs: (N,3), out: (N,3).

    mats columns are in Eigen column-major order (as produced by as_floats on
    a mat3f_dtype array whose fields are laid out: m00,m01,m02,m10,m11,m12,...).
    The wrapper calls as_floats then passes the reordered columns from unbox_mat3f.
    """
    i = cuda.grid(1)
    if i >= vecs.shape[0]:
        return
    # mats row i has 9 values: c0r0,c0r1,c0r2, c1r0,c1r1,c1r2, c2r0,c2r1,c2r2
    M = Matrix3f(
        mats[i, 0], mats[i, 1], mats[i, 2],   # column 0
        mats[i, 3], mats[i, 4], mats[i, 5],   # column 1
        mats[i, 6], mats[i, 7], mats[i, 8],   # column 2
    )
    v      = Vector3f(vecs[i, 0], vecs[i, 1], vecs[i, 2])
    result = M @ v
    e0     = Vector3f(1.0, 0.0, 0.0)
    e1     = Vector3f(0.0, 1.0, 0.0)
    e2     = Vector3f(0.0, 0.0, 1.0)
    out[i, 0] = dot(result, e0)
    out[i, 1] = dot(result, e1)
    out[i, 2] = dot(result, e2)


def matvec(mats, vecs):
    """Multiply each Mat3f by the corresponding Vec3f.  Returns Vec3f struct array."""
    assert mats.dtype == mat3f_dtype and vecs.dtype == vec3f_dtype
    N = len(vecs)
    out = np.empty(N, dtype=vec3f_dtype)
    TPB = 256
    # mat3f_dtype field order is row-major (m00,m01,m02,m10,...) but Eigen
    # needs column-major.  Reorder into a fresh (N,9) contiguous array.
    from eigenprim.numpy_utils import unbox_mat3f
    cols = unbox_mat3f(mats)   # tuple of 9 (N,) arrays in column-major order
    mats_cm = np.stack(cols, axis=1)   # (N, 9) column-major
    _matvec_kernel[(N + TPB - 1) // TPB, TPB](mats_cm, as_floats(vecs), as_floats(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pattern B — Element-wise device function (ufunc-style)
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _normalize_element(px, py, pz):
    """Normalize one vector.  Pure per-element logic — no grid index, no bounds.

    This is the unit of user logic.  The batch kernel below is boilerplate.
    """
    p = Vector3f(px, py, pz)
    u = normalized(p)
    e0 = Vector3f(1.0, 0.0, 0.0)
    e1 = Vector3f(0.0, 1.0, 0.0)
    e2 = Vector3f(0.0, 0.0, 1.0)
    return dot(u, e0), dot(u, e1), dot(u, e2)


@cuda.jit(link=links())
def _normalize_batch(pts, out):
    """Iteration scaffolding: dispatch _normalize_element for each thread."""
    i = cuda.grid(1)
    if i >= pts.shape[0]:
        return
    out[i, 0], out[i, 1], out[i, 2] = _normalize_element(
        pts[i, 0], pts[i, 1], pts[i, 2]
    )


def normalize_ufunc(pts):
    """Same as normalize() but using the element-wise device function."""
    assert pts.dtype == vec3f_dtype
    N = len(pts)
    out = np.empty(N, dtype=vec3f_dtype)
    TPB = 256
    _normalize_batch[(N + TPB - 1) // TPB, TPB](as_floats(pts), as_floats(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Host setup
# ─────────────────────────────────────────────────────────────────────────────

N   = 1024
rng = np.random.default_rng(42)

def random_vec3f(n):
    pts = np.empty(n, dtype=vec3f_dtype)
    pts["x"] = rng.standard_normal(n).astype(np.float32)
    pts["y"] = rng.standard_normal(n).astype(np.float32)
    pts["z"] = rng.standard_normal(n).astype(np.float32)
    return pts

def random_mat3f(n):
    mats = np.empty(n, dtype=mat3f_dtype)
    for f in mat3f_dtype.names:
        mats[f] = rng.standard_normal(n).astype(np.float32)
    return mats

raw_pts  = random_vec3f(N)
a_pts    = random_vec3f(N)
b_pts    = random_vec3f(N)
raw_mats = random_mat3f(N)

norm_result  = normalize(raw_pts)         # Pattern A
dot_result   = pairwise_dot(a_pts, b_pts)
mv_result    = matvec(raw_mats, a_pts)
norm_result2 = normalize_ufunc(raw_pts)  # Pattern B


# ─────────────────────────────────────────────────────────────────────────────
# Verify
# ─────────────────────────────────────────────────────────────────────────────

def to_f64(pts):
    return np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64)

def mat_to_f64(mats):
    out = np.empty((len(mats), 3, 3), dtype=np.float64)
    for i, f in enumerate(["m00","m01","m02","m10","m11","m12","m20","m21","m22"]):
        out[:, i // 3, i % 3] = mats[f]
    return out

raw_np  = to_f64(raw_pts)
a_np    = to_f64(a_pts)
b_np    = to_f64(b_pts)
mats_np = mat_to_f64(raw_mats)

expected_norm = (raw_np / np.linalg.norm(raw_np, axis=1, keepdims=True)).astype(np.float32)
expected_dot  = np.einsum("ni,ni->n", a_np, b_np).astype(np.float32)
expected_mv   = np.einsum("nij,nj->ni", mats_np, a_np).astype(np.float32)

got_norm  = as_floats(norm_result)
got_norm2 = as_floats(norm_result2)
got_mv    = as_floats(mv_result)

ok_a  = np.allclose(got_norm,  expected_norm, atol=1e-5)
ok_b  = np.allclose(dot_result, expected_dot, rtol=1e-5)
ok_c  = np.allclose(got_mv,    expected_mv,  rtol=1e-4)
ok_a2 = np.allclose(got_norm2, expected_norm, atol=1e-5)

print("Pattern A — Structured dtype + as_floats (zero-copy view)")
print(f"  normalize   {'PASS' if ok_a  else 'FAIL'}  max err: {np.max(np.abs(got_norm  - expected_norm)):.2e}")
print(f"  pairwise_dot{'PASS' if ok_b  else 'FAIL'}  max err: {np.max(np.abs(dot_result - expected_dot)):.2e}")
print(f"  matvec      {'PASS' if ok_c  else 'FAIL'}  max err: {np.max(np.abs(got_mv    - expected_mv )):.2e}")
print()
print("Pattern B — Element-wise device function (ufunc-style)")
print(f"  normalize   {'PASS' if ok_a2 else 'FAIL'}  max err: {np.max(np.abs(got_norm2 - expected_norm)):.2e}")

all_pass = ok_a and ok_b and ok_c and ok_a2
print(f"\n{'All passed.' if all_pass else 'FAILURES detected.'}")
if not all_pass:
    raise AssertionError("One or more kernels produced incorrect results")
