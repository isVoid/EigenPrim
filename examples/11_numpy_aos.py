"""
eigenprim: numpy packed-struct arrays and element-wise kernels.

This example shows two patterns for integrating eigenprim with numpy at
a higher level than raw Structure-of-Arrays (SoA).

─────────────────────────────────────────────────────────────────────────────
Pattern A — Structured dtype + unboxing wrapper
─────────────────────────────────────────────────────────────────────────────
numpy's structured dtype creates a true array-of-structs in memory, one
packed record per element:

    vec3f_dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    pts = np.empty(N, dtype=vec3f_dtype)
    pts['x'] = rng.standard_normal(N).astype(np.float32)
    ...

A Python **wrapper** is the public API.  It accepts the struct array,
unboxes the field arrays (``unbox_vec3f``), calls the CUDA kernel with
contiguous per-component arrays, and re-packs the output into a struct
array (``box_vec3f``).  The caller never sees SoA:

    result = normalize(pts)          # in:  vec3f struct array
                                     # out: vec3f struct array

─────────────────────────────────────────────────────────────────────────────
Pattern B — Element-wise device function (ufunc-style)
─────────────────────────────────────────────────────────────────────────────
For complex per-element logic, separate the computation from the iteration:

    @cuda.jit(device=True)                 # element kernel — no grid index,
    def normalize_element(px, py, pz):     # no bounds check, pure logic
        p = Vector3f(px, py, pz)
        u = normalized(p)
        ...
        return ux, uy, uz

    @cuda.jit(link=links())                # iteration scaffolding only
    def _normalize_batch(px, py, pz, ox, oy, oz):
        i = cuda.grid(1)
        if i >= px.shape[0]: return
        ox[i], oy[i], oz[i] = normalize_element(px[i], py[i], pz[i])

The element function is the unit of user logic; the batch kernel is
boilerplate.  The Python wrapper composes them with structured I/O.

Run:  pixi run python examples/11_numpy_aos.py
"""

import numpy as np
from numba import cuda

from eigenprim import Vector3f, Matrix3f, dot, normalized, links
from eigenprim.numpy_utils import (
    vec3f_dtype, mat3f_dtype,
    unbox_vec3f, box_vec3f,
    unbox_mat3f,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern A — Structured dtype + unboxing wrapper
#
# Kernel: normalize a batch of Vec3f.  Receives and returns contiguous SoA
# arrays — the structured-dtype interface is handled entirely by the wrapper.
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(link=links())
def _normalize_kernel(px, py, pz, ox, oy, oz):
    i = cuda.grid(1)
    if i >= px.shape[0]:
        return
    p = Vector3f(px[i], py[i], pz[i])
    u = normalized(p)
    e0 = Vector3f(1.0, 0.0, 0.0)
    e1 = Vector3f(0.0, 1.0, 0.0)
    e2 = Vector3f(0.0, 0.0, 1.0)
    ox[i] = dot(u, e0)
    oy[i] = dot(u, e1)
    oz[i] = dot(u, e2)


def normalize(pts):
    """Normalize an array of Vec3f structs.

    Parameters
    ----------
    pts : np.ndarray with dtype vec3f_dtype, shape (N,)
        Input vectors as a packed struct array.

    Returns
    -------
    np.ndarray with dtype vec3f_dtype, shape (N,)
        Unit vectors, packed into a struct array.
    """
    assert pts.dtype == vec3f_dtype, f"expected vec3f_dtype, got {pts.dtype}"
    N = len(pts)
    TPB = 256

    # Unbox: extract contiguous per-component arrays from the struct array.
    # (Struct field views are strided; np.ascontiguousarray packs each field.)
    px, py, pz = unbox_vec3f(pts)

    ox = np.empty(N, dtype=np.float32)
    oy = np.empty(N, dtype=np.float32)
    oz = np.empty(N, dtype=np.float32)

    _normalize_kernel[(N + TPB - 1) // TPB, TPB](px, py, pz, ox, oy, oz)

    # Box: pack the output components back into a struct array.
    return box_vec3f(ox, oy, oz)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern B — Element-wise device function (ufunc-style)
#
# The element function holds all per-element logic; the batch kernel is
# pure iteration scaffolding.  The Python wrapper provides structured I/O.
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _dot_element(ax, ay, az, bx, by, bz):
    """Dot product of two Vec3f, expressed as scalar components.

    This is the element-level logic — no thread index, no bounds check.
    It is called once per element by the batch kernel below.
    """
    a = Vector3f(ax, ay, az)
    b = Vector3f(bx, by, bz)
    return dot(a, b)


@cuda.jit(link=links())
def _dot_batch(ax, ay, az, bx, by, bz, out):
    """Iteration scaffolding: call _dot_element for each thread."""
    i = cuda.grid(1)
    if i >= ax.shape[0]:
        return
    out[i] = _dot_element(ax[i], ay[i], az[i], bx[i], by[i], bz[i])


def pairwise_dot(a, b):
    """Compute dot(a[i], b[i]) for each element.

    Parameters
    ----------
    a, b : np.ndarray with dtype vec3f_dtype, shape (N,)

    Returns
    -------
    np.ndarray, shape (N,), dtype float32
    """
    assert a.dtype == vec3f_dtype and b.dtype == vec3f_dtype
    N = len(a)
    TPB = 256

    ax, ay, az = unbox_vec3f(a)
    bx, by, bz = unbox_vec3f(b)
    out = np.empty(N, dtype=np.float32)

    _dot_batch[(N + TPB - 1) // TPB, TPB](ax, ay, az, bx, by, bz, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pattern B extended — element-wise mat-vec multiply
# ─────────────────────────────────────────────────────────────────────────────

@cuda.jit(device=True)
def _matvec_element(m00, m10, m20, m01, m11, m21, m02, m12, m22,
                    vx, vy, vz):
    """Matrix-vector multiply for one element (column-major Eigen layout)."""
    M = Matrix3f(m00, m10, m20, m01, m11, m21, m02, m12, m22)
    v = Vector3f(vx, vy, vz)
    r = M @ v
    e0 = Vector3f(1.0, 0.0, 0.0)
    e1 = Vector3f(0.0, 1.0, 0.0)
    e2 = Vector3f(0.0, 0.0, 1.0)
    return dot(r, e0), dot(r, e1), dot(r, e2)


@cuda.jit(link=links())
def _matvec_batch(m00, m10, m20, m01, m11, m21, m02, m12, m22,
                  vx, vy, vz, ox, oy, oz):
    """Iteration scaffolding for mat-vec multiply."""
    i = cuda.grid(1)
    if i >= vx.shape[0]:
        return
    ox[i], oy[i], oz[i] = _matvec_element(
        m00[i], m10[i], m20[i], m01[i], m11[i], m21[i], m02[i], m12[i], m22[i],
        vx[i], vy[i], vz[i],
    )


def matvec(mats, vecs):
    """Multiply each matrix by the corresponding vector.

    Parameters
    ----------
    mats : np.ndarray with dtype mat3f_dtype, shape (N,)
        Packed Mat3f struct array (field names: m_ij, row i col j).
    vecs : np.ndarray with dtype vec3f_dtype, shape (N,)
        Packed Vec3f struct array.

    Returns
    -------
    np.ndarray with dtype vec3f_dtype, shape (N,)
    """
    assert mats.dtype == mat3f_dtype and vecs.dtype == vec3f_dtype
    N = len(vecs)
    TPB = 256

    # Unbox both struct arrays
    m00, m10, m20, m01, m11, m21, m02, m12, m22 = unbox_mat3f(mats)
    vx, vy, vz = unbox_vec3f(vecs)

    ox = np.empty(N, dtype=np.float32)
    oy = np.empty(N, dtype=np.float32)
    oz = np.empty(N, dtype=np.float32)

    _matvec_batch[(N + TPB - 1) // TPB, TPB](
        m00, m10, m20, m01, m11, m21, m02, m12, m22,
        vx, vy, vz, ox, oy, oz,
    )
    return box_vec3f(ox, oy, oz)


# ─────────────────────────────────────────────────────────────────────────────
# Host setup — create packed struct arrays
# ─────────────────────────────────────────────────────────────────────────────

N   = 1024
rng = np.random.default_rng(42)

# Vec3f struct arrays
def random_vec3f(n):
    pts = np.empty(n, dtype=vec3f_dtype)
    pts["x"] = rng.standard_normal(n).astype(np.float32)
    pts["y"] = rng.standard_normal(n).astype(np.float32)
    pts["z"] = rng.standard_normal(n).astype(np.float32)
    return pts

# Mat3f struct arrays (row-major field names)
def random_mat3f(n):
    mats = np.empty(n, dtype=mat3f_dtype)
    for f in mat3f_dtype.names:
        mats[f] = rng.standard_normal(n).astype(np.float32)
    return mats

a_pts  = random_vec3f(N)
b_pts  = random_vec3f(N)
raw_pts = random_vec3f(N)
raw_mats = random_mat3f(N)

# Run
norm_result = normalize(raw_pts)         # Pattern A: struct → struct
dot_result  = pairwise_dot(a_pts, b_pts) # Pattern B: struct → scalar
mv_result   = matvec(raw_mats, a_pts)    # Pattern B ext: struct × struct → struct


# ─────────────────────────────────────────────────────────────────────────────
# Verify against numpy references
# ─────────────────────────────────────────────────────────────────────────────

def vec3f_to_array(pts):
    """Convert struct array to (N, 3) for numpy reference."""
    return np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64)

def mat3f_to_array(mats):
    """Convert struct array to (N, 3, 3) for numpy reference."""
    out = np.empty((len(mats), 3, 3), dtype=np.float64)
    for i, f in enumerate(["m00","m01","m02","m10","m11","m12","m20","m21","m22"]):
        out[:, i // 3, i % 3] = mats[f]
    return out

raw_np = vec3f_to_array(raw_pts)
a_np   = vec3f_to_array(a_pts)
b_np   = vec3f_to_array(b_pts)
mats_np = mat3f_to_array(raw_mats)

# normalize
norms64 = np.linalg.norm(raw_np, axis=1, keepdims=True)
expected_norm = (raw_np / norms64).astype(np.float32)
got_norm = np.stack([norm_result["x"], norm_result["y"], norm_result["z"]], axis=1)
ok_a = np.allclose(got_norm, expected_norm, atol=1e-5)

# pairwise dot
expected_dot = np.einsum("ni,ni->n", a_np, b_np).astype(np.float32)
ok_b = np.allclose(dot_result, expected_dot, rtol=1e-5)

# matvec
expected_mv = np.einsum("nij,nj->ni", mats_np, a_np).astype(np.float32)
got_mv = np.stack([mv_result["x"], mv_result["y"], mv_result["z"]], axis=1)
ok_c = np.allclose(got_mv, expected_mv, rtol=1e-4)

print("Pattern A — Structured dtype + unboxing wrapper")
print(f"  normalize({N} Vec3f structs) → Vec3f struct array")
print(f"  {'PASS' if ok_a else 'FAIL'}  max err: {np.max(np.abs(got_norm - expected_norm)):.2e}")

print()
print("Pattern B — Element-wise device function (ufunc-style)")
print(f"  pairwise_dot({N} Vec3f pairs) → float32 array")
print(f"  {'PASS' if ok_b else 'FAIL'}  max err: {np.max(np.abs(dot_result - expected_dot)):.2e}")
print()
print(f"  matvec({N} Mat3f × Vec3f) → Vec3f struct array")
print(f"  {'PASS' if ok_c else 'FAIL'}  max err: {np.max(np.abs(got_mv - expected_mv)):.2e}")

all_pass = ok_a and ok_b and ok_c
print(f"\n{'All passed.' if all_pass else 'FAILURES detected.'}")
if not all_pass:
    raise AssertionError("One or more kernels produced incorrect results")
