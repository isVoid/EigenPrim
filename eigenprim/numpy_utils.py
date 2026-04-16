"""eigenprim.numpy_utils — numpy structured dtypes and AoS ↔ SoA helpers.

**Structured dtype arrays** are the natural numpy representation of an array
of packed C structs — one struct per element, all components stored together
in memory.  eigenprim provides a dtype for each bound type:

    vec3f_dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    pts = np.empty(N, dtype=vec3f_dtype)   # N packed Vec3f structs
    pts['x'] = ...                         # field view (strided)

These are **not** passed directly to ``@cuda.jit`` kernels.  Instead, a
Python wrapper function **unboxes** the struct fields into contiguous 1-D
arrays before calling the kernel, then **boxes** the output components back
into a struct array before returning to the caller::

    def normalize(pts: np.ndarray) -> np.ndarray:
        \"\"\"Normalize a Vec3f struct array; returns a Vec3f struct array.\"\"\"
        px, py, pz = unbox_vec3f(pts)           # extract contiguous fields
        ox = np.empty(len(pts), dtype=np.float32)
        oy = np.empty(len(pts), dtype=np.float32)
        oz = np.empty(len(pts), dtype=np.float32)
        _normalize_kernel[...](px, py, pz, ox, oy, oz)
        return box_vec3f(ox, oy, oz)            # pack result into struct array

This module also keeps the AoS ↔ SoA helpers (``vec_to_soa`` / ``soa_to_vec``,
``mat_to_soa`` / ``soa_to_mat``) for interoperating with SoA-style kernels.

None of the functions here are callable inside ``@cuda.jit`` kernels.
"""

import numpy as np

__all__ = [
    # Structured dtypes — float32 vectors
    "vec2f_dtype", "vec3f_dtype", "vec4f_dtype",
    # Structured dtypes — float64 vectors
    "vec2d_dtype", "vec3d_dtype", "vec4d_dtype",
    # Structured dtypes — float32 matrices (row-major field names, m_ij: row i, col j)
    "mat2f_dtype", "mat3f_dtype", "mat4f_dtype",
    # Structured dtypes — float64 matrices
    "mat2d_dtype", "mat3d_dtype", "mat4d_dtype",
    # Unbox/box helpers for float32 vectors
    "unbox_vec2f", "box_vec2f",
    "unbox_vec3f", "box_vec3f",
    "unbox_vec4f", "box_vec4f",
    # Unbox/box helpers for float64 vectors
    "unbox_vec2d", "box_vec2d",
    "unbox_vec3d", "box_vec3d",
    "unbox_vec4d", "box_vec4d",
    # Unbox/box helpers for float32 matrices
    "unbox_mat3f", "box_mat3f",
    # AoS ↔ SoA conversion (plain 2-D / 3-D arrays)
    "vec_to_soa", "soa_to_vec",
    "mat_to_soa", "soa_to_mat",
]


# ── Structured dtype definitions ──────────────────────────────────────────────
# Each dtype matches the memory layout of a packed C struct.
# Field names for vectors: x, y, z, w
# Field names for matrices: m_ij (i = row, j = col, 0-indexed)

vec2f_dtype = np.dtype([("x", "f4"), ("y", "f4")])
vec3f_dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
vec4f_dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("w", "f4")])

vec2d_dtype = np.dtype([("x", "f8"), ("y", "f8")])
vec3d_dtype = np.dtype([("x", "f8"), ("y", "f8"), ("z", "f8")])
vec4d_dtype = np.dtype([("x", "f8"), ("y", "f8"), ("z", "f8"), ("w", "f8")])

mat2f_dtype = np.dtype([
    ("m00", "f4"), ("m01", "f4"),
    ("m10", "f4"), ("m11", "f4"),
])
mat3f_dtype = np.dtype([
    ("m00", "f4"), ("m01", "f4"), ("m02", "f4"),
    ("m10", "f4"), ("m11", "f4"), ("m12", "f4"),
    ("m20", "f4"), ("m21", "f4"), ("m22", "f4"),
])
mat4f_dtype = np.dtype([
    ("m00", "f4"), ("m01", "f4"), ("m02", "f4"), ("m03", "f4"),
    ("m10", "f4"), ("m11", "f4"), ("m12", "f4"), ("m13", "f4"),
    ("m20", "f4"), ("m21", "f4"), ("m22", "f4"), ("m23", "f4"),
    ("m30", "f4"), ("m31", "f4"), ("m32", "f4"), ("m33", "f4"),
])

mat2d_dtype = np.dtype([
    ("m00", "f8"), ("m01", "f8"),
    ("m10", "f8"), ("m11", "f8"),
])
mat3d_dtype = np.dtype([
    ("m00", "f8"), ("m01", "f8"), ("m02", "f8"),
    ("m10", "f8"), ("m11", "f8"), ("m12", "f8"),
    ("m20", "f8"), ("m21", "f8"), ("m22", "f8"),
])
mat4d_dtype = np.dtype([
    ("m00", "f8"), ("m01", "f8"), ("m02", "f8"), ("m03", "f8"),
    ("m10", "f8"), ("m11", "f8"), ("m12", "f8"), ("m13", "f8"),
    ("m20", "f8"), ("m21", "f8"), ("m22", "f8"), ("m23", "f8"),
    ("m30", "f8"), ("m31", "f8"), ("m32", "f8"), ("m33", "f8"),
])


# ── Unbox / box helpers ───────────────────────────────────────────────────────
# "Unbox": extract contiguous (N,) field arrays from a struct array.
#   Field views on a structured array are strided and not GPU-friendly;
#   np.ascontiguousarray produces packed copies suitable for cuda.jit kernels.
#
# "Box": pack per-component (N,) output arrays back into a struct array.
#   This is a cheap assignment into the pre-allocated struct array fields.

def unbox_vec2f(pts):
    """Unbox a vec2f struct array into two contiguous (N,) float32 arrays."""
    return np.ascontiguousarray(pts["x"]), np.ascontiguousarray(pts["y"])


def box_vec2f(x, y):
    """Pack two (N,) float32 arrays into a vec2f struct array."""
    out = np.empty(len(x), dtype=vec2f_dtype)
    out["x"], out["y"] = x, y
    return out


def unbox_vec3f(pts):
    """Unbox a vec3f struct array into three contiguous (N,) float32 arrays."""
    return (np.ascontiguousarray(pts["x"]),
            np.ascontiguousarray(pts["y"]),
            np.ascontiguousarray(pts["z"]))


def box_vec3f(x, y, z):
    """Pack three (N,) float32 arrays into a vec3f struct array."""
    out = np.empty(len(x), dtype=vec3f_dtype)
    out["x"], out["y"], out["z"] = x, y, z
    return out


def unbox_vec4f(pts):
    """Unbox a vec4f struct array into four contiguous (N,) float32 arrays."""
    return (np.ascontiguousarray(pts["x"]),
            np.ascontiguousarray(pts["y"]),
            np.ascontiguousarray(pts["z"]),
            np.ascontiguousarray(pts["w"]))


def box_vec4f(x, y, z, w):
    """Pack four (N,) float32 arrays into a vec4f struct array."""
    out = np.empty(len(x), dtype=vec4f_dtype)
    out["x"], out["y"], out["z"], out["w"] = x, y, z, w
    return out


def unbox_vec2d(pts):
    """Unbox a vec2d struct array into two contiguous (N,) float64 arrays."""
    return np.ascontiguousarray(pts["x"]), np.ascontiguousarray(pts["y"])


def box_vec2d(x, y):
    """Pack two (N,) float64 arrays into a vec2d struct array."""
    out = np.empty(len(x), dtype=vec2d_dtype)
    out["x"], out["y"] = x, y
    return out


def unbox_vec3d(pts):
    """Unbox a vec3d struct array into three contiguous (N,) float64 arrays."""
    return (np.ascontiguousarray(pts["x"]),
            np.ascontiguousarray(pts["y"]),
            np.ascontiguousarray(pts["z"]))


def box_vec3d(x, y, z):
    """Pack three (N,) float64 arrays into a vec3d struct array."""
    out = np.empty(len(x), dtype=vec3d_dtype)
    out["x"], out["y"], out["z"] = x, y, z
    return out


def unbox_vec4d(pts):
    """Unbox a vec4d struct array into four contiguous (N,) float64 arrays."""
    return (np.ascontiguousarray(pts["x"]),
            np.ascontiguousarray(pts["y"]),
            np.ascontiguousarray(pts["z"]),
            np.ascontiguousarray(pts["w"]))


def box_vec4d(x, y, z, w):
    """Pack four (N,) float64 arrays into a vec4d struct array."""
    out = np.empty(len(x), dtype=vec4d_dtype)
    out["x"], out["y"], out["z"], out["w"] = x, y, z, w
    return out


def unbox_mat3f(mats):
    """Unbox a mat3f struct array into nine contiguous (N,) float32 arrays.

    Returns arrays in **column-major** order matching the Eigen Matrix3f
    constructor:  col0row0, col0row1, col0row2, col1row0, col1row1, col1row2,
    col2row0, col2row1, col2row2.

    Field names use row-major notation (m_ij: row i, col j), so:
        col 0 = m00, m10, m20
        col 1 = m01, m11, m21
        col 2 = m02, m12, m22
    """
    return (
        np.ascontiguousarray(mats["m00"]), np.ascontiguousarray(mats["m10"]),
        np.ascontiguousarray(mats["m20"]),
        np.ascontiguousarray(mats["m01"]), np.ascontiguousarray(mats["m11"]),
        np.ascontiguousarray(mats["m21"]),
        np.ascontiguousarray(mats["m02"]), np.ascontiguousarray(mats["m12"]),
        np.ascontiguousarray(mats["m22"]),
    )


def box_mat3f(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2):
    """Pack nine (N,) float32 arrays (column-major order) into a mat3f struct array."""
    out = np.empty(len(c0r0), dtype=mat3f_dtype)
    out["m00"], out["m10"], out["m20"] = c0r0, c0r1, c0r2
    out["m01"], out["m11"], out["m21"] = c1r0, c1r1, c1r2
    out["m02"], out["m12"], out["m22"] = c2r0, c2r1, c2r2
    return out


# ── AoS ↔ SoA helpers (plain 2-D / 3-D arrays) ───────────────────────────────

def vec_to_soa(arr):
    """Convert an ``(N, k)`` array into *k* contiguous ``(N,)`` arrays."""
    if arr.ndim != 2:
        raise ValueError(
            f"vec_to_soa expects a 2-D array of shape (N, k); got shape {arr.shape}"
        )
    return tuple(np.ascontiguousarray(arr[:, j]) for j in range(arr.shape[1]))


def soa_to_vec(*arrays):
    """Convert *k* ``(N,)`` arrays into a single contiguous ``(N, k)`` array."""
    if not arrays:
        raise ValueError("soa_to_vec requires at least one array")
    return np.ascontiguousarray(np.stack(arrays, axis=1))


def mat_to_soa(arr):
    """Convert an ``(N, R, C)`` array into *R×C* contiguous ``(N,)`` arrays.

    Emits in **column-major** order matching the Eigen Matrix constructor:
    col0row0, col0row1, …, col1row0, …
    """
    if arr.ndim != 3:
        raise ValueError(
            f"mat_to_soa expects a 3-D array of shape (N, R, C); got shape {arr.shape}"
        )
    _, R, C = arr.shape
    return tuple(
        np.ascontiguousarray(arr[:, row, col])
        for col in range(C)
        for row in range(R)
    )


def soa_to_mat(R, C, *arrays):
    """Convert *R×C* ``(N,)`` arrays (column-major) into a ``(N, R, C)`` array.

    Inverse of :func:`mat_to_soa`.
    """
    expected = R * C
    if len(arrays) != expected:
        raise ValueError(
            f"soa_to_mat(R={R}, C={C}) expects {expected} arrays, got {len(arrays)}"
        )
    N = arrays[0].shape[0]
    result = np.empty((N, R, C), dtype=arrays[0].dtype)
    for k, arr_k in enumerate(arrays):
        col = k // R
        row = k % R
        result[:, row, col] = arr_k
    return np.ascontiguousarray(result)
