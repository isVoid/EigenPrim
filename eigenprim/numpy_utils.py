"""eigenprim.numpy_utils — Host-side numpy helpers for AoS ↔ SoA conversion.

Array-of-Structures (AoS) and Structure-of-Arrays (SoA) are two layouts
for batched vector/matrix data:

* **SoA** (current eigenprim default): separate 1-D arrays per component::

      px, py, pz = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()
      kernel[blocks, tpb](px, py, pz, ...)

* **AoS** (numpy-natural): a single 2-D array, one row per element::

      pts = np.zeros((N, 3), dtype=np.float32)  # shape (N, k)
      kernel[blocks, tpb](pts, ...)              # index as pts[i, 0] inside kernel

These utilities convert between the two layouts on the **host (CPU)**.
They are **not callable inside** ``@cuda.jit`` **kernels**.

Typical usage::

    from eigenprim.numpy_utils import vec_to_soa, soa_to_vec
    from eigenprim.numpy_utils import mat_to_soa, soa_to_mat

    # AoS → SoA  (prepare data for a SoA-style kernel)
    pts = rng.standard_normal((N, 3)).astype(np.float32)
    px, py, pz = vec_to_soa(pts)
    soa_kernel[blocks, tpb](px, py, pz, ...)

    # SoA → AoS  (collect per-component output arrays into a 2-D result)
    out = soa_to_vec(nx, ny, nz)   # shape (N, 3)

    # AoS matrix → SoA  (column-major order matching Eigen's Matrix constructor)
    mats = rng.standard_normal((N, 3, 3)).astype(np.float32)
    cols = mat_to_soa(mats)        # 9 contiguous (N,) arrays
    # order: c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2
    soa_mat_kernel[blocks, tpb](*cols, ...)

    # SoA → AoS matrix  (inverse of mat_to_soa)
    mats_out = soa_to_mat(3, 3, *cols)   # shape (N, 3, 3)
"""

import numpy as np

__all__ = ["vec_to_soa", "soa_to_vec", "mat_to_soa", "soa_to_mat"]


def vec_to_soa(arr):
    """Convert an ``(N, k)`` array into *k* contiguous ``(N,)`` arrays.

    Parameters
    ----------
    arr : np.ndarray, shape (N, k)
        AoS vector array.  dtype is preserved.

    Returns
    -------
    tuple of k np.ndarray objects, each shape (N,) and C-contiguous.

    Examples
    --------
    >>> pts = np.zeros((1024, 3), dtype=np.float32)
    >>> px, py, pz = vec_to_soa(pts)
    >>> soa_kernel[blocks, tpb](px, py, pz, ...)
    """
    if arr.ndim != 2:
        raise ValueError(
            f"vec_to_soa expects a 2-D array of shape (N, k); got shape {arr.shape}"
        )
    return tuple(np.ascontiguousarray(arr[:, j]) for j in range(arr.shape[1]))


def soa_to_vec(*arrays):
    """Convert *k* ``(N,)`` arrays into a single contiguous ``(N, k)`` array.

    Parameters
    ----------
    *arrays : k np.ndarray objects, each shape (N,)
        All arrays must have the same dtype and length.

    Returns
    -------
    np.ndarray, shape (N, k), C-contiguous, same dtype as inputs.

    Examples
    --------
    >>> out = soa_to_vec(nx, ny, nz)   # shape (N, 3)
    """
    if not arrays:
        raise ValueError("soa_to_vec requires at least one array")
    return np.ascontiguousarray(np.stack(arrays, axis=1))


def mat_to_soa(arr):
    """Convert an ``(N, R, C)`` array into *R×C* contiguous ``(N,)`` arrays.

    Arrays are emitted in **column-major** order to match the argument order
    of Eigen's ``Matrix`` constructor::

        Matrix3f(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)

    For a 3×3 matrix the nine returned arrays correspond to:
    ``c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2``.

    Parameters
    ----------
    arr : np.ndarray, shape (N, R, C)
        AoS matrix array.  dtype is preserved.

    Returns
    -------
    tuple of R*C np.ndarray objects, each shape (N,) and C-contiguous,
    in column-major order.

    Examples
    --------
    >>> mats = np.zeros((N, 3, 3), dtype=np.float32)
    >>> c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 = mat_to_soa(mats)
    >>> soa_mat_kernel[blocks, tpb](c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2, ...)
    """
    if arr.ndim != 3:
        raise ValueError(
            f"mat_to_soa expects a 3-D array of shape (N, R, C); got shape {arr.shape}"
        )
    _, R, C = arr.shape
    # Column-major: outer loop over columns, inner loop over rows
    return tuple(
        np.ascontiguousarray(arr[:, row, col])
        for col in range(C)
        for row in range(R)
    )


def soa_to_mat(R, C, *arrays):
    """Convert *R×C* ``(N,)`` arrays into a contiguous ``(N, R, C)`` array.

    This is the exact inverse of :func:`mat_to_soa`; arrays must be provided
    in the same column-major order that :func:`mat_to_soa` produces.

    Parameters
    ----------
    R : int
        Number of rows.
    C : int
        Number of columns.
    *arrays : R*C np.ndarray objects, each shape (N,)
        Column-major order: ``c0r0, c0r1, …, c(C-1)r(R-1)``.
        All must have the same dtype and length.

    Returns
    -------
    np.ndarray, shape (N, R, C), C-contiguous, same dtype as inputs.

    Examples
    --------
    >>> mats_out = soa_to_mat(3, 3, c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
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
