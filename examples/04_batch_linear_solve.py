"""
Eigenprim: Batch linear system solve on the GPU.

Each CUDA thread solves Ax = b for a different right-hand side b,
using inverse(A) @ b. This is efficient for small fixed-size matrices
where Eigen computes the inverse via cofactor expansion.

Run:  pixi run python examples/04_batch_linear_solve.py
"""

import numpy as np
from numba import cuda

from eigenprim import Vector3f, Matrix3f, inverse, norm, links


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


# ── Host-side setup ──────────────────────────────────────────────

N = 1024
rng = np.random.default_rng(42)

# A = diag(2, 3, 4) — simple invertible matrix
A_np = np.diag([2.0, 3.0, 4.0]).astype(np.float32)

# N random right-hand sides
rhs = rng.standard_normal((N, 3)).astype(np.float32)
bx, by, bz = rhs[:, 0].copy(), rhs[:, 1].copy(), rhs[:, 2].copy()

solution_norms = np.zeros(N, dtype=np.float32)

threads = 256
blocks = (N + threads - 1) // threads
solve_kernel[blocks, threads](
    bx, by, bz,
    solution_norms,
    # A column-major: col0=(2,0,0), col1=(0,3,0), col2=(0,0,4)
    2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0,
)

# ── Verify ───────────────────────────────────────────────────────

expected_x = np.linalg.solve(A_np, rhs.T).T
expected_norms = np.linalg.norm(expected_x, axis=1)

print(f"Batch linear solve ({N} systems)")
ok = np.allclose(solution_norms, expected_norms, rtol=1e-4)
print(f"  Solution norms: {'PASS' if ok else 'FAIL'}  (max err: {np.max(np.abs(solution_norms - expected_norms)):.2e})")

print(f"\n{'All passed.' if ok else 'FAILURES!'}")
if not ok:
    raise AssertionError("Example failures")
