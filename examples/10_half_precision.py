"""
Eigenprim: Half-precision (fp16) vector and matrix operations.

Demonstrates Vector3h and Matrix3h with the same operations as float
types, using types.float16() for scalar construction. Half-precision
is useful for ML inference and memory-bandwidth-limited kernels.

Demonstrates: Vector3h, Matrix3h, dot, norm, cross, inverse, @, +.

Run:  pixi run python examples/10_half_precision.py
"""

import numpy as np
from numba import cuda, types

from eigenprim import (
    Vector3h, Matrix3h,
    eigen_vec3h_dot, eigen_vec3h_norm, eigen_vec3h_cross,
    eigen_mat3h_determinant, eigen_mat3h_inverse,
    links,
)

h = types.float16


@cuda.jit(link=links())
def kernel(out):
    a = Vector3h(h(1.0), h(2.0), h(3.0))
    b = Vector3h(h(4.0), h(5.0), h(6.0))

    # Vector ops
    out[0] = eigen_vec3h_dot(a, b)                          # 32
    out[1] = eigen_vec3h_norm(Vector3h(h(3.0), h(4.0), h(0.0)))  # 5
    cr = eigen_vec3h_cross(Vector3h(h(1.0), h(0.0), h(0.0)),
                           Vector3h(h(0.0), h(1.0), h(0.0)))
    out[2] = eigen_vec3h_norm(cr)                            # 1

    # Operator syntax works with half types too
    c = a + b
    out[3] = eigen_vec3h_dot(c, Vector3h(h(1.0), h(1.0), h(1.0)))  # 21

    # Matrix ops
    D = Matrix3h(h(2.0), h(0.0), h(0.0),
                 h(0.0), h(3.0), h(0.0),
                 h(0.0), h(0.0), h(4.0))
    out[4] = eigen_mat3h_determinant(D)                      # 24

    # Solve via inverse: x = D^{-1} * a
    x = eigen_mat3h_inverse(D) @ a
    out[5] = eigen_vec3h_dot(x, Vector3h(h(1.0), h(1.0), h(1.0)))  # 0.5+0.667+0.75 ≈ 1.917


out = np.zeros(6, dtype=np.float16)
kernel[1, 1](out)

print("Half-precision (fp16) results:")
all_pass = True
for label, got, expected, tol in [
    ("dot((1,2,3),(4,5,6))", out[0], 32.0, 0.01),
    ("norm((3,4,0))", out[1], 5.0, 0.01),
    ("norm(cross(x,y))", out[2], 1.0, 0.01),
    ("(1,2,3)+(4,5,6) dot (1,1,1)", out[3], 21.0, 0.01),
    ("det(diag(2,3,4))", out[4], 24.0, 0.01),
    ("sum(inv(diag)*a)", out[5], 1.0/2 + 2.0/3 + 3.0/4, 0.02),
]:
    ok = np.isclose(float(got), expected, rtol=tol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} = {got:.4f}  (expected {expected:.4f})")
    all_pass &= ok

print(f"\n{'All passed.' if all_pass else 'FAILURES!'}")
if not all_pass:
    raise AssertionError("Example failures")
