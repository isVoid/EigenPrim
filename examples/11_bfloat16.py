"""
Eigenprim: Bfloat16 vector and matrix operations.

Demonstrates Vector3bf and Matrix3bf. Bfloat16 has the same exponent
range as float32 (8 bits) but only 7 bits of mantissa, making it
popular for ML training where dynamic range matters more than precision.

Demonstrates: Vector3bf, Matrix3bf, dot, norm, inverse, determinant, @.

Run:  pixi run python examples/11_bfloat16.py
"""

import numpy as np
from numba import cuda
from numba.cuda import types as cuda_types

from eigenprim import (
    Vector3bf, Matrix3bf,
    eigen_vec3bf_dot, eigen_vec3bf_norm,
    eigen_mat3bf_determinant, eigen_mat3bf_inverse,
    links,
)

bf = cuda_types.bfloat16


@cuda.jit(link=links())
def kernel(out):
    a = Vector3bf(bf(1.0), bf(2.0), bf(3.0))
    b = Vector3bf(bf(4.0), bf(5.0), bf(6.0))

    # Vector ops
    out[0] = eigen_vec3bf_dot(a, b)                            # 32
    out[1] = eigen_vec3bf_norm(Vector3bf(bf(3.0), bf(4.0), bf(0.0)))  # 5

    # Operator syntax
    c = a + b
    out[2] = eigen_vec3bf_dot(c, Vector3bf(bf(1.0), bf(1.0), bf(1.0)))  # 21

    # Matrix ops
    D = Matrix3bf(bf(2.0), bf(0.0), bf(0.0),
                  bf(0.0), bf(3.0), bf(0.0),
                  bf(0.0), bf(0.0), bf(4.0))
    out[3] = eigen_mat3bf_determinant(D)                        # 24

    # Solve via inverse
    x = eigen_mat3bf_inverse(D) @ a
    out[4] = eigen_vec3bf_dot(x, Vector3bf(bf(1.0), bf(1.0), bf(1.0)))  # ≈ 1.917


# bfloat16 arrays need float32 on host (numpy has no native bf16)
out = np.zeros(5, dtype=np.float32)
kernel[1, 1](out)

print("Bfloat16 results:")
all_pass = True
for label, got, expected, tol in [
    ("dot((1,2,3),(4,5,6))", out[0], 32.0, 0.02),
    ("norm((3,4,0))", out[1], 5.0, 0.02),
    ("(1,2,3)+(4,5,6) dot (1,1,1)", out[2], 21.0, 0.02),
    ("det(diag(2,3,4))", out[3], 24.0, 0.02),
    ("sum(inv(diag)*a)", out[4], 1.0/2 + 2.0/3 + 3.0/4, 0.05),
]:
    ok = np.isclose(float(got), expected, rtol=tol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} = {got:.4f}  (expected {expected:.4f})")
    all_pass &= ok

print(f"\n{'All passed.' if all_pass else 'FAILURES!'}")
if not all_pass:
    raise AssertionError("Example failures")
