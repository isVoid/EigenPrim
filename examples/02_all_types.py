"""
Eigenprim: Direct Eigen::Matrix types with operator syntax.

Demonstrates Vector2f..Vector4f, Matrix2f..Matrix4f.
Uses +, -, *, @ operators where possible, named functions for the rest.

Run:  pixi run python examples/02_all_types.py
"""

import numpy as np
from numba import cuda

from eigenprim import (
    Vector2f, Vector3f, Vector4f, Matrix2f, Matrix3f, Matrix4f,
    eigen_vec2f_dot, eigen_vec2f_norm,
    eigen_vec3f_dot, eigen_vec3f_cross, eigen_vec3f_norm,
    eigen_vec4f_dot,
    eigen_mat2f_determinant, eigen_mat2f_inverse,
    eigen_mat3f_determinant, eigen_mat3f_transpose, eigen_mat3f_trace,
    eigen_mat4f_determinant, eigen_mat4f_trace,
    links,
)


@cuda.jit(link=links())
def kernel_vectors(out):
    # Vec2f
    a2 = Vector2f(3.0, 4.0)
    out[0] = eigen_vec2f_norm(a2)                                    # 5.0

    # Vec3f — operator syntax
    a3 = Vector3f(1.0, 2.0, 3.0)
    b3 = Vector3f(4.0, 5.0, 6.0)
    out[1] = eigen_vec3f_dot(a3, b3)                                 # 32.0
    cr = eigen_vec3f_cross(Vector3f(1.0, 0.0, 0.0),
                           Vector3f(0.0, 1.0, 0.0))
    out[2] = eigen_vec3f_norm(cr)                                    # 1.0
    s = a3 + b3                                                      # operator +
    out[3] = eigen_vec3f_dot(s, Vector3f(1.0, 1.0, 1.0))           # 21.0

    # Vec4f — scalar multiply with *
    a4 = Vector4f(1.0, 2.0, 3.0, 4.0)
    b4 = Vector4f(5.0, 6.0, 7.0, 8.0)
    out[4] = eigen_vec4f_dot(a4, b4)                                 # 70.0
    c4 = a4 * 2.0                                                    # operator *
    out[5] = eigen_vec4f_dot(c4, Vector4f(1.0, 1.0, 1.0, 1.0))     # 20.0


@cuda.jit(link=links())
def kernel_matrices(out):
    # Mat2f — @ for matrix-vector multiply
    D2 = Matrix2f(3.0, 0.0, 0.0, 4.0)
    out[0] = eigen_mat2f_determinant(D2)                              # 12.0
    v2 = eigen_mat2f_inverse(D2) @ Vector2f(6.0, 8.0)               # operator @
    out[1] = eigen_vec2f_dot(v2, Vector2f(1.0, 1.0))                # 4.0

    # Mat3f — @ for matrix multiply and matrix-vector multiply
    D3 = Matrix3f(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0)
    out[2] = eigen_mat3f_determinant(D3)                              # 24.0
    out[3] = eigen_vec3f_dot(
        D3 @ Vector3f(1.0, 1.0, 1.0),                               # operator @
        Vector3f(1.0, 1.0, 1.0))                                     # 9.0
    out[4] = eigen_mat3f_determinant(D3 @ D3)                       # 576.0 (@ mat*mat)
    M = Matrix3f(1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0)
    Mt = eigen_mat3f_transpose(M)
    out[5] = eigen_vec3f_dot(
        Mt @ Vector3f(1.0, 0.0, 0.0),                               # operator @
        Vector3f(1.0, 1.0, 1.0))                                     # 6.0

    # Mat4f
    D4 = Matrix4f(1.0, 0.0, 0.0, 0.0,
                   0.0, 2.0, 0.0, 0.0,
                   0.0, 0.0, 3.0, 0.0,
                   0.0, 0.0, 0.0, 4.0)
    out[6] = eigen_mat4f_determinant(D4)                              # 24.0
    out[7] = eigen_mat4f_trace(D4)                                    # 10.0


# ── Run ──────────────────────────────────────────────────────────

out_v = np.zeros(6, dtype=np.float32)
kernel_vectors[1, 1](out_v)

out_m = np.zeros(8, dtype=np.float32)
kernel_matrices[1, 1](out_m)

# ── Results ──────────────────────────────────────────────────────

all_pass = True

print("Vector results:")
for label, got, expected in [
    ("norm((3,4))", out_v[0], 5.0),
    ("dot((1,2,3),(4,5,6))", out_v[1], 32.0),
    ("norm(cross(x,y))", out_v[2], 1.0),
    ("(1,2,3)+(4,5,6) dot (1,1,1)", out_v[3], 21.0),
    ("dot4((1,2,3,4),(5,6,7,8))", out_v[4], 70.0),
    ("(1,2,3,4)*2 dot (1,1,1,1)", out_v[5], 20.0),
]:
    ok = np.isclose(got, expected, rtol=1e-5)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} = {got:.4f}")
    all_pass &= ok

print("\nMatrix results:")
for label, got, expected in [
    ("det(diag2(3,4))", out_m[0], 12.0),
    ("inv(diag2)@(6,8) dot (1,1)", out_m[1], 4.0),
    ("det(diag3(2,3,4))", out_m[2], 24.0),
    ("diag3@(1,1,1) dot (1,1,1)", out_m[3], 9.0),
    ("det(D3@D3)", out_m[4], 576.0),
    ("Mt@(1,0,0) dot (1,1,1)", out_m[5], 6.0),
    ("det(diag4(1,2,3,4))", out_m[6], 24.0),
    ("trace(diag4(1,2,3,4))", out_m[7], 10.0),
]:
    ok = np.isclose(got, expected, rtol=1e-4)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} = {got:.4f}")
    all_pass &= ok

print(f"\n{'All tests passed.' if all_pass else 'FAILURES!'}")
if not all_pass:
    raise AssertionError("Example failures")
