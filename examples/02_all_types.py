"""
Eigenprim: All supported dtypes and sizes.

Demonstrates float32, float64, float16 (half), and bfloat16 vector/matrix
types with operators (+, *, @) and named functions.

Run:  python examples/02_all_types.py
"""

import numpy as np
from numba import cuda, types as nbtypes
from numba.cuda import types as cuda_types

from eigenprim import (
    # float32
    Vector2f, Vector3f, Vector4f, Matrix2f, Matrix3f, Matrix4f,
    eigen_vec2f_dot, eigen_vec2f_norm,
    eigen_vec3f_dot, eigen_vec3f_cross, eigen_vec3f_norm,
    eigen_vec4f_dot,
    eigen_mat2f_determinant, eigen_mat2f_inverse,
    eigen_mat3f_determinant, eigen_mat3f_transpose, eigen_mat3f_trace,
    eigen_mat4f_determinant, eigen_mat4f_trace,
    # float16 (half)
    Vector3h, Matrix3h,
    eigen_vec3h_dot, eigen_vec3h_norm, eigen_vec3h_cross,
    eigen_mat3h_determinant, eigen_mat3h_inverse,
    # bfloat16
    Vector3bf, Matrix3bf,
    eigen_vec3bf_dot, eigen_vec3bf_norm,
    eigen_mat3bf_determinant, eigen_mat3bf_inverse,
    links,
)

h = nbtypes.float16
bf = cuda_types.bfloat16


# ── Float32 kernels ──────────────────────────────────────────────

@cuda.jit(link=links())
def kernel_f32_vectors(out):
    a2 = Vector2f(3.0, 4.0)
    out[0] = eigen_vec2f_norm(a2)                                    # 5.0

    a3 = Vector3f(1.0, 2.0, 3.0)
    b3 = Vector3f(4.0, 5.0, 6.0)
    out[1] = eigen_vec3f_dot(a3, b3)                                 # 32.0
    cr = eigen_vec3f_cross(Vector3f(1.0, 0.0, 0.0),
                           Vector3f(0.0, 1.0, 0.0))
    out[2] = eigen_vec3f_norm(cr)                                    # 1.0
    s = a3 + b3
    out[3] = eigen_vec3f_dot(s, Vector3f(1.0, 1.0, 1.0))           # 21.0

    a4 = Vector4f(1.0, 2.0, 3.0, 4.0)
    b4 = Vector4f(5.0, 6.0, 7.0, 8.0)
    out[4] = eigen_vec4f_dot(a4, b4)                                 # 70.0
    c4 = a4 * 2.0
    out[5] = eigen_vec4f_dot(c4, Vector4f(1.0, 1.0, 1.0, 1.0))     # 20.0


@cuda.jit(link=links())
def kernel_f32_matrices(out):
    D2 = Matrix2f(3.0, 0.0, 0.0, 4.0)
    out[0] = eigen_mat2f_determinant(D2)                              # 12.0
    v2 = eigen_mat2f_inverse(D2) @ Vector2f(6.0, 8.0)
    out[1] = eigen_vec2f_dot(v2, Vector2f(1.0, 1.0))                # 4.0

    D3 = Matrix3f(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0)
    out[2] = eigen_mat3f_determinant(D3)                              # 24.0
    out[3] = eigen_vec3f_dot(
        D3 @ Vector3f(1.0, 1.0, 1.0),
        Vector3f(1.0, 1.0, 1.0))                                     # 9.0
    out[4] = eigen_mat3f_determinant(D3 @ D3)                       # 576.0

    D4 = Matrix4f(1.0, 0.0, 0.0, 0.0,
                   0.0, 2.0, 0.0, 0.0,
                   0.0, 0.0, 3.0, 0.0,
                   0.0, 0.0, 0.0, 4.0)
    out[5] = eigen_mat4f_determinant(D4)                              # 24.0
    out[6] = eigen_mat4f_trace(D4)                                    # 10.0


# ── Half-precision kernel ────────────────────────────────────────

@cuda.jit(link=links())
def kernel_half(out):
    a = Vector3h(h(1.0), h(2.0), h(3.0))
    b = Vector3h(h(4.0), h(5.0), h(6.0))
    out[0] = eigen_vec3h_dot(a, b)                                    # 32.0
    out[1] = eigen_vec3h_norm(Vector3h(h(3.0), h(4.0), h(0.0)))     # 5.0
    cr = eigen_vec3h_cross(Vector3h(h(1.0), h(0.0), h(0.0)),
                           Vector3h(h(0.0), h(1.0), h(0.0)))
    out[2] = eigen_vec3h_norm(cr)                                     # 1.0
    c = a + b
    out[3] = eigen_vec3h_dot(c, Vector3h(h(1.0), h(1.0), h(1.0)))  # 21.0

    D = Matrix3h(h(2.0), h(0.0), h(0.0),
                 h(0.0), h(3.0), h(0.0),
                 h(0.0), h(0.0), h(4.0))
    out[4] = eigen_mat3h_determinant(D)                               # 24.0


# ── Bfloat16 kernel ──────────────────────────────────────────────

@cuda.jit(link=links())
def kernel_bf16(out):
    a = Vector3bf(bf(1.0), bf(2.0), bf(3.0))
    b = Vector3bf(bf(4.0), bf(5.0), bf(6.0))
    out[0] = eigen_vec3bf_dot(a, b)                                   # 32.0
    out[1] = eigen_vec3bf_norm(Vector3bf(bf(3.0), bf(4.0), bf(0.0))) # 5.0
    c = a + b
    out[2] = eigen_vec3bf_dot(c, Vector3bf(bf(1.0), bf(1.0), bf(1.0)))  # 21.0

    D = Matrix3bf(bf(2.0), bf(0.0), bf(0.0),
                  bf(0.0), bf(3.0), bf(0.0),
                  bf(0.0), bf(0.0), bf(4.0))
    out[3] = eigen_mat3bf_determinant(D)                              # 24.0


# ── Run ──────────────────────────────────────────────────────────

out_fv = np.zeros(6, dtype=np.float32)
kernel_f32_vectors[1, 1](out_fv)

out_fm = np.zeros(7, dtype=np.float32)
kernel_f32_matrices[1, 1](out_fm)

out_h = np.zeros(5, dtype=np.float16)
kernel_half[1, 1](out_h)

out_bf = np.zeros(4, dtype=np.float32)  # no native bf16 in numpy
kernel_bf16[1, 1](out_bf)

# ── Results ──────────────────────────────────────────────────────

all_pass = True

print("Float32 vectors:")
for label, got, expected in [
    ("norm((3,4))", out_fv[0], 5.0),
    ("dot((1,2,3),(4,5,6))", out_fv[1], 32.0),
    ("norm(cross(x,y))", out_fv[2], 1.0),
    ("(1,2,3)+(4,5,6) dot (1,1,1)", out_fv[3], 21.0),
    ("dot4((1,2,3,4),(5,6,7,8))", out_fv[4], 70.0),
    ("(1,2,3,4)*2 dot (1,1,1,1)", out_fv[5], 20.0),
]:
    ok = np.isclose(got, expected, rtol=1e-5)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} = {got:.4f}")
    all_pass &= ok

print("\nFloat32 matrices:")
for label, got, expected in [
    ("det(diag2(3,4))", out_fm[0], 12.0),
    ("inv(diag2)@(6,8) dot (1,1)", out_fm[1], 4.0),
    ("det(diag3(2,3,4))", out_fm[2], 24.0),
    ("diag3@(1,1,1) dot (1,1,1)", out_fm[3], 9.0),
    ("det(D3@D3)", out_fm[4], 576.0),
    ("det(diag4(1,2,3,4))", out_fm[5], 24.0),
    ("trace(diag4(1,2,3,4))", out_fm[6], 10.0),
]:
    ok = np.isclose(got, expected, rtol=1e-4)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} = {got:.4f}")
    all_pass &= ok

print("\nHalf-precision (fp16):")
for label, got, expected in [
    ("dot((1,2,3),(4,5,6))", out_h[0], 32.0),
    ("norm((3,4,0))", out_h[1], 5.0),
    ("norm(cross(x,y))", out_h[2], 1.0),
    ("(1,2,3)+(4,5,6) dot (1,1,1)", out_h[3], 21.0),
    ("det(diag(2,3,4))", out_h[4], 24.0),
]:
    ok = np.isclose(float(got), expected, rtol=0.01)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} = {float(got):.4f}")
    all_pass &= ok

print("\nBfloat16:")
for label, got, expected in [
    ("dot((1,2,3),(4,5,6))", out_bf[0], 32.0),
    ("norm((3,4,0))", out_bf[1], 5.0),
    ("(1,2,3)+(4,5,6) dot (1,1,1)", out_bf[2], 21.0),
    ("det(diag(2,3,4))", out_bf[3], 24.0),
]:
    ok = np.isclose(float(got), expected, rtol=0.02)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} = {float(got):.4f}")
    all_pass &= ok

print(f"\n{'All tests passed.' if all_pass else 'FAILURES!'}")
if not all_pass:
    raise AssertionError("Example failures")
