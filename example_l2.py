"""
Numbast × Eigen Level 2: Direct Eigen::Matrix types.

Uses bind_eigen_header() for a one-call binding of 9 Eigen functions
that take Eigen::Matrix<float,3,1> and Eigen::Matrix<float,3,3> directly.

Run:  pixi run python example_l2.py
"""

import os
import numpy as np
from numba import types, cuda
from eigenprim import bind_eigen_header

ROOT = os.path.dirname(os.path.abspath(__file__))
INCLUDE = os.path.join(ROOT, "include")
EIGEN_INC = os.path.join(ROOT, ".pixi", "envs", "default", "include", "eigen3")

# ── One-call binding ──────────────────────────────────────────────

bindings = bind_eigen_header(
    header=os.path.join(INCLUDE, "eigen_wrapper_l2.cuh"),
    decl_header=os.path.join(INCLUDE, "eigen_wrapper_l2_decl.cuh"),
    impl_cu=os.path.join(INCLUDE, "eigen_wrapper_l2_impl.cu"),
    eigen_include=EIGEN_INC,
    stub_header=os.path.join(INCLUDE, "eigen_stub.cuh"),
    type_map={
        "Eigen::Matrix<float, 3, 1, 0, 3, 1>": "Eigen::Matrix<float, 3, 1>",
        "Eigen::Matrix<float, 3, 3, 0, 3, 3>": "Eigen::Matrix<float, 3, 3>",
    },
)

print(f"Types: {list(bindings.types.keys())}")
print(f"Functions: {list(bindings.functions.keys())}")

# ── Extract bindings ──────────────────────────────────────────────

EigenVec3f = bindings.types["Eigen::Matrix<float, 3, 1>"]
EigenMat3f = bindings.types["Eigen::Matrix<float, 3, 3>"]

eigen_vec3f_add = bindings.functions["eigen_vec3f_add"]
eigen_vec3f_dot = bindings.functions["eigen_vec3f_dot"]
eigen_vec3f_cross = bindings.functions["eigen_vec3f_cross"]
eigen_vec3f_norm = bindings.functions["eigen_vec3f_norm"]
eigen_mat3f_vec3f_mul = bindings.functions["eigen_mat3f_vec3f_mul"]
eigen_mat3f_mul = bindings.functions["eigen_mat3f_mul"]
eigen_mat3f_determinant = bindings.functions["eigen_mat3f_determinant"]
eigen_mat3f_inverse = bindings.functions["eigen_mat3f_inverse"]
eigen_mat3f_transpose = bindings.functions["eigen_mat3f_transpose"]

# ── Vec3f kernel (4 functions) ────────────────────────────────────

@cuda.jit(link=bindings.links())
def kernel_vec3f(out):
    a = EigenVec3f(1.0, 2.0, 3.0)
    b = EigenVec3f(4.0, 5.0, 6.0)
    out[0] = eigen_vec3f_dot(a, b)                              # 32
    out[1] = eigen_vec3f_norm(EigenVec3f(3.0, 4.0, 0.0))       # 5
    s = eigen_vec3f_add(a, b)
    out[2] = eigen_vec3f_dot(s, EigenVec3f(1.0, 1.0, 1.0))     # 21
    cr = eigen_vec3f_cross(EigenVec3f(1.0, 0.0, 0.0), EigenVec3f(0.0, 1.0, 0.0))
    out[3] = eigen_vec3f_norm(cr)                                # 1

out_v = np.zeros(4, dtype=np.float32)
kernel_vec3f[1, 1](out_v)

# ── Mat3f kernel (5 functions) ────────────────────────────────────
# Column-major: Matrix(col0_row0, col0_row1, col0_row2, col1_row0, ...)

@cuda.jit(link=bindings.links())
def kernel_mat3f(out):
    D = EigenMat3f(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0)  # diag(2,3,4)
    ones = EigenVec3f(1.0, 1.0, 1.0)

    out[0] = eigen_mat3f_determinant(D)                                    # 24
    out[1] = eigen_vec3f_dot(eigen_mat3f_vec3f_mul(D, ones), ones)         # 9
    out[2] = eigen_vec3f_dot(
        eigen_mat3f_vec3f_mul(eigen_mat3f_inverse(D), EigenVec3f(6.0, 6.0, 6.0)),
        ones)                                                               # 6.5
    out[3] = eigen_mat3f_determinant(eigen_mat3f_mul(D, D))               # 576
    M = EigenMat3f(1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0)
    Mt = eigen_mat3f_transpose(M)
    out[4] = eigen_vec3f_dot(
        eigen_mat3f_vec3f_mul(Mt, EigenVec3f(1.0, 0.0, 0.0)), ones)       # 6

out_m = np.zeros(5, dtype=np.float32)
kernel_mat3f[1, 1](out_m)

# ── Results ───────────────────────────────────────────────────────

all_pass = True
print("\nVec3f results:")
for label, got, expected in [
    ("dot((1,2,3),(4,5,6))", out_v[0], 32.0),
    ("norm((3,4,0))", out_v[1], 5.0),
    ("sum(add((1,2,3),(4,5,6)))", out_v[2], 21.0),
    ("norm(cross((1,0,0),(0,1,0)))", out_v[3], 1.0),
]:
    ok = np.isclose(got, expected, rtol=1e-5)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} = {got:.4f}")
    all_pass &= ok

print("\nMat3f results:")
for label, got, expected in [
    ("det(diag(2,3,4))", out_m[0], 24.0),
    ("sum(diag(2,3,4)*(1,1,1))", out_m[1], 9.0),
    ("sum(inv(diag(2,3,4))*(6,6,6))", out_m[2], 6.5),
    ("det(diag(2,3,4)^2)", out_m[3], 576.0),
    ("sum(transpose(M)*(1,0,0))", out_m[4], 6.0),
]:
    ok = np.isclose(got, expected, rtol=1e-4)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} = {got:.4f}")
    all_pass &= ok

print(f"\n{'All 9 functions passed.' if all_pass else 'FAILURES!'}")
if not all_pass:
    raise AssertionError("L2 failures")
