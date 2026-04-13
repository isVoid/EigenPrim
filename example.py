"""
Numbast × Eigen Level 1: Thin wrappers.

Vec3f struct + 6 device functions that internally call Eigen.
Simplest case — no Eigen types in the API surface.

Run:  pixi run python example.py
"""

import os
import numpy as np
from numba import cuda
from eigenprim import bind_eigen_header

ROOT = os.path.dirname(os.path.abspath(__file__))
INCLUDE = os.path.join(ROOT, "include")
EIGEN_INC = os.path.join(ROOT, ".pixi", "envs", "default", "include", "eigen3")

bindings = bind_eigen_header(
    header=os.path.join(INCLUDE, "eigen_wrapper_l1.cuh"),
    decl_header=os.path.join(INCLUDE, "eigen_wrapper_l1_decl.cuh"),
    impl_cu=os.path.join(INCLUDE, "eigen_wrapper_l1_impl.cu"),
    eigen_include=EIGEN_INC,
    extra_retain=["eigen_wrapper_l1_decl.cuh"],
)

print(f"Functions: {list(bindings.functions.keys())}")

Vec3f = bindings.types.get("Vec3f")  # L1 struct is in the decl header
vec3f_dot = bindings.functions["vec3f_dot"]
vec3f_add = bindings.functions["vec3f_add"]
vec3f_norm = bindings.functions["vec3f_norm"]

@cuda.jit(link=bindings.links())
def kernel(out_dot, out_norm):
    a = Vec3f(1.0, 2.0, 3.0)
    b = Vec3f(4.0, 5.0, 6.0)
    out_dot[0] = vec3f_dot(a, b)
    c = vec3f_add(a, b)
    out_norm[0] = vec3f_norm(c)

out_dot = np.zeros(1, dtype=np.float32)
out_norm = np.zeros(1, dtype=np.float32)
kernel[1, 1](out_dot, out_norm)

expected_dot = 32.0
expected_norm = np.sqrt(5**2 + 7**2 + 9**2)

print(f"\nResults:")
print(f"  dot = {out_dot[0]:.1f}  (expected {expected_dot:.1f})")
print(f"  norm = {out_norm[0]:.4f}  (expected {expected_norm:.4f})")

np.testing.assert_allclose(out_dot[0], expected_dot, rtol=1e-5)
np.testing.assert_allclose(out_norm[0], expected_norm, rtol=1e-4)
print("\nAll assertions passed.")
