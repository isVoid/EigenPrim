"""
Numbast × Eigen Level 3: Template bindings.

Function template templated_dot3<Scalar> with split-header/fatbin pattern.
Users get generic templates — Numba deduces <float> from argument types at JIT time.

Run:  pixi run python example_l3.py
"""

import os
import numpy as np
from numba import types, cuda
from eigenprim import bind_eigen_header

ROOT = os.path.dirname(os.path.abspath(__file__))
INCLUDE = os.path.join(ROOT, "include")
EIGEN_INC = os.path.join(ROOT, ".pixi", "envs", "default", "include", "eigen3")

bindings = bind_eigen_header(
    header=os.path.join(INCLUDE, "eigen_wrapper_l3.cuh"),
    decl_header=os.path.join(INCLUDE, "eigen_wrapper_l3_decl.cuh"),
    impl_cu=os.path.join(INCLUDE, "eigen_wrapper_l3_impl.cu"),
    eigen_include=EIGEN_INC,
)

print(f"Functions: {list(bindings.functions.keys())}")

templated_dot3 = bindings.functions["templated_dot3"]

@cuda.jit(link=bindings.links())
def kernel(out):
    out[0] = templated_dot3(
        types.float32(1.0), types.float32(2.0), types.float32(3.0),
        types.float32(4.0), types.float32(5.0), types.float32(6.0),
    )

out = np.zeros(1, dtype=np.float32)
kernel[1, 1](out)

expected = 32.0
ok = np.isclose(out[0], expected, rtol=1e-5)
print(f"\n[{'PASS' if ok else 'FAIL'}] templated_dot3<float>(1,2,3, 4,5,6) = {out[0]:.1f}  (expected {expected:.1f})")
if not ok:
    raise AssertionError("L3 failure")
