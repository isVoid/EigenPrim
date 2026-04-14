"""
Eigenprim: Vec3f — thin wrappers over Eigen.

Just import and use. No header paths, no manual binding.

Run:  pixi run python example.py
"""

import numpy as np
from numba import cuda

from eigenprim import Vec3f, vec3f_add, vec3f_dot, vec3f_norm, links


@cuda.jit(link=links())
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

print(f"Results:")
print(f"  dot = {out_dot[0]:.1f}  (expected {expected_dot:.1f})")
print(f"  norm = {out_norm[0]:.4f}  (expected {expected_norm:.4f})")

np.testing.assert_allclose(out_dot[0], expected_dot, rtol=1e-5)
np.testing.assert_allclose(out_norm[0], expected_norm, rtol=1e-4)
print("\nAll assertions passed.")
