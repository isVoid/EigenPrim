"""
Eigenprim: Vector3f basics.

Just import and use. Operators (+, -, *, @) work naturally.

Run:  python examples/01_vector_basics.py
"""

import numpy as np
from numba import cuda

from eigenprim import Vector3f, eigen_vec3f_dot, eigen_vec3f_norm, links


@cuda.jit(link=links())
def kernel(out_dot, out_norm):
    a = Vector3f(1.0, 2.0, 3.0)
    b = Vector3f(4.0, 5.0, 6.0)
    out_dot[0] = eigen_vec3f_dot(a, b)
    c = a + b
    out_norm[0] = eigen_vec3f_norm(c)


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
