"""
Eigenprim: Method invocation on vector and matrix types.

Demonstrates that all operations are available as methods on instances,
mirroring the generic dispatch functions from dispatch.py.

Run:  pixi run python examples/10_methods.py
"""

import numpy as np
from numba import cuda

from eigenprim import Vector3f, Matrix3f, links


@cuda.jit(link=links())
def kernel(results):
    a = Vector3f(1.0, 2.0, 3.0)
    b = Vector3f(4.0, 5.0, 6.0)

    # Vector methods
    results[0] = a.dot(b)                    # 1*4+2*5+3*6 = 32
    results[1] = a.norm()                    # sqrt(1+4+9) = 3.7417
    results[2] = a.squared_norm()            # 1+4+9 = 14

    c = a.cross(b)                           # cross product
    results[3] = c.dot(a)                    # orthogonal → ~0

    d = a.normalized()
    results[4] = d.norm()                    # unit vector → 1.0

    # Matrix methods
    M = Matrix3f(
        2.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 2.0,
    )
    results[5] = M.determinant()             # 8.0
    results[6] = M.trace()                   # 6.0

    Minv = M.inverse()
    results[7] = Minv.trace()                # 1/2 * 3 = 1.5

    v = M.vec_mul(a)                         # 2*[1,2,3] = [2,4,6]
    results[8] = v.dot(a)                    # 2*(1+4+9) = 28.0

    MT = M.transpose()
    results[9] = MT.trace()                  # same as M (symmetric) = 6.0


out = np.zeros(10, dtype=np.float32)
kernel[1, 1](out)

checks = [
    ("a.dot(b)",          out[0], 32.0,   1e-4),
    ("a.norm()",          out[1], 3.7417, 1e-3),
    ("a.squared_norm()",  out[2], 14.0,   1e-4),
    ("cross orthogonal",  out[3], 0.0,    1e-3),
    ("normalized norm",   out[4], 1.0,    1e-4),
    ("M.determinant()",   out[5], 8.0,    1e-4),
    ("M.trace()",         out[6], 6.0,    1e-4),
    ("M.inverse().trace()", out[7], 1.5,  1e-4),
    ("M.vec_mul(a).dot(a)", out[8], 28.0, 1e-4),
    ("M.transpose().trace()", out[9], 6.0, 1e-4),
]

all_ok = True
for name, got, expected, tol in checks:
    ok = abs(got - expected) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name} = {got:.4f}  (expected {expected:.4f})")
    all_ok = all_ok and ok

if not all_ok:
    raise AssertionError("One or more method invocation checks failed")
