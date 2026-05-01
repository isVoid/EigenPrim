"""
Eigenprim: Template bindings.

Generic templated_dot3<Scalar> — Numba deduces <float> from argument types.

Run:  python examples/05_templates.py
"""

import numpy as np
from numba import cuda, types

from eigenprim import templated_dot3, links


@cuda.jit(link=links())
def kernel(out):
    out[0] = templated_dot3(
        types.float32(1.0), types.float32(2.0), types.float32(3.0),
        types.float32(4.0), types.float32(5.0), types.float32(6.0),
    )


out = np.zeros(1, dtype=np.float32)
kernel[1, 1](out)

expected = 32.0
ok = np.isclose(out[0], expected, rtol=1e-5)
print(f"[{'PASS' if ok else 'FAIL'}] templated_dot3<float>(1,2,3, 4,5,6) = {out[0]:.1f}  (expected {expected:.1f})")
if not ok:
    raise AssertionError("L3 failure")
