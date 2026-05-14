"""Pre-bound L3 bindings: function templates (templated_dot3, etc.).

Usage::

    from eigenprim.templates import templated_dot3
    from eigenprim import links
    from numba import cuda, types

    @cuda.jit(link=links())
    def kernel(out):
        out[0] = templated_dot3(
            types.float32(1.0), types.float32(2.0), types.float32(3.0),
            types.float32(4.0), types.float32(5.0), types.float32(6.0),
        )
"""

import os

from eigenprim._env import fatbin_dir, include_dir
from eigenprim._registry import register
from eigenprim.bind import bind_eigen_header

_INCLUDE = include_dir()

_bindings = bind_eigen_header(
    header=os.path.join(_INCLUDE, "generic.cuh"),
    decl_header=os.path.join(_INCLUDE, "generic_decl.cuh"),
    fatbin=os.path.join(fatbin_dir(), "generic.fatbin"),
)
register(_bindings)

templated_dot3 = _bindings.functions["templated_dot3"]
