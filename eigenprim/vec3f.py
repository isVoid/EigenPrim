"""Pre-bound L1 bindings: Vec3f struct and 6 device functions.

Usage::

    from eigenprim.vec3f import Vec3f, vec3f_dot, vec3f_add
    from eigenprim import links
    from numba import cuda

    @cuda.jit(link=links())
    def kernel(out):
        a = Vec3f(1.0, 2.0, 3.0)
        b = Vec3f(4.0, 5.0, 6.0)
        out[0] = vec3f_dot(a, b)
"""

import os

from eigenprim._env import find_eigen_include, include_dir
from eigenprim._registry import register
from eigenprim.bind import bind_eigen_header

_INCLUDE = include_dir()

_bindings = bind_eigen_header(
    header=os.path.join(_INCLUDE, "vec3f.cuh"),
    decl_header=os.path.join(_INCLUDE, "vec3f_decl.cuh"),
    impl_cu=os.path.join(_INCLUDE, "vec3f.cu"),
    eigen_include=find_eigen_include(),
    extra_retain=["vec3f_decl.cuh"],
)
register(_bindings)

Vec3f = _bindings.types["Vec3f"]
vec3f_add = _bindings.functions["vec3f_add"]
vec3f_dot = _bindings.functions["vec3f_dot"]
vec3f_cross = _bindings.functions["vec3f_cross"]
vec3f_norm = _bindings.functions["vec3f_norm"]
vec3f_normalized = _bindings.functions["vec3f_normalized"]
vec3f_scale = _bindings.functions["vec3f_scale"]
