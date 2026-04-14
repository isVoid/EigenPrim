"""Eigenprim: Eigen device primitives for CUDA Python.

Import types and functions directly::

    from eigenprim import Vector3f, Matrix4f, eigen_vec3f_dot, links
    from numba import cuda

    @cuda.jit(link=links())
    def kernel(out):
        a = Vector3f(1.0, 2.0, 3.0)
        b = Vector3f(4.0, 5.0, 6.0)
        out[0] = eigen_vec3f_dot(a, b)

For custom bindings, use the low-level API::

    from eigenprim import bind_eigen_header
"""

import importlib

from eigenprim._registry import links
from eigenprim.bind import EigenBindings, bind_eigen_header

# ── Build lazy import map ─────────────────────────────────────────

# Lazy import map: attribute name -> (module, attribute)
# Sub-modules bind on first import, so only the modules you use get compiled.
_LAZY_IMPORTS = {
    # Templates (eigenprim.templates)
    "templated_dot3": ("eigenprim.templates", "templated_dot3"),
}

# Eigen matrix types (eigenprim.matrix) — 12 types
for _type_name in [
    "Vector2f", "Vector3f", "Vector4f", "Vector2d", "Vector3d", "Vector4d",
    "Matrix2f", "Matrix3f", "Matrix4f", "Matrix2d", "Matrix3d", "Matrix4d",
]:
    _LAZY_IMPORTS[_type_name] = ("eigenprim.matrix", _type_name)

# Eigen matrix functions (eigenprim.matrix) — ~92 functions
_VEC_OPS = [
    "add", "sub", "dot", "norm", "squared_norm", "normalized", "scale",
    "cwise_product", "cwise_abs", "cwise_min", "cwise_max",
    "sum", "min_coeff", "max_coeff", "outer",
]
_VEC_TYPES = ["vec2f", "vec3f", "vec4f", "vec2d", "vec3d", "vec4d"]
_MAT_OPS = [
    "add", "sub", "mul", "determinant", "inverse", "transpose", "trace",
    "cwise_product", "scale", "norm", "squared_norm", "diagonal",
]
_MAT_VEC_PAIRS = [
    ("mat2f", "vec2f"), ("mat3f", "vec3f"), ("mat4f", "vec4f"),
    ("mat2d", "vec2d"), ("mat3d", "vec3d"), ("mat4d", "vec4d"),
]

for _vt in _VEC_TYPES:
    for _op in _VEC_OPS:
        _name = f"eigen_{_vt}_{_op}"
        _LAZY_IMPORTS[_name] = ("eigenprim.matrix", _name)
    if "3" in _vt:
        _name = f"eigen_{_vt}_cross"
        _LAZY_IMPORTS[_name] = ("eigenprim.matrix", _name)

for _mt, _vt in _MAT_VEC_PAIRS:
    for _op in _MAT_OPS:
        _name = f"eigen_{_mt}_{_op}"
        _LAZY_IMPORTS[_name] = ("eigenprim.matrix", _name)
    _name = f"eigen_{_mt}_{_vt}_mul"
    _LAZY_IMPORTS[_name] = ("eigenprim.matrix", _name)

# ── Public API ────────────────────────────────────────────────────

__all__ = [
    "bind_eigen_header",
    "EigenBindings",
    "links",
    *_LAZY_IMPORTS.keys(),
]


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        mod_name, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_name)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'eigenprim' has no attribute {name!r}")
