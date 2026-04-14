"""Pre-bound Eigen vector and matrix bindings.

12 types (Vector2f..Vector4d, Matrix2f..Matrix4d) and ~92 device functions.

Usage::

    from eigenprim.matrix import Vector3f, Matrix4f, eigen_vec3f_dot
    from eigenprim import links
    from numba import cuda

    @cuda.jit(link=links())
    def kernel(out):
        a = Vector3f(1.0, 2.0, 3.0)
        b = Vector3f(4.0, 5.0, 6.0)
        out[0] = eigen_vec3f_dot(a, b)
"""

import os

from eigenprim._env import find_eigen_include, include_dir
from eigenprim._registry import register
from eigenprim.bind import bind_eigen_header

_INCLUDE = include_dir()

# All 12 types: ast_canopy fully-qualified name -> short name
_TYPE_MAP = {
    # Vectors
    "Eigen::Matrix<float, 2, 1, 0, 2, 1>": "Eigen::Matrix<float, 2, 1>",
    "Eigen::Matrix<float, 3, 1, 0, 3, 1>": "Eigen::Matrix<float, 3, 1>",
    "Eigen::Matrix<float, 4, 1, 0, 4, 1>": "Eigen::Matrix<float, 4, 1>",
    "Eigen::Matrix<double, 2, 1, 0, 2, 1>": "Eigen::Matrix<double, 2, 1>",
    "Eigen::Matrix<double, 3, 1, 0, 3, 1>": "Eigen::Matrix<double, 3, 1>",
    "Eigen::Matrix<double, 4, 1, 0, 4, 1>": "Eigen::Matrix<double, 4, 1>",
    # Matrices
    "Eigen::Matrix<float, 2, 2, 0, 2, 2>": "Eigen::Matrix<float, 2, 2>",
    "Eigen::Matrix<float, 3, 3, 0, 3, 3>": "Eigen::Matrix<float, 3, 3>",
    "Eigen::Matrix<float, 4, 4, 0, 4, 4>": "Eigen::Matrix<float, 4, 4>",
    "Eigen::Matrix<double, 2, 2, 0, 2, 2>": "Eigen::Matrix<double, 2, 2>",
    "Eigen::Matrix<double, 3, 3, 0, 3, 3>": "Eigen::Matrix<double, 3, 3>",
    "Eigen::Matrix<double, 4, 4, 0, 4, 4>": "Eigen::Matrix<double, 4, 4>",
}

_bindings = bind_eigen_header(
    header=os.path.join(_INCLUDE, "matrix.cuh"),
    decl_header=os.path.join(_INCLUDE, "matrix_decl.cuh"),
    impl_cu=os.path.join(_INCLUDE, "matrix.cu"),
    eigen_include=find_eigen_include(),
    stub_header=os.path.join(_INCLUDE, "eigen_stub.cuh"),
    type_map=_TYPE_MAP,
)
register(_bindings)

# ── Type aliases ──────────────────────────────────────────────────

Vector2f = _bindings.types["Eigen::Matrix<float, 2, 1>"]
Vector3f = _bindings.types["Eigen::Matrix<float, 3, 1>"]
Vector4f = _bindings.types["Eigen::Matrix<float, 4, 1>"]
Vector2d = _bindings.types["Eigen::Matrix<double, 2, 1>"]
Vector3d = _bindings.types["Eigen::Matrix<double, 3, 1>"]
Vector4d = _bindings.types["Eigen::Matrix<double, 4, 1>"]

Matrix2f = _bindings.types["Eigen::Matrix<float, 2, 2>"]
Matrix3f = _bindings.types["Eigen::Matrix<float, 3, 3>"]
Matrix4f = _bindings.types["Eigen::Matrix<float, 4, 4>"]
Matrix2d = _bindings.types["Eigen::Matrix<double, 2, 2>"]
Matrix3d = _bindings.types["Eigen::Matrix<double, 3, 3>"]
Matrix4d = _bindings.types["Eigen::Matrix<double, 4, 4>"]

# ── Function bindings ─────────────────────────────────────────────

# Build the complete function name list programmatically.
_VEC_OPS = ["add", "sub", "dot", "norm", "squared_norm", "normalized", "scale"]
_VEC_TYPES = ["vec2f", "vec3f", "vec4f", "vec2d", "vec3d", "vec4d"]

_MAT_OPS = ["add", "sub", "mul", "determinant", "inverse", "transpose", "trace"]
_MAT_VEC_PAIRS = [
    ("mat2f", "vec2f"), ("mat3f", "vec3f"), ("mat4f", "vec4f"),
    ("mat2d", "vec2d"), ("mat3d", "vec3d"), ("mat4d", "vec4d"),
]

FUNCTION_NAMES = []

for vt in _VEC_TYPES:
    for op in _VEC_OPS:
        FUNCTION_NAMES.append(f"eigen_{vt}_{op}")
    if "3" in vt:
        FUNCTION_NAMES.append(f"eigen_{vt}_cross")

for mt, vt in _MAT_VEC_PAIRS:
    for op in _MAT_OPS:
        FUNCTION_NAMES.append(f"eigen_{mt}_{op}")
    FUNCTION_NAMES.append(f"eigen_{mt}_{vt}_mul")

# Populate module namespace with all bound functions.
_ns = globals()
for _name in FUNCTION_NAMES:
    _ns[_name] = _bindings.functions[_name]

TYPE_NAMES = [
    "Vector2f", "Vector3f", "Vector4f", "Vector2d", "Vector3d", "Vector4d",
    "Matrix2f", "Matrix3f", "Matrix4f", "Matrix2d", "Matrix3d", "Matrix4d",
]

# ── Operator overloads ────────────────────────────────────────────

from eigenprim.operators import register_operators
register_operators(_bindings.types, _bindings.functions)

__all__ = TYPE_NAMES + FUNCTION_NAMES
