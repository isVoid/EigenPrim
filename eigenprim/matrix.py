"""Pre-bound Eigen vector and matrix bindings.

12 types (Vector2f..Vector4d, Matrix2f..Matrix4d) and 170 device functions.

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

# Register Eigen scalar types that map to Numba built-in types.
# Must happen before bind_eigen_header so numbast resolves Eigen::half
# in function signatures to numba.types.float16 instead of Opaque.
from numba import types as _nbtypes
from numba.cuda import types as _cuda_types
from numbast.types import register_cxx_type as _register_cxx_type
_register_cxx_type("Eigen::half", _nbtypes.float16)
_register_cxx_type("Eigen::bfloat16", _cuda_types.bfloat16)

# All 14 types: ast_canopy fully-qualified name -> short name
_TYPE_MAP = {
    # Vectors
    "Eigen::Matrix<float, 2, 1, 0, 2, 1>": "Eigen::Matrix<float, 2, 1>",
    "Eigen::Matrix<float, 3, 1, 0, 3, 1>": "Eigen::Matrix<float, 3, 1>",
    "Eigen::Matrix<float, 4, 1, 0, 4, 1>": "Eigen::Matrix<float, 4, 1>",
    "Eigen::Matrix<double, 2, 1, 0, 2, 1>": "Eigen::Matrix<double, 2, 1>",
    "Eigen::Matrix<double, 3, 1, 0, 3, 1>": "Eigen::Matrix<double, 3, 1>",
    "Eigen::Matrix<double, 4, 1, 0, 4, 1>": "Eigen::Matrix<double, 4, 1>",
    # Half-precision
    "Eigen::Matrix<struct Eigen::half, 3, 1, 0, 3, 1>": "Eigen::Matrix<Eigen::half, 3, 1>",
    "Eigen::Matrix<struct Eigen::half, 3, 3, 0, 3, 3>": "Eigen::Matrix<Eigen::half, 3, 3>",
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

Vector3h = _bindings.types["Eigen::Matrix<Eigen::half, 3, 1>"]
Matrix3h = _bindings.types["Eigen::Matrix<Eigen::half, 3, 3>"]

Matrix2f = _bindings.types["Eigen::Matrix<float, 2, 2>"]
Matrix3f = _bindings.types["Eigen::Matrix<float, 3, 3>"]
Matrix4f = _bindings.types["Eigen::Matrix<float, 4, 4>"]
Matrix2d = _bindings.types["Eigen::Matrix<double, 2, 2>"]
Matrix3d = _bindings.types["Eigen::Matrix<double, 3, 3>"]
Matrix4d = _bindings.types["Eigen::Matrix<double, 4, 4>"]

# ── Function bindings ─────────────────────────────────────────────

# Build the complete function name list programmatically.
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

# Half-precision (subset of ops)
for op in ["add", "sub", "dot", "norm", "normalized", "cross"]:
    FUNCTION_NAMES.append(f"eigen_vec3h_{op}")
for op in ["mul", "determinant", "inverse", "transpose"]:
    FUNCTION_NAMES.append(f"eigen_mat3h_{op}")
FUNCTION_NAMES.append("eigen_mat3h_vec3h_mul")

# Populate module namespace with all bound functions.
_ns = globals()
for _name in FUNCTION_NAMES:
    _ns[_name] = _bindings.functions[_name]

TYPE_NAMES = [
    "Vector2f", "Vector3f", "Vector4f", "Vector2d", "Vector3d", "Vector4d",
    "Matrix2f", "Matrix3f", "Matrix4f", "Matrix2d", "Matrix3d", "Matrix4d",
    "Vector3h", "Matrix3h",
]

# ── Operator overloads ────────────────────────────────────────────

from eigenprim.operators import register_operators
register_operators(_bindings.types, _bindings.functions)

from eigenprim.dispatch import register_dispatch
register_dispatch(_bindings.types, _bindings.functions)

__all__ = TYPE_NAMES + FUNCTION_NAMES
