"""Operator overloads for Eigen types.

Registers @overload(operator.add/sub/mul/matmul) so that Eigen types
support natural Python syntax in @cuda.jit kernels:

    a + b          # vector/matrix add
    a - b          # vector/matrix sub
    v * 2.0        # scalar multiply
    M @ N          # matrix multiply
    M @ v          # matrix-vector multiply
"""

import operator

from numba import types as nbtypes
from numba.extending import overload


def register_operators(types, functions):
    """Register operator overloads from bound Eigen types and functions.

    Called by eigenprim.matrix after binding completes.

    Parameters:
        types: dict of short_name -> bound type (from EigenBindings.types)
        functions: dict of func_name -> bound function (from EigenBindings.functions)
    """

    # ── Build dispatch tables ─────────────────────────────────────

    # (nbtype_a, nbtype_b) -> bound_function
    _add_table = {}
    _sub_table = {}
    _matmul_table = {}
    _mul_table = {}  # scalar multiply: (vec_nbtype, scalar_nbtype) and reverse

    _VEC_SPECS = [
        ("vec2f", "Eigen::Matrix<float, 2, 1>", nbtypes.float32),
        ("vec3f", "Eigen::Matrix<float, 3, 1>", nbtypes.float32),
        ("vec4f", "Eigen::Matrix<float, 4, 1>", nbtypes.float32),
        ("vec2d", "Eigen::Matrix<double, 2, 1>", nbtypes.float64),
        ("vec3d", "Eigen::Matrix<double, 3, 1>", nbtypes.float64),
        ("vec4d", "Eigen::Matrix<double, 4, 1>", nbtypes.float64),
    ]

    _MAT_VEC_SPECS = [
        ("mat2f", "Eigen::Matrix<float, 2, 2>", "vec2f", "Eigen::Matrix<float, 2, 1>"),
        ("mat3f", "Eigen::Matrix<float, 3, 3>", "vec3f", "Eigen::Matrix<float, 3, 1>"),
        ("mat4f", "Eigen::Matrix<float, 4, 4>", "vec4f", "Eigen::Matrix<float, 4, 1>"),
        ("mat2d", "Eigen::Matrix<double, 2, 2>", "vec2d", "Eigen::Matrix<double, 2, 1>"),
        ("mat3d", "Eigen::Matrix<double, 3, 3>", "vec3d", "Eigen::Matrix<double, 3, 1>"),
        ("mat4d", "Eigen::Matrix<double, 4, 4>", "vec4d", "Eigen::Matrix<double, 4, 1>"),
    ]

    for vname, tkey, scalar_nbt in _VEC_SPECS:
        vt = types[tkey]._nbtype
        _add_table[(vt, vt)] = functions[f"eigen_{vname}_add"]
        _sub_table[(vt, vt)] = functions[f"eigen_{vname}_sub"]
        fn = functions[f"eigen_{vname}_scale"]
        # Register both float32 and float64 scalars (Numba literals are float64)
        for snbt in (nbtypes.float32, nbtypes.float64):
            _mul_table[(vt, snbt)] = fn
            _mul_table[(snbt, vt)] = (fn, True)  # swapped

    for mname, mtkey, vname, vtkey in _MAT_VEC_SPECS:
        mt = types[mtkey]._nbtype
        vt = types[vtkey]._nbtype
        _add_table[(mt, mt)] = functions[f"eigen_{mname}_add"]
        _sub_table[(mt, mt)] = functions[f"eigen_{mname}_sub"]
        _matmul_table[(mt, mt)] = functions[f"eigen_{mname}_mul"]
        _matmul_table[(mt, vt)] = functions[f"eigen_{mname}_{vname}_mul"]

    # ── Register overloads ────────────────────────────────────────

    @overload(operator.add)
    def _eigen_add(a, b):
        fn = _add_table.get((a, b))
        if fn is not None:
            def impl(a, b):
                return fn(a, b)
            return impl

    @overload(operator.sub)
    def _eigen_sub(a, b):
        fn = _sub_table.get((a, b))
        if fn is not None:
            def impl(a, b):
                return fn(a, b)
            return impl

    @overload(operator.matmul)
    def _eigen_matmul(a, b):
        fn = _matmul_table.get((a, b))
        if fn is not None:
            def impl(a, b):
                return fn(a, b)
            return impl

    @overload(operator.mul)
    def _eigen_mul(a, b):
        entry = _mul_table.get((a, b))
        if entry is not None:
            if isinstance(entry, tuple):
                # (scalar, vec) — swap args to call scale(vec, scalar)
                fn, _ = entry
                def impl(a, b):
                    return fn(b, a)
                return impl
            else:
                # (vec, scalar) — direct call
                fn = entry
                def impl(a, b):
                    return fn(a, b)
                return impl
