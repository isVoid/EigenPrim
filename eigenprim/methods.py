"""Method overloads for Eigen types.

Registers @overload_method so Eigen types support method-call syntax
inside @cuda.jit kernels::

    a.dot(b)            # vector dot product → scalar
    a.cross(b)          # 3D cross product → vector
    v.norm()            # Euclidean norm → scalar
    v.normalized()      # unit vector → vector
    v.outer(b)          # outer product → matrix
    v.scale(s)          # scalar multiply → vector/matrix
    m.inverse()         # matrix inverse
    m.transpose()       # matrix transpose
    m.diagonal()        # diagonal → vector
    m.vec_mul(v)        # matrix-vector multiply
    ...

All methods correspond 1-to-1 with the generic functions in dispatch.py.
"""

from numba import types as nbtypes
from numba.extending import overload_method


def register_methods(types, functions):
    """Register @overload_method for all bound Eigen types.

    Called by eigenprim.matrix after binding completes.

    Parameters:
        types: dict of short_name -> bound type (from EigenBindings.types)
        functions: dict of func_name -> bound function (from EigenBindings.functions)
    """

    _VEC_SPECS = [
        ("vec2f",  "Eigen::Matrix<float, 2, 1>",           nbtypes.float32),
        ("vec3f",  "Eigen::Matrix<float, 3, 1>",           nbtypes.float32),
        ("vec4f",  "Eigen::Matrix<float, 4, 1>",           nbtypes.float32),
        ("vec2d",  "Eigen::Matrix<double, 2, 1>",          nbtypes.float64),
        ("vec3d",  "Eigen::Matrix<double, 3, 1>",          nbtypes.float64),
        ("vec4d",  "Eigen::Matrix<double, 4, 1>",          nbtypes.float64),
        ("vec2h",  "Eigen::Matrix<Eigen::half, 2, 1>",     nbtypes.float16),
        ("vec3h",  "Eigen::Matrix<Eigen::half, 3, 1>",     nbtypes.float16),
        ("vec4h",  "Eigen::Matrix<Eigen::half, 4, 1>",     nbtypes.float16),
        ("vec2bf", "Eigen::Matrix<Eigen::bfloat16, 2, 1>", nbtypes.float16),
        ("vec3bf", "Eigen::Matrix<Eigen::bfloat16, 3, 1>", nbtypes.float16),
        ("vec4bf", "Eigen::Matrix<Eigen::bfloat16, 4, 1>", nbtypes.float16),
    ]

    _MAT_VEC_SPECS = [
        ("mat2f",  "Eigen::Matrix<float, 2, 2>",           "vec2f",  "Eigen::Matrix<float, 2, 1>",           nbtypes.float32),
        ("mat3f",  "Eigen::Matrix<float, 3, 3>",           "vec3f",  "Eigen::Matrix<float, 3, 1>",           nbtypes.float32),
        ("mat4f",  "Eigen::Matrix<float, 4, 4>",           "vec4f",  "Eigen::Matrix<float, 4, 1>",           nbtypes.float32),
        ("mat2d",  "Eigen::Matrix<double, 2, 2>",          "vec2d",  "Eigen::Matrix<double, 2, 1>",          nbtypes.float64),
        ("mat3d",  "Eigen::Matrix<double, 3, 3>",          "vec3d",  "Eigen::Matrix<double, 3, 1>",          nbtypes.float64),
        ("mat4d",  "Eigen::Matrix<double, 4, 4>",          "vec4d",  "Eigen::Matrix<double, 4, 1>",          nbtypes.float64),
        ("mat2h",  "Eigen::Matrix<Eigen::half, 2, 2>",     "vec2h",  "Eigen::Matrix<Eigen::half, 2, 1>",     nbtypes.float16),
        ("mat3h",  "Eigen::Matrix<Eigen::half, 3, 3>",     "vec3h",  "Eigen::Matrix<Eigen::half, 3, 1>",     nbtypes.float16),
        ("mat4h",  "Eigen::Matrix<Eigen::half, 4, 4>",     "vec4h",  "Eigen::Matrix<Eigen::half, 4, 1>",     nbtypes.float16),
        ("mat2bf", "Eigen::Matrix<Eigen::bfloat16, 2, 2>", "vec2bf", "Eigen::Matrix<Eigen::bfloat16, 2, 1>", nbtypes.float16),
        ("mat3bf", "Eigen::Matrix<Eigen::bfloat16, 3, 3>", "vec3bf", "Eigen::Matrix<Eigen::bfloat16, 3, 1>", nbtypes.float16),
        ("mat4bf", "Eigen::Matrix<Eigen::bfloat16, 4, 4>", "vec4bf", "Eigen::Matrix<Eigen::bfloat16, 4, 1>", nbtypes.float16),
    ]

    # ── Helper registrars (factory pattern avoids closure-over-loop bugs) ──

    def _unary(VT, method, fn):
        @overload_method(VT, method)
        def _impl(self):
            def impl(self):
                return fn(self)
            return impl

    def _binary_same(VT, method, vt, fn):
        @overload_method(VT, method)
        def _impl(self, other):
            if other == vt:
                def impl(self, other):
                    return fn(self, other)
                return impl

    def _binary_other(VT, method, other_vt, fn):
        @overload_method(VT, method)
        def _impl(self, other):
            if other == other_vt:
                def impl(self, other):
                    return fn(self, other)
                return impl

    def _scale(VT, fn):
        @overload_method(VT, "scale")
        def _impl(self, s):
            if s in (nbtypes.float32, nbtypes.float64):
                def impl(self, s):
                    return fn(self, s)
                return impl

    # ── Vector methods ──────────────────────────────────────────────

    for vname, tkey, _ in _VEC_SPECS:
        vt = types[tkey]._nbtype
        VT = type(vt)

        _binary_same(VT, "dot",           vt, functions[f"eigen_{vname}_dot"])
        _binary_same(VT, "cwise_product", vt, functions[f"eigen_{vname}_cwise_product"])
        _binary_same(VT, "cwise_min",     vt, functions[f"eigen_{vname}_cwise_min"])
        _binary_same(VT, "cwise_max",     vt, functions[f"eigen_{vname}_cwise_max"])
        _binary_same(VT, "outer",         vt, functions[f"eigen_{vname}_outer"])

        _unary(VT, "norm",         functions[f"eigen_{vname}_norm"])
        _unary(VT, "squared_norm", functions[f"eigen_{vname}_squared_norm"])
        _unary(VT, "normalized",   functions[f"eigen_{vname}_normalized"])
        _unary(VT, "cwise_abs",    functions[f"eigen_{vname}_cwise_abs"])
        _unary(VT, "sum",          functions[f"eigen_{vname}_sum"])
        _unary(VT, "min_coeff",    functions[f"eigen_{vname}_min_coeff"])
        _unary(VT, "max_coeff",    functions[f"eigen_{vname}_max_coeff"])

        _scale(VT, functions[f"eigen_{vname}_scale"])

        if "3" in vname:
            _binary_same(VT, "cross", vt, functions[f"eigen_{vname}_cross"])

    # ── Matrix methods ──────────────────────────────────────────────

    for mname, mtkey, vname, vtkey, _ in _MAT_VEC_SPECS:
        mt = types[mtkey]._nbtype
        vt = types[vtkey]._nbtype
        MT = type(mt)

        _binary_same(MT,  "cwise_product", mt, functions[f"eigen_{mname}_cwise_product"])
        _binary_other(MT, "vec_mul",       vt, functions[f"eigen_{mname}_{vname}_mul"])

        _unary(MT, "determinant",  functions[f"eigen_{mname}_determinant"])
        _unary(MT, "inverse",      functions[f"eigen_{mname}_inverse"])
        _unary(MT, "transpose",    functions[f"eigen_{mname}_transpose"])
        _unary(MT, "trace",        functions[f"eigen_{mname}_trace"])
        _unary(MT, "norm",         functions[f"eigen_{mname}_norm"])
        _unary(MT, "squared_norm", functions[f"eigen_{mname}_squared_norm"])
        _unary(MT, "diagonal",     functions[f"eigen_{mname}_diagonal"])

        _scale(MT, functions[f"eigen_{mname}_scale"])
