"""Type-dispatched generic functions for Eigen types.

Provides unprefixed functions like ``dot``, ``norm``, ``inverse`` that
dispatch to the correct ``eigen_{type}_{op}`` based on argument types
at Numba JIT compile time.

Usage::

    from eigenprim import Vector3f, dot, norm, inverse, links

    @cuda.jit(link=links())
    def kernel(out):
        a = Vector3f(1.0, 2.0, 3.0)
        out[0] = dot(a, Vector3f(4.0, 5.0, 6.0))
"""

from numba import types as nbtypes
from numba.extending import overload


# ── Generic function stubs ────────────────────────────────────────
# These exist so Numba can resolve @overload(fn). Outside JIT they
# raise — the real work happens in the overloaded implementations.

def add(a, b):
    raise NotImplementedError("eigenprim.add is only usable inside @cuda.jit kernels")

def sub(a, b):
    raise NotImplementedError("eigenprim.sub is only usable inside @cuda.jit kernels")

def dot(a, b):
    raise NotImplementedError("eigenprim.dot is only usable inside @cuda.jit kernels")

def cross(a, b):
    raise NotImplementedError("eigenprim.cross is only usable inside @cuda.jit kernels")

def norm(v):
    raise NotImplementedError("eigenprim.norm is only usable inside @cuda.jit kernels")

def squared_norm(v):
    raise NotImplementedError("eigenprim.squared_norm is only usable inside @cuda.jit kernels")

def normalized(v):
    raise NotImplementedError("eigenprim.normalized is only usable inside @cuda.jit kernels")

def scale(a, s):
    raise NotImplementedError("eigenprim.scale is only usable inside @cuda.jit kernels")

def cwise_product(a, b):
    raise NotImplementedError("eigenprim.cwise_product is only usable inside @cuda.jit kernels")

def cwise_abs(v):
    raise NotImplementedError("eigenprim.cwise_abs is only usable inside @cuda.jit kernels")

def cwise_min(a, b):
    raise NotImplementedError("eigenprim.cwise_min is only usable inside @cuda.jit kernels")

def cwise_max(a, b):
    raise NotImplementedError("eigenprim.cwise_max is only usable inside @cuda.jit kernels")

def sum(v):
    raise NotImplementedError("eigenprim.sum is only usable inside @cuda.jit kernels")

def min_coeff(v):
    raise NotImplementedError("eigenprim.min_coeff is only usable inside @cuda.jit kernels")

def max_coeff(v):
    raise NotImplementedError("eigenprim.max_coeff is only usable inside @cuda.jit kernels")

def outer(a, b):
    raise NotImplementedError("eigenprim.outer is only usable inside @cuda.jit kernels")

def mul(a, b):
    raise NotImplementedError("eigenprim.mul is only usable inside @cuda.jit kernels")

def determinant(m):
    raise NotImplementedError("eigenprim.determinant is only usable inside @cuda.jit kernels")

def inverse(m):
    raise NotImplementedError("eigenprim.inverse is only usable inside @cuda.jit kernels")

def transpose(m):
    raise NotImplementedError("eigenprim.transpose is only usable inside @cuda.jit kernels")

def trace(m):
    raise NotImplementedError("eigenprim.trace is only usable inside @cuda.jit kernels")

def diagonal(m):
    raise NotImplementedError("eigenprim.diagonal is only usable inside @cuda.jit kernels")

def vec_mul(m, v):
    raise NotImplementedError("eigenprim.vec_mul is only usable inside @cuda.jit kernels")


# All generic function names for __all__ / lazy imports.
DISPATCH_NAMES = [
    "add", "sub", "dot", "cross", "norm", "squared_norm", "normalized",
    "scale", "cwise_product", "cwise_abs", "cwise_min", "cwise_max",
    "sum", "min_coeff", "max_coeff", "outer", "mul",
    "determinant", "inverse", "transpose", "trace", "diagonal", "vec_mul",
]


# ── Overload registration ────────────────────────────────────────

def register_dispatch(types, functions):
    """Register @overload for all generic functions.

    Called by eigenprim.matrix after binding completes.
    """

    _VEC_SPECS = [
        ("vec2f", "Eigen::Matrix<float, 2, 1>", nbtypes.float32),
        ("vec3f", "Eigen::Matrix<float, 3, 1>", nbtypes.float32),
        ("vec4f", "Eigen::Matrix<float, 4, 1>", nbtypes.float32),
        ("vec2d", "Eigen::Matrix<double, 2, 1>", nbtypes.float64),
        ("vec3d", "Eigen::Matrix<double, 3, 1>", nbtypes.float64),
        ("vec4d", "Eigen::Matrix<double, 4, 1>", nbtypes.float64),
        ("vec2h", "Eigen::Matrix<Eigen::half, 2, 1>", nbtypes.float16),
        ("vec3h", "Eigen::Matrix<Eigen::half, 3, 1>", nbtypes.float16),
        ("vec4h", "Eigen::Matrix<Eigen::half, 4, 1>", nbtypes.float16),
        ("vec2bf", "Eigen::Matrix<Eigen::bfloat16, 2, 1>", nbtypes.float16),  # closest scalar
        ("vec3bf", "Eigen::Matrix<Eigen::bfloat16, 3, 1>", nbtypes.float16),
        ("vec4bf", "Eigen::Matrix<Eigen::bfloat16, 4, 1>", nbtypes.float16),
    ]

    _MAT_VEC_SPECS = [
        ("mat2f", "Eigen::Matrix<float, 2, 2>", "vec2f", "Eigen::Matrix<float, 2, 1>", nbtypes.float32),
        ("mat3f", "Eigen::Matrix<float, 3, 3>", "vec3f", "Eigen::Matrix<float, 3, 1>", nbtypes.float32),
        ("mat4f", "Eigen::Matrix<float, 4, 4>", "vec4f", "Eigen::Matrix<float, 4, 1>", nbtypes.float32),
        ("mat2d", "Eigen::Matrix<double, 2, 2>", "vec2d", "Eigen::Matrix<double, 2, 1>", nbtypes.float64),
        ("mat3d", "Eigen::Matrix<double, 3, 3>", "vec3d", "Eigen::Matrix<double, 3, 1>", nbtypes.float64),
        ("mat4d", "Eigen::Matrix<double, 4, 4>", "vec4d", "Eigen::Matrix<double, 4, 1>", nbtypes.float64),
        ("mat2h", "Eigen::Matrix<Eigen::half, 2, 2>", "vec2h", "Eigen::Matrix<Eigen::half, 2, 1>", nbtypes.float16),
        ("mat3h", "Eigen::Matrix<Eigen::half, 3, 3>", "vec3h", "Eigen::Matrix<Eigen::half, 3, 1>", nbtypes.float16),
        ("mat4h", "Eigen::Matrix<Eigen::half, 4, 4>", "vec4h", "Eigen::Matrix<Eigen::half, 4, 1>", nbtypes.float16),
        ("mat2bf", "Eigen::Matrix<Eigen::bfloat16, 2, 2>", "vec2bf", "Eigen::Matrix<Eigen::bfloat16, 2, 1>", nbtypes.float16),
        ("mat3bf", "Eigen::Matrix<Eigen::bfloat16, 3, 3>", "vec3bf", "Eigen::Matrix<Eigen::bfloat16, 3, 1>", nbtypes.float16),
        ("mat4bf", "Eigen::Matrix<Eigen::bfloat16, 4, 4>", "vec4bf", "Eigen::Matrix<Eigen::bfloat16, 4, 1>", nbtypes.float16),
    ]

    # ── Binary (same, same) → same ───────────────────────────────

    _binary_same = {}  # op_name -> {(nbt, nbt): fn}
    # Ops available for both vectors and matrices
    for op in ["add", "sub", "cwise_product"]:
        table = {}
        for vname, tkey, _ in _VEC_SPECS:
            vt = types[tkey]._nbtype
            table[(vt, vt)] = functions[f"eigen_{vname}_{op}"]
        for mname, mtkey, _, _, _ in _MAT_VEC_SPECS:
            mt = types[mtkey]._nbtype
            table[(mt, mt)] = functions[f"eigen_{mname}_{op}"]
        _binary_same[op] = table
    # Ops available for vectors only
    for op in ["cwise_min", "cwise_max"]:
        table = {}
        for vname, tkey, _ in _VEC_SPECS:
            vt = types[tkey]._nbtype
            table[(vt, vt)] = functions[f"eigen_{vname}_{op}"]
        _binary_same[op] = table

    # ── Binary (same, same) → scalar (dot only) ──────────────────

    _dot_table = {}
    for vname, tkey, _ in _VEC_SPECS:
        vt = types[tkey]._nbtype
        _dot_table[(vt, vt)] = functions[f"eigen_{vname}_dot"]

    # ── Cross (3D only) ──────────────────────────────────────────

    _cross_table = {}
    for vname, tkey, _ in _VEC_SPECS:
        if "3" in vname:
            vt = types[tkey]._nbtype
            _cross_table[(vt, vt)] = functions[f"eigen_{vname}_cross"]

    # ── Outer (vec, vec) → mat ───────────────────────────────────

    _outer_table = {}
    for vname, tkey, _ in _VEC_SPECS:
        vt = types[tkey]._nbtype
        _outer_table[(vt, vt)] = functions[f"eigen_{vname}_outer"]

    # ── Unary → same type ────────────────────────────────────────

    _unary_same = {}  # op_name -> {nbt: fn}
    for op in ["normalized", "cwise_abs"]:
        table = {}
        for vname, tkey, _ in _VEC_SPECS:
            vt = types[tkey]._nbtype
            table[vt] = functions[f"eigen_{vname}_{op}"]
        _unary_same[op] = table

    # inverse/transpose for matrices
    for op in ["inverse", "transpose"]:
        table = {}
        for mname, mtkey, _, _, _ in _MAT_VEC_SPECS:
            mt = types[mtkey]._nbtype
            table[mt] = functions[f"eigen_{mname}_{op}"]
        _unary_same[op] = table

    # ── Unary → scalar ───────────────────────────────────────────

    _unary_scalar = {}  # op_name -> {nbt: fn}
    for op in ["norm", "squared_norm", "sum", "min_coeff", "max_coeff"]:
        table = {}
        for vname, tkey, _ in _VEC_SPECS:
            vt = types[tkey]._nbtype
            table[vt] = functions[f"eigen_{vname}_{op}"]
        _unary_scalar[op] = table

    # norm/squared_norm also for matrices
    for op in ["norm", "squared_norm"]:
        for mname, mtkey, _, _, _ in _MAT_VEC_SPECS:
            mt = types[mtkey]._nbtype
            _unary_scalar[op][mt] = functions[f"eigen_{mname}_{op}"]

    # determinant/trace for matrices only
    for op in ["determinant", "trace"]:
        table = {}
        for mname, mtkey, _, _, _ in _MAT_VEC_SPECS:
            mt = types[mtkey]._nbtype
            table[mt] = functions[f"eigen_{mname}_{op}"]
        _unary_scalar[op] = table

    # ── Diagonal: mat → vec ──────────────────────────────────────

    _diagonal_table = {}
    for mname, mtkey, _, _, _ in _MAT_VEC_SPECS:
        mt = types[mtkey]._nbtype
        _diagonal_table[mt] = functions[f"eigen_{mname}_diagonal"]

    # ── Scale: (type, scalar) → same type ────────────────────────

    _scale_table = {}
    for vname, tkey, scalar_nbt in _VEC_SPECS:
        vt = types[tkey]._nbtype
        fn = functions[f"eigen_{vname}_scale"]
        for snbt in (nbtypes.float32, nbtypes.float64):
            _scale_table[(vt, snbt)] = fn
    for mname, mtkey, _, _, scalar_nbt in _MAT_VEC_SPECS:
        mt = types[mtkey]._nbtype
        fn = functions[f"eigen_{mname}_scale"]
        for snbt in (nbtypes.float32, nbtypes.float64):
            _scale_table[(mt, snbt)] = fn

    # ── Matrix multiply: mat × mat, mat × vec ────────────────────

    _mul_table = {}
    _vec_mul_table = {}
    for mname, mtkey, vname, vtkey, _ in _MAT_VEC_SPECS:
        mt = types[mtkey]._nbtype
        vt = types[vtkey]._nbtype
        _mul_table[(mt, mt)] = functions[f"eigen_{mname}_mul"]
        _vec_mul_table[(mt, vt)] = functions[f"eigen_{mname}_{vname}_mul"]

    # ── Register overloads ────────────────────────────────────────

    def _make_binary_overload(target_fn, table):
        @overload(target_fn)
        def _dispatch(a, b):
            fn = table.get((a, b))
            if fn is not None:
                def impl(a, b):
                    return fn(a, b)
                return impl

    def _make_unary_overload(target_fn, table):
        @overload(target_fn)
        def _dispatch(a):
            fn = table.get(a)
            if fn is not None:
                def impl(a):
                    return fn(a)
                return impl

    _make_binary_overload(add, _binary_same["add"])
    _make_binary_overload(sub, _binary_same["sub"])
    _make_binary_overload(cwise_product, _binary_same["cwise_product"])
    _make_binary_overload(cwise_min, _binary_same["cwise_min"])
    _make_binary_overload(cwise_max, _binary_same["cwise_max"])
    _make_binary_overload(dot, _dot_table)
    _make_binary_overload(cross, _cross_table)
    _make_binary_overload(outer, _outer_table)
    _make_binary_overload(scale, _scale_table)
    _make_binary_overload(mul, _mul_table)
    _make_binary_overload(vec_mul, _vec_mul_table)

    _make_unary_overload(normalized, _unary_same["normalized"])
    _make_unary_overload(cwise_abs, _unary_same["cwise_abs"])
    _make_unary_overload(inverse, _unary_same["inverse"])
    _make_unary_overload(transpose, _unary_same["transpose"])
    _make_unary_overload(norm, _unary_scalar["norm"])
    _make_unary_overload(squared_norm, _unary_scalar["squared_norm"])
    _make_unary_overload(sum, _unary_scalar["sum"])
    _make_unary_overload(min_coeff, _unary_scalar["min_coeff"])
    _make_unary_overload(max_coeff, _unary_scalar["max_coeff"])
    _make_unary_overload(determinant, _unary_scalar["determinant"])
    _make_unary_overload(trace, _unary_scalar["trace"])
    _make_unary_overload(diagonal, _diagonal_table)
