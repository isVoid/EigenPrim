"""Broad API surface checks for EigenPrim vector and matrix bindings."""

import numpy as np
import pytest
from numba import cuda


VECTOR_SPECS = [
    ("vec2f", "Vector2f", "mat2f", 2, np.float32, 1e-4, 1e-5),
    ("vec3f", "Vector3f", "mat3f", 3, np.float32, 1e-4, 1e-5),
    ("vec4f", "Vector4f", "mat4f", 4, np.float32, 1e-4, 1e-5),
    ("vec2d", "Vector2d", "mat2d", 2, np.float64, 1e-10, 1e-12),
    ("vec3d", "Vector3d", "mat3d", 3, np.float64, 1e-10, 1e-12),
    ("vec4d", "Vector4d", "mat4d", 4, np.float64, 1e-10, 1e-12),
    ("vec2h", "Vector2h", "mat2h", 2, np.float16, 2e-2, 2e-2),
    ("vec3h", "Vector3h", "mat3h", 3, np.float16, 2e-2, 2e-2),
    ("vec4h", "Vector4h", "mat4h", 4, np.float16, 2e-2, 2e-2),
    ("vec2bf", "Vector2bf", "mat2bf", 2, np.float32, 3e-2, 3e-2),
    ("vec3bf", "Vector3bf", "mat3bf", 3, np.float32, 3e-2, 3e-2),
    ("vec4bf", "Vector4bf", "mat4bf", 4, np.float32, 3e-2, 3e-2),
]

MATRIX_SPECS = [
    ("mat2f", "Matrix2f", "vec2f", "Vector2f", 2, np.float32, 1e-4, 1e-5),
    ("mat3f", "Matrix3f", "vec3f", "Vector3f", 3, np.float32, 1e-4, 1e-5),
    ("mat4f", "Matrix4f", "vec4f", "Vector4f", 4, np.float32, 1e-4, 1e-5),
    ("mat2d", "Matrix2d", "vec2d", "Vector2d", 2, np.float64, 1e-10, 1e-12),
    ("mat3d", "Matrix3d", "vec3d", "Vector3d", 3, np.float64, 1e-10, 1e-12),
    ("mat4d", "Matrix4d", "vec4d", "Vector4d", 4, np.float64, 1e-10, 1e-12),
    ("mat2h", "Matrix2h", "vec2h", "Vector2h", 2, np.float16, 2e-2, 2e-2),
    ("mat3h", "Matrix3h", "vec3h", "Vector3h", 3, np.float16, 2e-2, 2e-2),
    ("mat4h", "Matrix4h", "vec4h", "Vector4h", 4, np.float16, 2e-2, 2e-2),
    ("mat2bf", "Matrix2bf", "vec2bf", "Vector2bf", 2, np.float32, 3e-2, 3e-2),
    ("mat3bf", "Matrix3bf", "vec3bf", "Vector3bf", 3, np.float32, 3e-2, 3e-2),
    ("mat4bf", "Matrix4bf", "vec4bf", "Vector4bf", 4, np.float32, 3e-2, 3e-2),
]


@pytest.fixture(scope="module")
def matrix_module():
    from eigenprim import matrix as m

    return m


@pytest.fixture(scope="module")
def kernel_links():
    from eigenprim import links

    return links()


def _vector_expected(dim):
    a = np.arange(1, dim + 1, dtype=np.float64)
    b = np.arange(dim + 1, 2 * dim + 1, dtype=np.float64)

    labels = []
    expected = []

    def add_variants(name, value, variants=("direct", "dispatch", "method")):
        for variant in variants:
            labels.append(f"{variant}.{name}")
            expected.append(value)

    add_variants("add", np.sum(a + b), ("direct", "dispatch", "operator"))
    add_variants("sub", np.sum(b - a), ("direct", "dispatch", "operator"))
    add_variants("dot", np.dot(a, b))
    add_variants("norm", np.linalg.norm(a))
    add_variants("squared_norm", np.dot(a, a))
    add_variants("normalized", 1.0)
    add_variants("scale", 2.0 * np.sum(a), ("direct", "dispatch", "method", "operator_rhs", "operator_lhs"))
    add_variants("cwise_product", np.dot(a, b))
    add_variants("cwise_abs", np.sum(np.abs(-a)))
    add_variants("cwise_min", np.sum(np.minimum(a, b)))
    add_variants("cwise_max", np.sum(np.maximum(a, b)))
    add_variants("sum", np.sum(a))
    add_variants("min_coeff", np.min(a))
    add_variants("max_coeff", np.max(a))
    add_variants("outer", np.dot(a, b))

    if dim == 3:
        add_variants("cross", 1.0)

    return labels, np.asarray(expected, dtype=np.float64)


def _matrix_inputs(dim):
    if dim == 2:
        a = np.asarray([
            2.0, 1.0,
            0.0, 3.0,
        ])
        b = np.asarray([
            5.0, 2.0,
            0.0, 7.0,
        ])
        v = np.asarray([11.0, 13.0])
        return a.reshape((2, 2), order="F"), b.reshape((2, 2), order="F"), v
    if dim == 3:
        a = np.asarray([
            2.0, 1.0, 0.0,
            0.0, 3.0, 1.0,
            0.0, 0.0, 4.0,
        ])
        b = np.asarray([
            5.0, 2.0, 0.0,
            0.0, 7.0, 2.0,
            0.0, 0.0, 11.0,
        ])
        v = np.asarray([13.0, 17.0, 19.0])
        return a.reshape((3, 3), order="F"), b.reshape((3, 3), order="F"), v

    a = np.asarray([
        2.0, 1.0, 0.0, 0.0,
        0.0, 3.0, 1.0, 0.0,
        0.0, 0.0, 5.0, 1.0,
        0.0, 0.0, 0.0, 7.0,
    ])
    b = np.asarray([
        11.0, 2.0, 0.0, 0.0,
        0.0, 13.0, 2.0, 0.0,
        0.0, 0.0, 17.0, 2.0,
        0.0, 0.0, 0.0, 19.0,
    ])
    v = np.asarray([23.0, 29.0, 31.0, 37.0])
    return a.reshape((4, 4), order="F"), b.reshape((4, 4), order="F"), v


def _matrix_expected(dim):
    a, b, v = _matrix_inputs(dim)

    labels = []
    expected = []

    def apply_sum(matrix):
        return np.sum(matrix @ v)

    def add_variants(name, value, variants=("direct", "dispatch", "method")):
        for variant in variants:
            labels.append(f"{variant}.{name}")
            expected.append(value)

    add_variants("add", apply_sum(a + b), ("direct", "dispatch", "operator"))
    add_variants("sub", apply_sum(b - a), ("direct", "dispatch", "operator"))
    add_variants("mul", apply_sum(a @ b), ("direct", "dispatch", "operator"))
    add_variants("determinant", np.linalg.det(a))
    add_variants("inverse", apply_sum(np.linalg.inv(a)))
    add_variants("transpose", apply_sum(a.T))
    add_variants("trace", np.trace(a))
    add_variants("cwise_product", apply_sum(a * b))
    add_variants("scale", apply_sum(2.0 * a))
    add_variants("norm", np.linalg.norm(a))
    add_variants("squared_norm", np.sum(a * a))
    add_variants("diagonal", np.trace(a))
    add_variants("vec_mul", np.sum(a @ v), ("direct", "dispatch", "method", "operator"))

    return labels, np.asarray(expected, dtype=np.float64)


def _assert_surface(out, expected, labels, rtol, atol):
    got = np.asarray(out, dtype=np.float64)
    for label, actual, want in zip(labels, got, expected):
        np.testing.assert_allclose(
            actual,
            want,
            rtol=rtol,
            atol=atol,
            err_msg=label,
        )


def _make_vector_kernel(spec, m, kernel_links):
    import eigenprim as ep

    tag, type_name, mat_tag, dim, *_ = spec
    Vec = getattr(m, type_name)

    fn_add = getattr(m, f"eigen_{tag}_add")
    fn_sub = getattr(m, f"eigen_{tag}_sub")
    fn_dot = getattr(m, f"eigen_{tag}_dot")
    fn_norm = getattr(m, f"eigen_{tag}_norm")
    fn_squared_norm = getattr(m, f"eigen_{tag}_squared_norm")
    fn_normalized = getattr(m, f"eigen_{tag}_normalized")
    fn_scale = getattr(m, f"eigen_{tag}_scale")
    fn_cwise_product = getattr(m, f"eigen_{tag}_cwise_product")
    fn_cwise_abs = getattr(m, f"eigen_{tag}_cwise_abs")
    fn_cwise_min = getattr(m, f"eigen_{tag}_cwise_min")
    fn_cwise_max = getattr(m, f"eigen_{tag}_cwise_max")
    fn_sum = getattr(m, f"eigen_{tag}_sum")
    fn_min_coeff = getattr(m, f"eigen_{tag}_min_coeff")
    fn_max_coeff = getattr(m, f"eigen_{tag}_max_coeff")
    fn_outer = getattr(m, f"eigen_{tag}_outer")
    fn_mat_trace = getattr(m, f"eigen_{mat_tag}_trace")
    fn_cross = getattr(m, f"eigen_{tag}_cross") if dim == 3 else None

    dadd = ep.add
    dsub = ep.sub
    ddot = ep.dot
    dcross = ep.cross
    dnorm = ep.norm
    dsquared_norm = ep.squared_norm
    dnormalized = ep.normalized
    dscale = ep.scale
    dcwise_product = ep.cwise_product
    dcwise_abs = ep.cwise_abs
    dcwise_min = ep.cwise_min
    dcwise_max = ep.cwise_max
    dsum = ep.sum
    dmin_coeff = ep.min_coeff
    dmax_coeff = ep.max_coeff
    douter = ep.outer

    @cuda.jit(link=kernel_links)
    def _kernel(out):
        if dim == 2:
            a = Vec(1.0, 2.0)
            b = Vec(3.0, 4.0)
            neg = Vec(-1.0, -2.0)
        elif dim == 3:
            a = Vec(1.0, 2.0, 3.0)
            b = Vec(4.0, 5.0, 6.0)
            neg = Vec(-1.0, -2.0, -3.0)
        else:
            a = Vec(1.0, 2.0, 3.0, 4.0)
            b = Vec(5.0, 6.0, 7.0, 8.0)
            neg = Vec(-1.0, -2.0, -3.0, -4.0)

        out[0] = fn_sum(fn_add(a, b))
        out[1] = fn_sum(dadd(a, b))
        out[2] = fn_sum(a + b)

        out[3] = fn_sum(fn_sub(b, a))
        out[4] = fn_sum(dsub(b, a))
        out[5] = fn_sum(b - a)

        out[6] = fn_dot(a, b)
        out[7] = ddot(a, b)
        out[8] = a.dot(b)

        out[9] = fn_norm(a)
        out[10] = dnorm(a)
        out[11] = a.norm()

        out[12] = fn_squared_norm(a)
        out[13] = dsquared_norm(a)
        out[14] = a.squared_norm()

        out[15] = fn_norm(fn_normalized(a))
        out[16] = fn_norm(dnormalized(a))
        out[17] = fn_norm(a.normalized())

        out[18] = fn_sum(fn_scale(a, 2.0))
        out[19] = fn_sum(dscale(a, 2.0))
        out[20] = fn_sum(a.scale(2.0))
        out[21] = fn_sum(a * 2.0)
        out[22] = fn_sum(2.0 * a)

        out[23] = fn_sum(fn_cwise_product(a, b))
        out[24] = fn_sum(dcwise_product(a, b))
        out[25] = fn_sum(a.cwise_product(b))

        out[26] = fn_sum(fn_cwise_abs(neg))
        out[27] = fn_sum(dcwise_abs(neg))
        out[28] = fn_sum(neg.cwise_abs())

        out[29] = fn_sum(fn_cwise_min(a, b))
        out[30] = fn_sum(dcwise_min(a, b))
        out[31] = fn_sum(a.cwise_min(b))

        out[32] = fn_sum(fn_cwise_max(a, b))
        out[33] = fn_sum(dcwise_max(a, b))
        out[34] = fn_sum(a.cwise_max(b))

        out[35] = fn_sum(a)
        out[36] = dsum(a)
        out[37] = a.sum()

        out[38] = fn_min_coeff(a)
        out[39] = dmin_coeff(a)
        out[40] = a.min_coeff()

        out[41] = fn_max_coeff(a)
        out[42] = dmax_coeff(a)
        out[43] = a.max_coeff()

        out[44] = fn_mat_trace(fn_outer(a, b))
        out[45] = fn_mat_trace(douter(a, b))
        out[46] = fn_mat_trace(a.outer(b))

        if dim == 3:
            x = Vec(1.0, 0.0, 0.0)
            y = Vec(0.0, 1.0, 0.0)
            out[47] = fn_norm(fn_cross(x, y))
            out[48] = fn_norm(dcross(x, y))
            out[49] = fn_norm(x.cross(y))

    return _kernel


def _make_matrix_kernel(spec, m, kernel_links):
    import eigenprim as ep

    tag, type_name, vec_tag, vec_type_name, dim, *_ = spec
    Mat = getattr(m, type_name)
    Vec = getattr(m, vec_type_name)

    fn_add = getattr(m, f"eigen_{tag}_add")
    fn_sub = getattr(m, f"eigen_{tag}_sub")
    fn_mul = getattr(m, f"eigen_{tag}_mul")
    fn_determinant = getattr(m, f"eigen_{tag}_determinant")
    fn_inverse = getattr(m, f"eigen_{tag}_inverse")
    fn_transpose = getattr(m, f"eigen_{tag}_transpose")
    fn_trace = getattr(m, f"eigen_{tag}_trace")
    fn_cwise_product = getattr(m, f"eigen_{tag}_cwise_product")
    fn_scale = getattr(m, f"eigen_{tag}_scale")
    fn_norm = getattr(m, f"eigen_{tag}_norm")
    fn_squared_norm = getattr(m, f"eigen_{tag}_squared_norm")
    fn_diagonal = getattr(m, f"eigen_{tag}_diagonal")
    fn_vec_mul = getattr(m, f"eigen_{tag}_{vec_tag}_mul")
    fn_vec_sum = getattr(m, f"eigen_{vec_tag}_sum")

    dadd = ep.add
    dsub = ep.sub
    dmul = ep.mul
    ddeterminant = ep.determinant
    dinverse = ep.inverse
    dtranspose = ep.transpose
    dtrace = ep.trace
    dcwise_product = ep.cwise_product
    dscale = ep.scale
    dnorm = ep.norm
    dsquared_norm = ep.squared_norm
    ddiagonal = ep.diagonal
    dvec_mul = ep.vec_mul

    @cuda.jit(link=kernel_links)
    def _kernel(out):
        if dim == 2:
            a = Mat(2.0, 1.0, 0.0, 3.0)
            b = Mat(5.0, 2.0, 0.0, 7.0)
            v = Vec(11.0, 13.0)
        elif dim == 3:
            a = Mat(
                2.0, 1.0, 0.0,
                0.0, 3.0, 1.0,
                0.0, 0.0, 4.0,
            )
            b = Mat(
                5.0, 2.0, 0.0,
                0.0, 7.0, 2.0,
                0.0, 0.0, 11.0,
            )
            v = Vec(13.0, 17.0, 19.0)
        else:
            a = Mat(
                2.0, 1.0, 0.0, 0.0,
                0.0, 3.0, 1.0, 0.0,
                0.0, 0.0, 5.0, 1.0,
                0.0, 0.0, 0.0, 7.0,
            )
            b = Mat(
                11.0, 2.0, 0.0, 0.0,
                0.0, 13.0, 2.0, 0.0,
                0.0, 0.0, 17.0, 2.0,
                0.0, 0.0, 0.0, 19.0,
            )
            v = Vec(23.0, 29.0, 31.0, 37.0)

        out[0] = fn_vec_sum(fn_vec_mul(fn_add(a, b), v))
        out[1] = fn_vec_sum(fn_vec_mul(dadd(a, b), v))
        out[2] = fn_vec_sum(fn_vec_mul(a + b, v))

        out[3] = fn_vec_sum(fn_vec_mul(fn_sub(b, a), v))
        out[4] = fn_vec_sum(fn_vec_mul(dsub(b, a), v))
        out[5] = fn_vec_sum(fn_vec_mul(b - a, v))

        out[6] = fn_vec_sum(fn_vec_mul(fn_mul(a, b), v))
        out[7] = fn_vec_sum(fn_vec_mul(dmul(a, b), v))
        out[8] = fn_vec_sum(fn_vec_mul(a @ b, v))

        out[9] = fn_determinant(a)
        out[10] = ddeterminant(a)
        out[11] = a.determinant()

        out[12] = fn_vec_sum(fn_vec_mul(fn_inverse(a), v))
        out[13] = fn_vec_sum(fn_vec_mul(dinverse(a), v))
        out[14] = fn_vec_sum(fn_vec_mul(a.inverse(), v))

        out[15] = fn_vec_sum(fn_vec_mul(fn_transpose(a), v))
        out[16] = fn_vec_sum(fn_vec_mul(dtranspose(a), v))
        out[17] = fn_vec_sum(fn_vec_mul(a.transpose(), v))

        out[18] = fn_trace(a)
        out[19] = dtrace(a)
        out[20] = a.trace()

        out[21] = fn_vec_sum(fn_vec_mul(fn_cwise_product(a, b), v))
        out[22] = fn_vec_sum(fn_vec_mul(dcwise_product(a, b), v))
        out[23] = fn_vec_sum(fn_vec_mul(a.cwise_product(b), v))

        out[24] = fn_vec_sum(fn_vec_mul(fn_scale(a, 2.0), v))
        out[25] = fn_vec_sum(fn_vec_mul(dscale(a, 2.0), v))
        out[26] = fn_vec_sum(fn_vec_mul(a.scale(2.0), v))

        out[27] = fn_norm(a)
        out[28] = dnorm(a)
        out[29] = a.norm()

        out[30] = fn_squared_norm(a)
        out[31] = dsquared_norm(a)
        out[32] = a.squared_norm()

        out[33] = fn_vec_sum(fn_diagonal(a))
        out[34] = fn_vec_sum(ddiagonal(a))
        out[35] = fn_vec_sum(a.diagonal())

        out[36] = fn_vec_sum(fn_vec_mul(a, v))
        out[37] = fn_vec_sum(dvec_mul(a, v))
        out[38] = fn_vec_sum(a.vec_mul(v))
        out[39] = fn_vec_sum(a @ v)

    return _kernel


@pytest.mark.parametrize("spec", VECTOR_SPECS, ids=lambda s: s[0])
def test_vector_api_surface(spec, matrix_module, kernel_links):
    labels, expected = _vector_expected(spec[3])
    kernel = _make_vector_kernel(spec, matrix_module, kernel_links)
    out = np.zeros(len(labels), dtype=spec[4])

    kernel[1, 1](out)

    _assert_surface(out, expected, labels, spec[5], spec[6])


@pytest.mark.parametrize("spec", MATRIX_SPECS, ids=lambda s: s[0])
def test_matrix_api_surface(spec, matrix_module, kernel_links):
    labels, expected = _matrix_expected(spec[4])
    kernel = _make_matrix_kernel(spec, matrix_module, kernel_links)
    out = np.zeros(len(labels), dtype=spec[5])

    kernel[1, 1](out)

    _assert_surface(out, expected, labels, spec[6], spec[7])
