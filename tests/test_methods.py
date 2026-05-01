"""Integration tests for method invocation on Eigen types.

Each test runs a @cuda.jit kernel and checks that method-call syntax
(e.g. a.dot(b), v.norm(), M.inverse()) produces results matching
the corresponding generic dispatch functions (dot(a, b), norm(v), etc.).
"""

import numpy as np
import pytest
from numba import cuda


@pytest.fixture(scope="module")
def bindings():
    from eigenprim import matrix as m
    return m._bindings


@pytest.fixture(scope="module")
def kernel_links():
    from eigenprim import links
    return links()


# ── Vector methods ────────────────────────────────────────────────────────────

class TestVectorMethods:
    """Method calls on Vector3f; verifies parity with dispatch functions."""

    def test_dot_method(self, kernel_links):
        from eigenprim import Vector3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            a = Vector3f(1.0, 2.0, 3.0)
            b = Vector3f(4.0, 5.0, 6.0)
            out[0] = a.dot(b)

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 32.0, rtol=1e-5), f"dot: got {out[0]}, expected 32.0"

    def test_norm_method(self, kernel_links):
        from eigenprim import Vector3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            v = Vector3f(3.0, 4.0, 0.0)
            out[0] = v.norm()

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 5.0, rtol=1e-5), f"norm: got {out[0]}, expected 5.0"

    def test_squared_norm_method(self, kernel_links):
        from eigenprim import Vector3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            v = Vector3f(1.0, 2.0, 3.0)
            out[0] = v.squared_norm()

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 14.0, rtol=1e-5), f"squared_norm: got {out[0]}, expected 14.0"

    def test_normalized_method(self, kernel_links):
        from eigenprim import Vector3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            v = Vector3f(3.0, 4.0, 0.0)
            u = v.normalized()
            out[0] = u.norm()

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 1.0, rtol=1e-5), f"normalized().norm(): got {out[0]}, expected 1.0"

    def test_cross_method(self, kernel_links):
        from eigenprim import Vector3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            a = Vector3f(1.0, 0.0, 0.0)
            b = Vector3f(0.0, 1.0, 0.0)
            c = a.cross(b)          # should be (0, 0, 1)
            out[0] = c.dot(a)       # orthogonal -> 0
            out[1] = c.dot(b)       # orthogonal -> 0
            out[2] = c.norm()       # unit cross product -> 1

        out = np.zeros(3, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 0.0, atol=1e-5), f"cross orthogonality (a): {out[0]}"
        assert np.isclose(out[1], 0.0, atol=1e-5), f"cross orthogonality (b): {out[1]}"
        assert np.isclose(out[2], 1.0, rtol=1e-5), f"cross norm: {out[2]}"

    def test_scale_method(self, kernel_links):
        from eigenprim import Vector3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            v = Vector3f(1.0, 2.0, 3.0)
            w = v.scale(2.0)
            out[0] = w.dot(v)      # <2v, v> = 2 * |v|^2 = 2 * 14 = 28

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 28.0, rtol=1e-5), f"scale: got {out[0]}, expected 28.0"

    def test_cwise_product_method(self, kernel_links):
        from eigenprim import Vector3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            a = Vector3f(1.0, 2.0, 3.0)
            b = Vector3f(4.0, 5.0, 6.0)
            c = a.cwise_product(b)
            out[0] = c.sum()       # 4 + 10 + 18 = 32

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 32.0, rtol=1e-5), f"cwise_product: got {out[0]}, expected 32.0"

    def test_sum_min_max_methods(self, kernel_links):
        from eigenprim import Vector3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            v = Vector3f(1.0, 5.0, 3.0)
            out[0] = v.sum()
            out[1] = v.min_coeff()
            out[2] = v.max_coeff()

        out = np.zeros(3, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 9.0,  rtol=1e-5), f"sum: {out[0]}"
        assert np.isclose(out[1], 1.0,  rtol=1e-5), f"min_coeff: {out[1]}"
        assert np.isclose(out[2], 5.0,  rtol=1e-5), f"max_coeff: {out[2]}"

    def test_outer_method(self, kernel_links):
        from eigenprim import Vector3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            a = Vector3f(1.0, 0.0, 0.0)
            b = Vector3f(0.0, 1.0, 0.0)
            M = a.outer(b)         # rank-1 matrix: M[0,1]=1, else 0
            out[0] = M.trace()     # trace = 0 (off-diagonal outer product)

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 0.0, atol=1e-5), f"outer.trace: {out[0]}"

    def test_chained_methods(self, kernel_links):
        """Method results can be chained: a.cross(b).norm() etc."""
        from eigenprim import Vector3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            a = Vector3f(3.0, 4.0, 0.0)
            out[0] = a.normalized().norm()          # always 1.0
            out[1] = a.cwise_abs().min_coeff()      # min(3, 4, 0) = 0

        out = np.zeros(2, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 1.0, rtol=1e-5), f"normalized().norm(): {out[0]}"
        assert np.isclose(out[1], 0.0, atol=1e-5), f"cwise_abs().min_coeff(): {out[1]}"


# ── Matrix methods ────────────────────────────────────────────────────────────

class TestMatrixMethods:
    """Method calls on Matrix3f; verifies parity with dispatch functions."""

    def test_determinant_method(self, kernel_links):
        from eigenprim import Matrix3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            M = Matrix3f(
                2.0, 0.0, 0.0,
                0.0, 3.0, 0.0,
                0.0, 0.0, 4.0,
            )
            out[0] = M.determinant()  # 2*3*4 = 24

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 24.0, rtol=1e-4), f"determinant: got {out[0]}, expected 24.0"

    def test_trace_method(self, kernel_links):
        from eigenprim import Matrix3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            M = Matrix3f(
                1.0, 0.0, 0.0,
                0.0, 2.0, 0.0,
                0.0, 0.0, 3.0,
            )
            out[0] = M.trace()  # 1+2+3 = 6

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 6.0, rtol=1e-5), f"trace: got {out[0]}, expected 6.0"

    def test_inverse_method(self, kernel_links):
        from eigenprim import Matrix3f, Vector3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            M = Matrix3f(
                2.0, 0.0, 0.0,
                0.0, 2.0, 0.0,
                0.0, 0.0, 2.0,
            )
            Minv = M.inverse()
            out[0] = Minv.trace()  # 3 * (1/2) = 1.5

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 1.5, rtol=1e-4), f"inverse trace: got {out[0]}, expected 1.5"

    def test_transpose_method(self, kernel_links):
        from eigenprim import Matrix3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            # Upper-triangular: off-diagonal element at [0,1] = 7
            # After transpose it moves to [1,0]; trace unchanged
            M = Matrix3f(
                1.0, 0.0, 0.0,
                7.0, 2.0, 0.0,
                0.0, 0.0, 3.0,
            )
            MT = M.transpose()
            out[0] = MT.trace()    # trace preserved = 6

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 6.0, rtol=1e-5), f"transpose trace: {out[0]}"

    def test_vec_mul_method(self, kernel_links):
        from eigenprim import Matrix3f, Vector3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            M = Matrix3f(
                2.0, 0.0, 0.0,
                0.0, 2.0, 0.0,
                0.0, 0.0, 2.0,
            )
            v = Vector3f(1.0, 2.0, 3.0)
            w = M.vec_mul(v)       # 2 * v
            out[0] = w.dot(v)      # <2v, v> = 2|v|^2 = 2*14 = 28

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 28.0, rtol=1e-4), f"vec_mul: {out[0]}"

    def test_diagonal_method(self, kernel_links):
        from eigenprim import Matrix3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            M = Matrix3f(
                1.0, 0.0, 0.0,
                0.0, 2.0, 0.0,
                0.0, 0.0, 3.0,
            )
            d = M.diagonal()
            out[0] = d.sum()  # 1+2+3 = 6

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 6.0, rtol=1e-5), f"diagonal.sum: {out[0]}"

    def test_norm_method(self, kernel_links):
        from eigenprim import Matrix3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            # Identity matrix: Frobenius norm = sqrt(3)
            M = Matrix3f(
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            )
            out[0] = M.norm()

        import math
        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], np.sqrt(3), rtol=1e-4), f"norm: {out[0]}"

    def test_scale_method(self, kernel_links):
        from eigenprim import Matrix3f

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            M = Matrix3f(
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            )
            S = M.scale(3.0)
            out[0] = S.trace()  # 3 * 3 = 9

        out = np.zeros(1, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], 9.0, rtol=1e-5), f"scale trace: {out[0]}"


# ── Cross-type parity ─────────────────────────────────────────────────────────

class TestMethodDispatchParity:
    """Method calls produce identical results to the dispatch functions."""

    def test_dot_method_matches_dispatch(self, kernel_links):
        from eigenprim import Vector3f, dot

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            a = Vector3f(1.0, 2.0, 3.0)
            b = Vector3f(4.0, 5.0, 6.0)
            out[0] = a.dot(b)
            out[1] = dot(a, b)

        out = np.zeros(2, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], out[1], rtol=1e-6), (
            f"method result {out[0]} != dispatch result {out[1]}"
        )

    def test_norm_method_matches_dispatch(self, kernel_links):
        from eigenprim import Vector3f, norm

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            v = Vector3f(1.0, 2.0, 3.0)
            out[0] = v.norm()
            out[1] = norm(v)

        out = np.zeros(2, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], out[1], rtol=1e-6), (
            f"method result {out[0]} != dispatch result {out[1]}"
        )

    def test_inverse_method_matches_dispatch(self, kernel_links):
        from eigenprim import Matrix3f, inverse

        @cuda.jit(link=kernel_links)
        def _kernel(out):
            M = Matrix3f(
                2.0, 0.0, 0.0,
                0.0, 2.0, 0.0,
                0.0, 0.0, 2.0,
            )
            out[0] = M.inverse().trace()
            out[1] = inverse(M).trace()

        out = np.zeros(2, dtype=np.float32)
        _kernel[1, 1](out)
        assert np.isclose(out[0], out[1], rtol=1e-5), (
            f"method result {out[0]} != dispatch result {out[1]}"
        )
