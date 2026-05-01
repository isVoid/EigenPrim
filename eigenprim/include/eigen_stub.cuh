// NVRTC-safe stub declarations for Eigen::Matrix.
//
// These provide just enough type info for NVRTC to compile shim functions
// that accept/return Eigen::Matrix by pointer. The memory layout matches
// the real Eigen::Matrix for fixed-size types (column-major DenseStorage):
//
//   Matrix<Scalar,R,C> = sizeof(Scalar) * R * C bytes, align = alignof(Scalar)
//
// The actual Eigen implementations are compiled separately by nvcc
// into a fatbin and linked at kernel JIT time.

#pragma once

namespace Eigen {

// Stub half/bfloat16 — 2-byte types layout-compatible with real Eigen scalars.
// The real Eigen::half wraps __half; the real Eigen::bfloat16 wraps __nv_bfloat16.
// These stubs match sizeof=2, alignof=2 so Matrix<half,R,C> has correct layout.
struct half { unsigned short x; __host__ __device__ half() {} };
struct bfloat16 { unsigned short x; __host__ __device__ bfloat16() {} };

template <typename Scalar, int Rows, int Cols,
          int Options = 0, int MaxRows = Rows, int MaxCols = Cols>
struct Matrix {
  Scalar m_storage[Rows * Cols];

  __host__ __device__ Matrix() {}

  // Constructor for 2-element types (Vector2)
  __host__ __device__ Matrix(Scalar v0, Scalar v1) {
    m_storage[0] = v0; m_storage[1] = v1;
  }

  // Constructor for 3-element types (Vector3)
  __host__ __device__ Matrix(Scalar v0, Scalar v1, Scalar v2) {
    m_storage[0] = v0; m_storage[1] = v1; m_storage[2] = v2;
  }

  // Constructor for 4-element types (Vector4 / Matrix2x2), column-major
  __host__ __device__ Matrix(Scalar v0, Scalar v1, Scalar v2, Scalar v3) {
    m_storage[0] = v0; m_storage[1] = v1;
    m_storage[2] = v2; m_storage[3] = v3;
  }

  // Constructor for 9-element types (Matrix3x3), column-major
  __host__ __device__ Matrix(Scalar c0r0, Scalar c0r1, Scalar c0r2,
                             Scalar c1r0, Scalar c1r1, Scalar c1r2,
                             Scalar c2r0, Scalar c2r1, Scalar c2r2) {
    m_storage[0] = c0r0; m_storage[1] = c0r1; m_storage[2] = c0r2;
    m_storage[3] = c1r0; m_storage[4] = c1r1; m_storage[5] = c1r2;
    m_storage[6] = c2r0; m_storage[7] = c2r1; m_storage[8] = c2r2;
  }

  // Constructor for 16-element types (Matrix4x4), column-major
  __host__ __device__ Matrix(Scalar c0r0, Scalar c0r1, Scalar c0r2, Scalar c0r3,
                             Scalar c1r0, Scalar c1r1, Scalar c1r2, Scalar c1r3,
                             Scalar c2r0, Scalar c2r1, Scalar c2r2, Scalar c2r3,
                             Scalar c3r0, Scalar c3r1, Scalar c3r2, Scalar c3r3) {
    m_storage[0]  = c0r0; m_storage[1]  = c0r1; m_storage[2]  = c0r2; m_storage[3]  = c0r3;
    m_storage[4]  = c1r0; m_storage[5]  = c1r1; m_storage[6]  = c1r2; m_storage[7]  = c1r3;
    m_storage[8]  = c2r0; m_storage[9]  = c2r1; m_storage[10] = c2r2; m_storage[11] = c2r3;
    m_storage[12] = c3r0; m_storage[13] = c3r1; m_storage[14] = c3r2; m_storage[15] = c3r3;
  }
};

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
struct Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols> {
  Scalar m_storage[Rows];

  __host__ __device__ Matrix() {}

  __host__ __device__ Matrix(Scalar v0, Scalar v1) {
    m_storage[0] = v0; m_storage[1] = v1;
  }

  __host__ __device__ Matrix(Scalar v0, Scalar v1, Scalar v2) {
    m_storage[0] = v0; m_storage[1] = v1; m_storage[2] = v2;
  }

  __host__ __device__ Matrix(Scalar v0, Scalar v1, Scalar v2, Scalar v3) {
    m_storage[0] = v0; m_storage[1] = v1;
    m_storage[2] = v2; m_storage[3] = v3;
  }

  __device__ Scalar dot(Matrix other) const;
  __device__ Scalar norm() const;
  __device__ Scalar squared_norm() const;
  __device__ Matrix normalized() const;
  __device__ Matrix scale(Scalar s) const;
  __device__ Matrix cwise_product(Matrix other) const;
  __device__ Matrix cwise_abs() const;
  __device__ Matrix cwise_min(Matrix other) const;
  __device__ Matrix cwise_max(Matrix other) const;
  __device__ Scalar sum() const;
  __device__ Scalar min_coeff() const;
  __device__ Scalar max_coeff() const;
  __device__ Matrix<Scalar, Rows, Rows> outer(Matrix other) const;
};

template <typename Scalar, int Options, int MaxRows, int MaxCols>
struct Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols> {
  Scalar m_storage[3];

  __host__ __device__ Matrix() {}

  __host__ __device__ Matrix(Scalar v0, Scalar v1) {
    m_storage[0] = v0; m_storage[1] = v1;
  }

  __host__ __device__ Matrix(Scalar v0, Scalar v1, Scalar v2) {
    m_storage[0] = v0; m_storage[1] = v1; m_storage[2] = v2;
  }

  __host__ __device__ Matrix(Scalar v0, Scalar v1, Scalar v2, Scalar v3) {
    m_storage[0] = v0; m_storage[1] = v1;
    m_storage[2] = v2; m_storage[3] = v3;
  }

  __device__ Scalar dot(Matrix other) const;
  __device__ Scalar norm() const;
  __device__ Scalar squared_norm() const;
  __device__ Matrix normalized() const;
  __device__ Matrix scale(Scalar s) const;
  __device__ Matrix cross(Matrix other) const;
  __device__ Matrix cwise_product(Matrix other) const;
  __device__ Matrix cwise_abs() const;
  __device__ Matrix cwise_min(Matrix other) const;
  __device__ Matrix cwise_max(Matrix other) const;
  __device__ Scalar sum() const;
  __device__ Scalar min_coeff() const;
  __device__ Scalar max_coeff() const;
  __device__ Matrix<Scalar, 3, 3> outer(Matrix other) const;
};

template <typename Scalar, int Dim, int Options, int MaxRows, int MaxCols>
struct Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols> {
  Scalar m_storage[Dim * Dim];

  __host__ __device__ Matrix() {}

  __host__ __device__ Matrix(Scalar v0, Scalar v1, Scalar v2, Scalar v3) {
    m_storage[0] = v0; m_storage[1] = v1;
    m_storage[2] = v2; m_storage[3] = v3;
  }

  __host__ __device__ Matrix(Scalar c0r0, Scalar c0r1, Scalar c0r2,
                             Scalar c1r0, Scalar c1r1, Scalar c1r2,
                             Scalar c2r0, Scalar c2r1, Scalar c2r2) {
    m_storage[0] = c0r0; m_storage[1] = c0r1; m_storage[2] = c0r2;
    m_storage[3] = c1r0; m_storage[4] = c1r1; m_storage[5] = c1r2;
    m_storage[6] = c2r0; m_storage[7] = c2r1; m_storage[8] = c2r2;
  }

  __host__ __device__ Matrix(Scalar c0r0, Scalar c0r1, Scalar c0r2, Scalar c0r3,
                             Scalar c1r0, Scalar c1r1, Scalar c1r2, Scalar c1r3,
                             Scalar c2r0, Scalar c2r1, Scalar c2r2, Scalar c2r3,
                             Scalar c3r0, Scalar c3r1, Scalar c3r2, Scalar c3r3) {
    m_storage[0]  = c0r0; m_storage[1]  = c0r1; m_storage[2]  = c0r2; m_storage[3]  = c0r3;
    m_storage[4]  = c1r0; m_storage[5]  = c1r1; m_storage[6]  = c1r2; m_storage[7]  = c1r3;
    m_storage[8]  = c2r0; m_storage[9]  = c2r1; m_storage[10] = c2r2; m_storage[11] = c2r3;
    m_storage[12] = c3r0; m_storage[13] = c3r1; m_storage[14] = c3r2; m_storage[15] = c3r3;
  }

  __device__ Scalar determinant() const;
  __device__ Matrix inverse() const;
  __device__ Matrix transpose() const;
  __device__ Scalar trace() const;
  __device__ Matrix cwise_product(Matrix other) const;
  __device__ Matrix scale(Scalar s) const;
  __device__ Scalar norm() const;
  __device__ Scalar squared_norm() const;
  __device__ Matrix<Scalar, Dim, 1> diagonal() const;
  __device__ Matrix<Scalar, Dim, 1> vec_mul(Matrix<Scalar, Dim, 1> v) const;
};

}  // namespace Eigen
