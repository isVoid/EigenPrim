// NVRTC-safe stub declarations for Eigen::Matrix.
//
// These provide just enough type info for NVRTC to compile shim functions
// that accept/return Eigen::Matrix by pointer. The memory layout matches
// the real Eigen::Matrix for fixed-size types (verified via ast_canopy
// parsing of DenseStorage.h):
//
//   Matrix<float,3,1> = 12 bytes (float[3]), align 4
//   Matrix<float,3,3> = 36 bytes (float[9]), align 4
//
// The actual Eigen implementations are compiled separately by nvcc
// into a fatbin and linked at kernel JIT time.

#pragma once

namespace Eigen {

template <typename Scalar, int Rows, int Cols,
          int Options = 0, int MaxRows = Rows, int MaxCols = Cols>
struct Matrix {
  Scalar m_storage[Rows * Cols];

  __host__ __device__ Matrix() {}

  // Constructor for 3-vectors (Matrix<Scalar,3,1>)
  __host__ __device__ Matrix(Scalar v0, Scalar v1, Scalar v2) {
    m_storage[0] = v0; m_storage[1] = v1; m_storage[2] = v2;
  }

  // Constructor for 3x3 matrices (Matrix<Scalar,3,3>), column-major order
  // m_storage layout: [col0_row0, col0_row1, col0_row2, col1_row0, ...]
  __host__ __device__ Matrix(Scalar c0r0, Scalar c0r1, Scalar c0r2,
                             Scalar c1r0, Scalar c1r1, Scalar c1r2,
                             Scalar c2r0, Scalar c2r1, Scalar c2r2) {
    m_storage[0] = c0r0; m_storage[1] = c0r1; m_storage[2] = c0r2;
    m_storage[3] = c1r0; m_storage[4] = c1r1; m_storage[5] = c1r2;
    m_storage[6] = c2r0; m_storage[7] = c2r1; m_storage[8] = c2r2;
  }
};

}  // namespace Eigen
