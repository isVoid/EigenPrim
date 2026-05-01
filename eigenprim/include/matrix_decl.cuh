// Matrix: Declarations-only header for Eigen vector and matrix functions.
// Uses the NVRTC-safe Eigen stub (no real Eigen dependency).
// Implementations are in matrix.cuh (compiled separately by nvcc).

#pragma once

#include "eigen_stub.cuh"

// ── Type aliases ─────────────────────────────────────────────────

// Vectors
using EigenVec2f = Eigen::Matrix<float, 2, 1>;
using EigenVec3f = Eigen::Matrix<float, 3, 1>;
using EigenVec4f = Eigen::Matrix<float, 4, 1>;
using EigenVec2d = Eigen::Matrix<double, 2, 1>;
using EigenVec3d = Eigen::Matrix<double, 3, 1>;
using EigenVec4d = Eigen::Matrix<double, 4, 1>;

// Matrices
using EigenMat2f = Eigen::Matrix<float, 2, 2>;
using EigenMat3f = Eigen::Matrix<float, 3, 3>;
using EigenMat4f = Eigen::Matrix<float, 4, 4>;
using EigenMat2d = Eigen::Matrix<double, 2, 2>;
using EigenMat3d = Eigen::Matrix<double, 3, 3>;
using EigenMat4d = Eigen::Matrix<double, 4, 4>;

template <typename MatT, typename VecT, int Dim>
static __device__ __forceinline__ MatT eigenprim_stub_outer(VecT a, VecT b) {
  MatT result;
#pragma unroll
  for (int col = 0; col < Dim; ++col) {
#pragma unroll
    for (int row = 0; row < Dim; ++row) {
      result.m_storage[row + col * Dim] = a.m_storage[row] * b.m_storage[col];
    }
  }
  return result;
}

// Vector4f/Vector4d outer products stay in this NVRTC-safe header because the
// fatbin return-by-value path for these large matrices is not stable.

// ── Vector2f ─────────────────────────────────────────────────────

__device__ EigenVec2f eigen_vec2f_add(EigenVec2f a, EigenVec2f b);
__device__ EigenVec2f eigen_vec2f_sub(EigenVec2f a, EigenVec2f b);
__device__ float eigen_vec2f_dot(EigenVec2f a, EigenVec2f b);
__device__ float eigen_vec2f_norm(EigenVec2f v);
__device__ float eigen_vec2f_squared_norm(EigenVec2f v);
__device__ EigenVec2f eigen_vec2f_normalized(EigenVec2f v);
__device__ EigenVec2f eigen_vec2f_scale(EigenVec2f v, float s);
__device__ EigenVec2f eigen_vec2f_cwise_product(EigenVec2f a, EigenVec2f b);
__device__ EigenVec2f eigen_vec2f_cwise_abs(EigenVec2f v);
__device__ EigenVec2f eigen_vec2f_cwise_min(EigenVec2f a, EigenVec2f b);
__device__ EigenVec2f eigen_vec2f_cwise_max(EigenVec2f a, EigenVec2f b);
__device__ float eigen_vec2f_sum(EigenVec2f v);
__device__ float eigen_vec2f_min_coeff(EigenVec2f v);
__device__ float eigen_vec2f_max_coeff(EigenVec2f v);
__device__ EigenMat2f eigen_vec2f_outer(EigenVec2f a, EigenVec2f b);

// ── Vector3f ─────────────────────────────────────────────────────

__device__ EigenVec3f eigen_vec3f_add(EigenVec3f a, EigenVec3f b);
__device__ EigenVec3f eigen_vec3f_sub(EigenVec3f a, EigenVec3f b);
__device__ float eigen_vec3f_dot(EigenVec3f a, EigenVec3f b);
__device__ float eigen_vec3f_norm(EigenVec3f v);
__device__ float eigen_vec3f_squared_norm(EigenVec3f v);
__device__ EigenVec3f eigen_vec3f_normalized(EigenVec3f v);
__device__ EigenVec3f eigen_vec3f_scale(EigenVec3f v, float s);
__device__ EigenVec3f eigen_vec3f_cross(EigenVec3f a, EigenVec3f b);
__device__ EigenVec3f eigen_vec3f_cwise_product(EigenVec3f a, EigenVec3f b);
__device__ EigenVec3f eigen_vec3f_cwise_abs(EigenVec3f v);
__device__ EigenVec3f eigen_vec3f_cwise_min(EigenVec3f a, EigenVec3f b);
__device__ EigenVec3f eigen_vec3f_cwise_max(EigenVec3f a, EigenVec3f b);
__device__ float eigen_vec3f_sum(EigenVec3f v);
__device__ float eigen_vec3f_min_coeff(EigenVec3f v);
__device__ float eigen_vec3f_max_coeff(EigenVec3f v);
__device__ EigenMat3f eigen_vec3f_outer(EigenVec3f a, EigenVec3f b);

// ── Vector4f ─────────────────────────────────────────────────────

__device__ EigenVec4f eigen_vec4f_add(EigenVec4f a, EigenVec4f b);
__device__ EigenVec4f eigen_vec4f_sub(EigenVec4f a, EigenVec4f b);
__device__ float eigen_vec4f_dot(EigenVec4f a, EigenVec4f b);
__device__ float eigen_vec4f_norm(EigenVec4f v);
__device__ float eigen_vec4f_squared_norm(EigenVec4f v);
__device__ EigenVec4f eigen_vec4f_normalized(EigenVec4f v);
__device__ EigenVec4f eigen_vec4f_scale(EigenVec4f v, float s);
__device__ EigenVec4f eigen_vec4f_cwise_product(EigenVec4f a, EigenVec4f b);
__device__ EigenVec4f eigen_vec4f_cwise_abs(EigenVec4f v);
__device__ EigenVec4f eigen_vec4f_cwise_min(EigenVec4f a, EigenVec4f b);
__device__ EigenVec4f eigen_vec4f_cwise_max(EigenVec4f a, EigenVec4f b);
__device__ float eigen_vec4f_sum(EigenVec4f v);
__device__ float eigen_vec4f_min_coeff(EigenVec4f v);
__device__ float eigen_vec4f_max_coeff(EigenVec4f v);
static __device__ __forceinline__ EigenMat4f eigen_vec4f_outer(EigenVec4f a, EigenVec4f b) {
  return eigenprim_stub_outer<EigenMat4f, EigenVec4f, 4>(a, b);
}

// ── Vector2d ─────────────────────────────────────────────────────

__device__ EigenVec2d eigen_vec2d_add(EigenVec2d a, EigenVec2d b);
__device__ EigenVec2d eigen_vec2d_sub(EigenVec2d a, EigenVec2d b);
__device__ double eigen_vec2d_dot(EigenVec2d a, EigenVec2d b);
__device__ double eigen_vec2d_norm(EigenVec2d v);
__device__ double eigen_vec2d_squared_norm(EigenVec2d v);
__device__ EigenVec2d eigen_vec2d_normalized(EigenVec2d v);
__device__ EigenVec2d eigen_vec2d_scale(EigenVec2d v, double s);
__device__ EigenVec2d eigen_vec2d_cwise_product(EigenVec2d a, EigenVec2d b);
__device__ EigenVec2d eigen_vec2d_cwise_abs(EigenVec2d v);
__device__ EigenVec2d eigen_vec2d_cwise_min(EigenVec2d a, EigenVec2d b);
__device__ EigenVec2d eigen_vec2d_cwise_max(EigenVec2d a, EigenVec2d b);
__device__ double eigen_vec2d_sum(EigenVec2d v);
__device__ double eigen_vec2d_min_coeff(EigenVec2d v);
__device__ double eigen_vec2d_max_coeff(EigenVec2d v);
__device__ EigenMat2d eigen_vec2d_outer(EigenVec2d a, EigenVec2d b);

// ── Vector3d ─────────────────────────────────────────────────────

__device__ EigenVec3d eigen_vec3d_add(EigenVec3d a, EigenVec3d b);
__device__ EigenVec3d eigen_vec3d_sub(EigenVec3d a, EigenVec3d b);
__device__ double eigen_vec3d_dot(EigenVec3d a, EigenVec3d b);
__device__ double eigen_vec3d_norm(EigenVec3d v);
__device__ double eigen_vec3d_squared_norm(EigenVec3d v);
__device__ EigenVec3d eigen_vec3d_normalized(EigenVec3d v);
__device__ EigenVec3d eigen_vec3d_scale(EigenVec3d v, double s);
__device__ EigenVec3d eigen_vec3d_cross(EigenVec3d a, EigenVec3d b);
__device__ EigenVec3d eigen_vec3d_cwise_product(EigenVec3d a, EigenVec3d b);
__device__ EigenVec3d eigen_vec3d_cwise_abs(EigenVec3d v);
__device__ EigenVec3d eigen_vec3d_cwise_min(EigenVec3d a, EigenVec3d b);
__device__ EigenVec3d eigen_vec3d_cwise_max(EigenVec3d a, EigenVec3d b);
__device__ double eigen_vec3d_sum(EigenVec3d v);
__device__ double eigen_vec3d_min_coeff(EigenVec3d v);
__device__ double eigen_vec3d_max_coeff(EigenVec3d v);
__device__ EigenMat3d eigen_vec3d_outer(EigenVec3d a, EigenVec3d b);

// ── Vector4d ─────────────────────────────────────────────────────

__device__ EigenVec4d eigen_vec4d_add(EigenVec4d a, EigenVec4d b);
__device__ EigenVec4d eigen_vec4d_sub(EigenVec4d a, EigenVec4d b);
__device__ double eigen_vec4d_dot(EigenVec4d a, EigenVec4d b);
__device__ double eigen_vec4d_norm(EigenVec4d v);
__device__ double eigen_vec4d_squared_norm(EigenVec4d v);
__device__ EigenVec4d eigen_vec4d_normalized(EigenVec4d v);
__device__ EigenVec4d eigen_vec4d_scale(EigenVec4d v, double s);
__device__ EigenVec4d eigen_vec4d_cwise_product(EigenVec4d a, EigenVec4d b);
__device__ EigenVec4d eigen_vec4d_cwise_abs(EigenVec4d v);
__device__ EigenVec4d eigen_vec4d_cwise_min(EigenVec4d a, EigenVec4d b);
__device__ EigenVec4d eigen_vec4d_cwise_max(EigenVec4d a, EigenVec4d b);
__device__ double eigen_vec4d_sum(EigenVec4d v);
__device__ double eigen_vec4d_min_coeff(EigenVec4d v);
__device__ double eigen_vec4d_max_coeff(EigenVec4d v);
static __device__ __forceinline__ EigenMat4d eigen_vec4d_outer(EigenVec4d a, EigenVec4d b) {
  return eigenprim_stub_outer<EigenMat4d, EigenVec4d, 4>(a, b);
}

// ── Matrix2f ─────────────────────────────────────────────────────

__device__ EigenMat2f eigen_mat2f_add(EigenMat2f a, EigenMat2f b);
__device__ EigenMat2f eigen_mat2f_sub(EigenMat2f a, EigenMat2f b);
__device__ EigenMat2f eigen_mat2f_mul(EigenMat2f a, EigenMat2f b);
__device__ EigenVec2f eigen_mat2f_vec2f_mul(EigenMat2f m, EigenVec2f v);
__device__ float eigen_mat2f_determinant(EigenMat2f m);
__device__ EigenMat2f eigen_mat2f_inverse(EigenMat2f m);
__device__ EigenMat2f eigen_mat2f_transpose(EigenMat2f m);
__device__ float eigen_mat2f_trace(EigenMat2f m);
__device__ EigenMat2f eigen_mat2f_cwise_product(EigenMat2f a, EigenMat2f b);
__device__ EigenMat2f eigen_mat2f_scale(EigenMat2f m, float s);
__device__ float eigen_mat2f_norm(EigenMat2f m);
__device__ float eigen_mat2f_squared_norm(EigenMat2f m);
__device__ EigenVec2f eigen_mat2f_diagonal(EigenMat2f m);

// ── Matrix3f ─────────────────────────────────────────────────────

__device__ EigenMat3f eigen_mat3f_add(EigenMat3f a, EigenMat3f b);
__device__ EigenMat3f eigen_mat3f_sub(EigenMat3f a, EigenMat3f b);
__device__ EigenMat3f eigen_mat3f_mul(EigenMat3f a, EigenMat3f b);
__device__ EigenVec3f eigen_mat3f_vec3f_mul(EigenMat3f m, EigenVec3f v);
__device__ float eigen_mat3f_determinant(EigenMat3f m);
__device__ EigenMat3f eigen_mat3f_inverse(EigenMat3f m);
__device__ EigenMat3f eigen_mat3f_transpose(EigenMat3f m);
__device__ float eigen_mat3f_trace(EigenMat3f m);
__device__ EigenMat3f eigen_mat3f_cwise_product(EigenMat3f a, EigenMat3f b);
__device__ EigenMat3f eigen_mat3f_scale(EigenMat3f m, float s);
__device__ float eigen_mat3f_norm(EigenMat3f m);
__device__ float eigen_mat3f_squared_norm(EigenMat3f m);
__device__ EigenVec3f eigen_mat3f_diagonal(EigenMat3f m);

// ── Matrix4f ─────────────────────────────────────────────────────

__device__ EigenMat4f eigen_mat4f_add(EigenMat4f a, EigenMat4f b);
__device__ EigenMat4f eigen_mat4f_sub(EigenMat4f a, EigenMat4f b);
__device__ EigenMat4f eigen_mat4f_mul(EigenMat4f a, EigenMat4f b);
__device__ EigenVec4f eigen_mat4f_vec4f_mul(EigenMat4f m, EigenVec4f v);
__device__ float eigen_mat4f_determinant(EigenMat4f m);
__device__ EigenMat4f eigen_mat4f_inverse(EigenMat4f m);
__device__ EigenMat4f eigen_mat4f_transpose(EigenMat4f m);
__device__ float eigen_mat4f_trace(EigenMat4f m);
__device__ EigenMat4f eigen_mat4f_cwise_product(EigenMat4f a, EigenMat4f b);
__device__ EigenMat4f eigen_mat4f_scale(EigenMat4f m, float s);
__device__ float eigen_mat4f_norm(EigenMat4f m);
__device__ float eigen_mat4f_squared_norm(EigenMat4f m);
__device__ EigenVec4f eigen_mat4f_diagonal(EigenMat4f m);

// ── Matrix2d ─────────────────────────────────────────────────────

__device__ EigenMat2d eigen_mat2d_add(EigenMat2d a, EigenMat2d b);
__device__ EigenMat2d eigen_mat2d_sub(EigenMat2d a, EigenMat2d b);
__device__ EigenMat2d eigen_mat2d_mul(EigenMat2d a, EigenMat2d b);
__device__ EigenVec2d eigen_mat2d_vec2d_mul(EigenMat2d m, EigenVec2d v);
__device__ double eigen_mat2d_determinant(EigenMat2d m);
__device__ EigenMat2d eigen_mat2d_inverse(EigenMat2d m);
__device__ EigenMat2d eigen_mat2d_transpose(EigenMat2d m);
__device__ double eigen_mat2d_trace(EigenMat2d m);
__device__ EigenMat2d eigen_mat2d_cwise_product(EigenMat2d a, EigenMat2d b);
__device__ EigenMat2d eigen_mat2d_scale(EigenMat2d m, double s);
__device__ double eigen_mat2d_norm(EigenMat2d m);
__device__ double eigen_mat2d_squared_norm(EigenMat2d m);
__device__ EigenVec2d eigen_mat2d_diagonal(EigenMat2d m);

// ── Matrix3d ─────────────────────────────────────────────────────

__device__ EigenMat3d eigen_mat3d_add(EigenMat3d a, EigenMat3d b);
__device__ EigenMat3d eigen_mat3d_sub(EigenMat3d a, EigenMat3d b);
__device__ EigenMat3d eigen_mat3d_mul(EigenMat3d a, EigenMat3d b);
__device__ EigenVec3d eigen_mat3d_vec3d_mul(EigenMat3d m, EigenVec3d v);
__device__ double eigen_mat3d_determinant(EigenMat3d m);
__device__ EigenMat3d eigen_mat3d_inverse(EigenMat3d m);
__device__ EigenMat3d eigen_mat3d_transpose(EigenMat3d m);
__device__ double eigen_mat3d_trace(EigenMat3d m);
__device__ EigenMat3d eigen_mat3d_cwise_product(EigenMat3d a, EigenMat3d b);
__device__ EigenMat3d eigen_mat3d_scale(EigenMat3d m, double s);
__device__ double eigen_mat3d_norm(EigenMat3d m);
__device__ double eigen_mat3d_squared_norm(EigenMat3d m);
__device__ EigenVec3d eigen_mat3d_diagonal(EigenMat3d m);

// ── Matrix4d ─────────────────────────────────────────────────────

__device__ EigenMat4d eigen_mat4d_add(EigenMat4d a, EigenMat4d b);
__device__ EigenMat4d eigen_mat4d_sub(EigenMat4d a, EigenMat4d b);
__device__ EigenMat4d eigen_mat4d_mul(EigenMat4d a, EigenMat4d b);
__device__ EigenVec4d eigen_mat4d_vec4d_mul(EigenMat4d m, EigenVec4d v);
__device__ double eigen_mat4d_determinant(EigenMat4d m);
__device__ EigenMat4d eigen_mat4d_inverse(EigenMat4d m);
__device__ EigenMat4d eigen_mat4d_transpose(EigenMat4d m);
__device__ double eigen_mat4d_trace(EigenMat4d m);
__device__ EigenMat4d eigen_mat4d_cwise_product(EigenMat4d a, EigenMat4d b);
__device__ EigenMat4d eigen_mat4d_scale(EigenMat4d m, double s);
__device__ double eigen_mat4d_norm(EigenMat4d m);
__device__ double eigen_mat4d_squared_norm(EigenMat4d m);
__device__ EigenVec4d eigen_mat4d_diagonal(EigenMat4d m);

// ── Half-precision type aliases ──────────────────────────────────

using EigenVec2h = Eigen::Matrix<Eigen::half, 2, 1>;
using EigenVec3h = Eigen::Matrix<Eigen::half, 3, 1>;
using EigenVec4h = Eigen::Matrix<Eigen::half, 4, 1>;
using EigenMat2h = Eigen::Matrix<Eigen::half, 2, 2>;
using EigenMat3h = Eigen::Matrix<Eigen::half, 3, 3>;
using EigenMat4h = Eigen::Matrix<Eigen::half, 4, 4>;

using EigenVec2bf = Eigen::Matrix<Eigen::bfloat16, 2, 1>;
using EigenVec3bf = Eigen::Matrix<Eigen::bfloat16, 3, 1>;
using EigenVec4bf = Eigen::Matrix<Eigen::bfloat16, 4, 1>;
using EigenMat2bf = Eigen::Matrix<Eigen::bfloat16, 2, 2>;
using EigenMat3bf = Eigen::Matrix<Eigen::bfloat16, 3, 3>;
using EigenMat4bf = Eigen::Matrix<Eigen::bfloat16, 4, 4>;

// ── Vector2h ─────────────────────────────────────────────────────

__device__ EigenVec2h eigen_vec2h_add(EigenVec2h a, EigenVec2h b);
__device__ EigenVec2h eigen_vec2h_sub(EigenVec2h a, EigenVec2h b);
__device__ Eigen::half eigen_vec2h_dot(EigenVec2h a, EigenVec2h b);
__device__ Eigen::half eigen_vec2h_norm(EigenVec2h v);
__device__ Eigen::half eigen_vec2h_squared_norm(EigenVec2h v);
__device__ EigenVec2h eigen_vec2h_normalized(EigenVec2h v);
__device__ EigenVec2h eigen_vec2h_scale(EigenVec2h v, Eigen::half s);
__device__ EigenVec2h eigen_vec2h_cwise_product(EigenVec2h a, EigenVec2h b);
__device__ EigenVec2h eigen_vec2h_cwise_abs(EigenVec2h v);
__device__ EigenVec2h eigen_vec2h_cwise_min(EigenVec2h a, EigenVec2h b);
__device__ EigenVec2h eigen_vec2h_cwise_max(EigenVec2h a, EigenVec2h b);
__device__ Eigen::half eigen_vec2h_sum(EigenVec2h v);
__device__ Eigen::half eigen_vec2h_min_coeff(EigenVec2h v);
__device__ Eigen::half eigen_vec2h_max_coeff(EigenVec2h v);
__device__ EigenMat2h eigen_vec2h_outer(EigenVec2h a, EigenVec2h b);

// ── Vector3h ─────────────────────────────────────────────────────

__device__ EigenVec3h eigen_vec3h_add(EigenVec3h a, EigenVec3h b);
__device__ EigenVec3h eigen_vec3h_sub(EigenVec3h a, EigenVec3h b);
__device__ Eigen::half eigen_vec3h_dot(EigenVec3h a, EigenVec3h b);
__device__ Eigen::half eigen_vec3h_norm(EigenVec3h v);
__device__ Eigen::half eigen_vec3h_squared_norm(EigenVec3h v);
__device__ EigenVec3h eigen_vec3h_normalized(EigenVec3h v);
__device__ EigenVec3h eigen_vec3h_scale(EigenVec3h v, Eigen::half s);
__device__ EigenVec3h eigen_vec3h_cross(EigenVec3h a, EigenVec3h b);
__device__ EigenVec3h eigen_vec3h_cwise_product(EigenVec3h a, EigenVec3h b);
__device__ EigenVec3h eigen_vec3h_cwise_abs(EigenVec3h v);
__device__ EigenVec3h eigen_vec3h_cwise_min(EigenVec3h a, EigenVec3h b);
__device__ EigenVec3h eigen_vec3h_cwise_max(EigenVec3h a, EigenVec3h b);
__device__ Eigen::half eigen_vec3h_sum(EigenVec3h v);
__device__ Eigen::half eigen_vec3h_min_coeff(EigenVec3h v);
__device__ Eigen::half eigen_vec3h_max_coeff(EigenVec3h v);
__device__ EigenMat3h eigen_vec3h_outer(EigenVec3h a, EigenVec3h b);

// ── Vector4h ─────────────────────────────────────────────────────

__device__ EigenVec4h eigen_vec4h_add(EigenVec4h a, EigenVec4h b);
__device__ EigenVec4h eigen_vec4h_sub(EigenVec4h a, EigenVec4h b);
__device__ Eigen::half eigen_vec4h_dot(EigenVec4h a, EigenVec4h b);
__device__ Eigen::half eigen_vec4h_norm(EigenVec4h v);
__device__ Eigen::half eigen_vec4h_squared_norm(EigenVec4h v);
__device__ EigenVec4h eigen_vec4h_normalized(EigenVec4h v);
__device__ EigenVec4h eigen_vec4h_scale(EigenVec4h v, Eigen::half s);
__device__ EigenVec4h eigen_vec4h_cwise_product(EigenVec4h a, EigenVec4h b);
__device__ EigenVec4h eigen_vec4h_cwise_abs(EigenVec4h v);
__device__ EigenVec4h eigen_vec4h_cwise_min(EigenVec4h a, EigenVec4h b);
__device__ EigenVec4h eigen_vec4h_cwise_max(EigenVec4h a, EigenVec4h b);
__device__ Eigen::half eigen_vec4h_sum(EigenVec4h v);
__device__ Eigen::half eigen_vec4h_min_coeff(EigenVec4h v);
__device__ Eigen::half eigen_vec4h_max_coeff(EigenVec4h v);
__device__ EigenMat4h eigen_vec4h_outer(EigenVec4h a, EigenVec4h b);

// ── Matrix2h ─────────────────────────────────────────────────────

__device__ EigenMat2h eigen_mat2h_add(EigenMat2h a, EigenMat2h b);
__device__ EigenMat2h eigen_mat2h_sub(EigenMat2h a, EigenMat2h b);
__device__ EigenMat2h eigen_mat2h_mul(EigenMat2h a, EigenMat2h b);
__device__ EigenVec2h eigen_mat2h_vec2h_mul(EigenMat2h m, EigenVec2h v);
__device__ Eigen::half eigen_mat2h_determinant(EigenMat2h m);
__device__ EigenMat2h eigen_mat2h_inverse(EigenMat2h m);
__device__ EigenMat2h eigen_mat2h_transpose(EigenMat2h m);
__device__ Eigen::half eigen_mat2h_trace(EigenMat2h m);
__device__ EigenMat2h eigen_mat2h_cwise_product(EigenMat2h a, EigenMat2h b);
__device__ EigenMat2h eigen_mat2h_scale(EigenMat2h m, Eigen::half s);
__device__ Eigen::half eigen_mat2h_norm(EigenMat2h m);
__device__ Eigen::half eigen_mat2h_squared_norm(EigenMat2h m);
__device__ EigenVec2h eigen_mat2h_diagonal(EigenMat2h m);

// ── Matrix3h ─────────────────────────────────────────────────────

__device__ EigenMat3h eigen_mat3h_add(EigenMat3h a, EigenMat3h b);
__device__ EigenMat3h eigen_mat3h_sub(EigenMat3h a, EigenMat3h b);
__device__ EigenMat3h eigen_mat3h_mul(EigenMat3h a, EigenMat3h b);
__device__ EigenVec3h eigen_mat3h_vec3h_mul(EigenMat3h m, EigenVec3h v);
__device__ Eigen::half eigen_mat3h_determinant(EigenMat3h m);
__device__ EigenMat3h eigen_mat3h_inverse(EigenMat3h m);
__device__ EigenMat3h eigen_mat3h_transpose(EigenMat3h m);
__device__ Eigen::half eigen_mat3h_trace(EigenMat3h m);
__device__ EigenMat3h eigen_mat3h_cwise_product(EigenMat3h a, EigenMat3h b);
__device__ EigenMat3h eigen_mat3h_scale(EigenMat3h m, Eigen::half s);
__device__ Eigen::half eigen_mat3h_norm(EigenMat3h m);
__device__ Eigen::half eigen_mat3h_squared_norm(EigenMat3h m);
__device__ EigenVec3h eigen_mat3h_diagonal(EigenMat3h m);

// ── Matrix4h ─────────────────────────────────────────────────────

__device__ EigenMat4h eigen_mat4h_add(EigenMat4h a, EigenMat4h b);
__device__ EigenMat4h eigen_mat4h_sub(EigenMat4h a, EigenMat4h b);
__device__ EigenMat4h eigen_mat4h_mul(EigenMat4h a, EigenMat4h b);
__device__ EigenVec4h eigen_mat4h_vec4h_mul(EigenMat4h m, EigenVec4h v);
__device__ Eigen::half eigen_mat4h_determinant(EigenMat4h m);
__device__ EigenMat4h eigen_mat4h_inverse(EigenMat4h m);
__device__ EigenMat4h eigen_mat4h_transpose(EigenMat4h m);
__device__ Eigen::half eigen_mat4h_trace(EigenMat4h m);
__device__ EigenMat4h eigen_mat4h_cwise_product(EigenMat4h a, EigenMat4h b);
__device__ EigenMat4h eigen_mat4h_scale(EigenMat4h m, Eigen::half s);
__device__ Eigen::half eigen_mat4h_norm(EigenMat4h m);
__device__ Eigen::half eigen_mat4h_squared_norm(EigenMat4h m);
__device__ EigenVec4h eigen_mat4h_diagonal(EigenMat4h m);

// ── Vector2bf ────────────────────────────────────────────────────

__device__ EigenVec2bf eigen_vec2bf_add(EigenVec2bf a, EigenVec2bf b);
__device__ EigenVec2bf eigen_vec2bf_sub(EigenVec2bf a, EigenVec2bf b);
__device__ Eigen::bfloat16 eigen_vec2bf_dot(EigenVec2bf a, EigenVec2bf b);
__device__ Eigen::bfloat16 eigen_vec2bf_norm(EigenVec2bf v);
__device__ Eigen::bfloat16 eigen_vec2bf_squared_norm(EigenVec2bf v);
__device__ EigenVec2bf eigen_vec2bf_normalized(EigenVec2bf v);
__device__ EigenVec2bf eigen_vec2bf_scale(EigenVec2bf v, Eigen::bfloat16 s);
__device__ EigenVec2bf eigen_vec2bf_cwise_product(EigenVec2bf a, EigenVec2bf b);
__device__ EigenVec2bf eigen_vec2bf_cwise_abs(EigenVec2bf v);
__device__ EigenVec2bf eigen_vec2bf_cwise_min(EigenVec2bf a, EigenVec2bf b);
__device__ EigenVec2bf eigen_vec2bf_cwise_max(EigenVec2bf a, EigenVec2bf b);
__device__ Eigen::bfloat16 eigen_vec2bf_sum(EigenVec2bf v);
__device__ Eigen::bfloat16 eigen_vec2bf_min_coeff(EigenVec2bf v);
__device__ Eigen::bfloat16 eigen_vec2bf_max_coeff(EigenVec2bf v);
__device__ EigenMat2bf eigen_vec2bf_outer(EigenVec2bf a, EigenVec2bf b);

// ── Vector3bf ────────────────────────────────────────────────────

__device__ EigenVec3bf eigen_vec3bf_add(EigenVec3bf a, EigenVec3bf b);
__device__ EigenVec3bf eigen_vec3bf_sub(EigenVec3bf a, EigenVec3bf b);
__device__ Eigen::bfloat16 eigen_vec3bf_dot(EigenVec3bf a, EigenVec3bf b);
__device__ Eigen::bfloat16 eigen_vec3bf_norm(EigenVec3bf v);
__device__ Eigen::bfloat16 eigen_vec3bf_squared_norm(EigenVec3bf v);
__device__ EigenVec3bf eigen_vec3bf_normalized(EigenVec3bf v);
__device__ EigenVec3bf eigen_vec3bf_scale(EigenVec3bf v, Eigen::bfloat16 s);
__device__ EigenVec3bf eigen_vec3bf_cross(EigenVec3bf a, EigenVec3bf b);
__device__ EigenVec3bf eigen_vec3bf_cwise_product(EigenVec3bf a, EigenVec3bf b);
__device__ EigenVec3bf eigen_vec3bf_cwise_abs(EigenVec3bf v);
__device__ EigenVec3bf eigen_vec3bf_cwise_min(EigenVec3bf a, EigenVec3bf b);
__device__ EigenVec3bf eigen_vec3bf_cwise_max(EigenVec3bf a, EigenVec3bf b);
__device__ Eigen::bfloat16 eigen_vec3bf_sum(EigenVec3bf v);
__device__ Eigen::bfloat16 eigen_vec3bf_min_coeff(EigenVec3bf v);
__device__ Eigen::bfloat16 eigen_vec3bf_max_coeff(EigenVec3bf v);
__device__ EigenMat3bf eigen_vec3bf_outer(EigenVec3bf a, EigenVec3bf b);

// ── Vector4bf ────────────────────────────────────────────────────

__device__ EigenVec4bf eigen_vec4bf_add(EigenVec4bf a, EigenVec4bf b);
__device__ EigenVec4bf eigen_vec4bf_sub(EigenVec4bf a, EigenVec4bf b);
__device__ Eigen::bfloat16 eigen_vec4bf_dot(EigenVec4bf a, EigenVec4bf b);
__device__ Eigen::bfloat16 eigen_vec4bf_norm(EigenVec4bf v);
__device__ Eigen::bfloat16 eigen_vec4bf_squared_norm(EigenVec4bf v);
__device__ EigenVec4bf eigen_vec4bf_normalized(EigenVec4bf v);
__device__ EigenVec4bf eigen_vec4bf_scale(EigenVec4bf v, Eigen::bfloat16 s);
__device__ EigenVec4bf eigen_vec4bf_cwise_product(EigenVec4bf a, EigenVec4bf b);
__device__ EigenVec4bf eigen_vec4bf_cwise_abs(EigenVec4bf v);
__device__ EigenVec4bf eigen_vec4bf_cwise_min(EigenVec4bf a, EigenVec4bf b);
__device__ EigenVec4bf eigen_vec4bf_cwise_max(EigenVec4bf a, EigenVec4bf b);
__device__ Eigen::bfloat16 eigen_vec4bf_sum(EigenVec4bf v);
__device__ Eigen::bfloat16 eigen_vec4bf_min_coeff(EigenVec4bf v);
__device__ Eigen::bfloat16 eigen_vec4bf_max_coeff(EigenVec4bf v);
__device__ EigenMat4bf eigen_vec4bf_outer(EigenVec4bf a, EigenVec4bf b);

// ── Matrix2bf ────────────────────────────────────────────────────

__device__ EigenMat2bf eigen_mat2bf_add(EigenMat2bf a, EigenMat2bf b);
__device__ EigenMat2bf eigen_mat2bf_sub(EigenMat2bf a, EigenMat2bf b);
__device__ EigenMat2bf eigen_mat2bf_mul(EigenMat2bf a, EigenMat2bf b);
__device__ EigenVec2bf eigen_mat2bf_vec2bf_mul(EigenMat2bf m, EigenVec2bf v);
__device__ Eigen::bfloat16 eigen_mat2bf_determinant(EigenMat2bf m);
__device__ EigenMat2bf eigen_mat2bf_inverse(EigenMat2bf m);
__device__ EigenMat2bf eigen_mat2bf_transpose(EigenMat2bf m);
__device__ Eigen::bfloat16 eigen_mat2bf_trace(EigenMat2bf m);
__device__ EigenMat2bf eigen_mat2bf_cwise_product(EigenMat2bf a, EigenMat2bf b);
__device__ EigenMat2bf eigen_mat2bf_scale(EigenMat2bf m, Eigen::bfloat16 s);
__device__ Eigen::bfloat16 eigen_mat2bf_norm(EigenMat2bf m);
__device__ Eigen::bfloat16 eigen_mat2bf_squared_norm(EigenMat2bf m);
__device__ EigenVec2bf eigen_mat2bf_diagonal(EigenMat2bf m);

// ── Matrix3bf ────────────────────────────────────────────────────

__device__ EigenMat3bf eigen_mat3bf_add(EigenMat3bf a, EigenMat3bf b);
__device__ EigenMat3bf eigen_mat3bf_sub(EigenMat3bf a, EigenMat3bf b);
__device__ EigenMat3bf eigen_mat3bf_mul(EigenMat3bf a, EigenMat3bf b);
__device__ EigenVec3bf eigen_mat3bf_vec3bf_mul(EigenMat3bf m, EigenVec3bf v);
__device__ Eigen::bfloat16 eigen_mat3bf_determinant(EigenMat3bf m);
__device__ EigenMat3bf eigen_mat3bf_inverse(EigenMat3bf m);
__device__ EigenMat3bf eigen_mat3bf_transpose(EigenMat3bf m);
__device__ Eigen::bfloat16 eigen_mat3bf_trace(EigenMat3bf m);
__device__ EigenMat3bf eigen_mat3bf_cwise_product(EigenMat3bf a, EigenMat3bf b);
__device__ EigenMat3bf eigen_mat3bf_scale(EigenMat3bf m, Eigen::bfloat16 s);
__device__ Eigen::bfloat16 eigen_mat3bf_norm(EigenMat3bf m);
__device__ Eigen::bfloat16 eigen_mat3bf_squared_norm(EigenMat3bf m);
__device__ EigenVec3bf eigen_mat3bf_diagonal(EigenMat3bf m);

// ── Matrix4bf ────────────────────────────────────────────────────

__device__ EigenMat4bf eigen_mat4bf_add(EigenMat4bf a, EigenMat4bf b);
__device__ EigenMat4bf eigen_mat4bf_sub(EigenMat4bf a, EigenMat4bf b);
__device__ EigenMat4bf eigen_mat4bf_mul(EigenMat4bf a, EigenMat4bf b);
__device__ EigenVec4bf eigen_mat4bf_vec4bf_mul(EigenMat4bf m, EigenVec4bf v);
__device__ Eigen::bfloat16 eigen_mat4bf_determinant(EigenMat4bf m);
__device__ EigenMat4bf eigen_mat4bf_inverse(EigenMat4bf m);
__device__ EigenMat4bf eigen_mat4bf_transpose(EigenMat4bf m);
__device__ Eigen::bfloat16 eigen_mat4bf_trace(EigenMat4bf m);
__device__ EigenMat4bf eigen_mat4bf_cwise_product(EigenMat4bf a, EigenMat4bf b);
__device__ EigenMat4bf eigen_mat4bf_scale(EigenMat4bf m, Eigen::bfloat16 s);
__device__ Eigen::bfloat16 eigen_mat4bf_norm(EigenMat4bf m);
__device__ Eigen::bfloat16 eigen_mat4bf_squared_norm(EigenMat4bf m);
__device__ EigenVec4bf eigen_mat4bf_diagonal(EigenMat4bf m);

// NVRTC-safe member trampolines. Eigen::Matrix stub methods forward to these
// overloads, and these overloads call the standalone functions defined in the
// NVCC-compiled fatbin.
namespace eigenprim_detail {

#define EIGENPRIM_VECTOR_TRAMPOLINES(tag, VecT, MatT, ScalarT)                           \
__device__ inline ScalarT dot(VecT a, VecT b) { return eigen_##tag##_dot(a, b); }         \
__device__ inline ScalarT norm(VecT v) { return eigen_##tag##_norm(v); }                  \
__device__ inline ScalarT squared_norm(VecT v) { return eigen_##tag##_squared_norm(v); }  \
__device__ inline VecT normalized(VecT v) { return eigen_##tag##_normalized(v); }         \
__device__ inline VecT scale(VecT v, ScalarT s) { return eigen_##tag##_scale(v, s); }     \
__device__ inline VecT cwise_product(VecT a, VecT b) { return eigen_##tag##_cwise_product(a, b); } \
__device__ inline VecT cwise_abs(VecT v) { return eigen_##tag##_cwise_abs(v); }           \
__device__ inline VecT cwise_min(VecT a, VecT b) { return eigen_##tag##_cwise_min(a, b); } \
__device__ inline VecT cwise_max(VecT a, VecT b) { return eigen_##tag##_cwise_max(a, b); } \
__device__ inline ScalarT sum(VecT v) { return eigen_##tag##_sum(v); }                    \
__device__ inline ScalarT min_coeff(VecT v) { return eigen_##tag##_min_coeff(v); }        \
__device__ inline ScalarT max_coeff(VecT v) { return eigen_##tag##_max_coeff(v); }        \
__device__ inline MatT outer(VecT a, VecT b) { return eigen_##tag##_outer(a, b); }

#define EIGENPRIM_VECTOR3_TRAMPOLINES(tag, VecT, MatT, ScalarT)                          \
EIGENPRIM_VECTOR_TRAMPOLINES(tag, VecT, MatT, ScalarT)                                   \
__device__ inline VecT cross(VecT a, VecT b) { return eigen_##tag##_cross(a, b); }

#define EIGENPRIM_MATRIX_TRAMPOLINES(mtag, MatT, vtag, VecT, ScalarT)                    \
__device__ inline ScalarT determinant(MatT m) { return eigen_##mtag##_determinant(m); }   \
__device__ inline MatT inverse(MatT m) { return eigen_##mtag##_inverse(m); }              \
__device__ inline MatT transpose(MatT m) { return eigen_##mtag##_transpose(m); }          \
__device__ inline ScalarT trace(MatT m) { return eigen_##mtag##_trace(m); }               \
__device__ inline MatT cwise_product(MatT a, MatT b) { return eigen_##mtag##_cwise_product(a, b); } \
__device__ inline MatT scale(MatT m, ScalarT s) { return eigen_##mtag##_scale(m, s); }    \
__device__ inline ScalarT norm(MatT m) { return eigen_##mtag##_norm(m); }                 \
__device__ inline ScalarT squared_norm(MatT m) { return eigen_##mtag##_squared_norm(m); } \
__device__ inline VecT diagonal(MatT m) { return eigen_##mtag##_diagonal(m); }            \
__device__ inline VecT vec_mul(MatT m, VecT v) { return eigen_##mtag##_##vtag##_mul(m, v); }

EIGENPRIM_VECTOR_TRAMPOLINES(vec2f, EigenVec2f, EigenMat2f, float)
EIGENPRIM_VECTOR3_TRAMPOLINES(vec3f, EigenVec3f, EigenMat3f, float)
EIGENPRIM_VECTOR_TRAMPOLINES(vec4f, EigenVec4f, EigenMat4f, float)
EIGENPRIM_VECTOR_TRAMPOLINES(vec2d, EigenVec2d, EigenMat2d, double)
EIGENPRIM_VECTOR3_TRAMPOLINES(vec3d, EigenVec3d, EigenMat3d, double)
EIGENPRIM_VECTOR_TRAMPOLINES(vec4d, EigenVec4d, EigenMat4d, double)
EIGENPRIM_VECTOR_TRAMPOLINES(vec2h, EigenVec2h, EigenMat2h, Eigen::half)
EIGENPRIM_VECTOR3_TRAMPOLINES(vec3h, EigenVec3h, EigenMat3h, Eigen::half)
EIGENPRIM_VECTOR_TRAMPOLINES(vec4h, EigenVec4h, EigenMat4h, Eigen::half)
EIGENPRIM_VECTOR_TRAMPOLINES(vec2bf, EigenVec2bf, EigenMat2bf, Eigen::bfloat16)
EIGENPRIM_VECTOR3_TRAMPOLINES(vec3bf, EigenVec3bf, EigenMat3bf, Eigen::bfloat16)
EIGENPRIM_VECTOR_TRAMPOLINES(vec4bf, EigenVec4bf, EigenMat4bf, Eigen::bfloat16)

EIGENPRIM_MATRIX_TRAMPOLINES(mat2f, EigenMat2f, vec2f, EigenVec2f, float)
EIGENPRIM_MATRIX_TRAMPOLINES(mat3f, EigenMat3f, vec3f, EigenVec3f, float)
EIGENPRIM_MATRIX_TRAMPOLINES(mat4f, EigenMat4f, vec4f, EigenVec4f, float)
EIGENPRIM_MATRIX_TRAMPOLINES(mat2d, EigenMat2d, vec2d, EigenVec2d, double)
EIGENPRIM_MATRIX_TRAMPOLINES(mat3d, EigenMat3d, vec3d, EigenVec3d, double)
EIGENPRIM_MATRIX_TRAMPOLINES(mat4d, EigenMat4d, vec4d, EigenVec4d, double)
EIGENPRIM_MATRIX_TRAMPOLINES(mat2h, EigenMat2h, vec2h, EigenVec2h, Eigen::half)
EIGENPRIM_MATRIX_TRAMPOLINES(mat3h, EigenMat3h, vec3h, EigenVec3h, Eigen::half)
EIGENPRIM_MATRIX_TRAMPOLINES(mat4h, EigenMat4h, vec4h, EigenVec4h, Eigen::half)
EIGENPRIM_MATRIX_TRAMPOLINES(mat2bf, EigenMat2bf, vec2bf, EigenVec2bf, Eigen::bfloat16)
EIGENPRIM_MATRIX_TRAMPOLINES(mat3bf, EigenMat3bf, vec3bf, EigenVec3bf, Eigen::bfloat16)
EIGENPRIM_MATRIX_TRAMPOLINES(mat4bf, EigenMat4bf, vec4bf, EigenVec4bf, Eigen::bfloat16)

#undef EIGENPRIM_MATRIX_TRAMPOLINES
#undef EIGENPRIM_VECTOR3_TRAMPOLINES
#undef EIGENPRIM_VECTOR_TRAMPOLINES

}  // namespace eigenprim_detail

namespace Eigen {

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::dot(
    Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols> other) const {
  return eigenprim_detail::dot(*this, other);
}

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::norm() const {
  return eigenprim_detail::norm(*this);
}

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::squared_norm() const {
  return eigenprim_detail::squared_norm(*this);
}

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::normalized() const {
  return eigenprim_detail::normalized(*this);
}

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::scale(Scalar s) const {
  return eigenprim_detail::scale(*this, s);
}

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::cwise_product(
    Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols> other) const {
  return eigenprim_detail::cwise_product(*this, other);
}

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::cwise_abs() const {
  return eigenprim_detail::cwise_abs(*this);
}

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::cwise_min(
    Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols> other) const {
  return eigenprim_detail::cwise_min(*this, other);
}

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::cwise_max(
    Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols> other) const {
  return eigenprim_detail::cwise_max(*this, other);
}

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::sum() const {
  return eigenprim_detail::sum(*this);
}

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::min_coeff() const {
  return eigenprim_detail::min_coeff(*this);
}

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::max_coeff() const {
  return eigenprim_detail::max_coeff(*this);
}

template <typename Scalar, int Rows, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Rows, Rows>
Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols>::outer(
    Matrix<Scalar, Rows, 1, Options, MaxRows, MaxCols> other) const {
  return eigenprim_detail::outer(*this, other);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::dot(
    Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols> other) const {
  return eigenprim_detail::dot(*this, other);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::norm() const {
  return eigenprim_detail::norm(*this);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::squared_norm() const {
  return eigenprim_detail::squared_norm(*this);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::normalized() const {
  return eigenprim_detail::normalized(*this);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::scale(Scalar s) const {
  return eigenprim_detail::scale(*this, s);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::cross(
    Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols> other) const {
  return eigenprim_detail::cross(*this, other);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::cwise_product(
    Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols> other) const {
  return eigenprim_detail::cwise_product(*this, other);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::cwise_abs() const {
  return eigenprim_detail::cwise_abs(*this);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::cwise_min(
    Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols> other) const {
  return eigenprim_detail::cwise_min(*this, other);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>
Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::cwise_max(
    Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols> other) const {
  return eigenprim_detail::cwise_max(*this, other);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::sum() const {
  return eigenprim_detail::sum(*this);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::min_coeff() const {
  return eigenprim_detail::min_coeff(*this);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::max_coeff() const {
  return eigenprim_detail::max_coeff(*this);
}

template <typename Scalar, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, 3, 3>
Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols>::outer(
    Matrix<Scalar, 3, 1, Options, MaxRows, MaxCols> other) const {
  return eigenprim_detail::outer(*this, other);
}

template <typename Scalar, int Dim, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>::determinant() const {
  return eigenprim_detail::determinant(*this);
}

template <typename Scalar, int Dim, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>
Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>::inverse() const {
  return eigenprim_detail::inverse(*this);
}

template <typename Scalar, int Dim, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>
Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>::transpose() const {
  return eigenprim_detail::transpose(*this);
}

template <typename Scalar, int Dim, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>::trace() const {
  return eigenprim_detail::trace(*this);
}

template <typename Scalar, int Dim, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>
Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>::cwise_product(
    Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols> other) const {
  return eigenprim_detail::cwise_product(*this, other);
}

template <typename Scalar, int Dim, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>
Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>::scale(Scalar s) const {
  return eigenprim_detail::scale(*this, s);
}

template <typename Scalar, int Dim, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>::norm() const {
  return eigenprim_detail::norm(*this);
}

template <typename Scalar, int Dim, int Options, int MaxRows, int MaxCols>
__device__ Scalar Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>::squared_norm() const {
  return eigenprim_detail::squared_norm(*this);
}

template <typename Scalar, int Dim, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Dim, 1>
Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>::diagonal() const {
  return eigenprim_detail::diagonal(*this);
}

template <typename Scalar, int Dim, int Options, int MaxRows, int MaxCols>
__device__ Matrix<Scalar, Dim, 1>
Matrix<Scalar, Dim, Dim, Options, MaxRows, MaxCols>::vec_mul(
    Matrix<Scalar, Dim, 1> v) const {
  return eigenprim_detail::vec_mul(*this, v);
}

}  // namespace Eigen
