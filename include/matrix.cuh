// Matrix: Eigen vector and matrix device functions.
// Uses Eigen::Matrix types directly as function parameters.
// Declarations are in matrix_decl.cuh (NVRTC-safe, uses eigen_stub).

#pragma once

#include <Eigen/Dense>

// ── Type aliases ─────────────────────────────────────────────────

using EigenVec2f = Eigen::Matrix<float, 2, 1>;
using EigenVec3f = Eigen::Matrix<float, 3, 1>;
using EigenVec4f = Eigen::Matrix<float, 4, 1>;
using EigenVec2d = Eigen::Matrix<double, 2, 1>;
using EigenVec3d = Eigen::Matrix<double, 3, 1>;
using EigenVec4d = Eigen::Matrix<double, 4, 1>;

using EigenMat2f = Eigen::Matrix<float, 2, 2>;
using EigenMat3f = Eigen::Matrix<float, 3, 3>;
using EigenMat4f = Eigen::Matrix<float, 4, 4>;
using EigenMat2d = Eigen::Matrix<double, 2, 2>;
using EigenMat3d = Eigen::Matrix<double, 3, 3>;
using EigenMat4d = Eigen::Matrix<double, 4, 4>;

// ── Vector2f ─────────────────────────────────────────────────────

__device__ EigenVec2f eigen_vec2f_add(EigenVec2f a, EigenVec2f b) { return a + b; }
__device__ EigenVec2f eigen_vec2f_sub(EigenVec2f a, EigenVec2f b) { return a - b; }
__device__ float eigen_vec2f_dot(EigenVec2f a, EigenVec2f b) { return a.dot(b); }
__device__ float eigen_vec2f_norm(EigenVec2f v) { return v.norm(); }
__device__ float eigen_vec2f_squared_norm(EigenVec2f v) { return v.squaredNorm(); }
__device__ EigenVec2f eigen_vec2f_normalized(EigenVec2f v) { return v.normalized(); }
__device__ EigenVec2f eigen_vec2f_scale(EigenVec2f v, float s) { return v * s; }
__device__ EigenVec2f eigen_vec2f_cwise_product(EigenVec2f a, EigenVec2f b) { return a.cwiseProduct(b); }
__device__ EigenVec2f eigen_vec2f_cwise_abs(EigenVec2f v) { return v.cwiseAbs(); }
__device__ EigenVec2f eigen_vec2f_cwise_min(EigenVec2f a, EigenVec2f b) { return a.cwiseMin(b); }
__device__ EigenVec2f eigen_vec2f_cwise_max(EigenVec2f a, EigenVec2f b) { return a.cwiseMax(b); }
__device__ float eigen_vec2f_sum(EigenVec2f v) { return v.sum(); }
__device__ float eigen_vec2f_min_coeff(EigenVec2f v) { return v.minCoeff(); }
__device__ float eigen_vec2f_max_coeff(EigenVec2f v) { return v.maxCoeff(); }
__device__ EigenMat2f eigen_vec2f_outer(EigenVec2f a, EigenVec2f b) { return a * b.transpose(); }

// ── Vector3f ─────────────────────────────────────────────────────

__device__ EigenVec3f eigen_vec3f_add(EigenVec3f a, EigenVec3f b) { return a + b; }
__device__ EigenVec3f eigen_vec3f_sub(EigenVec3f a, EigenVec3f b) { return a - b; }
__device__ float eigen_vec3f_dot(EigenVec3f a, EigenVec3f b) { return a.dot(b); }
__device__ float eigen_vec3f_norm(EigenVec3f v) { return v.norm(); }
__device__ float eigen_vec3f_squared_norm(EigenVec3f v) { return v.squaredNorm(); }
__device__ EigenVec3f eigen_vec3f_normalized(EigenVec3f v) { return v.normalized(); }
__device__ EigenVec3f eigen_vec3f_scale(EigenVec3f v, float s) { return v * s; }
__device__ EigenVec3f eigen_vec3f_cross(EigenVec3f a, EigenVec3f b) { return a.cross(b); }
__device__ EigenVec3f eigen_vec3f_cwise_product(EigenVec3f a, EigenVec3f b) { return a.cwiseProduct(b); }
__device__ EigenVec3f eigen_vec3f_cwise_abs(EigenVec3f v) { return v.cwiseAbs(); }
__device__ EigenVec3f eigen_vec3f_cwise_min(EigenVec3f a, EigenVec3f b) { return a.cwiseMin(b); }
__device__ EigenVec3f eigen_vec3f_cwise_max(EigenVec3f a, EigenVec3f b) { return a.cwiseMax(b); }
__device__ float eigen_vec3f_sum(EigenVec3f v) { return v.sum(); }
__device__ float eigen_vec3f_min_coeff(EigenVec3f v) { return v.minCoeff(); }
__device__ float eigen_vec3f_max_coeff(EigenVec3f v) { return v.maxCoeff(); }
__device__ EigenMat3f eigen_vec3f_outer(EigenVec3f a, EigenVec3f b) { return a * b.transpose(); }

// ── Vector4f ─────────────────────────────────────────────────────

__device__ EigenVec4f eigen_vec4f_add(EigenVec4f a, EigenVec4f b) { return a + b; }
__device__ EigenVec4f eigen_vec4f_sub(EigenVec4f a, EigenVec4f b) { return a - b; }
__device__ float eigen_vec4f_dot(EigenVec4f a, EigenVec4f b) { return a.dot(b); }
__device__ float eigen_vec4f_norm(EigenVec4f v) { return v.norm(); }
__device__ float eigen_vec4f_squared_norm(EigenVec4f v) { return v.squaredNorm(); }
__device__ EigenVec4f eigen_vec4f_normalized(EigenVec4f v) { return v.normalized(); }
__device__ EigenVec4f eigen_vec4f_scale(EigenVec4f v, float s) { return v * s; }
__device__ EigenVec4f eigen_vec4f_cwise_product(EigenVec4f a, EigenVec4f b) { return a.cwiseProduct(b); }
__device__ EigenVec4f eigen_vec4f_cwise_abs(EigenVec4f v) { return v.cwiseAbs(); }
__device__ EigenVec4f eigen_vec4f_cwise_min(EigenVec4f a, EigenVec4f b) { return a.cwiseMin(b); }
__device__ EigenVec4f eigen_vec4f_cwise_max(EigenVec4f a, EigenVec4f b) { return a.cwiseMax(b); }
__device__ float eigen_vec4f_sum(EigenVec4f v) { return v.sum(); }
__device__ float eigen_vec4f_min_coeff(EigenVec4f v) { return v.minCoeff(); }
__device__ float eigen_vec4f_max_coeff(EigenVec4f v) { return v.maxCoeff(); }
__device__ EigenMat4f eigen_vec4f_outer(EigenVec4f a, EigenVec4f b) { return a * b.transpose(); }

// ── Vector2d ─────────────────────────────────────────────────────

__device__ EigenVec2d eigen_vec2d_add(EigenVec2d a, EigenVec2d b) { return a + b; }
__device__ EigenVec2d eigen_vec2d_sub(EigenVec2d a, EigenVec2d b) { return a - b; }
__device__ double eigen_vec2d_dot(EigenVec2d a, EigenVec2d b) { return a.dot(b); }
__device__ double eigen_vec2d_norm(EigenVec2d v) { return v.norm(); }
__device__ double eigen_vec2d_squared_norm(EigenVec2d v) { return v.squaredNorm(); }
__device__ EigenVec2d eigen_vec2d_normalized(EigenVec2d v) { return v.normalized(); }
__device__ EigenVec2d eigen_vec2d_scale(EigenVec2d v, double s) { return v * s; }
__device__ EigenVec2d eigen_vec2d_cwise_product(EigenVec2d a, EigenVec2d b) { return a.cwiseProduct(b); }
__device__ EigenVec2d eigen_vec2d_cwise_abs(EigenVec2d v) { return v.cwiseAbs(); }
__device__ EigenVec2d eigen_vec2d_cwise_min(EigenVec2d a, EigenVec2d b) { return a.cwiseMin(b); }
__device__ EigenVec2d eigen_vec2d_cwise_max(EigenVec2d a, EigenVec2d b) { return a.cwiseMax(b); }
__device__ double eigen_vec2d_sum(EigenVec2d v) { return v.sum(); }
__device__ double eigen_vec2d_min_coeff(EigenVec2d v) { return v.minCoeff(); }
__device__ double eigen_vec2d_max_coeff(EigenVec2d v) { return v.maxCoeff(); }
__device__ EigenMat2d eigen_vec2d_outer(EigenVec2d a, EigenVec2d b) { return a * b.transpose(); }

// ── Vector3d ─────────────────────────────────────────────────────

__device__ EigenVec3d eigen_vec3d_add(EigenVec3d a, EigenVec3d b) { return a + b; }
__device__ EigenVec3d eigen_vec3d_sub(EigenVec3d a, EigenVec3d b) { return a - b; }
__device__ double eigen_vec3d_dot(EigenVec3d a, EigenVec3d b) { return a.dot(b); }
__device__ double eigen_vec3d_norm(EigenVec3d v) { return v.norm(); }
__device__ double eigen_vec3d_squared_norm(EigenVec3d v) { return v.squaredNorm(); }
__device__ EigenVec3d eigen_vec3d_normalized(EigenVec3d v) { return v.normalized(); }
__device__ EigenVec3d eigen_vec3d_scale(EigenVec3d v, double s) { return v * s; }
__device__ EigenVec3d eigen_vec3d_cross(EigenVec3d a, EigenVec3d b) { return a.cross(b); }
__device__ EigenVec3d eigen_vec3d_cwise_product(EigenVec3d a, EigenVec3d b) { return a.cwiseProduct(b); }
__device__ EigenVec3d eigen_vec3d_cwise_abs(EigenVec3d v) { return v.cwiseAbs(); }
__device__ EigenVec3d eigen_vec3d_cwise_min(EigenVec3d a, EigenVec3d b) { return a.cwiseMin(b); }
__device__ EigenVec3d eigen_vec3d_cwise_max(EigenVec3d a, EigenVec3d b) { return a.cwiseMax(b); }
__device__ double eigen_vec3d_sum(EigenVec3d v) { return v.sum(); }
__device__ double eigen_vec3d_min_coeff(EigenVec3d v) { return v.minCoeff(); }
__device__ double eigen_vec3d_max_coeff(EigenVec3d v) { return v.maxCoeff(); }
__device__ EigenMat3d eigen_vec3d_outer(EigenVec3d a, EigenVec3d b) { return a * b.transpose(); }

// ── Vector4d ─────────────────────────────────────────────────────

__device__ EigenVec4d eigen_vec4d_add(EigenVec4d a, EigenVec4d b) { return a + b; }
__device__ EigenVec4d eigen_vec4d_sub(EigenVec4d a, EigenVec4d b) { return a - b; }
__device__ double eigen_vec4d_dot(EigenVec4d a, EigenVec4d b) { return a.dot(b); }
__device__ double eigen_vec4d_norm(EigenVec4d v) { return v.norm(); }
__device__ double eigen_vec4d_squared_norm(EigenVec4d v) { return v.squaredNorm(); }
__device__ EigenVec4d eigen_vec4d_normalized(EigenVec4d v) { return v.normalized(); }
__device__ EigenVec4d eigen_vec4d_scale(EigenVec4d v, double s) { return v * s; }
__device__ EigenVec4d eigen_vec4d_cwise_product(EigenVec4d a, EigenVec4d b) { return a.cwiseProduct(b); }
__device__ EigenVec4d eigen_vec4d_cwise_abs(EigenVec4d v) { return v.cwiseAbs(); }
__device__ EigenVec4d eigen_vec4d_cwise_min(EigenVec4d a, EigenVec4d b) { return a.cwiseMin(b); }
__device__ EigenVec4d eigen_vec4d_cwise_max(EigenVec4d a, EigenVec4d b) { return a.cwiseMax(b); }
__device__ double eigen_vec4d_sum(EigenVec4d v) { return v.sum(); }
__device__ double eigen_vec4d_min_coeff(EigenVec4d v) { return v.minCoeff(); }
__device__ double eigen_vec4d_max_coeff(EigenVec4d v) { return v.maxCoeff(); }
__device__ EigenMat4d eigen_vec4d_outer(EigenVec4d a, EigenVec4d b) { return a * b.transpose(); }

// ── Matrix2f ─────────────────────────────────────────────────────

__device__ EigenMat2f eigen_mat2f_add(EigenMat2f a, EigenMat2f b) { return a + b; }
__device__ EigenMat2f eigen_mat2f_sub(EigenMat2f a, EigenMat2f b) { return a - b; }
__device__ EigenMat2f eigen_mat2f_mul(EigenMat2f a, EigenMat2f b) { return a * b; }
__device__ EigenVec2f eigen_mat2f_vec2f_mul(EigenMat2f m, EigenVec2f v) { return m * v; }
__device__ float eigen_mat2f_determinant(EigenMat2f m) { return m.determinant(); }
__device__ EigenMat2f eigen_mat2f_inverse(EigenMat2f m) { return m.inverse(); }
__device__ EigenMat2f eigen_mat2f_transpose(EigenMat2f m) { return m.transpose(); }
__device__ float eigen_mat2f_trace(EigenMat2f m) { return m.trace(); }
__device__ EigenMat2f eigen_mat2f_cwise_product(EigenMat2f a, EigenMat2f b) { return a.cwiseProduct(b); }
__device__ EigenMat2f eigen_mat2f_scale(EigenMat2f m, float s) { return m * s; }
__device__ float eigen_mat2f_norm(EigenMat2f m) { return m.norm(); }
__device__ float eigen_mat2f_squared_norm(EigenMat2f m) { return m.squaredNorm(); }
__device__ EigenVec2f eigen_mat2f_diagonal(EigenMat2f m) { return m.diagonal(); }

// ── Matrix3f ─────────────────────────────────────────────────────

__device__ EigenMat3f eigen_mat3f_add(EigenMat3f a, EigenMat3f b) { return a + b; }
__device__ EigenMat3f eigen_mat3f_sub(EigenMat3f a, EigenMat3f b) { return a - b; }
__device__ EigenMat3f eigen_mat3f_mul(EigenMat3f a, EigenMat3f b) { return a * b; }
__device__ EigenVec3f eigen_mat3f_vec3f_mul(EigenMat3f m, EigenVec3f v) { return m * v; }
__device__ float eigen_mat3f_determinant(EigenMat3f m) { return m.determinant(); }
__device__ EigenMat3f eigen_mat3f_inverse(EigenMat3f m) { return m.inverse(); }
__device__ EigenMat3f eigen_mat3f_transpose(EigenMat3f m) { return m.transpose(); }
__device__ float eigen_mat3f_trace(EigenMat3f m) { return m.trace(); }
__device__ EigenMat3f eigen_mat3f_cwise_product(EigenMat3f a, EigenMat3f b) { return a.cwiseProduct(b); }
__device__ EigenMat3f eigen_mat3f_scale(EigenMat3f m, float s) { return m * s; }
__device__ float eigen_mat3f_norm(EigenMat3f m) { return m.norm(); }
__device__ float eigen_mat3f_squared_norm(EigenMat3f m) { return m.squaredNorm(); }
__device__ EigenVec3f eigen_mat3f_diagonal(EigenMat3f m) { return m.diagonal(); }

// ── Matrix4f ─────────────────────────────────────────────────────

__device__ EigenMat4f eigen_mat4f_add(EigenMat4f a, EigenMat4f b) { return a + b; }
__device__ EigenMat4f eigen_mat4f_sub(EigenMat4f a, EigenMat4f b) { return a - b; }
__device__ EigenMat4f eigen_mat4f_mul(EigenMat4f a, EigenMat4f b) { return a * b; }
__device__ EigenVec4f eigen_mat4f_vec4f_mul(EigenMat4f m, EigenVec4f v) { return m * v; }
__device__ float eigen_mat4f_determinant(EigenMat4f m) { return m.determinant(); }
__device__ EigenMat4f eigen_mat4f_inverse(EigenMat4f m) { return m.inverse(); }
__device__ EigenMat4f eigen_mat4f_transpose(EigenMat4f m) { return m.transpose(); }
__device__ float eigen_mat4f_trace(EigenMat4f m) { return m.trace(); }
__device__ EigenMat4f eigen_mat4f_cwise_product(EigenMat4f a, EigenMat4f b) { return a.cwiseProduct(b); }
__device__ EigenMat4f eigen_mat4f_scale(EigenMat4f m, float s) { return m * s; }
__device__ float eigen_mat4f_norm(EigenMat4f m) { return m.norm(); }
__device__ float eigen_mat4f_squared_norm(EigenMat4f m) { return m.squaredNorm(); }
__device__ EigenVec4f eigen_mat4f_diagonal(EigenMat4f m) { return m.diagonal(); }

// ── Matrix2d ─────────────────────────────────────────────────────

__device__ EigenMat2d eigen_mat2d_add(EigenMat2d a, EigenMat2d b) { return a + b; }
__device__ EigenMat2d eigen_mat2d_sub(EigenMat2d a, EigenMat2d b) { return a - b; }
__device__ EigenMat2d eigen_mat2d_mul(EigenMat2d a, EigenMat2d b) { return a * b; }
__device__ EigenVec2d eigen_mat2d_vec2d_mul(EigenMat2d m, EigenVec2d v) { return m * v; }
__device__ double eigen_mat2d_determinant(EigenMat2d m) { return m.determinant(); }
__device__ EigenMat2d eigen_mat2d_inverse(EigenMat2d m) { return m.inverse(); }
__device__ EigenMat2d eigen_mat2d_transpose(EigenMat2d m) { return m.transpose(); }
__device__ double eigen_mat2d_trace(EigenMat2d m) { return m.trace(); }
__device__ EigenMat2d eigen_mat2d_cwise_product(EigenMat2d a, EigenMat2d b) { return a.cwiseProduct(b); }
__device__ EigenMat2d eigen_mat2d_scale(EigenMat2d m, double s) { return m * s; }
__device__ double eigen_mat2d_norm(EigenMat2d m) { return m.norm(); }
__device__ double eigen_mat2d_squared_norm(EigenMat2d m) { return m.squaredNorm(); }
__device__ EigenVec2d eigen_mat2d_diagonal(EigenMat2d m) { return m.diagonal(); }

// ── Matrix3d ─────────────────────────────────────────────────────

__device__ EigenMat3d eigen_mat3d_add(EigenMat3d a, EigenMat3d b) { return a + b; }
__device__ EigenMat3d eigen_mat3d_sub(EigenMat3d a, EigenMat3d b) { return a - b; }
__device__ EigenMat3d eigen_mat3d_mul(EigenMat3d a, EigenMat3d b) { return a * b; }
__device__ EigenVec3d eigen_mat3d_vec3d_mul(EigenMat3d m, EigenVec3d v) { return m * v; }
__device__ double eigen_mat3d_determinant(EigenMat3d m) { return m.determinant(); }
__device__ EigenMat3d eigen_mat3d_inverse(EigenMat3d m) { return m.inverse(); }
__device__ EigenMat3d eigen_mat3d_transpose(EigenMat3d m) { return m.transpose(); }
__device__ double eigen_mat3d_trace(EigenMat3d m) { return m.trace(); }
__device__ EigenMat3d eigen_mat3d_cwise_product(EigenMat3d a, EigenMat3d b) { return a.cwiseProduct(b); }
__device__ EigenMat3d eigen_mat3d_scale(EigenMat3d m, double s) { return m * s; }
__device__ double eigen_mat3d_norm(EigenMat3d m) { return m.norm(); }
__device__ double eigen_mat3d_squared_norm(EigenMat3d m) { return m.squaredNorm(); }
__device__ EigenVec3d eigen_mat3d_diagonal(EigenMat3d m) { return m.diagonal(); }

// ── Matrix4d ─────────────────────────────────────────────────────

__device__ EigenMat4d eigen_mat4d_add(EigenMat4d a, EigenMat4d b) { return a + b; }
__device__ EigenMat4d eigen_mat4d_sub(EigenMat4d a, EigenMat4d b) { return a - b; }
__device__ EigenMat4d eigen_mat4d_mul(EigenMat4d a, EigenMat4d b) { return a * b; }
__device__ EigenVec4d eigen_mat4d_vec4d_mul(EigenMat4d m, EigenVec4d v) { return m * v; }
__device__ double eigen_mat4d_determinant(EigenMat4d m) { return m.determinant(); }
__device__ EigenMat4d eigen_mat4d_inverse(EigenMat4d m) { return m.inverse(); }
__device__ EigenMat4d eigen_mat4d_transpose(EigenMat4d m) { return m.transpose(); }
__device__ double eigen_mat4d_trace(EigenMat4d m) { return m.trace(); }
__device__ EigenMat4d eigen_mat4d_cwise_product(EigenMat4d a, EigenMat4d b) { return a.cwiseProduct(b); }
__device__ EigenMat4d eigen_mat4d_scale(EigenMat4d m, double s) { return m * s; }
__device__ double eigen_mat4d_norm(EigenMat4d m) { return m.norm(); }
__device__ double eigen_mat4d_squared_norm(EigenMat4d m) { return m.squaredNorm(); }
__device__ EigenVec4d eigen_mat4d_diagonal(EigenMat4d m) { return m.diagonal(); }

// ── Vector3h (half-precision) ────────────────────────────────────

using EigenVec3h = Eigen::Matrix<Eigen::half, 3, 1>;
using EigenMat3h = Eigen::Matrix<Eigen::half, 3, 3>;

__device__ EigenVec3h eigen_vec3h_add(EigenVec3h a, EigenVec3h b) { return a + b; }
__device__ EigenVec3h eigen_vec3h_sub(EigenVec3h a, EigenVec3h b) { return a - b; }
__device__ Eigen::half eigen_vec3h_dot(EigenVec3h a, EigenVec3h b) { return a.dot(b); }
__device__ Eigen::half eigen_vec3h_norm(EigenVec3h v) { return v.norm(); }
__device__ EigenVec3h eigen_vec3h_normalized(EigenVec3h v) { return v.normalized(); }
__device__ EigenVec3h eigen_vec3h_cross(EigenVec3h a, EigenVec3h b) { return a.cross(b); }

// ── Matrix3h (half-precision) ────────────────────────────────────

__device__ EigenMat3h eigen_mat3h_mul(EigenMat3h a, EigenMat3h b) { return a * b; }
__device__ EigenVec3h eigen_mat3h_vec3h_mul(EigenMat3h m, EigenVec3h v) { return m * v; }
__device__ Eigen::half eigen_mat3h_determinant(EigenMat3h m) { return m.determinant(); }
__device__ EigenMat3h eigen_mat3h_inverse(EigenMat3h m) { return m.inverse(); }
__device__ EigenMat3h eigen_mat3h_transpose(EigenMat3h m) { return m.transpose(); }
