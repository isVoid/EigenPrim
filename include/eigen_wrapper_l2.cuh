// Level 2: Expose Eigen's own types directly.
// This tests whether ast_canopy can parse Eigen template specializations
// like Matrix<float,3,1> when used as function parameters/return types.

#pragma once

#include <Eigen/Dense>

// Expose concrete Eigen types via typedef
using EigenVec3f = Eigen::Matrix<float, 3, 1>;
using EigenVec4f = Eigen::Matrix<float, 4, 1>;
using EigenMat3f = Eigen::Matrix<float, 3, 3>;

// Functions that use Eigen types directly
__device__ EigenVec3f eigen_vec3f_add(EigenVec3f a, EigenVec3f b) {
  return a + b;
}

__device__ float eigen_vec3f_dot(EigenVec3f a, EigenVec3f b) {
  return a.dot(b);
}

__device__ EigenVec3f eigen_vec3f_cross(EigenVec3f a, EigenVec3f b) {
  return a.cross(b);
}

__device__ float eigen_vec3f_norm(EigenVec3f v) {
  return v.norm();
}

__device__ EigenVec3f eigen_mat3f_vec3f_mul(EigenMat3f m, EigenVec3f v) {
  return m * v;
}

__device__ EigenMat3f eigen_mat3f_mul(EigenMat3f a, EigenMat3f b) {
  return a * b;
}

__device__ float eigen_mat3f_determinant(EigenMat3f m) {
  return m.determinant();
}

__device__ EigenMat3f eigen_mat3f_inverse(EigenMat3f m) {
  return m.inverse();
}

__device__ EigenMat3f eigen_mat3f_transpose(EigenMat3f m) {
  return m.transpose();
}
