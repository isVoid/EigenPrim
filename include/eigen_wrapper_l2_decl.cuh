// Level 2: Declarations-only header for Eigen-type functions.
// Uses the NVRTC-safe Eigen stub (no real Eigen dependency).
// The implementations live in eigen_wrapper_l2.cuh (compiled separately by nvcc).

#pragma once

#include "eigen_stub.cuh"

using EigenVec3f = Eigen::Matrix<float, 3, 1>;
using EigenMat3f = Eigen::Matrix<float, 3, 3>;

__device__ EigenVec3f eigen_vec3f_add(EigenVec3f a, EigenVec3f b);
__device__ float eigen_vec3f_dot(EigenVec3f a, EigenVec3f b);
__device__ EigenVec3f eigen_vec3f_cross(EigenVec3f a, EigenVec3f b);
__device__ float eigen_vec3f_norm(EigenVec3f v);
__device__ EigenVec3f eigen_mat3f_vec3f_mul(EigenMat3f m, EigenVec3f v);
__device__ EigenMat3f eigen_mat3f_mul(EigenMat3f a, EigenMat3f b);
__device__ float eigen_mat3f_determinant(EigenMat3f m);
__device__ EigenMat3f eigen_mat3f_inverse(EigenMat3f m);
__device__ EigenMat3f eigen_mat3f_transpose(EigenMat3f m);
