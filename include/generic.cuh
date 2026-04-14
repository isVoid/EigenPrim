// Generic: Template wrapper functions over Eigen.
//
// Declarations are in generic_decl.cuh (Eigen-free, safe for NVRTC).
// This file provides the Eigen-based implementations.

#pragma once

#include <Eigen/Dense>
#include "generic_decl.cuh"

template <typename Scalar>
__device__ Scalar templated_dot3(
    Scalar x1, Scalar y1, Scalar z1,
    Scalar x2, Scalar y2, Scalar z2)
{
  Eigen::Matrix<Scalar, 3, 1> a;
  a << x1, y1, z1;
  Eigen::Matrix<Scalar, 3, 1> b;
  b << x2, y2, z2;
  return a.dot(b);
}

template <typename Scalar>
__device__ EigenVecWrapper<Scalar, 3> templated_cross(
    EigenVecWrapper<Scalar, 3> a, EigenVecWrapper<Scalar, 3> b)
{
  Eigen::Matrix<Scalar, 3, 1> ea(a.data[0], a.data[1], a.data[2]);
  Eigen::Matrix<Scalar, 3, 1> eb(b.data[0], b.data[1], b.data[2]);
  Eigen::Matrix<Scalar, 3, 1> result = ea.cross(eb);
  EigenVecWrapper<Scalar, 3> out;
  out.data[0] = result(0);
  out.data[1] = result(1);
  out.data[2] = result(2);
  return out;
}

// Explicit instantiation definitions
template __device__ float templated_dot3<float>(float, float, float, float, float, float);
template __device__ double templated_dot3<double>(double, double, double, double, double, double);
template __device__ EigenVecWrapper<float, 3> templated_cross<float>(
    EigenVecWrapper<float, 3>, EigenVecWrapper<float, 3>);
