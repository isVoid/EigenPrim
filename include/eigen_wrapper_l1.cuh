// Level 1: Thin wrapper functions around Eigen fixed-size types.
// These expose Eigen functionality through plain __device__ functions
// with concrete types that numbast can more easily parse.
//
// The Vec3f struct and function declarations are in eigen_wrapper_l1_decl.cuh
// (Eigen-free, safe for NVRTC). This file provides the implementations.

#pragma once

#include <Eigen/Dense>
#include "eigen_wrapper_l1_decl.cuh"

// ---- Vector3f wrapper implementations ----

__device__ Vec3f vec3f_add(Vec3f a, Vec3f b) {
  Eigen::Vector3f ea(a.x, a.y, a.z);
  Eigen::Vector3f eb(b.x, b.y, b.z);
  Eigen::Vector3f result = ea + eb;
  return Vec3f(result(0), result(1), result(2));
}

__device__ float vec3f_dot(Vec3f a, Vec3f b) {
  Eigen::Vector3f ea(a.x, a.y, a.z);
  Eigen::Vector3f eb(b.x, b.y, b.z);
  return ea.dot(eb);
}

__device__ Vec3f vec3f_cross(Vec3f a, Vec3f b) {
  Eigen::Vector3f ea(a.x, a.y, a.z);
  Eigen::Vector3f eb(b.x, b.y, b.z);
  Eigen::Vector3f result = ea.cross(eb);
  return Vec3f(result(0), result(1), result(2));
}

__device__ float vec3f_norm(Vec3f v) {
  Eigen::Vector3f ev(v.x, v.y, v.z);
  return ev.norm();
}

__device__ Vec3f vec3f_normalized(Vec3f v) {
  Eigen::Vector3f ev(v.x, v.y, v.z);
  Eigen::Vector3f result = ev.normalized();
  return Vec3f(result(0), result(1), result(2));
}

__device__ Vec3f vec3f_scale(Vec3f v, float s) {
  Eigen::Vector3f ev(v.x, v.y, v.z);
  Eigen::Vector3f result = ev * s;
  return Vec3f(result(0), result(1), result(2));
}
