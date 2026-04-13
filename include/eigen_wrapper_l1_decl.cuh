// Level 1: Declarations-only header for Vec3f and device functions.
// No Eigen dependency — safe for NVRTC compilation (used by numbast shims).
// The implementations live in eigen_wrapper_l1.cuh (compiled separately by nvcc).

#pragma once

struct Vec3f {
  float x, y, z;

  __host__ __device__ Vec3f() : x(0), y(0), z(0) {}
  __host__ __device__ Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

  __host__ __device__ operator float() const { return x; }
};

__device__ Vec3f vec3f_add(Vec3f a, Vec3f b);
__device__ float vec3f_dot(Vec3f a, Vec3f b);
__device__ Vec3f vec3f_cross(Vec3f a, Vec3f b);
__device__ float vec3f_norm(Vec3f v);
__device__ Vec3f vec3f_normalized(Vec3f v);
__device__ Vec3f vec3f_scale(Vec3f v, float s);
