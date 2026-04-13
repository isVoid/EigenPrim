// Level 3: Declarations-only header for template wrappers.
// No Eigen dependency — safe for NVRTC compilation (used by numbast shims).
// The implementations live in eigen_wrapper_l3.cuh (compiled separately by nvcc).

#pragma once

template <typename Scalar, int N>
struct EigenVecWrapper {
  Scalar data[N];

  __host__ __device__ EigenVecWrapper() {
    for (int i = 0; i < N; i++) data[i] = Scalar(0);
  }

  // Convenience constructor for N=3
  __host__ __device__ EigenVecWrapper(Scalar v0, Scalar v1, Scalar v2) {
    data[0] = v0; data[1] = v1; data[2] = v2;
  }
};

// Function template declarations (no body — implementations use Eigen)

template <typename Scalar>
__device__ Scalar templated_dot3(
    Scalar x1, Scalar y1, Scalar z1,
    Scalar x2, Scalar y2, Scalar z2);

template <typename Scalar>
__device__ EigenVecWrapper<Scalar, 3> templated_cross(
    EigenVecWrapper<Scalar, 3> a, EigenVecWrapper<Scalar, 3> b);

// Explicit instantiation declarations — these exist in the fatbin
extern template __device__ float templated_dot3<float>(float, float, float, float, float, float);
extern template __device__ double templated_dot3<double>(double, double, double, double, double, double);
extern template __device__ EigenVecWrapper<float, 3> templated_cross<float>(
    EigenVecWrapper<float, 3>, EigenVecWrapper<float, 3>);
