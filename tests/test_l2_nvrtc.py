"""Verify matrix shim compiles with NVRTC using the stub header."""

import os
from numba.cuda import config as cuda_config
from numba.cuda.cudadrv import nvrtc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE_DIR = os.path.join(PROJECT_ROOT, "include")

cuda_config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = INCLUDE_DIR

decl = os.path.join(INCLUDE_DIR, "matrix_decl.cuh")

# Simulate a shim that numbast would generate for eigen_vec3f_dot
shim_source = f"""\
#include "{decl}"

extern "C" __device__ int
test_dot_shim(float &retval,
              Eigen::Matrix<float, 3, 1>* a,
              Eigen::Matrix<float, 3, 1>* b) {{
    retval = eigen_vec3f_dot(*a, *b);
    return 0;
}}

extern "C" __device__ int
test_add_shim(Eigen::Matrix<float, 3, 1> &retval,
              Eigen::Matrix<float, 3, 1>* a,
              Eigen::Matrix<float, 3, 1>* b) {{
    retval = eigen_vec3f_add(*a, *b);
    return 0;
}}
"""

print("Compiling matrix shim with NVRTC (stub Eigen types)...")
try:
    result, log = nvrtc.compile(shim_source.encode(), "test_l2_shim.cu", (8, 0))
    print(f"SUCCESS! Compiled to {type(result).__name__}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    raise
