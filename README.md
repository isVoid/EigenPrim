# Numbast × Eigen: Binding Experiment

An experiment to generate Numba CUDA bindings for Eigen library device functions using [numbast](https://github.com/NVIDIA/numbast).

## Goal

Investigate what works out of the box when using numbast/ast_canopy to parse and bind Eigen's `__device__`-annotated linear algebra functions, identify feature gaps, and develop tests and documentation.

## Project Structure

```
numbast-eigen/
├── pixi.toml                     # Pixi workspace (eigen, numbast deps, CUDA 12.9)
├── include/
│   ├── eigen_wrapper_l1_decl.cuh # Level 1: declarations only (NVRTC-safe, no Eigen)
│   ├── eigen_wrapper_l1.cuh      # Level 1: Eigen implementations (includes decl header)
│   ├── eigen_wrapper_l1_impl.cu  # Level 1: nvcc compilation unit
│   ├── eigen_wrapper_l2.cuh      # Level 2: direct Eigen types as params
│   └── eigen_wrapper_l3.cuh      # Level 3: function/class templates
├── tests/
│   ├── test_parse_eigen.py       # ast_canopy parsing tests (all levels)
│   └── test_bind_eigen.py        # numbast binding generation tests
├── TODO.md
├── LOG.md
└── README.md
```

## Approach: Three Levels of Complexity

### Level 1 — Thin Wrappers (WORKS)

Define a simple `Vec3f` struct with `float x, y, z` and plain `__device__` functions (`vec3f_add`, `vec3f_dot`, `vec3f_cross`, etc.) that internally use Eigen but expose only simple types.

- **Parsing**: All 6 functions and the struct parse correctly, including execution space detection.
- **Binding**: `bind_cxx_struct` and `bind_cxx_function` succeed for all declarations.
- **Kernel**: Uses split-header approach — shim uses NVRTC-safe decl header, implementations pre-compiled with nvcc to fatbin. Needs GPU to validate end-to-end.

### Level 2 — Direct Eigen Types (BINDS, KERNEL BLOCKED)

Functions that use `Eigen::Matrix<float, 3, 1>` directly as parameter and return types.

- **Parsing**: All 9 functions parse. ast_canopy correctly reports param/return types as `Eigen::Matrix<float, 3, 1>` and `Eigen::Matrix<float, 3, 3>`.
- **Binding**: All 9 functions bind successfully (overturns earlier assumption — numbast generates shim text with arbitrary type strings without NVRTC validation).
- **Kernel**: Blocked — shim CUSource would reference Eigen types that NVRTC cannot compile.

### Level 3 — Template Wrappers (PARSES, BINDING UNTESTED)

Function templates (`templated_dot3<Scalar>`) and class templates (`EigenVecWrapper<Scalar, N>`) wrapping Eigen.

- **Parsing**: Both function templates and class templates parse correctly with all template parameters identified.
- **Binding**: Untested — requires `bind_cxx_function_templates` / `bind_cxx_class_templates`.

## Dependencies

- **numbast** (from `/home/wangm/numbast/`) — ast_canopy + numbast Python packages
- **eigen** — via conda-forge
- **CUDA Toolkit 12.9** — via pixi

## Quick Start

```bash
cd /home/wangm/numbast-eigen
pixi install
pixi run build-ast-canopy
pixi run install-numbast
pixi run python -m pytest tests/test_parse_eigen.py -v -s
pixi run python -m pytest tests/test_bind_eigen.py -v -s
```

## Key Findings Summary

| Capability                          | Status     |
|-------------------------------------|------------|
| Parse thin-wrapper device functions | ✅ Works    |
| Parse functions with Eigen types    | ✅ Works    |
| Parse template wrappers             | ✅ Works    |
| Bind thin-wrapper struct            | ✅ Works    |
| Bind thin-wrapper functions         | ✅ Works    |
| Bind functions with Eigen types     | ✅ Works (binding only) |
| End-to-end kernel (thin wrapper)    | ✅ Implemented (needs GPU) |
| End-to-end kernel (Eigen types)     | ❌ Blocked (NVRTC) |
| Typedef capture (`using` aliases)   | ❌ Not captured |
| Direct Eigen/Dense parsing          | ❌ 0 decls (filtering issue) |

## Key Technical Finding: NVRTC Limitation

NVRTC (NVIDIA Runtime Compiler, used by numba-cuda for CUSource compilation) cannot compile Eigen headers because Eigen transitively includes system C library headers (`stdlib.h` via `cuda.h`) that NVRTC doesn't have access to.

**Workaround**: The split-header approach separates declarations (NVRTC-safe) from implementations (compiled by nvcc), then links them together via CUDA's relocatable device code (RDC) mechanism.
