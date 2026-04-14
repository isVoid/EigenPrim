# TODO — Eigenprim

_Last updated: 2026-04-13_

## Completed

- [x] Set up project: `pixi.toml` with eigen, numbast deps, CUDA 12.9
- [x] Build ast_canopy + install numbast into pixi env
- [x] Create Vec3f header: thin wrapper (`Vec3f` struct + `__device__` functions calling Eigen internally)
- [x] Create Matrix header: functions using `Eigen::Matrix<float,3,1>` directly as params/returns
- [x] Create Generic header: function templates + class templates wrapping Eigen
- [x] Test ast_canopy **parsing** — all three header sets parse successfully (20 tests)
- [x] Test numbast **binding generation** — Vec3f struct + all 6 device functions bind successfully
- [x] Test Matrix binding generation — all 9 Eigen-type functions bind successfully
- [x] Diagnose end-to-end kernel failure: NVRTC cannot compile Eigen (system headers missing)
- [x] Implement split-header approach for Vec3f end-to-end kernel:
  - Created `vec3f_decl.cuh` (declarations only, no Eigen, NVRTC-safe)
  - Refactored `vec3f.cuh` to include decl header
  - Created `vec3f.cu` (nvcc compilation unit)
  - Test pre-compiles with nvcc, shim uses decl header, links fatbin
- [x] Restructure as importable library (direct `from eigenprim import Vec3f, links`)

## Needs GPU to Validate

- [ ] Run end-to-end kernel test on a machine with a GPU
  - NVRTC compilation of shim: verified working (no GPU needed)
  - nvcc compilation of fatbin: verified working (no GPU needed)
  - Kernel launch + correctness check: requires GPU

## Remaining Work

- [ ] Test Generic binding with `bind_cxx_function_templates` / `bind_cxx_class_templates`
- [ ] Investigate whether Eigen `Matrix<float,3,1>` can be bound as a numbast struct (parse with `files_to_retain` including Eigen internals)
- [ ] Benchmark: measure overhead of thin-wrapper approach vs direct Eigen type binding
- [ ] Document all feature gaps found

## Feature Gaps Identified

1. **NVRTC cannot compile Eigen headers**: Eigen → cuda.h → `stdlib.h` — NVRTC doesn't have system C library headers. Workaround: split-header + nvcc pre-compilation.
2. **`files_to_retain` filtering**: parsing `Eigen/Dense` directly yields 0 declarations; struct definitions in included headers need those files in `files_to_retain` too.
3. **~~Eigen types as function params~~** (RESOLVED): `bind_cxx_function` DOES succeed — the gap is at NVRTC compilation, not at bind time.
4. **Typedefs not captured**: `using EigenVec3f = Eigen::Matrix<float,3,1>` not appearing in `decls.typedefs`.
