# LOG — Numbast × Eigen Implementation Details

_Last updated: 2026-04-13_

---

## 2026-04-13: Initial Investigation & Setup

### Environment

- Workspace: `/home/wangm/numbast-eigen/` (fresh git repo, no prior content)
- Numbast source: `/home/wangm/numbast/` (monorepo: ast_canopy + numbast + numbast_extensions)
- Pixi 0.59.0, Python 3.14.4, CUDA 12.9
- Eigen installed via conda-forge at `.pixi/envs/default/include/eigen3/`

### Architecture Understanding

**numbast** works in this pipeline:

1. **ast_canopy** invokes Clang in CUDA mode (`-xcuda`) to parse C++ headers
2. Clang AST is walked by `libastcanopy` (C++ library) to extract declarations: structs, functions, templates, enums, typedefs
3. Python bindings (`pylibastcanopy`) expose these as Python objects
4. **numbast** consumes declarations and generates:
   - Numba type definitions (`bind_cxx_struct`)
   - Typing/lowering for functions (`bind_cxx_function`)
   - Shim functions (extern "C" `__device__` wrappers that call original C++ functions by mangled name)
5. Shim source is compiled via `MemoryShimWriter` → `cuda.CUSource` → linked into `@cuda.jit(link=...)`

### Key Eigen Behavior

- `EIGEN_DEVICE_FUNC` expands to `__host__ __device__` when compiled in CUDA mode
- Eigen is header-only, heavily template-based (Matrix class template, expression templates, CRTP)
- Fixed-size types (`Matrix<float,3,1>`, `Vector3f`) are the typical choice for device code
- `Eigen/Dense` is a proxy header that `#include`s internal headers from `Eigen/src/`

---

## 2026-04-13: Parsing Tests (All Levels)

### Level 1 — Thin Wrappers: ALL 8 TESTS PASS

Header: `include/eigen_wrapper_l1.cuh`

```
Vec3f struct (12 bytes, align 4):
  fields: x (float), y (float), z (float)
  constructors: Vec3f() [host_device], Vec3f(float,float,float) [host_device]
  conversion: operator float() [host_device]

Device functions (all exec_space=device):
  vec3f_add(Vec3f, Vec3f) -> Vec3f
  vec3f_dot(Vec3f, Vec3f) -> float
  vec3f_cross(Vec3f, Vec3f) -> Vec3f
  vec3f_norm(Vec3f) -> float
  vec3f_normalized(Vec3f) -> Vec3f
  vec3f_scale(Vec3f, float) -> Vec3f
```

**Design decision**: Use plain C-style struct (`Vec3f`) as the API surface. Eigen is used only internally in function bodies. This maximizes compatibility with numbast's type system.

### Level 2 — Direct Eigen Types: ALL 6 TESTS PASS

Header: `include/eigen_wrapper_l2.cuh`

Key finding: **ast_canopy correctly parses Eigen template specializations as parameter/return types**.

```
Functions found (all 9):
  eigen_vec3f_add: returns Eigen::Matrix<float, 3, 1>, params Eigen::Matrix<float, 3, 1>
  eigen_vec3f_dot: returns float, params Eigen::Matrix<float, 3, 1>
  eigen_vec3f_cross: returns Eigen::Matrix<float, 3, 1>
  eigen_vec3f_norm: returns float
  eigen_mat3f_vec3f_mul: returns Eigen::Matrix<float, 3, 1>, params (Eigen::Matrix<float, 3, 3>, Eigen::Matrix<float, 3, 1>)
  eigen_mat3f_mul: returns Eigen::Matrix<float, 3, 3>
  eigen_mat3f_determinant: returns float
  eigen_mat3f_inverse: returns Eigen::Matrix<float, 3, 3>
  eigen_mat3f_transpose: returns Eigen::Matrix<float, 3, 3>
```

**Observations**:
- Typedefs (`using EigenVec3f = Eigen::Matrix<float,3,1>`) are NOT captured in `decls.typedefs` (empty list)
- Class templates and class template specializations are empty (because `files_to_retain` only includes the wrapper header, not Eigen's internal headers)
- The type names in param/return are fully qualified: `Eigen::Matrix<float, 3, 1>`

### Level 3 — Template Wrappers: ALL 5 TESTS PASS

Header: `include/eigen_wrapper_l3.cuh`

```
Function templates: ['templated_dot3', 'templated_cross']
  templated_dot3 template params: ['Scalar']
Class templates: ['EigenVecWrapper']
  EigenVecWrapper template params: ['Scalar', 'N']
```

### Direct Eigen/Dense Parsing: 0 DECLARATIONS

Parsing `Eigen/Dense` with `files_to_retain=[Eigen/Dense]` yields all zeros. This is because `Eigen/Dense` is just a proxy header — all real declarations live in files like `Eigen/src/Core/Matrix.h`. Would need to enumerate and include those paths in `files_to_retain`.

---

## 2026-04-13: Binding Generation Tests

### Level 1 Struct Binding: SUCCESS

```python
Vec3fType = bind_cxx_struct(shim_writer, vec3f_struct, types.Type, StructModel)
# Returns: <class 'numbast.struct.Vec3f'>
```

### Level 1 Function Binding: SUCCESS (6/6)

```
Bound: vec3f_add
Bound: vec3f_dot
Bound: vec3f_cross
Bound: vec3f_norm
Bound: vec3f_normalized
Bound: vec3f_scale
```

All 6 device functions bound successfully.

### Level 1 End-to-End Kernel: SOLVED (split-header approach)

**Original problem**: The shim CUSource includes `eigen_wrapper_l1.cuh` which has `#include <Eigen/Dense>`. NVRTC cannot compile Eigen because:
1. NVRTC lacks system include paths (`stdlib.h`, `gnu/stubs.h`, etc.)
2. Even with `config.CUDA_NVRTC_EXTRA_SEARCH_PATHS` pointing to system includes, NVRTC chokes on GCC-specific system headers

**Solution — split-header + separate compilation**:
1. Created `eigen_wrapper_l1_decl.cuh`: Vec3f struct + function prototypes only (no Eigen, NVRTC-safe)
2. Refactored `eigen_wrapper_l1.cuh`: includes the decl header + provides Eigen-based implementations
3. Created `eigen_wrapper_l1_impl.cu`: thin compilation unit for nvcc
4. Test pre-compiles implementations with `nvcc -rdc=true -fatbin` (which has full system + Eigen access)
5. Shim uses the decl header for `preceding_text` → NVRTC compiles cleanly
6. Kernel links both: `@cuda.jit(link=[*shim_writer.links(), fatbin_path])`
7. CUDA linker resolves device function calls across modules (RDC enabled by default in numba-cuda's NVRTC: `nvrtc.py:154`)

**Status**: Test skipped on current machine (no GPU), but:
- NVRTC compilation of shim + decl header: verified working
- nvcc compilation of Eigen implementations to fatbin: verified working
- The linking and kernel execution requires a GPU to validate

**Key discovery — `NUMBA_CUDA_NVRTC_EXTRA_SEARCH_PATHS`**: numba-cuda has a built-in mechanism to add include paths to NVRTC via `config.CUDA_NVRTC_EXTRA_SEARCH_PATHS` (colon-separated). This works for CUDA-compatible headers but NOT for headers that transitively include system headers (like Eigen → cuda.h → stdlib.h).

### Level 2 Binding: SUCCESS (9/9) — overturns earlier assumption

**Surprise finding**: `bind_cxx_function` succeeds for ALL 9 functions with `Eigen::Matrix<float,3,1>` parameters!

```
Bound: eigen_vec3f_add, eigen_vec3f_dot, eigen_vec3f_cross, eigen_vec3f_norm,
       eigen_mat3f_vec3f_mul, eigen_mat3f_mul, eigen_mat3f_determinant,
       eigen_mat3f_inverse, eigen_mat3f_transpose
```

**Why it works**: `bind_cxx_function` only generates shim source text and Numba typing/lowering at bind time. It does NOT validate that NVRTC can compile the types. The actual failure point would be at kernel JIT time when NVRTC attempts to compile the shim CUSource referencing `Eigen::Matrix`.

**Implication**: The gap is not in numbast's type resolution (which handles arbitrary type name strings) but in NVRTC's inability to compile Eigen headers. The same split-header approach could potentially work for Level 2 if:
- A declarations-only header declared `Eigen::Matrix<float,3,1>` types (or a typedef)
- Implementations were pre-compiled with nvcc
- But this would require numbast to know the sizeof/alignof of Eigen types for correct lowering

### Level 3 Binding: NOT YET TESTED

Would use `bind_cxx_function_templates` / `bind_cxx_class_templates` from numbast. The `EigenVecWrapper` class template should be bindable (it's a simple struct with array fields). The `templated_dot3` function template may work after class template binding.

---

## Feature Gaps Summary

### 1. NVRTC Cannot Compile Eigen Headers

**Where**: `MemoryShimWriter` → `cuda.CUSource` → numba-cuda NVRTC compilation
**Impact**: Cannot include Eigen headers in shim CUSource. Eigen → cuda.h → `stdlib.h` — NVRTC doesn't have system C library headers, and system headers aren't compatible with NVRTC even when paths are added.
**Workaround**: **Split-header approach** — declarations-only header for the shim (NVRTC-safe), implementations pre-compiled with nvcc to fatbin, linked together at kernel JIT time.
**Note**: `config.CUDA_NVRTC_EXTRA_SEARCH_PATHS` exists for adding include paths to NVRTC, but is insufficient for Eigen due to transitive system header dependencies.

### 2. Typedef Capture for `using` Aliases

**Where**: ast_canopy parser
**Impact**: `using EigenVec3f = Eigen::Matrix<float,3,1>` not captured in `decls.typedefs`
**Note**: The underlying types are still correctly resolved in function signatures

### 3. `files_to_retain` Proxy Header Issue

**Where**: ast_canopy `parse_declarations_from_source`
**Impact**: Parsing a proxy header like `Eigen/Dense` yields 0 declarations because real declarations are in transitive includes
**Workaround**: List actual source files in `files_to_retain`, or use wrapper headers

### 4. `files_to_retain` and Include Chains

**Where**: ast_canopy `parse_declarations_from_source`
**Impact**: When a header `#include`s another header, declarations from the included file are only in the result if that file is also in `files_to_retain`. Splitting struct definitions into a separate decl header requires adding both files.
**Workaround**: Include all relevant files in the `files_to_retain` list.

### ~~5. Non-Numbast Types in Function Signatures~~ (RESOLVED)

**Previously expected**: `bind_cxx_function` would fail for `Eigen::Matrix<float,3,1>` params.
**Actual behavior**: Binding succeeds for all 9 L2 functions. numbast generates shim text with arbitrary type name strings — no validation that NVRTC can compile them. The real gap is #1 (NVRTC compilation), not type resolution.

### 5. Expression Templates / Complex Metaprogramming

**Where**: Potential issue for direct Eigen type binding
**Impact**: Eigen's expression templates (`CwiseBinaryOp`, `Product`, etc.) produce intermediate types that are not the same as the final `Matrix` type. These would be extremely difficult to bind.
**Note**: Not directly tested, but expected to be a fundamental limitation

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `/home/wangm/numbast/ast_canopy/ast_canopy/api.py` | `parse_declarations_from_source` — main parsing entry point |
| `/home/wangm/numbast/ast_canopy/ast_canopy/decl.py` | Python declaration types (Function, Struct, Template, etc.) |
| `/home/wangm/numbast/numbast/src/numbast/function.py` | `bind_cxx_function` — shim template: `extern "C" __device__ int {shim}(retval&, args...) { retval = method(args); }` |
| `/home/wangm/numbast/numbast/src/numbast/struct.py` | `bind_cxx_struct` — generates Numba type + data model |
| `/home/wangm/numbast/numbast/src/numbast/shim_writer.py` | `MemoryShimWriter` — accumulates shim source as `cuda.CUSource` |
| `/home/wangm/numbast/numbast/src/numbast/callconv.py` | `FunctionCallConv` — LLVM IR lowering for shim calls |
| `.pixi/.../numba_cuda/numba/cuda/cudadrv/nvrtc.py` | NVRTC compilation: `config.CUDA_NVRTC_EXTRA_SEARCH_PATHS`, `relocatable_device_code=True` |
| `.pixi/envs/default/include/eigen3/Eigen/src/Core/util/Macros.h` | `EIGEN_DEVICE_FUNC` → `__host__ __device__` in CUDA mode |
