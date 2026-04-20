# Eigenprim: Slide Deck Outline

Generate a presentation slide deck from the following outline. Each section is one slide. Use clean, minimal design with code snippets where indicated.

---

## Slide 1: Title

**Eigenprim — Eigen Device Primitives for CUDA Python**

Subtitle: Numba CUDA bindings for Eigen's fixed-size linear algebra, powered by numbast.

Author: Michael Wang

---

## Slide 2: The Problem

- Eigen is the standard C++ library for fixed-size linear algebra (robotics, graphics, physics)
- CUDA Python (via Numba) has no access to Eigen's device-side operations
- Users either rewrite math in raw CUDA or can't use Eigen at all from Python
- Eigen headers can't compile under NVRTC (Numba's runtime compiler) because Eigen transitively includes system headers NVRTC doesn't have

---

## Slide 3: The Solution — Split-Header Architecture

Diagram showing the two compilation paths:

1. **Declaration header** (NVRTC-safe) → compiled by NVRTC at JIT time → shim functions
2. **Implementation header** (real Eigen) → compiled by nvcc ahead of time → fatbin

Both linked together via CUDA's relocatable device code (RDC) at kernel launch.

Key insight: separate what NVRTC sees (stub types, declarations only) from what nvcc compiles (real Eigen implementations).

---

## Slide 4: User Experience — Before vs After

**Before** — what it takes to bind ONE Eigen function by hand:

1. Write a C++ implementation header (`my_ops.cuh`) that includes `<Eigen/Dense>` and defines your `__device__` function
2. Write a separate NVRTC-safe declaration header (`my_ops_decl.cuh`) with a layout-compatible stub type — must match Eigen's internal `DenseStorage` byte-for-byte
3. Write a `.cu` compilation unit for nvcc
4. Compile the `.cu` to a fatbin with `nvcc -rdc=true -fatbin`
5. Parse the header with `ast_canopy` (Clang-based C++ parser), choosing the right `files_to_retain` and `bypass_parse_error` flags
6. Create a `MemoryShimWriter` pointed at the decl header
7. For each Eigen type: create a temp header forcing template instantiation, parse it, call `bind_cxx_struct`, call `register_cxx_type` with the correct fully-qualified name mapping (e.g. `"Eigen::Matrix<float, 3, 1, 0, 3, 1>"` → `"Eigen::Matrix<float, 3, 1>"`)
8. For each function: call `bind_cxx_function` with the parsed declaration
9. Assemble a lazy link generator that yields the shim CUSource + fatbin path
10. Pass the generator to `@cuda.jit(link=...)`

Repeat for every type and function you need. Getting any step wrong gives cryptic NVRTC or Numba type errors.

**After** — eigenprim:
```python
from eigenprim import Vector3f, dot, norm, inverse, links

@cuda.jit(link=links())
def kernel(out):
    a = Vector3f(1.0, 2.0, 3.0)
    out[0] = dot(a, a)
    out[1] = norm(a + a)
```

All 10 steps happen automatically at import time. 24 types and 340 functions, ready to use.

---

## Slide 5: Supported Types — 24 Types Across 4 Dtypes

Table:

|  | float32 | float64 | float16 | bfloat16 |
|---|---|---|---|---|
| Vector2 | Vector2f | Vector2d | Vector2h | Vector2bf |
| Vector3 | Vector3f | Vector3d | Vector3h | Vector3bf |
| Vector4 | Vector4f | Vector4d | Vector4h | Vector4bf |
| Matrix2 | Matrix2f | Matrix2d | Matrix2h | Matrix2bf |
| Matrix3 | Matrix3f | Matrix3d | Matrix3h | Matrix3bf |
| Matrix4 | Matrix4f | Matrix4d | Matrix4h | Matrix4bf |

Half and bfloat16 support required bridging Eigen::half → numba.float16 via numbast's `register_cxx_type` API.

---

## Slide 6: Supported Operations — 340 Functions

**Vector operations** (per type): add, sub, dot, norm, squared_norm, normalized, scale, cross (3D), cwise_product, cwise_abs, cwise_min, cwise_max, sum, min_coeff, max_coeff, outer

**Matrix operations** (per type): add, sub, mul, vec_mul, determinant, inverse, transpose, trace, cwise_product, scale, norm, squared_norm, diagonal

Note: Decomposition solvers (LU, QR, Cholesky, SVD) are host-only in Eigen and not available on device.

---

## Slide 7: Three Ways to Use Operations

**1. Python operators** — most natural:
```python
c = a + b        # vector/matrix add
R = M @ N        # matrix multiply
v = M @ a        # matrix-vector multiply
s = v * 2.0      # scalar multiply
```

**2. Generic dispatched functions** — no type prefix:
```python
from eigenprim import dot, norm, inverse, determinant
out = dot(a, b)          # dispatches by type
inv_M = inverse(M)
```

**3. Explicit functions** — full control:
```python
from eigenprim import eigen_vec3f_dot, eigen_mat3f_inverse
```

Operators and generic functions are implemented via Numba's `@overload` mechanism, dispatching at JIT compile time.

---

## Slide 8: Real Example — Point Cloud Transform

```python
@cuda.jit(link=links())
def transform_kernel(px, py, pz, distances, projections, ...):
    i = cuda.grid(1)
    p = Vector3f(px[i], py[i], pz[i])
    R = Matrix3f(...)
    t = Vector3f(...)

    q = R @ p + t               # rigid-body transform
    distances[i] = norm(q)      # distance from origin
    projections[i] = dot(q, d)  # projection onto direction
```

- 1024 points transformed in parallel
- Verified against numpy: max error ~5e-7
- Also demonstrated: batch linear solve via `inverse(A) @ b`

---

## Slide 9: Real Example — Triangle Normals & Covariance

**Triangle normals** (1024 triangles):
```python
e1 = p1 - p0          # edge vectors
e2 = p2 - p0
n = cross(e1, e2)     # normal direction
unit_n = normalized(n) # unit normal
area = norm(n) * 0.5   # triangle area
```

**Covariance matrix** (2048 points):
```python
centered = p - mean
P = outer(centered, centered)  # rank-1 outer product
d = diagonal(P)                # per-axis variances
total_var = trace(P)           # total variance
```

---

## Slide 10: Architecture Stack

Diagram showing layers from top to bottom:

1. **User code** — `from eigenprim import Vector3f, dot`
2. **Generic dispatch** (`dispatch.py`) — `@overload` routes `dot(a,b)` → `eigen_vec3f_dot`
3. **Operator overloads** (`operators.py`) — `@overload(operator.add)` for `a + b`
4. **Binding layer** (`matrix.py`) — `bind_eigen_header()` → 340 bound functions
5. **numbast** — `bind_cxx_function` generates Numba typing + LLVM lowering + shim code
6. **NVRTC shim** — `extern "C" __device__` wrapper calling mangled C++ symbol
7. **nvcc fatbin** — pre-compiled Eigen implementations (matrix.cu → matrix.cuh → Eigen/Dense)
8. **CUDA RDC linker** — links shim + fatbin at kernel JIT time

---

## Slide 11: Key Technical Decisions

- **Stub types for NVRTC**: `eigen_stub.cuh` defines layout-compatible `Eigen::Matrix` with `Scalar m_storage[R*C]` — matches real Eigen's DenseStorage
- **Half/bf16 via `register_cxx_type`**: Maps `Eigen::half` → `numba.float16` before binding, so numbast resolves scalar types correctly
- **Lazy module imports**: `import eigenprim` is instant; binding only triggers when you access a type — avoids compiling all 340 functions if you only need Vector3f
- **Fatbin caching**: nvcc compilation cached by file mtime + GPU arch hash — only pays the cost once

---

## Slide 12: What's Next / Limitations

**Current limitations**:
- Decomposition solvers (LU, QR, SVD, eigenvalues) are host-only in Eigen
- Solve is done via `inverse(A) @ b` — efficient for 2x2/3x3/4x4 via cofactor expansion, but not numerically stable for ill-conditioned systems
- No dynamic-size matrix support (Eigen::MatrixXf) — fixed-size only

**Potential future work**:
- Geometry module: quaternions, rotations, affine transforms
- Integration with cuBLAS/cuSOLVER for larger matrices
- Packaging as pip-installable library

---

## Slide 13: Summary

- **24 types** across 4 dtypes (float32, float64, float16, bfloat16)
- **340 device functions** covering all common linear algebra operations
- **Natural syntax**: `a + b`, `M @ v`, `dot(a, b)`, `inverse(M)`
- **Split-header architecture** solves the NVRTC/Eigen incompatibility
- **9 verified examples** from basics to batch transforms to narrow dtypes
- Built on **numbast** + **ast_canopy** — the same pipeline can bind other C++ CUDA libraries
