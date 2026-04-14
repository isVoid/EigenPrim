# Eigenprim

Numba CUDA bindings for Eigen's fixed-size vector and matrix types, powered by [numbast](https://github.com/NVIDIA/numbast).

## Quick Start

```python
from eigenprim import Vector3f, Matrix3f, dot, norm, inverse, links
from numba import cuda
import numpy as np

@cuda.jit(link=links())
def kernel(out):
    a = Vector3f(1.0, 2.0, 3.0)
    b = Vector3f(4.0, 5.0, 6.0)
    out[0] = dot(a, b)     # type-dispatched
    out[1] = norm(a + b)   # operators + generic functions

out = np.zeros(2, dtype=np.float32)
kernel[1, 1](out)
```

Import types and functions, pass `links()` to `@cuda.jit`, and use them directly in the kernel. Three ways to call operations:

- **Operators**: `a + b`, `M @ v`, `v * 2.0`
- **Generic functions**: `eigenprim.dot(a, b)`, `eigenprim.inverse(M)` — type-dispatched
- **Explicit functions**: `eigen_vec3f_dot(a, b)` — when you need full control

## Available Types

### Vectors

| Python name | Eigen type | Constructor |
|---|---|---|
| `Vector2f` | `Matrix<float,2,1>` | `Vector2f(x, y)` |
| `Vector3f` | `Matrix<float,3,1>` | `Vector3f(x, y, z)` |
| `Vector4f` | `Matrix<float,4,1>` | `Vector4f(x, y, z, w)` |
| `Vector2d` | `Matrix<double,2,1>` | `Vector2d(x, y)` |
| `Vector3d` | `Matrix<double,3,1>` | `Vector3d(x, y, z)` |
| `Vector4d` | `Matrix<double,4,1>` | `Vector4d(x, y, z, w)` |

### Matrices

| Python name | Eigen type | Constructor |
|---|---|---|
| `Matrix2f` | `Matrix<float,2,2>` | `Matrix2f(c0r0, c0r1, c1r0, c1r1)` |
| `Matrix3f` | `Matrix<float,3,3>` | `Matrix3f(c0r0, c0r1, ..., c2r2)` — 9 args |
| `Matrix4f` | `Matrix<float,4,4>` | `Matrix4f(c0r0, c0r1, ..., c3r3)` — 16 args |
| `Matrix2d` | `Matrix<double,2,2>` | `Matrix2d(c0r0, c0r1, c1r0, c1r1)` |
| `Matrix3d` | `Matrix<double,3,3>` | `Matrix3d(c0r0, c0r1, ..., c2r2)` — 9 args |
| `Matrix4d` | `Matrix<double,4,4>` | `Matrix4d(c0r0, c0r1, ..., c3r3)` — 16 args |

### Column-Major Storage

Matrix constructors take elements in **column-major** order (Eigen's native layout). For a 3x3 matrix:

```
Matrix3f(
    col0_row0, col0_row1, col0_row2,   # first column
    col1_row0, col1_row1, col1_row2,   # second column
    col2_row0, col2_row1, col2_row2,   # third column
)
```

Example — a diagonal matrix `diag(2, 3, 4)`:

```python
D = Matrix3f(2.0, 0.0, 0.0,   # col 0
             0.0, 3.0, 0.0,   # col 1
             0.0, 0.0, 4.0)   # col 2
```

Example — the matrix `[[1,2,3],[4,5,6],[7,8,9]]` (rows are 1-2-3, 4-5-6, 7-8-9):

```python
M = Matrix3f(1.0, 4.0, 7.0,   # col 0: rows 0,1,2
             2.0, 5.0, 8.0,   # col 1: rows 0,1,2
             3.0, 6.0, 9.0)   # col 2: rows 0,1,2
```

## Available Operations

All functions follow the naming pattern `eigen_{type}_{op}`, where type is `vec2f`, `vec3f`, `vec4f`, `vec2d`, `vec3d`, `vec4d`, `mat2f`, `mat3f`, `mat4f`, `mat2d`, `mat3d`, or `mat4d`.

### Vector Operations

Available for all 6 vector types (`vec2f` through `vec4d`):

| Operation | Example | Returns |
|---|---|---|
| `add(a, b)` | `eigen_vec3f_add(a, b)` | vector |
| `sub(a, b)` | `eigen_vec3f_sub(a, b)` | vector |
| `dot(a, b)` | `eigen_vec3f_dot(a, b)` | scalar |
| `norm(v)` | `eigen_vec3f_norm(v)` | scalar |
| `squared_norm(v)` | `eigen_vec3f_squared_norm(v)` | scalar |
| `normalized(v)` | `eigen_vec3f_normalized(v)` | vector |
| `scale(v, s)` | `eigen_vec3f_scale(v, 2.0)` | vector |
| `cross(a, b)` | `eigen_vec3f_cross(a, b)` | vector |
| `cwise_product(a, b)` | `eigen_vec3f_cwise_product(a, b)` | vector |
| `cwise_abs(v)` | `eigen_vec3f_cwise_abs(v)` | vector |
| `cwise_min(a, b)` | `eigen_vec3f_cwise_min(a, b)` | vector |
| `cwise_max(a, b)` | `eigen_vec3f_cwise_max(a, b)` | vector |
| `sum(v)` | `eigen_vec3f_sum(v)` | scalar |
| `min_coeff(v)` | `eigen_vec3f_min_coeff(v)` | scalar |
| `max_coeff(v)` | `eigen_vec3f_max_coeff(v)` | scalar |
| `outer(a, b)` | `eigen_vec3f_outer(a, b)` | matrix |

`cross` is only available for 3D vectors (`vec3f`, `vec3d`).

Scalar return type matches the vector's element type: `float` for `*f` types, `double` for `*d` types.

### Matrix Operations

Available for all 6 matrix types (`mat2f` through `mat4d`):

| Operation | Example | Returns |
|---|---|---|
| `add(a, b)` | `eigen_mat3f_add(a, b)` | matrix |
| `sub(a, b)` | `eigen_mat3f_sub(a, b)` | matrix |
| `mul(a, b)` | `eigen_mat3f_mul(a, b)` | matrix |
| `{mat}_{vec}_mul(m, v)` | `eigen_mat3f_vec3f_mul(m, v)` | vector |
| `determinant(m)` | `eigen_mat3f_determinant(m)` | scalar |
| `inverse(m)` | `eigen_mat3f_inverse(m)` | matrix |
| `transpose(m)` | `eigen_mat3f_transpose(m)` | matrix |
| `trace(m)` | `eigen_mat3f_trace(m)` | scalar |
| `cwise_product(a, b)` | `eigen_mat3f_cwise_product(a, b)` | matrix |
| `scale(m, s)` | `eigen_mat3f_scale(m, 2.0)` | matrix |
| `norm(m)` | `eigen_mat3f_norm(m)` | scalar (Frobenius) |
| `squared_norm(m)` | `eigen_mat3f_squared_norm(m)` | scalar |
| `diagonal(m)` | `eigen_mat3f_diagonal(m)` | vector |

Matrix-vector multiply uses the pattern `eigen_{mattype}_{vectype}_mul` — for example, `eigen_mat4f_vec4f_mul(m, v)`.

### Operator Syntax

Standard Python operators are overloaded for all Eigen types, so you can write natural expressions in kernels:

| Syntax | Vectors | Matrices |
|---|---|---|
| `a + b` | vector add | matrix add |
| `a - b` | vector sub | matrix sub |
| `v * 2.0` or `2.0 * v` | scalar multiply | — |
| `M @ N` | — | matrix multiply |
| `M @ v` | — | matrix-vector multiply |

```python
@cuda.jit(link=links())
def kernel(out):
    a = Vector3f(1.0, 2.0, 3.0)
    b = Vector3f(4.0, 5.0, 6.0)
    c = a + b           # eigen_vec3f_add
    d = a * 2.0         # eigen_vec3f_scale
    e = 3.0 * a         # eigen_vec3f_scale (reversed)

    M = Matrix3f(...)
    v = M @ a            # eigen_mat3f_vec3f_mul
    R = M @ M            # eigen_mat3f_mul
```

### Generic Functions

All operations are available as type-dispatched generic functions, accessible as `eigenprim.{op}` or via direct import. They dispatch to the correct `eigen_{type}_{op}` at JIT compile time based on argument types.

```python
from eigenprim import dot, norm, inverse, determinant, outer, diagonal

@cuda.jit(link=links())
def kernel(out):
    a = Vector3f(1.0, 2.0, 3.0)
    b = Vector3f(4.0, 5.0, 6.0)
    out[0] = dot(a, b)              # -> eigen_vec3f_dot
    out[1] = norm(a)                # -> eigen_vec3f_norm

    M = Matrix3f(...)
    out[2] = determinant(M)         # -> eigen_mat3f_determinant
    inv_M = inverse(M)              # -> eigen_mat3f_inverse
    d = diagonal(M)                 # -> eigen_mat3f_diagonal (returns Vector3f)
    P = outer(a, b)                 # -> eigen_vec3f_outer (returns Matrix3f)
```

Full list: `add`, `sub`, `dot`, `cross`, `norm`, `squared_norm`, `normalized`, `scale`, `cwise_product`, `cwise_abs`, `cwise_min`, `cwise_max`, `sum`, `min_coeff`, `max_coeff`, `outer`, `mul`, `determinant`, `inverse`, `transpose`, `trace`, `diagonal`, `vec_mul`.

### Template Functions

Generic functions where Numba deduces the scalar type from arguments:

```python
from eigenprim import templated_dot3, links
from numba import types

@cuda.jit(link=links())
def kernel(out):
    out[0] = templated_dot3(
        types.float32(1.0), types.float32(2.0), types.float32(3.0),
        types.float32(4.0), types.float32(5.0), types.float32(6.0),
    )
```

## Examples

```bash
pixi run python example.py            # Vector3f basics with operators
pixi run python example_l2.py         # All types: Vector2f..Matrix4f
pixi run python example_l3.py         # Template bindings
pixi run python example_transform.py  # Point cloud transform + linear solve
```

`example_transform.py` demonstrates a realistic pattern: rotating a 1024-point cloud with a rigid-body transform kernel (`R @ p + t`), then solving per-element linear systems via `inverse(A) @ b`. Results are verified against numpy.

## Custom Bindings

For binding your own Eigen-wrapping CUDA headers, use the low-level API.

### Step 1: Write Three Files

**Implementation header** (`my_ops.cuh`) — uses real Eigen, compiled by nvcc:

```cuda
#pragma once
#include <Eigen/Dense>
#include "my_ops_decl.cuh"

using Vec = Eigen::Matrix<float, 3, 1>;

__device__ float my_dot(Vec a, Vec b) {
  return a.dot(b);
}
```

**Declaration header** (`my_ops_decl.cuh`) — NVRTC-safe, no Eigen:

```cuda
#pragma once
#include "eigen_stub.cuh"  // from eigenprim/include/

using Vec = Eigen::Matrix<float, 3, 1>;

__device__ float my_dot(Vec a, Vec b);
```

**Compilation unit** (`my_ops.cu`):

```cuda
#include "my_ops.cuh"
```

### Step 2: Bind

```python
from eigenprim import bind_eigen_header

bindings = bind_eigen_header(
    header="my_ops.cuh",
    decl_header="my_ops_decl.cuh",
    impl_cu="my_ops.cu",
    eigen_include="/path/to/eigen3",
    stub_header="path/to/eigen_stub.cuh",
    type_map={
        "Eigen::Matrix<float, 3, 1, 0, 3, 1>": "Eigen::Matrix<float, 3, 1>",
    },
)

MyVec = bindings.types["Eigen::Matrix<float, 3, 1>"]
my_dot = bindings.functions["my_dot"]
```

### Step 3: Use

```python
from numba import cuda

@cuda.jit(link=bindings.links())
def kernel(out):
    a = MyVec(1.0, 2.0, 3.0)
    b = MyVec(4.0, 5.0, 6.0)
    out[0] = my_dot(a, b)
```

The key requirement: NVRTC (used by numba-cuda) cannot compile Eigen headers. So you must separate declarations (NVRTC-safe, using `eigen_stub.cuh`) from implementations (compiled by nvcc into a fatbin). `links()` yields both for `@cuda.jit(link=...)` to link together.

## Setup

```bash
pixi install
pixi run build-ast-canopy
pixi run install-numbast
pixi run python example.py
```

### Dependencies

- [numbast](https://github.com/NVIDIA/numbast) — C++ to Numba binding generator
- [Eigen](https://eigen.tuxfamily.org/) — via conda-forge
- CUDA Toolkit 12.9 — via pixi

### Eigen Detection

The Eigen include path is auto-detected from `CONDA_PREFIX/include/eigen3`. Override with `EIGEN_INCLUDE_DIR` environment variable if needed.
