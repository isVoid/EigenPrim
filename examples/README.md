# Examples

Run all: `pixi run run-examples`

| Example | What it demonstrates |
|---|---|
| [01_vector_basics.py](01_vector_basics.py) | Construct a `Vector3f`, compute dot product and norm, vector addition with `+`. |
| [02_all_types.py](02_all_types.py) | All 4 dtypes (float32, float64, float16, bfloat16) with operators (`+`, `*`, `@`) and named functions. |
| [03_point_cloud_transform.py](03_point_cloud_transform.py) | 1024-point rigid-body transform (`R @ p + t`), distances and projections. |
| [04_batch_linear_solve.py](04_batch_linear_solve.py) | 1024 linear systems `Ax = b` solved via `inverse(A) @ b`. |
| [05_templates.py](05_templates.py) | `templated_dot3` — generic function template with Numba type deduction. Experimental. |
| [06_triangle_normals.py](06_triangle_normals.py) | Surface normals for 1024 triangles using `cross`, `normalized`, and `squared_norm`. |
| [07_covariance_matrix.py](07_covariance_matrix.py) | Per-point outer products for covariance matrix using `outer`, `diagonal`, and `trace`. |
| [08_aabb_reduction.py](08_aabb_reduction.py) | Axis-aligned bounding box using `cwise_min`, `cwise_max`, `min_coeff`, `max_coeff`. |
| [09_double_precision.py](09_double_precision.py) | Double-precision gravitational N-body with `Vector3d`, `squared_norm`, `norm`. |
| [10_methods.py](10_methods.py) | Method invocation on types: `a.dot(b)`, `v.norm()`, `M.inverse()`, `M.vec_mul(v)`, chaining. |
| [11_numpy_aos.py](11_numpy_aos.py) | numpy packed-struct dtype arrays (`vec3f_dtype`, `mat3f_dtype`) with unboxing wrappers; element-wise `@cuda.jit(device=True)` functions (ufunc-style). |
