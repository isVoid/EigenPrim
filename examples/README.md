# Examples

Run all: `pixi run run-examples`

| Example | What it demonstrates |
|---|---|
| [01_vector_basics.py](01_vector_basics.py) | Construct a `Vector3f`, compute dot product and norm, vector addition with `+`. |
| [02_all_types.py](02_all_types.py) | Float types (`Vector2f`..`Matrix4f`) with operators (`+`, `*`, `@`) and named functions. Half/bf16/double types follow the same pattern. |
| [03_point_cloud_transform.py](03_point_cloud_transform.py) | 1024-point rigid-body transform (`R @ p + t`), distances and projections. |
| [04_batch_linear_solve.py](04_batch_linear_solve.py) | 1024 linear systems `Ax = b` solved via `inverse(A) @ b`. |
| [05_templates.py](05_templates.py) | `templated_dot3` — generic function template with Numba type deduction. Experimental. |
| [06_triangle_normals.py](06_triangle_normals.py) | Surface normals for 1024 triangles using `cross`, `normalized`, and `squared_norm`. |
| [07_covariance_matrix.py](07_covariance_matrix.py) | Per-point outer products for covariance matrix using `outer`, `diagonal`, and `trace`. |
| [08_aabb_reduction.py](08_aabb_reduction.py) | Axis-aligned bounding box using `cwise_min`, `cwise_max`, `min_coeff`, `max_coeff`. |
| [09_double_precision.py](09_double_precision.py) | Double-precision gravitational N-body with `Vector3d`, `squared_norm`, `norm`. |
| [10_half_precision.py](10_half_precision.py) | Half-precision (fp16): `Vector3h`, `Matrix3h` — dot, norm, cross, inverse, operators. |
| [11_bfloat16.py](11_bfloat16.py) | Bfloat16: `Vector3bf`, `Matrix3bf` — dot, norm, determinant, inverse, operators. |
