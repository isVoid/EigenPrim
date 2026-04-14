# Examples

Run all: `pixi run run-examples`

| Example | What it demonstrates |
|---|---|
| [01_vector_basics.py](01_vector_basics.py) | Construct a `Vector3f`, compute dot product and norm using generic functions, vector addition with `+` operator. |
| [02_all_types.py](02_all_types.py) | All 12 types (`Vector2f`..`Matrix4f`) with operators (`+`, `*`, `@`) and named functions (dot, cross, norm, determinant, inverse, transpose, trace). |
| [03_templates.py](03_templates.py) | `templated_dot3` — a generic function template where Numba deduces `<float>` from argument types at JIT time. |
| [04_point_cloud_transform.py](04_point_cloud_transform.py) | 1024-point rigid-body transform: each CUDA thread computes `q = R @ p + t`, then distance from origin and projection onto a direction. Verified against numpy. |
| [05_batch_linear_solve.py](05_batch_linear_solve.py) | 1024 linear systems `Ax = b` solved in parallel via `inverse(A) @ b`. Uses Eigen's cofactor-based inverse, efficient for small fixed-size matrices. Verified against `numpy.linalg.solve`. |
