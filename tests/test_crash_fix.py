"""Test whether the ast_canopy typedef.cpp fix resolves the Eigen crash."""
import os
from ast_canopy import parse_declarations_from_source

EIGEN_INCLUDE = "/home/wangm/numbast-eigen/.pixi/envs/default/include/eigen3"
EIGEN_SRC = os.path.join(EIGEN_INCLUDE, "Eigen/src/Core")
L2 = "/home/wangm/numbast-eigen/include/matrix.cuh"

files = ["Matrix.h", "PlainObjectBase.h", "DenseBase.h",
         "MatrixBase.h", "EigenBase.h", "DenseCoeffsBase.h", "DenseStorage.h"]

for fname in files:
    path = os.path.join(EIGEN_SRC, fname)
    try:
        d = parse_declarations_from_source(
            L2, [L2, path], "sm_80",
            additional_includes=[EIGEN_INCLUDE],
            bypass_parse_error=True,
        )
        print(f"  OK  {fname}: {len(d.typedefs)} typedefs, {len(d.class_templates)} class_templates")
    except Exception as e:
        print(f"  FAIL {fname}: {type(e).__name__}: {str(e)[:100]}")
