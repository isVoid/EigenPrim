"""Phase 1 continued: isolate the ast_canopy crash and try alternative approaches.

1. Test individual Eigen files to find which causes the crash
2. Try a minimal header that explicitly instantiates Eigen::Matrix<float,3,1>
3. Check if class template specializations are reported for our header
"""

import os
from ast_canopy import parse_declarations_from_source

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE_DIR = os.path.join(PROJECT_ROOT, "include")
EIGEN_INCLUDE = os.path.join(
    PROJECT_ROOT, ".pixi", "envs", "default", "include", "eigen3"
)
EIGEN_SRC = os.path.join(EIGEN_INCLUDE, "Eigen", "src", "Core")
CC = "sm_80"

L2_HEADER = os.path.join(INCLUDE_DIR, "matrix.cuh")


def try_parse(files_to_retain, label, source=L2_HEADER):
    """Attempt parse and report success/failure."""
    print(f"\n--- {label} ---")
    try:
        decls = parse_declarations_from_source(
            source, files_to_retain, CC,
            additional_includes=[EIGEN_INCLUDE],
            bypass_parse_error=True,
        )
        print(f"  OK: {len(decls.structs)} structs, "
              f"{len(decls.class_templates)} class templates, "
              f"{len(decls.class_template_specializations)} specializations, "
              f"{len(decls.typedefs)} typedefs, "
              f"{len(decls.functions)} functions")
        return decls
    except Exception as e:
        print(f"  CRASH: {type(e).__name__}: {str(e)[:200]}")
        return None


# ── Test 1: Individual Eigen files ────────────────────────────────
print("=" * 60)
print("TEST 1: Which Eigen file crashes ast_canopy?")
print("=" * 60)

eigen_files = [
    "DenseStorage.h",
    "Matrix.h",
    "PlainObjectBase.h",
    "DenseBase.h",
    "MatrixBase.h",
    "DenseCoeffsBase.h",
    "EigenBase.h",
]

for f in eigen_files:
    path = os.path.join(EIGEN_SRC, f)
    if os.path.exists(path):
        try_parse([L2_HEADER, path], f"L2 + {f}")

# ── Test 2: Minimal explicit instantiation header ─────────────────
print("\n" + "=" * 60)
print("TEST 2: Minimal header with explicit Matrix instantiation")
print("=" * 60)

# Create a temporary header that forces Matrix<float,3,1> instantiation
import tempfile
tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.cuh', delete=False,
                                  dir=INCLUDE_DIR)
tmp.write("""\
#pragma once
#include <Eigen/Dense>

// Force explicit instantiation of the types we care about
template class Eigen::Matrix<float, 3, 1>;
template class Eigen::Matrix<float, 3, 3>;
template class Eigen::Matrix<float, 4, 1>;
""")
tmp.close()

try_parse([tmp.name], "Explicit Matrix instantiations only")

# Also try with the L2 header
try_parse([L2_HEADER, tmp.name], "L2 + explicit Matrix instantiations")

os.unlink(tmp.name)

# ── Test 3: Check class template specializations in baseline parse ─
print("\n" + "=" * 60)
print("TEST 3: Deep inspect of baseline L2 parse (what Clang sees)")
print("=" * 60)

decls = try_parse([L2_HEADER], "Baseline L2 (deep inspect)")
if decls:
    print(f"\n  All declaration counts:")
    print(f"    structs: {len(decls.structs)}")
    print(f"    functions: {len(decls.functions)}")
    print(f"    function_templates: {len(decls.function_templates)}")
    print(f"    class_templates: {len(decls.class_templates)}")
    print(f"    class_template_specializations: {len(decls.class_template_specializations)}")
    print(f"    typedefs: {len(decls.typedefs)}")
    print(f"    enums: {len(decls.enums)}")

    # Check if any functions have Eigen types we can inspect
    for func in decls.functions[:3]:
        print(f"\n  Function: {func.name}")
        print(f"    mangled: {func.mangled_name}")
        for p in func.params:
            t = p.type_
            print(f"    param {p.name}:")
            print(f"      type name: {t.unqualified_non_ref_type_name}")
            print(f"      is_left_reference: {t.is_left_reference()}")
            print(f"      is_right_reference: {t.is_right_reference()}")
            # Check available attributes
            for attr in dir(t):
                if not attr.startswith('_') and attr not in ('is_left_reference', 'is_right_reference'):
                    try:
                        val = getattr(t, attr)
                        if not callable(val):
                            print(f"      {attr}: {val}")
                    except Exception:
                        pass
