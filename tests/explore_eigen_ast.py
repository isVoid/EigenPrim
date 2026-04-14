"""Phase 1: Explore what ast_canopy extracts from Eigen's internal headers.

Goal: Understand whether Eigen::Matrix<float,3,1> can be captured as a
usable struct (with sizeof, fields, constructors) for numbast binding.

Run: pixi run python tests/explore_eigen_ast.py
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


def parse_with_retain(files_to_retain, label, bypass=True):
    """Parse the L2 header with various files_to_retain configs."""
    print(f"\n{'='*70}")
    print(f"Config: {label}")
    print(f"  files_to_retain: {[os.path.basename(f) for f in files_to_retain]}")
    print(f"{'='*70}")

    try:
        decls = parse_declarations_from_source(
            L2_HEADER,
            files_to_retain,
            CC,
            additional_includes=[EIGEN_INCLUDE],
            bypass_parse_error=bypass,
        )
    except Exception as e:
        print(f"  PARSE FAILED: {type(e).__name__}: {e}")
        return None

    print(f"\n  Structs ({len(decls.structs)}):")
    for s in decls.structs[:20]:
        print(f"    {s.name} (sizeof={s.sizeof_}, alignof={s.alignof_})")
        for f in s.fields[:10]:
            print(f"      field: {f.name} : {f.type_.unqualified_non_ref_type_name}")
        ctors = list(s.constructors())
        if ctors:
            print(f"      constructors: {len(ctors)}")
            for c in ctors[:5]:
                print(f"        {c}")

    print(f"\n  Class Templates ({len(decls.class_templates)}):")
    for ct in decls.class_templates[:20]:
        tparams = [tp.name for tp in ct.template_parameters]
        print(f"    {ct.qual_name} <{', '.join(tparams)}>")

    print(f"\n  Class Template Specializations ({len(decls.class_template_specializations)}):")
    for cts in decls.class_template_specializations[:20]:
        args = cts.actual_template_arguments
        print(f"    {cts.name} (sizeof={cts.record.sizeof_}, alignof={cts.record.alignof_})")
        print(f"      template args: {args}")
        for f in cts.record.fields[:10]:
            print(f"      field: {f.name} : {f.type_.unqualified_non_ref_type_name}")

    print(f"\n  Typedefs ({len(decls.typedefs)}):")
    for td in decls.typedefs[:20]:
        print(f"    {td.name} -> {td.underlying_type}")

    print(f"\n  Functions ({len(decls.functions)}):")
    for func in decls.functions[:10]:
        ret = func.return_type.unqualified_non_ref_type_name
        params = ", ".join(p.type_.unqualified_non_ref_type_name for p in func.params)
        print(f"    {func.name}({params}) -> {ret}")

    print(f"\n  Function Templates ({len(decls.function_templates)}):")
    for ft in decls.function_templates[:10]:
        print(f"    {ft.qual_name}")

    return decls


# ── Config 1: Just the L2 wrapper header (baseline) ──────────────

parse_with_retain(
    [L2_HEADER],
    "L2 header only (baseline)",
)

# ── Config 2: Add Matrix.h ───────────────────────────────────────

parse_with_retain(
    [L2_HEADER, os.path.join(EIGEN_SRC, "Matrix.h")],
    "L2 + Matrix.h",
)

# ── Config 3: Add Matrix.h + PlainObjectBase.h + DenseStorage.h ──

parse_with_retain(
    [
        L2_HEADER,
        os.path.join(EIGEN_SRC, "Matrix.h"),
        os.path.join(EIGEN_SRC, "PlainObjectBase.h"),
        os.path.join(EIGEN_SRC, "DenseStorage.h"),
    ],
    "L2 + Matrix.h + PlainObjectBase.h + DenseStorage.h",
)

# ── Config 4: Broad Eigen/src/Core/ ──────────────────────────────
# Include ALL files under Eigen/src/Core/ to catch whatever
# ast_canopy can extract

core_files = []
for root, dirs, files in os.walk(EIGEN_SRC):
    for f in files:
        if f.endswith(".h"):
            core_files.append(os.path.join(root, f))

parse_with_retain(
    [L2_HEADER] + core_files,
    f"L2 + ALL Eigen/src/Core/*.h ({len(core_files)} files)",
)
