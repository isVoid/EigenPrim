"""Phase 1 deep dive: DenseStorage.h analysis.

DenseStorage.h is the ONLY Eigen file that doesn't crash ast_canopy,
and it contains the actual data layout for fixed-size Eigen::Matrix types.
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
DENSE_STORAGE = os.path.join(EIGEN_SRC, "DenseStorage.h")

print("Parsing L2 + DenseStorage.h ...")
decls = parse_declarations_from_source(
    L2_HEADER,
    [L2_HEADER, DENSE_STORAGE],
    CC,
    additional_includes=[EIGEN_INCLUDE],
    bypass_parse_error=True,
)

print(f"\n{'='*60}")
print("CLASS TEMPLATES from DenseStorage.h")
print(f"{'='*60}")
for ct in decls.class_templates:
    tparams = [(tp.name, tp.kind) for tp in ct.template_parameters]
    print(f"\n  {ct.qual_name}")
    print(f"    template params: {tparams}")
    if hasattr(ct, 'record') and ct.record:
        r = ct.record
        print(f"    fields: {[(f.name, f.type_.unqualified_non_ref_type_name) for f in r.fields]}")

print(f"\n{'='*60}")
print("CLASS TEMPLATE SPECIALIZATIONS from DenseStorage.h")
print(f"{'='*60}")
for cts in decls.class_template_specializations:
    print(f"\n  {cts.name}")
    print(f"    template args: {cts.actual_template_arguments}")
    # Discover all available attributes
    print(f"    attributes: {[a for a in dir(cts) if not a.startswith('_')]}")
    # Try common attribute names
    for attr in ['record', 'struct_', 'sizeof_', 'alignof_', 'fields',
                 'constructors', 'methods', 'public_fields', 'name',
                 'qual_name', 'class_template']:
        if hasattr(cts, attr):
            val = getattr(cts, attr)
            if not callable(val):
                print(f"    {attr}: {val}")
