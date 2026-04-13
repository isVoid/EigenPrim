"""Type registration for external C++ types (e.g. Eigen::Matrix stubs)."""

import os
import tempfile

from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_struct, MemoryShimWriter
from numbast.types import register_cxx_type
from numba import types
from numba.cuda.datamodel.models import StructModel


def register_eigen_types(shim_writer, stub_header, type_map, cc="sm_80"):
    """Parse a stub header, bind each specialization, register in numbast.

    The stub header provides NVRTC-safe type definitions that match the
    memory layout of real Eigen types (e.g. Eigen::Matrix<float,3,1> as
    a simple struct with float m_storage[3]).

    Parameters:
        shim_writer: MemoryShimWriter instance.
        stub_header: Path to the stub .cuh file.
        type_map: Dict mapping full qualified name (from ast_canopy) to the
            short name used in function signatures.
            E.g. {"Eigen::Matrix<float, 3, 1, 0, 3, 1>": "Eigen::Matrix<float, 3, 1>"}
        cc: Compute capability string for parsing (default "sm_80").

    Returns:
        Dict mapping short_name -> Python API class.
    """
    # Create a temp header that forces instantiation of the stub types
    include_dir = os.path.dirname(stub_header)
    inst_lines = [f'#pragma once\n#include "{stub_header}"\n']
    for i, full_name in enumerate(type_map):
        # Create a wrapper struct that forces the specialization
        inst_lines.append(
            f"struct _inst_{i} {{ {full_name} v; }};\n"
        )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cuh", delete=False, dir=include_dir,
    )
    tmp.write("".join(inst_lines))
    tmp.close()

    try:
        stub_decls = parse_declarations_from_source(
            tmp.name, [tmp.name, stub_header], cc,
            bypass_parse_error=True,
        )
    finally:
        os.unlink(tmp.name)

    # Bind each Matrix specialization and register under both names
    eigen_types = {}
    for cts in stub_decls.class_template_specializations:
        full_name = cts.qual_name
        if full_name not in type_map:
            continue

        short_name = type_map[full_name]
        S = bind_cxx_struct(
            shim_writer, cts, types.Type, StructModel, name=full_name,
        )
        # Also register under the short name used in function signatures
        numba_type = S._nbtype
        register_cxx_type(short_name, numba_type)
        eigen_types[short_name] = S

    return eigen_types
