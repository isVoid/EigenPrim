"""Type registration for external C++ types (e.g. Eigen::Matrix stubs)."""

import os
import tempfile

from ast_canopy import parse_declarations_from_source
from numbast import MemoryShimWriter
from numbast.struct import (
    bind_cxx_struct_conversion_opeartors,
    bind_cxx_struct_ctors,
    bind_cxx_struct_regular_methods,
)
from numbast.types import CTYPE_MAPS, register_cxx_type, to_numba_type
from numba import types
from numba.cuda.cudadecl import register_attr
from numba.cuda.datamodel.models import StructModel
from numba.cuda.extending import make_attribute_wrapper, register_model
from numba.cuda.typing.templates import AttributeTemplate


def _register_struct_model(S_type, struct_decl):
    @register_model(S_type)
    class S_model(StructModel):
        def __init__(self, dmm, fe_type):
            members = [
                (
                    f.name,
                    to_numba_type(f.type_.unqualified_non_ref_type_name),
                )
                for f in struct_decl.fields
            ]
            super().__init__(dmm, fe_type, members)


def _register_struct_attrs(S_type, s_type, public_fields_tys, method_templates):
    @register_attr
    class S_attr(AttributeTemplate):
        key = s_type

        def _field_ty(self, attr):
            field_ty = public_fields_tys[attr]
            return to_numba_type(field_ty.unqualified_non_ref_type_name)

        def _method_ty(self, typ, attr):
            template = method_templates[attr]
            return types.BoundFunction(template, typ)

        def generic_resolve(self, typ, attr):
            if attr in public_fields_tys:
                return self._field_ty(attr)
            if attr in method_templates:
                return self._method_ty(typ, attr)
            return None

    for field_name in public_fields_tys:
        make_attribute_wrapper(S_type, field_name, field_name)


def _make_api_type(struct_decl, struct_name):
    class S_type(types.Type):
        def __init__(self):
            super().__init__(name=struct_name)
            self.alignof_ = struct_decl.alignof_
            self.bitwidth = struct_decl.sizeof_ * 8

    s_type = S_type()
    S = type(struct_name, (), {"_nbtype": s_type})
    return S, S_type, s_type


def _type_aliases(full_name, short_name):
    aliases = [full_name, short_name]
    without_struct = full_name.replace("struct Eigen::", "Eigen::")
    if without_struct not in aliases:
        aliases.append(without_struct)
    return aliases


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

    # Bind each Matrix specialization and register under both names.
    #
    # Numbast binds method signatures while binding a type. Eigen vector
    # methods can return matrices (outer), and matrix methods can return
    # vectors (diagonal/vec_mul), so all Eigen type names must resolve before
    # any method signatures are created.
    eigen_types = {}
    pending = []
    for cts in stub_decls.class_template_specializations:
        full_name = cts.qual_name
        if full_name not in type_map:
            continue

        short_name = type_map[full_name]
        S, S_type, numba_type = _make_api_type(cts, full_name)
        for alias in _type_aliases(full_name, short_name):
            CTYPE_MAPS[alias] = numba_type
            register_cxx_type(alias, numba_type)
        _register_struct_model(S_type, cts)
        eigen_types[short_name] = S
        pending.append((cts, full_name, S, S_type, numba_type))

    for cts, full_name, S, S_type, numba_type in pending:
        method_templates = bind_cxx_struct_regular_methods(
            cts, full_name, numba_type, shim_writer,
        )
        public_fields_tys = {
            f.name: f.type_ for f in cts.public_fields()
        }
        _register_struct_attrs(
            S_type, numba_type, public_fields_tys, method_templates,
        )
        bind_cxx_struct_ctors(cts, full_name, S, numba_type, shim_writer)
        bind_cxx_struct_conversion_opeartors(
            cts, full_name, numba_type, shim_writer,
        )

    return eigen_types
