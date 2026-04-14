"""High-level API for binding Eigen-wrapping CUDA headers."""

import os

from ast_canopy import parse_declarations_from_source
from ast_canopy.pylibastcanopy import execution_space
from numbast import bind_cxx_struct, bind_cxx_function, bind_cxx_function_templates, MemoryShimWriter
from numba import types as nbtypes
from numba.cuda import config as cuda_config
from numba.cuda.datamodel.models import StructModel

from eigenprim.compile import compile_fatbin
from eigenprim.types import register_eigen_types


class EigenBindings:
    """Result of bind_eigen_header — holds bound types, functions, and link info."""

    def __init__(self, types, functions, shim_writer, fatbin_path):
        self.types = types
        self.functions = functions
        self._shim_writer = shim_writer
        self._fatbin_path = fatbin_path

    def links(self):
        """Lazy generator for @cuda.jit(link=...).

        Yields the NVRTC-compiled shim CUSource followed by the
        nvcc-compiled fatbin. Must be lazy because shims are written
        during Numba's lowering phase, not at decoration time.
        """
        yield from self._shim_writer.links()
        yield self._fatbin_path


def bind_eigen_header(
    header,
    decl_header,
    impl_cu,
    eigen_include,
    stub_header=None,
    type_map=None,
    extra_retain=None,
    cc="sm_80",
):
    """One-call binding: parse -> register types -> bind functions -> compile -> link.

    Parameters:
        header: Path to the .cuh with __device__ functions (uses real Eigen).
        decl_header: Path to the NVRTC-safe declarations header.
        impl_cu: Path to the nvcc compilation unit (.cu file).
        eigen_include: Path to the Eigen include directory.
        stub_header: Optional path to NVRTC-safe stub for non-native types (L2).
            Required when type_map is provided.
        type_map: Optional dict mapping full qualified C++ type names to the
            short names used in function signatures. Required for L2-style
            bindings where Eigen types appear directly in function params.
            E.g. {"Eigen::Matrix<float, 3, 1, 0, 3, 1>": "Eigen::Matrix<float, 3, 1>"}
        extra_retain: Optional list of additional file basenames to include
            in files_to_retain when parsing.
        cc: Compute capability string (default "sm_80").

    Returns:
        EigenBindings with .types, .functions, and .links() method.
    """
    header = os.path.abspath(header)
    decl_header = os.path.abspath(decl_header)
    impl_cu = os.path.abspath(impl_cu)
    include_dir = os.path.dirname(header)

    # Set NVRTC search paths
    cuda_config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = include_dir

    # Parse the real header to get function declarations
    retain = [header]
    if extra_retain:
        retain += [os.path.join(include_dir, f) for f in extra_retain]
    # Also retain the decl header if it exists separately
    if decl_header not in retain:
        retain.append(decl_header)

    decls = parse_declarations_from_source(
        header, retain, cc,
        additional_includes=[eigen_include],
        bypass_parse_error=True,
    )

    # Create shim writer pointing at the NVRTC-safe decl header
    shim_writer = MemoryShimWriter(f'#include "{decl_header}"')

    # Register external types (L2: Eigen::Matrix stubs)
    bound_types = {}
    if stub_header and type_map:
        bound_types = register_eigen_types(shim_writer, stub_header, type_map, cc)

    # Bind structs from the parsed declarations
    for struct in decls.structs:
        try:
            S = bind_cxx_struct(shim_writer, struct, nbtypes.Type, StructModel)
            bound_types[struct.name] = S
        except Exception:
            pass

    # Bind device functions
    bound_functions = {}
    for func in decls.functions:
        if func.exec_space not in (execution_space.device, execution_space.host_device):
            continue
        try:
            result = bind_cxx_function(shim_writer, func)
            if result is not None:
                bound_functions[func.name] = result
        except Exception:
            pass

    # Bind function templates (L3)
    # Deduplicate by qualified name — when both the main header and decl header
    # are in files_to_retain, each function template appears twice.
    if decls.function_templates:
        seen = set()
        deduped = []
        for ft in decls.function_templates:
            if ft.qual_name not in seen:
                seen.add(ft.qual_name)
                deduped.append(ft)
        try:
            func_apis = bind_cxx_function_templates(
                function_templates=deduped,
                shim_writer=shim_writer,
            )
            for f in func_apis:
                name = getattr(f, "__name__", None)
                if name:
                    bound_functions[name] = f
        except Exception:
            pass

    # Compile fatbin
    fatbin_path = compile_fatbin(
        impl_cu, [eigen_include, include_dir],
    )

    return EigenBindings(bound_types, bound_functions, shim_writer, fatbin_path)
