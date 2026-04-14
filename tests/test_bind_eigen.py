"""Test numbast binding generation for Eigen-wrapping CUDA headers."""

import os
import pytest

from ast_canopy import parse_declarations_from_source

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE_DIR = os.path.join(PROJECT_ROOT, "include")
EIGEN_INCLUDE = os.path.join(
    PROJECT_ROOT, ".pixi", "envs", "default", "include", "eigen3"
)
CC = "sm_80"


def _parse(header_name, bypass_errors=False, extra_retain=None):
    source = os.path.join(INCLUDE_DIR, header_name)
    retain = [source]
    if extra_retain:
        retain += [os.path.join(INCLUDE_DIR, f) for f in extra_retain]
    return parse_declarations_from_source(
        source, retain, CC,
        additional_includes=[EIGEN_INCLUDE],
        bypass_parse_error=bypass_errors,
    )


# ---- Matrix Bindings ----

class TestMatrixBindings:
    """Test binding generation for functions using Eigen types directly."""

    @pytest.fixture(scope="class")
    def parsed(self):
        return _parse("matrix.cuh", bypass_errors=True)

    def test_bind_eigen_functions_succeeds(self, parsed):
        """bind_cxx_function succeeds for Eigen-type functions because it only
        generates shim source text + Numba typing/lowering at bind time.
        The actual failure would occur at NVRTC compilation time when the shim
        CUSource references Eigen types that NVRTC cannot compile."""
        from numbast import bind_cxx_function, MemoryShimWriter

        source = os.path.join(INCLUDE_DIR, "matrix.cuh")
        shim_writer = MemoryShimWriter(f'#include "{source}"')

        results = {"bound": [], "failed": []}
        for func in parsed.functions:
            if not func.name.startswith("eigen_"):
                continue
            try:
                bind_cxx_function(shim_writer, func)
                results["bound"].append(func.name)
            except Exception as e:
                results["failed"].append((func.name, type(e).__name__, str(e)[:200]))

        print(f"\nMatrix binding results:")
        print(f"  Bound: {results['bound']}")
        for name, etype, msg in results["failed"]:
            print(f"  Failed: {name} ({etype}): {msg}")

        assert len(results["bound"]) == 92, (
            f"Expected all 92 Eigen functions to bind, got {len(results['bound'])}"
        )
