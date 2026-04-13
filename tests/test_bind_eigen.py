"""Test numbast binding generation for Eigen-wrapping CUDA headers.

Level 1: Thin wrapper (Vec3f struct + plain device functions)
  - This should work end-to-end because Vec3f is a simple struct with scalar fields.
  - The end-to-end kernel uses a split-header approach:
    - Declarations-only header (no Eigen) for the NVRTC-compiled shim
    - Implementations pre-compiled to fatbin by nvcc (with Eigen)

Level 2: Direct Eigen types (Eigen::Matrix<float,3,1> as params)
  - This is expected to hit limitations: Eigen::Matrix is not a numbast-generated
    struct, so bind_cxx_function won't know how to lower it.
"""

import os
import shutil
import subprocess
import tempfile
import pytest
import numpy as np

from ast_canopy import parse_declarations_from_source
from ast_canopy.pylibastcanopy import execution_space

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE_DIR = os.path.join(PROJECT_ROOT, "include")
EIGEN_INCLUDE = os.path.join(
    PROJECT_ROOT, ".pixi", "envs", "default", "include", "eigen3"
)
DECL_HEADER = os.path.join(INCLUDE_DIR, "eigen_wrapper_l1_decl.cuh")
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


# ---- Level 1 Bindings: Thin wrappers ----

class TestLevel1Bindings:
    """Test numbast binding generation for Level 1 (thin wrappers).
    Vec3f is a simple struct with float x,y,z - should be bindable."""

    @pytest.fixture(scope="class")
    def parsed(self):
        return _parse("eigen_wrapper_l1.cuh", extra_retain=["eigen_wrapper_l1_decl.cuh"])

    def test_bind_vec3f_struct(self, parsed):
        from numbast import bind_cxx_struct, MemoryShimWriter
        from numba import types
        from numba.cuda.datamodel.models import StructModel

        source = os.path.join(INCLUDE_DIR, "eigen_wrapper_l1.cuh")
        shim_writer = MemoryShimWriter(f'#include "{source}"')

        vec3f_struct = [s for s in parsed.structs if s.name == "Vec3f"][0]

        print(f"\nVec3f struct info:")
        print(f"  name: {vec3f_struct.name}")
        print(f"  sizeof: {vec3f_struct.sizeof_}")
        print(f"  alignof: {vec3f_struct.alignof_}")
        print(f"  fields: {[(f.name, f.type_.unqualified_non_ref_type_name) for f in vec3f_struct.fields]}")
        print(f"  public_fields: {[(f.name, f.type_.unqualified_non_ref_type_name) for f in vec3f_struct.public_fields()]}")
        print(f"  constructors: {[(str(c), c.exec_space) for c in vec3f_struct.constructors()]}")
        print(f"  methods: {[(m.name, str(m)) for m in vec3f_struct.methods]}")

        try:
            Vec3fType = bind_cxx_struct(
                shim_writer, vec3f_struct, types.Type, StructModel
            )
            assert Vec3fType is not None
            print(f"  Vec3f type created: {Vec3fType}")
        except Exception as e:
            pytest.fail(f"Failed to bind Vec3f struct: {e}")

    def test_bind_device_functions(self, parsed):
        from numbast import bind_cxx_struct, bind_cxx_function, MemoryShimWriter
        from numba import types
        from numba.cuda.datamodel.models import StructModel

        source = os.path.join(INCLUDE_DIR, "eigen_wrapper_l1.cuh")
        shim_writer = MemoryShimWriter(f'#include "{source}"')

        vec3f_struct = [s for s in parsed.structs if s.name == "Vec3f"][0]
        Vec3fType = bind_cxx_struct(
            shim_writer, vec3f_struct, types.Type, StructModel
        )

        device_funcs = [
            f for f in parsed.functions
            if f.exec_space == execution_space.device
        ]

        bound = {}
        failed = {}
        for func in device_funcs:
            try:
                result = bind_cxx_function(shim_writer, func)
                bound[func.name] = result
                print(f"  Bound: {func.name}")
            except Exception as e:
                failed[func.name] = str(e)
                print(f"  FAILED to bind: {func.name}: {e}")

        print(f"\nBound {len(bound)}/{len(device_funcs)} device functions")
        if failed:
            print(f"Failed functions: {list(failed.keys())}")

        assert len(bound) > 0, "No functions could be bound"

    def test_end_to_end_kernel(self, parsed):
        """Full end-to-end: parse -> bind -> compile shim + fatbin -> run kernel.

        Uses the split-header approach:
        - Shim preceding_text uses the decl header (no Eigen, NVRTC-safe)
        - Eigen implementations are pre-compiled to fatbin by nvcc
        - Both are linked together at kernel JIT time
        """
        from numbast import bind_cxx_struct, bind_cxx_function, MemoryShimWriter
        from numba import types, cuda
        from numba.cuda.datamodel.models import StructModel

        # Check for GPU availability
        try:
            cuda.gpus[0]
        except cuda.cudadrv.error.CudaSupportError:
            pytest.skip("No CUDA GPU available")

        # Detect GPU compute capability for nvcc
        device = cuda.get_current_device()
        arch = f"sm_{device.compute_capability[0]}{device.compute_capability[1]}"

        # Pre-compile Eigen implementations with nvcc
        impl_cu = os.path.join(INCLUDE_DIR, "eigen_wrapper_l1_impl.cu")
        tmpdir = tempfile.mkdtemp(prefix="numbast_eigen_")
        fatbin_path = os.path.join(tmpdir, "eigen_wrapper_l1_impl.fatbin")
        try:
            nvcc = shutil.which("nvcc")
            if nvcc is None:
                pytest.skip("nvcc not found")
            result = subprocess.run(
                [nvcc, "-rdc=true", "-fatbin", f"-arch={arch}",
                 f"-I{EIGEN_INCLUDE}", f"-I{INCLUDE_DIR}",
                 impl_cu, "-o", fatbin_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                pytest.fail(f"nvcc compilation failed:\n{result.stderr[:2000]}")
        except subprocess.TimeoutExpired:
            pytest.fail("nvcc compilation timed out")

        # Use declarations-only header for shim (no Eigen, NVRTC-safe)
        shim_writer = MemoryShimWriter(f'#include "{DECL_HEADER}"')

        vec3f_struct = [s for s in parsed.structs if s.name == "Vec3f"][0]
        Vec3fType = bind_cxx_struct(
            shim_writer, vec3f_struct, types.Type, StructModel
        )

        device_funcs = {f.name: f for f in parsed.functions
                        if f.exec_space == execution_space.device}

        bound_funcs = {}
        for name, func in device_funcs.items():
            try:
                bound_funcs[name] = bind_cxx_function(shim_writer, func)
            except Exception:
                pass

        if "vec3f_dot" not in bound_funcs:
            pytest.fail("vec3f_dot could not be bound")

        vec3f_dot = bound_funcs["vec3f_dot"]

        # Link shim CUSource (NVRTC) + pre-compiled fatbin (nvcc) together.
        # links() is a lazy generator — shims are written during lowering,
        # so we must NOT expand it eagerly with [*links(), ...].
        def all_links():
            yield from shim_writer.links()
            yield fatbin_path

        @cuda.jit(link=all_links())
        def kernel(out):
            a = Vec3fType(1.0, 2.0, 3.0)
            b = Vec3fType(4.0, 5.0, 6.0)
            out[0] = vec3f_dot(a, b)

        out = np.zeros(1, dtype=np.float32)
        kernel[1, 1](out)
        expected = 1*4 + 2*5 + 3*6  # 32.0
        np.testing.assert_allclose(out[0], expected, rtol=1e-5)
        print(f"\nKernel executed successfully: dot({'{1,2,3}'}, {'{4,5,6}'}) = {out[0]}")


# ---- Level 2 Bindings: Direct Eigen types ----

class TestLevel2Bindings:
    """Test binding generation for functions using Eigen types directly.
    Expected to hit gaps: Eigen::Matrix is not a numbast-bound type."""

    @pytest.fixture(scope="class")
    def parsed(self):
        return _parse("eigen_wrapper_l2.cuh", bypass_errors=True)

    def test_bind_eigen_functions_succeeds(self, parsed):
        """bind_cxx_function succeeds for Eigen-type functions because it only
        generates shim source text + Numba typing/lowering at bind time.
        The actual failure would occur at NVRTC compilation time when the shim
        CUSource references Eigen types that NVRTC cannot compile."""
        from numbast import bind_cxx_function, MemoryShimWriter

        source = os.path.join(INCLUDE_DIR, "eigen_wrapper_l2.cuh")
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

        print(f"\nLevel 2 binding results:")
        print(f"  Bound: {results['bound']}")
        for name, etype, msg in results["failed"]:
            print(f"  Failed: {name} ({etype}): {msg}")

        # Binding generation succeeds — numbast generates shim text and
        # typing/lowering without checking if NVRTC can handle the types.
        # The gap is at shim compilation time, not bind time.
        assert len(results["bound"]) == 9, (
            f"Expected all 9 Eigen functions to bind, got {len(results['bound'])}"
        )
