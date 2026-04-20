"""Fatbin compilation utility for nvcc."""

import hashlib
import os
import subprocess
import tempfile

from numba import cuda


def _find_nvcc():
    """Locate the nvcc executable via cuda-pathfinder.

    Search order:
      1. NVIDIA Python wheels  (site-packages/nvidia/cu13/bin/ or nvidia/cuda_nvcc/bin/)
      2. Conda environment     (CONDA_PREFIX/bin/)
      3. CUDA Toolkit env vars (CUDA_HOME/bin/ or CUDA_PATH/bin/)

    Raises:
        RuntimeError: If nvcc cannot be found.
    """
    from cuda.pathfinder import find_nvidia_binary_utility
    path = find_nvidia_binary_utility("nvcc")
    if path is None:
        raise RuntimeError(
            "nvcc not found. Install the CUDA compiler wheel:\n"
            "  pip install nvidia-nvcc-cu13"
        )
    return path


def _detect_arch():
    """Detect GPU compute capability from the current CUDA device."""
    cc = cuda.get_current_device().compute_capability
    return f"sm_{cc[0]}{cc[1]}"


def compile_fatbin(impl_cu, include_dirs, arch=None, cache_dir=None):
    """Compile a .cu file to fatbin with nvcc -rdc=true.

    Parameters:
        impl_cu: Path to the .cu compilation unit.
        include_dirs: List of include directories for nvcc -I flags.
        arch: GPU architecture (e.g. "sm_80"). Auto-detected if None.
        cache_dir: Directory to cache fatbin by file mtime. Uses tempdir if None.

    Returns:
        Path to the compiled .fatbin file.

    Raises:
        RuntimeError: If nvcc is not found or compilation fails.
    """
    nvcc = _find_nvcc()  # raises RuntimeError if not found

    if arch is None:
        arch = _detect_arch()

    # Determine output path (with optional caching)
    impl_cu = os.path.abspath(impl_cu)
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="eigenprim_")

    # Cache key based on file path + mtime
    mtime = os.path.getmtime(impl_cu)
    key = hashlib.md5(f"{impl_cu}:{mtime}:{arch}".encode()).hexdigest()[:12]
    fatbin_path = os.path.join(cache_dir, f"{os.path.basename(impl_cu)}.{key}.fatbin")

    if os.path.exists(fatbin_path):
        return fatbin_path

    cmd = [
        nvcc, "-rdc=true", "-fatbin", f"-arch={arch}",
        *[f"-I{d}" for d in include_dirs],
        impl_cu, "-o", fatbin_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        # Filter to just errors (nvcc produces many Eigen warnings)
        errors = [l for l in result.stderr.splitlines() if "error" in l.lower()]
        error_summary = "\n".join(errors[:20]) if errors else result.stderr[:2000]
        raise RuntimeError(f"nvcc compilation failed:\n{error_summary}")

    return fatbin_path
