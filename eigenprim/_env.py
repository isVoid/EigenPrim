"""Environment detection for eigenprim."""

import os
import sys
import sysconfig


_INCLUDE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")
_FATBIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fatbin")


def find_eigen_include():
    """Auto-detect the Eigen include directory.

    Checks, in order:
    1. EIGEN_INCLUDE_DIR environment variable
    2. CONDA_PREFIX/include/eigen3
    3. sys.prefix/include/eigen3
    4. common system include paths

    Raises RuntimeError if Eigen is not found.
    """
    env = os.environ.get("EIGEN_INCLUDE_DIR")
    if env and os.path.isdir(env):
        return env

    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidate = os.path.join(prefix, "include", "eigen3")
        if os.path.isdir(candidate):
            return candidate

    candidates = [
        os.path.join(sys.prefix, "include", "eigen3"),
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    raise RuntimeError(
        "Cannot find Eigen include directory. "
        "Set EIGEN_INCLUDE_DIR or install Eigen headers via conda, pixi, "
        "or your system package manager."
    )


def include_dir():
    """Path to eigenprim's bundled include/ directory."""
    return _INCLUDE_DIR


def fatbin_dir():
    """Path to eigenprim's bundled fatbin/ directory."""
    if os.path.exists(os.path.join(_FATBIN_DIR, "matrix.fatbin")):
        return _FATBIN_DIR

    candidates = []
    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            candidates.append(os.path.join(path, "eigenprim", "fatbin"))
    candidates.extend(os.path.join(path, "eigenprim", "fatbin") for path in sys.path)

    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "matrix.fatbin")):
            return candidate

    return _FATBIN_DIR
