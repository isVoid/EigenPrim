"""Environment detection for eigenprim."""

import os


_INCLUDE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")


def find_eigen_include():
    """Auto-detect the Eigen include directory.

    Checks, in order:
    1. EIGEN_INCLUDE_DIR environment variable
    2. CONDA_PREFIX/include/eigen3

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

    raise RuntimeError(
        "Cannot find Eigen include directory. "
        "Set EIGEN_INCLUDE_DIR or install eigen via conda/pixi."
    )


def include_dir():
    """Path to eigenprim's bundled include/ directory."""
    return _INCLUDE_DIR
