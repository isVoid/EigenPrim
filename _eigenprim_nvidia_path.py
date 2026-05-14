from __future__ import annotations

import os


def install() -> None:
    try:
        from cuda.pathfinder import find_nvidia_binary_utility
    except Exception:
        return

    nvcc = find_nvidia_binary_utility("nvcc")
    if not nvcc:
        return

    bindir = os.path.dirname(nvcc)
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    if bindir not in parts:
        os.environ["PATH"] = bindir + (os.pathsep + path if path else "")

    os.environ.setdefault("CUDACXX", nvcc)
