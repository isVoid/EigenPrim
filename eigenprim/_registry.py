"""Global registry of EigenBindings for the unified links() function."""

_bindings = []


def register(binding):
    """Register an EigenBindings instance for link collection."""
    _bindings.append(binding)


def links():
    """Yield link dependencies for all imported eigenprim modules.

    Use as: @cuda.jit(link=links())
    """
    for b in _bindings:
        yield from b.links()
