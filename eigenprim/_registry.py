"""Global registry of EigenBindings for the unified links() function."""

_bindings = []


def register(binding):
    """Register an EigenBindings instance for link collection."""
    _bindings.append(binding)


def links():
    """Return reusable lazy link dependencies for imported eigenprim modules.

    Use as: @cuda.jit(link=links())
    """
    return _RegisteredLinks()


class _RegisteredLinks:
    def __iter__(self):
        for b in _bindings:
            yield from b.links()
