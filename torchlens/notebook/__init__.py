"""Notebook appliance for Treescope-style HTML reprs and IPython integration.
Gated by `pip install torchlens[notebook]`.
"""

import importlib

_REQUIRED_DEPS = ("IPython", "jupyter_client")
_missing_deps: list[str] = []

for _dep in _REQUIRED_DEPS:
    try:
        importlib.import_module(_dep)
    except ImportError:
        _missing_deps.append(_dep)

if _missing_deps:
    _missing = ", ".join(_missing_deps)
    raise ImportError(
        "torchlens.notebook requires extra: install with "
        f"`pip install torchlens[notebook]`. Missing deps: {_missing}"
    )

__all__: list[str] = []
