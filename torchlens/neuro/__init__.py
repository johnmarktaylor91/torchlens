"""Neuroscience appliance for RDM, CKA, Brain-Score, and representation helpers.
Gated by `pip install torchlens[neuro]`.
"""

import importlib

_REQUIRED_DEPS = ("rsatoolbox", "brainscore_core")
_missing_deps = []

for _dep in _REQUIRED_DEPS:
    try:
        importlib.import_module(_dep)
    except ImportError:
        _missing_deps.append(_dep)

if _missing_deps:
    _missing = ", ".join(_missing_deps)
    raise ImportError(
        "torchlens.neuro requires extra: install with "
        f"`pip install torchlens[neuro]`. Missing deps: {_missing}"
    )

__all__ = []
