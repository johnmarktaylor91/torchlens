"""Lazy alias module: ``torchlens.facets`` is :mod:`torchlens.semantic.facets`.

``facets`` is registered in ``torchlens.__init__._LAZY_ATTRS`` so the *attribute*
``torchlens.facets`` resolves on demand, but ``import torchlens.facets`` /
``importlib.import_module("torchlens.facets")`` go through the import system, which needs a
real submodule. This stub is imported only when someone explicitly imports
``torchlens.facets`` (never at ``import torchlens``), then replaces itself in ``sys.modules``
with the canonical ``torchlens.semantic.facets`` object so the two are identity-equal.
"""

import sys as _sys

from .semantic import facets as _facets

_sys.modules[__name__] = _facets
