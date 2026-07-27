"""Load/decode/exec-path re-export of the canonical ``torch_attr`` helper (r42/r45/r47 secC_1/secD_1).

The canonical implementation now lives in :mod:`torchlens.utils._torch_symbols` -- a low-level
module importing only ``torch`` -- so low-level importers can resolve it without pulling
``_io/__init__.py`` and risking an import cycle (r47 secD_1 / secF_1). This module preserves the
established ``from torchlens._io._torch_symbols import torch_attr`` spelling for the load/decode
importers (``rehydrate`` / ``runnable_load``) and for the AST immunizer's public entry point. The
public contract is ONE safe helper, not a physical module path -- both spellings return the same
object.
"""

from __future__ import annotations

from ..utils._torch_symbols import torch_attr

__all__ = ["torch_attr"]
