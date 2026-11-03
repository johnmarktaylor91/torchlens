"""Torchvision custom-op wrap targets, appended lazily once torchvision is imported.

Split out of ``_constants`` (whose ``get_orig_torch_funcs`` appends these
targets); stdlib-only basis module.
"""

from __future__ import annotations

import sys

__tl_layer__ = "L0"

# Torchvision ops accessed via torch.ops — decorated only if torchvision is installed.
TORCHVISION_FUNCS = [
    ("torch.ops.torchvision.nms", "_op"),
    ("torch.ops.torchvision.deform_conv2d", "_op"),
    ("torch.ops.torchvision.ps_roi_align", "_op"),
    ("torch.ops.torchvision.ps_roi_pool", "_op"),
    ("torch.ops.torchvision.roi_align", "_op"),
    ("torch.ops.torchvision.roi_pool", "_op"),
]

_TORCHVISION_FUNCS_CACHE: list[tuple[str, str]] | None = None


def _get_torchvision_funcs() -> list[tuple[str, str]]:
    """Return torchvision torch.ops targets if torchvision is already imported.

    TorchLens never imports torchvision itself: importing it costs ~2 s and
    ~150 MB RSS, and a model cannot call a torchvision custom op unless the
    user's process has already imported torchvision (the ops only register
    with the torch dispatcher during ``import torchvision``). The
    not-yet-imported answer is deliberately uncached so a later user import
    is picked up by ``wrap_torch()`` on the next capture.

    Returns
    -------
    list[tuple[str, str]]
        Torchvision operation targets for wrapper decoration, or an empty list
        while torchvision has not (finished) being imported.
    """

    global _TORCHVISION_FUNCS_CACHE
    if _TORCHVISION_FUNCS_CACHE is not None:
        return _TORCHVISION_FUNCS_CACHE
    torchvision_module = sys.modules.get("torchvision")
    if torchvision_module is None:
        return []
    spec = getattr(torchvision_module, "__spec__", None)
    if spec is not None and getattr(spec, "_initializing", False):
        # Mid-import (circular import): the C ops may not be registered yet.
        return []
    _TORCHVISION_FUNCS_CACHE = list(TORCHVISION_FUNCS)
    return _TORCHVISION_FUNCS_CACHE
