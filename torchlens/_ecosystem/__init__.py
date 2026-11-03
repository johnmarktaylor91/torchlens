"""TorchLens ecosystem runtime: provider discovery.

The ecosystem surface packages the entry-point provider machinery:
:mod:`torchlens._ecosystem.plugins` is the entry-point provider surface, with
metadata-only discovery and explicit distribution-scoped activation.

The artifact compatibility window and ``tl.migrate`` left with the 2.x saved-file
readers: torchlens 3 reads no file saved by torchlens 2.x.
"""

from __future__ import annotations

from . import plugins

__all__ = [
    "plugins",
]

__tl_layer__ = "L8"
