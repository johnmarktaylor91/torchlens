"""Whether this torch's weights-only ``torch.load`` is safe on untrusted bytes.

``torch.load(..., weights_only=True)`` can be bypassed by a crafted file on torch 2.5.1
and older (CVE-2025-32434). Loaders outside the torch-private licensed packages ask
:func:`weights_only_load_is_safe` instead of reading the capability flag themselves, so
the probe stays inside the licensed ``_utils`` package.
"""

from __future__ import annotations

from . import _torch_compat


def weights_only_load_is_safe() -> bool:
    """Return whether ``torch.load(..., weights_only=True)`` resists crafted files here.

    Returns
    -------
    bool
        The ``HAS_SAFE_WEIGHTS_ONLY_LOAD`` capability flag, read at call time.
    """

    return _torch_compat.HAS_SAFE_WEIGHTS_ONLY_LOAD
