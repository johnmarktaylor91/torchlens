"""Dispatch-mode pauses around TorchLens's own work on strict Tensor subclasses.

Torch 2.1 to 2.5 refuse to construct, or convert out of, a strict Tensor
subclass while a dispatch mode is active (``HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE``).
TorchLens's own payload copies and value comparisons on such a tensor run the
subclass's ``__torch_function__`` return conversion, so they pause TorchLens's
own modes on those builds.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

from ._torch_compat import HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE


def _pause_dispatch_modes_for_subclass_clone(x: Any) -> AbstractContextManager[Any]:
    """Pause TorchLens dispatch modes around a strict-subclass ``clone()``, when needed.

    Torch's default ``__torch_function__`` subclass-return conversion calls
    ``ret.as_subclass(cls)`` INSIDE ``x.clone()`` to reconstruct the subclass
    type (R16-5, see ``torchlens/backends/torch/wrappers.py``). On torch 2.1
    and 2.2 that reconstruction raises "Creating a new Tensor subclass ...
    already associated to a python object" whenever a TorchLens dispatch mode
    (the completeness witness, armed by default; an intervention-ready
    capture's mode) is active -- reproduced on stock torch with a no-op mode,
    not something TorchLens's own wrapping causes. Pausing ONCE here, around
    the whole ``clone()`` call, covers that nested reconstruction in a single
    non-reentrant bracket (see ``HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE``); a
    plain ``torch.Tensor``/``Parameter`` receiver, or a modern torch build
    that tolerates the reconstruction, pays nothing.

    Parameters
    ----------
    x:
        Tensor about to be cloned with its own type preserved.

    Returns
    -------
    AbstractContextManager[Any]
        The bracket for the ``clone()`` call: a no-op ``nullcontext`` unless a
        pause is needed, so the common case enters no generator per payload copy.
    """

    if (
        HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE
        or type(x) is torch.Tensor
        or type(x) is torch.nn.Parameter
    ):
        return nullcontext()
    # Deferred: backends.torch is upward of utils in the layer map (C01 item
    # 1); this branch is the floor-only (torch 2.1/2.2) fallback, so a plain
    # `import torchlens` never pays for it.
    from ..backends.torch._modes import pause_own_dispatch_modes

    return pause_own_dispatch_modes()


def _subclass_compare_pause(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor
) -> AbstractContextManager[Any]:
    """Pause TorchLens dispatch modes around a value comparison, when needed.

    A comparison such as ``tensor_a == 0`` on a strict subclass reconstructs
    the subclass through ``as_subclass`` exactly like ``clone()`` does, so it
    takes the same bracket as :func:`_pause_dispatch_modes_for_subclass_clone`
    keyed on whichever operand is a strict subclass.

    Parameters
    ----------
    tensor_a:
        First compared tensor.
    tensor_b:
        Second compared tensor.

    Returns
    -------
    AbstractContextManager[Any]
        The bracket for the comparison: a no-op unless a pause is needed.
    """

    plain_a = type(tensor_a) is torch.Tensor or type(tensor_a) is torch.nn.Parameter
    return _pause_dispatch_modes_for_subclass_clone(tensor_b if plain_a else tensor_a)
