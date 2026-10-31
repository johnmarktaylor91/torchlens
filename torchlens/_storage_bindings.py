"""Record and restore the storage binding of a parameter or buffer.

``Tensor.set_`` rebinds a tensor in place: ``buf.set_()`` empties it,
``buf.set_(source)`` points it at another tensor's storage (possibly growing
it), and ``buf.set_(storage, offset, size, stride)`` views a raw storage. The
tensor object stays the same, so a value restore that only copies bytes back
(``load_state_dict``, ``copy_``) either refuses the new shape or writes through
the new alias into another tensor's bytes. Every restore of model state
therefore puts a moved tensor back on its recorded storage first and restores
values second.

A binding is a detached alias of the tensor taken before the forward: it keeps
the original storage alive and remembers the offset, size and stride, and
``Tensor.is_set_to`` / ``Tensor.set_(binding)`` compare and restore all four.
"""

from __future__ import annotations

import torch
from torch import nn


def holds_rebindable_storage(tensor: torch.Tensor) -> bool:
    """Return whether ``Tensor.set_`` can put ``tensor`` back on a recorded storage.

    Parameters
    ----------
    tensor:
        A registered parameter or buffer.

    Returns
    -------
    bool
        ``True`` for plain dense tensors and parameters on a real device. Tensor
        subclasses, sparse, quantized, meta and inference tensors are left to
        their owners' own restore paths.
    """

    return (
        type(tensor) in (torch.Tensor, nn.Parameter)
        and tensor.layout == torch.strided
        and tensor.device.type != "meta"
        and not tensor.is_quantized
        and not tensor.is_inference()
    )


def record_storage_binding(tensor: torch.Tensor) -> torch.Tensor | None:
    """Return a binding for ``tensor``, or ``None`` when it cannot be rebound.

    Parameters
    ----------
    tensor:
        A registered parameter or buffer, before the forward runs.

    Returns
    -------
    torch.Tensor | None
        A detached alias holding the tensor's storage, offset, size and stride.
    """

    return tensor.detach() if holds_rebindable_storage(tensor) else None


def rebind_if_moved(tensor: torch.Tensor, binding: torch.Tensor) -> bool:
    """Put ``tensor`` back on its recorded storage if a forward moved it.

    The tensor object is kept, so references a caller holds (an optimizer's
    parameter list, a tied weight) stay valid. Values are not compared or
    written here.

    Parameters
    ----------
    tensor:
        The parameter or buffer whose binding was recorded.
    binding:
        The alias :func:`record_storage_binding` returned for it.

    Returns
    -------
    bool
        ``True`` when the tensor was rebound.
    """

    if tensor.is_set_to(binding):
        return False
    with torch.no_grad():
        tensor.set_(
            binding.untyped_storage(), binding.storage_offset(), binding.size(), binding.stride()
        )
    return True
