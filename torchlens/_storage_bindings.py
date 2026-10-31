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

STORAGE_CONTEXT_RECEIVER_FUNCS = frozenset({"as_strided_", "resize_", "resize_as_"})
"""In-place storage-geometry calls whose receiver is saved with its whole storage.

``as_strided_`` reads any bytes of the receiver's storage, and a ``resize_`` /
``resize_as_`` that grows inside a larger storage exposes the bytes past the
receiver's view, so a compact clone of the receiver cannot replay the call.
Capture saves the receiver through :func:`copy_with_storage_context` (the
save budget charges the whole storage, as it already projects) and validation
replays the call on a clone of that context.
"""


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


def views_part_of_its_storage(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` leaves bytes of its storage outside its own view.

    Parameters
    ----------
    tensor:
        A dense tensor.

    Returns
    -------
    bool
        ``True`` when the view starts past the storage's first element or ends
        before its last one, so a plain ``clone()`` would drop bytes a
        storage-geometry call (``as_strided_``, a ``resize_`` that grows inside
        the storage) can read.
    """

    if not holds_rebindable_storage(tensor) or tensor.numel() == 0:
        return False
    last = tensor.storage_offset() + sum(
        (size - 1) * stride for size, stride in zip(tensor.size(), tensor.stride())
    )
    spanned = (last + 1) * tensor.element_size()
    return tensor.storage_offset() != 0 or spanned != tensor.untyped_storage().nbytes()


def copy_with_storage_context(tensor: torch.Tensor) -> torch.Tensor:
    """Copy ``tensor`` together with the whole storage it views.

    The copy views a private clone of the full storage at the same offset,
    size and stride, so a storage-geometry call replayed on it reads the bytes
    the original call read. A tensor that spans its whole storage from offset
    zero gets a plain clone, which holds the same bytes.

    Parameters
    ----------
    tensor:
        A dense tensor on a real device.

    Returns
    -------
    torch.Tensor
        Detached copy carrying the original's ``requires_grad``.
    """

    if not views_part_of_its_storage(tensor):
        copy = tensor.detach().clone()
    else:
        copy = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        copy.set_(
            tensor.untyped_storage().clone(),
            tensor.storage_offset(),
            tensor.size(),
            tensor.stride(),
        )
    if tensor.requires_grad:
        copy.requires_grad_(True)
    return copy
