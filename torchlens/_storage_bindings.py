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

VIEW_CONTEXT_RECEIVER_FUNCS = frozenset(
    {"_view_func", "_view_func_unsafe", "_rev_view_func_unsafe"}
)
"""View-function methods whose receiver is saved with its view context.

Torch reads only the receiver's view chain, never its values, and returns ``None``
for a receiver that is not a view (``view_func_impl`` and ``rev_view_func_impl`` in
``torch/csrc/autograd/python_variable.cpp``), so a compact clone cannot replay the
call. Capture saves the receiver rebuilt on a private copy of its base
(:func:`copy_with_view_context`) and validation rebuilds it the same way before
the call.
"""


STORAGE_CONTEXT_NEW_VIEW_FUNCS = frozenset({"_rev_view_func_unsafe"})
"""View-function methods whose new view (positional argument 1) is saved with its whole storage.

``_rev_view_func_unsafe`` maps its new view back to the base geometry; a reverse
``select`` or ``slice`` is an ``as_strided`` over the new view's storage, so it reads
bytes outside the new view's own elements and a compact clone cannot replay it.
"""


def storage_context_arg_index(func_name: str | None) -> int | None:
    """Return the positional argument of ``func_name`` saved with its whole storage, if any.

    Parameters
    ----------
    func_name:
        Name of the captured function.

    Returns
    -------
    int | None
        ``0`` for a :data:`STORAGE_CONTEXT_RECEIVER_FUNCS` receiver, ``1`` for a
        :data:`STORAGE_CONTEXT_NEW_VIEW_FUNCS` new view, else ``None``.
    """

    if func_name in STORAGE_CONTEXT_RECEIVER_FUNCS:
        return 0
    if func_name in STORAGE_CONTEXT_NEW_VIEW_FUNCS:
        return 1
    return None


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
        (size - 1) * stride for size, stride in zip(tensor.size(), tensor.stride(), strict=True)
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


def has_view_geometry_of(rebuilt: torch.Tensor, recorded: torch.Tensor) -> bool:
    """Return whether ``rebuilt`` is a view with ``recorded``'s view geometry.

    Parameters
    ----------
    rebuilt:
        A receiver rebuilt from a saved view context.
    recorded:
        The receiver it stands for.

    Returns
    -------
    bool
        ``True`` when both are views with the same shape, strides and storage offset.
    """

    return (
        rebuilt._is_view()
        and recorded._is_view()
        and rebuilt.size() == recorded.size()
        and rebuilt.stride() == recorded.stride()
        and rebuilt.storage_offset() == recorded.storage_offset()
    )


def copy_with_view_context(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a view together with a private copy of its base, keeping the view relation.

    The base is copied with its whole storage (:func:`copy_with_storage_context`)
    and the receiver's own view function replays its view chain onto that copy
    under original-view tracking, so the copy carries the same view chain for a
    later ``_view_func`` or ``_rev_view_func_unsafe`` call. A receiver without a
    view chain (a ``None`` from torch) or one whose chain does not rebuild its
    geometry gets a compact detached clone, which replay refuses.

    Parameters
    ----------
    tensor:
        The receiver of a :data:`VIEW_CONTEXT_RECEIVER_FUNCS` call.

    Returns
    -------
    torch.Tensor
        The rebuilt view, or a compact detached clone as described above.
    """

    if tensor._is_view() and tensor._base is not None:
        base_copy = copy_with_storage_context(tensor._base.detach())
        with torch.autograd._force_original_view_tracking(True):
            rebuilt = tensor._view_func_unsafe(base_copy)
        if isinstance(rebuilt, torch.Tensor) and has_view_geometry_of(rebuilt, tensor):
            return rebuilt
    return tensor.detach().clone()
