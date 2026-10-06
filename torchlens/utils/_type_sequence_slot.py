"""CPython ``sq_item`` slot repair for ``torch.Tensor`` (a stdlib-only ctypes leaf).

Wrapping ``torch.Tensor.__getitem__`` on CPython can leave the sequence
protocol's ``sq_item`` slot populated, so scalar tensors look like sequences to
CPython C APIs after wrap/unwrap cycles (``torch.tensor([zero_dim_tensor])``
then breaks). This module holds the private-layout mirror and the slot write;
the ``HAS_TENSOR_SEQUENCE_SLOT_FIX`` flag, its probe and the capability warning
stay in ``torchlens.utils._torch_compat``, which imports this module at call
time (the sibling of ``_type_new_slot``, which patches ``tp_new``).
"""

from __future__ import annotations

import ctypes
import sys


class _PySequenceMethods(ctypes.Structure):
    """Minimal ctypes mirror of CPython's PySequenceMethods struct."""

    _fields_ = [
        ("sq_length", ctypes.c_void_p),
        ("sq_concat", ctypes.c_void_p),
        ("sq_repeat", ctypes.c_void_p),
        ("sq_item", ctypes.c_void_p),
        ("was_sq_slice", ctypes.c_void_p),
        ("sq_ass_item", ctypes.c_void_p),
        ("was_sq_ass_slice", ctypes.c_void_p),
        ("sq_contains", ctypes.c_void_p),
        ("sq_inplace_concat", ctypes.c_void_p),
        ("sq_inplace_repeat", ctypes.c_void_p),
    ]


class _PyTypeObject(ctypes.Structure):
    """Partial ctypes mirror of CPython's PyTypeObject up to tp_as_sequence."""

    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("ob_size", ctypes.c_ssize_t),
        ("tp_name", ctypes.c_char_p),
        ("tp_basicsize", ctypes.c_ssize_t),
        ("tp_itemsize", ctypes.c_ssize_t),
        ("tp_dealloc", ctypes.c_void_p),
        ("tp_vectorcall_offset", ctypes.c_ssize_t),
        ("tp_getattr", ctypes.c_void_p),
        ("tp_setattr", ctypes.c_void_p),
        ("tp_as_async", ctypes.c_void_p),
        ("tp_repr", ctypes.c_void_p),
        ("tp_as_number", ctypes.c_void_p),
        ("tp_as_sequence", ctypes.POINTER(_PySequenceMethods)),
        ("tp_as_mapping", ctypes.c_void_p),
    ]


def sequence_slot_layout_problem(cls: type, expected_name: bytes) -> str | None:
    """Return why ``cls``'s sequence slot cannot be repaired, or None when it can.

    Parameters
    ----------
    cls:
        The extension type whose ``PyTypeObject`` is read (``torch.Tensor``).
    expected_name:
        The ``tp_name`` the layout must show, guarding against a misread mirror.

    Returns
    -------
    str | None
        A short reason when the runtime is not CPython or the layout does not
        match; None when :func:`clear_sequence_item_slot` may write the slot.
    """

    if sys.implementation.name != "cpython":
        return "Tensor __getitem__ wrap/unwrap may leave scalar tensors sequence-like"
    try:
        type_obj = _PyTypeObject.from_address(id(cls))
    except Exception:
        return "Tensor sequence-slot layout could not be inspected"
    if type_obj.tp_name != expected_name or not type_obj.tp_as_sequence:
        return "Tensor sequence-slot layout did not match the expected CPython structure"
    return None


def clear_sequence_item_slot(cls: type) -> None:
    """Clear ``cls``'s ``sq_item`` slot; call only after a None layout problem.

    Parameters
    ----------
    cls:
        The extension type whose sequence slot is cleared.
    """

    _PyTypeObject.from_address(id(cls)).tp_as_sequence.contents.sq_item = None
