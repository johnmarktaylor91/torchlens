"""Foreign data round trips in plain forward code: departures, re-entries, replay.

Outside an ``autograd.Function`` a tensor can still leave torch (``.numpy()``,
``__array__``, ``Tensor.__dlpack__``, ``to_dlpack``, ``.tolist()``) and data can come
back through a host constructor (``from_numpy``, ``from_dlpack``, ``frombuffer``, or a
``tensor`` / ``as_tensor`` / ``asarray`` call with no tensor argument). Without this
module the graph has a hole where the round trip was: the constructor is a parentless
source op, the output has no input ancestor, and validation replays the constructor
from its recorded host argument, which is a silent constant for an ndarray and a dead
capsule for DLPack.

This module keeps a per-capture **departure journal** (:func:`note_plain_export`) and
resolves each parentless host constructor against it (:func:`resolve_host_entry`):

* a zero-copy round trip (the new tensor's storage lies inside a departed tensor's
  storage) or a copy round trip (the new tensor equals a departed tensor's value) is
  recorded as ONE ``foreigndata`` boundary op whose logged argument is the departed
  tensor, so the parent edge, saved args and replay all follow the generic op path
  (the shape of the torch.func and ``autogradfunction`` boundary ops);
* a re-entry that matches no departure while departures exist is recorded as an
  **unresolved** ``foreigndata`` op linked to every compatible departure; validation
  refuses to replay it with a typed failure (``foreign_data_roundtrip_unreplayable``)
  instead of blessing a constant;
* a re-entry in a capture with no departure stays the plain source op it is, except
  that a consumed DLPack capsule replays from the recorded output value.

Departures and re-entries inside a live ``Function.apply`` frame belong to the
foreign-kernel boundary (``_function_boundary``), which also routes plain-code exports
into the journal and owns the export's own dispatches (torch's ``toDLPack`` strides
fix) under the audited census token ``foreign_data:export``. This module imports no
capture machinery: the transform-metadata setter is handed in by the caller.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ... import _state
from ..._utils import _torch_compat
from ..._utils._foreign_data_replay import foreigndata
from ..._utils._tensor_label import get_tensor_label

MetadataSetter = Callable[..., None]
"""The transform-metadata setter (``_transform_wrappers._set_transform_metadata``)."""

FOREIGN_DATA_FUNC_NAME = "foreigndata"
"""Op type / func name of the round-trip boundary op."""

FOREIGN_DATA_TRANSFORM_KIND = "foreign_data"
"""``Op.transform_kind`` of the round-trip boundary op (``tl.func_transform`` selector)."""

FOREIGN_DATA_EXPORT_WRAPPER_NAME = "foreign_data:export"
"""Completeness-witness owner token of the dispatches inside a plain-code export."""

FOREIGN_DATA_RECORDER_ATTR = "__tl_foreign_data_recorder__"
"""Marker set on every ``foreigndata`` replay callable."""

FOREIGN_DATA_RESOLUTION_ATTR = "__tl_foreign_data_resolution__"
"""Resolution of the round trip on its callable: ``view``, ``copy``, ``constant``, ``unresolved``."""

FOREIGN_DATA_DEPARTURES_ATTR = "_foreign_departures"
"""Session-time Trace attribute: the departure journal, popped in postprocess step 3."""

FOREIGN_DATA_REPLAY_REASON = "foreign_data_roundtrip_unreplayable"
"""Validation reason of an unresolved round trip: the foreign step cannot be re-run."""

RESOLUTION_VIEW = "view"
RESOLUTION_COPY = "copy"
RESOLUTION_CONSTANT = "constant"
RESOLUTION_UNRESOLVED = "unresolved"

TOLIST_EXPORT_FUNC_NAME = "tolist"
"""Host export that copies values into Python containers (never zero-copy)."""

HOST_CONSTRUCTOR_FUNC_NAMES = frozenset(
    {"from_numpy", "from_dlpack", "frombuffer", "as_tensor", "asarray", "tensor"}
)
"""Constructors that can bring host data back into torch."""

_BUFFER_DEPARTURE_KINDS = frozenset({"numpy", "__array__", "__dlpack__", "to_dlpack"})
"""Zero-copy departures: the host holds the tensor's own memory."""


@dataclass(frozen=True)
class Departure:
    """One tensor that left torch in plain forward code."""

    label: str
    tensor: torch.Tensor
    """The departing tensor OBJECT (it carries the label the parent edge is read from)."""
    kind: str
    storage_ptr: int
    storage_nbytes: int


@dataclass(frozen=True)
class HostEntry:
    """How a parentless host constructor is logged: its callable and its logged args."""

    replay: Callable[..., Any]
    args: tuple[Any, ...]
    resolution: str


def departures(trace: Any) -> list[Departure]:
    """Return the capture's departure journal (created on first use)."""

    journal = trace.__dict__.get(FOREIGN_DATA_DEPARTURES_ATTR)
    if not isinstance(journal, list):
        journal = []
        trace.__dict__["_foreign_departures"] = journal  # literal: the ownership census reads it
    return journal


def is_foreign_data_recorder(func: Any) -> bool:
    """Return whether ``func`` is a ``foreigndata`` replay callable."""

    return getattr(func, FOREIGN_DATA_RECORDER_ATTR, False) is True


def foreign_data_resolution(func: Any) -> str | None:
    """Return the round-trip resolution recorded on a ``foreigndata`` callable, if any."""

    resolution = getattr(func, FOREIGN_DATA_RESOLUTION_ATTR, None)
    return resolution if isinstance(resolution, str) else None


def is_unresolved_foreign_data(func: Any) -> bool:
    """Return whether ``func`` is a round trip TorchLens cannot re-run (typed refusal)."""

    return foreign_data_resolution(func) == RESOLUTION_UNRESOLVED


def _storage_range(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Return ``(data pointer, nbytes)`` of a tensor's untyped storage, if readable."""

    try:
        storage = tensor.untyped_storage()
        return int(storage.data_ptr()), int(storage.nbytes())
    except (RuntimeError, TypeError, AttributeError, NotImplementedError):
        return None


def note_plain_export(trace: Any, kind: str, tensor: Any) -> None:
    """Record a tensor leaving torch in plain forward code.

    Parameters
    ----------
    trace:
        Active capture trace.
    kind:
        Export spelling (``numpy``, ``__array__``, ``__dlpack__``, ``to_dlpack``, ``tolist``).
    tensor:
        The departing tensor; anything without a captured label records nothing.
    """

    if not isinstance(tensor, torch.Tensor):
        return
    label = get_tensor_label(tensor)
    if not isinstance(label, str):
        return
    # Bookkeeping, not a user exposure: an unpaused ``untyped_storage().data_ptr()`` read
    # would trip the completeness witness's raw-pointer escape and poison every runnable
    # verdict of the capture (``_completeness_storage``).
    with _state.pause_logging():
        storage_range = _storage_range(tensor)
    if storage_range is None:
        return
    departures(trace).append(
        Departure(
            label=label,
            tensor=tensor,
            kind=kind,
            storage_ptr=storage_range[0],
            storage_nbytes=storage_range[1],
        )
    )


def _mark(
    func: Callable[..., Any],
    resolution: str,
    entry: str,
    exits: tuple[str, ...],
    mark: MetadataSetter,
) -> None:
    """Stamp a ``foreigndata`` callable with its marker and transform metadata."""

    setattr(func, FOREIGN_DATA_RECORDER_ATTR, True)
    setattr(func, FOREIGN_DATA_RESOLUTION_ATTR, resolution)
    mark(
        func,
        transform_kind=FOREIGN_DATA_TRANSFORM_KIND,
        tags=(FOREIGN_DATA_TRANSFORM_KIND,),
        transform_config={"entry": entry, "exits": list(exits), "resolution": resolution},
        inner_fn=func,
    )


def _flat_positions(size: tuple[int, ...], stride: tuple[int, ...], offset: int) -> np.ndarray:
    """Return each element's storage position for a (size, stride, offset) geometry, flattened.

    Host-side integer arithmetic on purpose: it runs inside the capture, where a torch
    computation would be logged and a tensor-to-scalar read counts as an escape.
    """

    positions = np.full(tuple(size), offset, dtype=np.int64)
    for axis, (extent, step) in enumerate(zip(size, stride, strict=True)):
        shape = [1] * len(size)
        shape[axis] = extent
        positions = positions + (np.arange(extent, dtype=np.int64) * step).reshape(shape)
    return positions.reshape(-1)


def _contiguous_strides(size: tuple[int, ...]) -> tuple[int, ...]:
    """Return the row-major strides of ``size`` (elements, not bytes)."""

    strides: list[int] = []
    step = 1
    for extent in reversed(size):
        strides.append(step)
        step *= max(extent, 1)
    return tuple(reversed(strides))


def _affine_fit(index: np.ndarray) -> tuple[tuple[int, ...], int] | None:
    """Return ``(stride, offset)`` when ``index`` (shaped like the output) is strided."""

    offset = int(index.reshape(-1)[0])
    stride: list[int] = []
    for axis, extent in enumerate(index.shape):
        step = 0
        if extent > 1:
            probe = [0] * index.ndim
            probe[axis] = 1
            step = int(index[tuple(probe)]) - offset
        stride.append(step)
    if any(step < 0 for step in stride):
        return None
    fitted = _flat_positions(tuple(index.shape), tuple(stride), offset)
    if not np.array_equal(fitted, index.reshape(-1)):
        return None
    return tuple(stride), offset


def _view_geometry(departure: Departure, out: torch.Tensor) -> tuple[str, tuple[Any, ...]] | None:
    """Describe how ``out`` reads the departed tensor's memory, or ``None`` if it does not.

    Returns the replay mode and literals of :func:`foreigndata` over the departed
    tensor's logical element order: ``("strided", (size, stride, offset))`` for any view
    a stride pattern describes (its own geometry, a reshape, a transpose, a slice), else
    ``("gather", (size, index))`` with one logical element index per output element.
    """

    source = departure.tensor
    if out.dtype != source.dtype or out.device != source.device or out.numel() == 0:
        return None
    size = tuple(out.size())
    delta_bytes = out.data_ptr() - source.data_ptr()
    element_size = out.element_size()
    if delta_bytes % element_size:
        return None
    source_offset = int(source.storage_offset())
    out_offset = source_offset + delta_bytes // element_size
    if out_offset < 0:
        return None
    same_start = out_offset == source_offset
    if same_start and size == tuple(source.size()) and out.stride() == source.stride():
        return "strided", (size, _contiguous_strides(size), 0)
    reshaped = out.numel() == source.numel() and out.is_contiguous() and source.is_contiguous()
    if same_start and reshaped:
        return "strided", (size, _contiguous_strides(size), 0)
    storage_numel = departure.storage_nbytes // element_size
    out_positions = _flat_positions(size, tuple(out.stride()), out_offset)
    if out_positions.size and int(out_positions.max()) >= storage_numel:
        return None
    source_positions = _flat_positions(tuple(source.size()), tuple(source.stride()), source_offset)
    inverse = np.full((storage_numel,), -1, dtype=np.int64)
    inverse[source_positions] = np.arange(source.numel(), dtype=np.int64)
    index = inverse[out_positions]
    if index.size and int(index.min()) < 0:
        return None
    fit = _affine_fit(index.reshape(size))
    if fit is not None:
        return "strided", (size, *fit)
    return "gather", (size, tuple(int(item) for item in index))


def _resolved_replay(
    resolution: str, entry: str, exits: tuple[str, ...], mark: MetadataSetter
) -> Any:
    """Return this op's ``foreigndata`` callable: a marked alias of the portable replay.

    The alias keeps :func:`foreigndata`'s module and qualname, so the op's registry key
    (and a runnable artifact's resolver) names the shared replay function, while the
    per-op marker and transform metadata live on the alias, never on the shared function.
    """

    @functools.wraps(foreigndata)
    def replay(source: torch.Tensor, mode: str, *literals: Any) -> torch.Tensor:
        """Re-run the resolved round trip (see :func:`foreigndata`)."""

        return foreigndata(source, mode, *literals)

    _mark(replay, resolution, entry, exits, mark)
    return replay


def _view_entry(
    departure: Departure, out: torch.Tensor, entry: str, mark: MetadataSetter
) -> HostEntry | None:
    """Return the zero-copy round-trip entry for ``out``, if it reads the departed memory."""

    described = _view_geometry(departure, out)
    if described is None:
        return None
    mode, literals = described
    replay = _resolved_replay(RESOLUTION_VIEW, entry, (departure.kind,), mark)
    return HostEntry(
        replay=replay, args=(departure.tensor, mode, *literals), resolution=RESOLUTION_VIEW
    )


def _values_equal(out: torch.Tensor, source: torch.Tensor) -> bool:
    """Return whether ``out`` holds exactly the departed tensor's values (cast to its dtype)."""

    if out.size() != source.size():
        return False
    try:
        with torch.no_grad():
            cast = source.to(device=out.device, dtype=out.dtype)
            return bool(torch.equal(cast, out))
    except (RuntimeError, TypeError):
        return False


def _copy_entry(
    departure: Departure, out: torch.Tensor, entry: str, mark: MetadataSetter
) -> HostEntry:
    """Return the copy round-trip entry: the output equals the departed tensor's value."""

    replay = _resolved_replay(RESOLUTION_COPY, entry, (departure.kind,), mark)
    return HostEntry(
        replay=replay,
        args=(departure.tensor, "copy", out.dtype, out.device),
        resolution=RESOLUTION_COPY,
    )


def _unresolved_entry(linked: list[Departure], entry: str, mark: MetadataSetter) -> HostEntry:
    """Return the entry for a round trip whose foreign step cannot be re-run."""

    labels = tuple(dict.fromkeys(departure.label for departure in linked))
    exits = tuple(dict.fromkeys(departure.kind for departure in linked))

    def foreigndata(*sources: torch.Tensor) -> torch.Tensor:
        """Refuse: the host computation between the departures and this re-entry is unknown."""

        raise RuntimeError(
            f"TorchLens cannot re-run the foreign data round trip recorded as {entry}: data "
            f"left torch through {', '.join(exits)} (ops {', '.join(labels)}) and came back "
            "after host computation TorchLens did not see. Wrap the foreign step in an "
            "autograd.Function (captured as one replayable boundary op) or build the "
            "constant before the forward pass."
        )

    _mark(foreigndata, RESOLUTION_UNRESOLVED, entry, exits, mark)
    seen: dict[str, torch.Tensor] = {}
    for departure in linked:
        seen.setdefault(departure.label, departure.tensor)
    return HostEntry(
        replay=foreigndata, args=tuple(seen.values()), resolution=RESOLUTION_UNRESOLVED
    )


def _constant_entry(out: torch.Tensor, entry: str, mark: MetadataSetter) -> HostEntry:
    """Return the entry for a consumed-capsule re-entry in a capture with no departure."""

    with torch.no_grad():
        value = out.clone()

    def foreigndata() -> torch.Tensor:
        """Return the recorded value: with no departure in the capture it is not torch data."""

        return value.clone()

    _mark(foreigndata, RESOLUTION_CONSTANT, entry, (), mark)
    return HostEntry(replay=foreigndata, args=(), resolution=RESOLUTION_CONSTANT)


def _is_capsule(value: Any) -> bool:
    """Return whether ``value`` is a DLPack capsule (single-use)."""

    return type(value).__name__ == "PyCapsule"


def _is_buffer_argument(value: Any) -> bool:
    """Return whether the constructor argument hands over a memory buffer (ndarray, DLPack)."""

    return (
        _is_capsule(value)
        or hasattr(value, "__array_interface__")
        or hasattr(value, "__cuda_array_interface__")
        or hasattr(value, "__dlpack__")
        or isinstance(value, (bytes, bytearray, memoryview))
    )


def _compatible(departure: Departure, host_argument: Any) -> bool:
    """Return whether a departure could have fed this host argument."""

    if _is_buffer_argument(host_argument):
        return True
    return departure.kind == TOLIST_EXPORT_FUNC_NAME


HostCall = tuple[tuple[Any, ...], dict[str, Any]]
"""A host constructor's ``(args, kwargs)``."""


def resolve_host_entry(
    trace: Any, func_name: str, call: HostCall, out: Any, mark: MetadataSetter
) -> HostEntry | None:
    """Resolve a parentless host constructor call against the departure journal.

    Parameters
    ----------
    trace:
        Active capture trace.
    func_name:
        The constructor's func name (one of :data:`HOST_CONSTRUCTOR_FUNC_NAMES`).
    call:
        The call's ``(args, kwargs)`` (no captured tensor among them).
    out:
        The tensor the constructor returned.
    mark:
        Transform-metadata setter stamped on the ``foreigndata`` callable.

    Returns
    -------
    HostEntry | None
        How to log the call as a ``foreigndata`` op, or ``None`` to log it unchanged.
    """

    if not isinstance(out, torch.Tensor):
        return None
    with _state.pause_logging():
        return _resolve_paused(trace, func_name, call, out, mark)


def _resolve_paused(
    trace: Any, func_name: str, call: HostCall, out: torch.Tensor, mark: MetadataSetter
) -> HostEntry | None:
    """Match ``out`` against the journal; the matching's own torch work is never logged."""

    args, kwargs = call
    host_argument = args[0] if args else kwargs.get("data", kwargs.get("obj"))
    journal = departures(trace)
    if not journal:
        if func_name == "from_dlpack" and _is_capsule(host_argument):
            return _constant_entry(out, func_name, mark)
        return None
    for departure in reversed(journal):
        if departure.kind in _BUFFER_DEPARTURE_KINDS:
            entry = _view_entry(departure, out, func_name, mark)
            if entry is not None:
                return entry
    for departure in reversed(journal):
        if _values_equal(out, departure.tensor):
            return _copy_entry(departure, out, func_name, mark)
    linked = [departure for departure in journal if _compatible(departure, host_argument)]
    if not linked:
        if func_name == "from_dlpack" and _is_capsule(host_argument):
            return _constant_entry(out, func_name, mark)
        return None
    return _unresolved_entry(linked, func_name, mark)


FLASH_ATTENTION_4_INTERFACE_MODULE = "flash_attn.cute.interface"
"""Module that defines the FlashAttention 4 ``autograd.Function`` entry points."""

FLASH_ATTENTION_4_EXIT = "flash_attn.cute"
"""Exit name recorded in the known foreign kernel's ``transform_config["exits"]`` and warning."""

_KNOWN_FOREIGN_KERNEL_MODULES: tuple[tuple[str, str, str], ...] = (
    (FLASH_ATTENTION_4_INTERFACE_MODULE, "HAS_FLASH_ATTENTION_4_INTERFACE", FLASH_ATTENTION_4_EXIT),
)
"""``(module, capability flag, exit name)`` per known foreign-kernel Function module.

The dynamic detector in ``_function_boundary`` collapses a Function whose forward left torch
through an export it can see (numpy, dlpack). FlashAttention 4 (``flash_attn.cute``) hands
q, k, v and its fresh ``torch.empty`` output to a CuTe kernel through TVM FFI: no export the
detector sees, no ``custom_op``, no aten op for the attention itself, so the trace used to
record the allocation in place of the attention and the completeness census reported the
hole (forward validation failed on ``bfs_completeness``). Every ``autograd.Function``
defined in ``flash_attn.cute.interface`` is therefore a known foreign kernel: its ``apply``
is recorded as one ``autogradfunction`` boundary op with the exit ``flash_attn.cute``, the
same op, census token and replay the dynamic path produces, so nothing is exempted: the
boundary op owns every dispatch inside the call and replays by calling ``apply`` again.
Gated by the capability flag, read at call time so a flipped flag is honoured.
"""


def known_foreign_kernel_exits(cls: type[Any]) -> tuple[str, ...]:
    """Return the exit names of a Function class known to leave torch below the dispatcher.

    Parameters
    ----------
    cls:
        The ``autograd.Function`` subclass being applied.

    Returns
    -------
    tuple[str, ...]
        ``("flash_attn.cute",)`` for a Function defined in the FlashAttention 4 interface
        module (or a submodule of it) while its capability flag is set; empty otherwise.
    """

    module = getattr(cls, "__module__", None)
    if not isinstance(module, str):
        return ()
    for known_module, flag, exit_name in _KNOWN_FOREIGN_KERNEL_MODULES:
        if module != known_module and not module.startswith(known_module + "."):
            continue
        if getattr(_torch_compat, flag, False):
            return (exit_name,)
    return ()
