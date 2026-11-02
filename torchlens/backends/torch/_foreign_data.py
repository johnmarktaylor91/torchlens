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
foreign-kernel boundary (``_function_boundary``) and never reach the journal. The
export's own dispatches (torch's ``toDLPack`` strides fix) are owned by the audited
census token ``foreign_data:export`` so the completeness census counts each export once.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from ... import _state

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
    args: tuple[torch.Tensor, ...]
    resolution: str


def departures(trace: Any) -> list[Departure]:
    """Return the capture's departure journal (created on first use)."""

    journal = trace.__dict__.get(FOREIGN_DATA_DEPARTURES_ATTR)
    if not isinstance(journal, list):
        journal = []
        trace.__dict__[FOREIGN_DATA_DEPARTURES_ATTR] = journal
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

    from ._tl import get_tensor_label

    if not isinstance(tensor, torch.Tensor):
        return
    label = get_tensor_label(tensor)
    if not isinstance(label, str):
        return
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


@contextmanager
def owned_export_dispatches(trace: Any) -> Iterator[None]:
    """Own every census dispatch inside a plain-code export (``toDLPack``'s strides fix).

    Parameters
    ----------
    trace:
        Active capture trace.
    """

    from ._function_boundary import _active_witness_state
    from .escape_detection import ExpectedOriginalToken

    state = _active_witness_state(trace)
    start = len(state.events) if state is not None else 0
    try:
        yield
    finally:
        if state is not None and len(state.events) > start:
            export = torch.utils.dlpack.to_dlpack
            token = ExpectedOriginalToken(
                raw_id=id(export),
                original=export,
                wrapper_name=FOREIGN_DATA_EXPORT_WRAPPER_NAME,
                wrapper_frame_id=0,
                owner_thread_id=threading.get_ident(),
                pending=False,
                func_name=FOREIGN_DATA_FUNC_NAME,
                func_call_id=None,
                census_scope="expected_opaque",
                capture_accounted=True,
            )
            for event in state.events[start:]:
                event.owner = token


def _mark(func: Callable[..., Any], resolution: str, entry: str, exits: tuple[str, ...]) -> None:
    """Stamp a ``foreigndata`` callable with its marker and transform metadata."""

    from ._transform_wrappers import _set_transform_metadata

    setattr(func, FOREIGN_DATA_RECORDER_ATTR, True)
    setattr(func, FOREIGN_DATA_RESOLUTION_ATTR, resolution)
    _set_transform_metadata(
        func,
        transform_kind=FOREIGN_DATA_TRANSFORM_KIND,
        tags=(FOREIGN_DATA_TRANSFORM_KIND,),
        transform_config={"entry": entry, "exits": list(exits), "resolution": resolution},
        inner_fn=func,
    )


def _view_gather_index(
    departure: Departure, out: torch.Tensor
) -> tuple[str, torch.Tensor | None] | None:
    """Describe how ``out`` reads the departed tensor's memory, or ``None`` if it does not.

    Returns ``("identity", None)`` when ``out`` is the departed tensor's own geometry,
    ``("reshape", None)`` when both are contiguous with the same element count, and
    ``("gather", index)`` with one flat index per output element otherwise.
    """

    source = departure.tensor
    if out.dtype != source.dtype or out.device != source.device or out.numel() == 0:
        return None
    delta_bytes = out.data_ptr() - source.data_ptr()
    element_size = out.element_size()
    if delta_bytes % element_size:
        return None
    out_offset = source.storage_offset() + delta_bytes // element_size
    if out_offset < 0:
        return None
    if (
        out_offset == source.storage_offset()
        and out.size() == source.size()
        and out.stride() == source.stride()
    ):
        return "identity", None
    if (
        out_offset == source.storage_offset()
        and out.numel() == source.numel()
        and out.is_contiguous()
        and source.is_contiguous()
    ):
        return "reshape", None
    storage_numel = departure.storage_nbytes // element_size
    last = out_offset + sum(
        (size - 1) * stride for size, stride in zip(out.size(), out.stride(), strict=True)
    )
    if last >= storage_numel:
        return None
    positions = torch.arange(storage_numel, dtype=torch.int64)
    source_positions = positions.as_strided(
        source.size(), source.stride(), source.storage_offset()
    ).reshape(-1)
    inverse = torch.full((storage_numel,), -1, dtype=torch.int64)
    inverse[source_positions] = torch.arange(source.numel(), dtype=torch.int64)
    out_positions = positions.as_strided(out.size(), out.stride(), out_offset).reshape(-1)
    index = inverse[out_positions]
    if bool((index < 0).any()):
        return None
    return "gather", index


def _view_entry(departure: Departure, out: torch.Tensor, entry: str) -> HostEntry | None:
    """Return the zero-copy round-trip entry for ``out``, if it reads the departed memory."""

    described = _view_gather_index(departure, out)
    if described is None:
        return None
    mode, index = described
    size = tuple(out.size())

    def foreigndata(source: torch.Tensor) -> torch.Tensor:
        """Re-read the departed tensor exactly as the host constructor exposed it."""

        value = source.detach()
        if mode == "identity":
            return value
        if mode == "reshape":
            return value.reshape(size)
        assert index is not None
        return value.reshape(-1)[index.to(value.device)].reshape(size)

    _mark(foreigndata, RESOLUTION_VIEW, entry, (departure.kind,))
    return HostEntry(replay=foreigndata, args=(departure.tensor,), resolution=RESOLUTION_VIEW)


def _values_equal(out: torch.Tensor, source: torch.Tensor) -> bool:
    """Return whether ``out`` holds exactly the departed tensor's values (cast to its dtype)."""

    if out.size() != source.size():
        return False
    try:
        cast = source.detach().to(device=out.device, dtype=out.dtype)
        return bool(torch.equal(cast, out.detach()))
    except (RuntimeError, TypeError):
        return False


def _copy_entry(departure: Departure, out: torch.Tensor, entry: str) -> HostEntry:
    """Return the copy round-trip entry: the output equals the departed tensor's value."""

    dtype, device = out.dtype, out.device

    def foreigndata(source: torch.Tensor) -> torch.Tensor:
        """Re-copy the departed tensor into the dtype and device the constructor produced."""

        return source.detach().to(device=device, dtype=dtype).clone()

    _mark(foreigndata, RESOLUTION_COPY, entry, (departure.kind,))
    return HostEntry(replay=foreigndata, args=(departure.tensor,), resolution=RESOLUTION_COPY)


def _unresolved_entry(linked: list[Departure], entry: str) -> HostEntry:
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

    _mark(foreigndata, RESOLUTION_UNRESOLVED, entry, exits)
    seen: dict[str, torch.Tensor] = {}
    for departure in linked:
        seen.setdefault(departure.label, departure.tensor)
    return HostEntry(
        replay=foreigndata, args=tuple(seen.values()), resolution=RESOLUTION_UNRESOLVED
    )


def _constant_entry(out: torch.Tensor, entry: str) -> HostEntry:
    """Return the entry for a consumed-capsule re-entry in a capture with no departure."""

    value = out.detach().clone()

    def foreigndata() -> torch.Tensor:
        """Return the recorded value: with no departure in the capture it is not torch data."""

        return value.clone()

    _mark(foreigndata, RESOLUTION_CONSTANT, entry, ())
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


def resolve_host_entry(
    trace: Any, func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any], out: Any
) -> HostEntry | None:
    """Resolve a parentless host constructor call against the departure journal.

    Parameters
    ----------
    trace:
        Active capture trace.
    func_name:
        The constructor's func name (one of :data:`HOST_CONSTRUCTOR_FUNC_NAMES`).
    args, kwargs:
        The call's arguments (no captured tensor among them).
    out:
        The tensor the constructor returned.

    Returns
    -------
    HostEntry | None
        How to log the call as a ``foreigndata`` op, or ``None`` to log it unchanged.
    """

    if not isinstance(out, torch.Tensor):
        return None
    host_argument = args[0] if args else kwargs.get("data", kwargs.get("obj"))
    journal = departures(trace)
    if not journal:
        if func_name == "from_dlpack" and _is_capsule(host_argument):
            return _constant_entry(out, func_name)
        return None
    for departure in reversed(journal):
        if departure.kind in _BUFFER_DEPARTURE_KINDS:
            entry = _view_entry(departure, out, func_name)
            if entry is not None:
                return entry
    for departure in reversed(journal):
        if _values_equal(out, departure.tensor):
            return _copy_entry(departure, out, func_name)
    linked = [departure for departure in journal if _compatible(departure, host_argument)]
    if not linked:
        if func_name == "from_dlpack" and _is_capsule(host_argument):
            return _constant_entry(out, func_name)
        return None
    return _unresolved_entry(linked, func_name)


def active_trace_for_export() -> Any | None:
    """Return the live capture trace when a plain-code export should be journaled."""

    from ._function_boundary import _FRAMES

    if _FRAMES:
        return None
    active_trace, logging_enabled = _state.active_capture()
    if not logging_enabled or active_trace is None:
        return None
    return active_trace
