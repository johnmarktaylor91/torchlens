"""State and value types shared by completeness-witness slices."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)

from .escape_detection import (
    ExpectedOriginalToken,
)

__all__ = (
    "AuditedCompletenessBoundary",
    "_DispatchCallsite",
    "_DispatchEvent",
    "_WitnessState",
    "_StorageOriginRegistry",
    "_TensorOriginRegistry",
    "_PlainScalarEscapeState",
)


@dataclass(frozen=True)
class AuditedCompletenessBoundary:
    """One exact wrapper/operator boundary that is intentionally not captured."""

    wrapper_name: str
    operator: str | None
    reason: str


@dataclass(frozen=True)
class _DispatchCallsite:
    """Stable user-side source location captured at dispatch time."""

    file: str
    line: int
    function: str


@dataclass
class _DispatchEvent:
    """One aten dispatcher event and its exact live wrapper owner, if any.

    r35 I2: every event carries a lifecycle ``outcome`` -- ``started`` (armed),
    then ``returned_tensor`` / ``returned_host_or_none`` / ``raised`` -- so the
    runnable ledger can discharge every observed event as an accounted call, an
    exact witness, an audited opaque boundary, or an explicit incomplete fact.
    Only safe facts are recorded for a raise: operator, owner identity, and the
    exception type module+qualname -- never exception objects/messages/tracebacks.
    """

    operator: str
    owner: ExpectedOriginalToken | None
    callsite: _DispatchCallsite | None
    in_replacement_hook: bool = False
    mutates: bool = False
    state_view_accessor: bool = False
    outcome: str = "started"
    exception_type: str | None = None
    contained_view: bool = False
    """``aten.as_strided`` whose result byte span is contained in its operand's (r37)."""
    metadata_witnessed: bool = False
    """r67 C3: a host-returning metadata dispatch (``aten.is_pinned``) whose receiver was
    positively attributed and recorded by the placement metadata net -- discharged as
    witnessed (the observed-value ledger / input fact owns it); an UNATTRIBUTED receiver
    keeps the incomplete fact (fail closed)."""
    output_refs: tuple[weakref.ref[torch.Tensor], ...] = ()
    """Weak references to the tensors this dispatch returned, recorded only when the owner's
    boundary credit is output-scoped (module-forward tokens). Weak, so the census never
    extends an intermediate's lifetime; the token holds its boundary tensors strongly, so a
    dead reference can never name one of them."""


@dataclass
class _WitnessState:
    """Per-forward dispatch census state."""

    trace: Any
    owner_thread_id: int
    guard_pass_index: int
    events: list[_DispatchEvent] = field(default_factory=list)
    callback_ns: int = 0
    census: bool = True
    record_escapes: bool = False
    ledger: bool = False
    record_aten: bool = False
    plane_p: bool = False
    """Merge-ranks C2 plane-P: record EVERY dispatched op (any namespace,
    paused windows included) for armed distributed captures. Feeds the
    session-time ``trace._distributed_plane_p`` journal the capture-fidelity
    census criteria 2-4 consume; never installed for unarmed captures."""
    plane_p_events: list[tuple[str, int | None, bool, bool, bool]] = field(default_factory=list)
    """Plane-P records: ``(qualified_op, owner_func_call_id, paused,
    discharged_inside_boundary, has_module_context)``."""
    aten_events: Any = None
    capture_phase: str = "forward"
    backward_epoch_index: int | None = None
    aten_storage_alias_groups: dict[tuple[str, int], int] = field(default_factory=dict)
    # r43: armed for the entire forward window (SAME lifetime as
    # ``_observe_invisible_host_escapes``), cleared in its ``finally``. The
    # non-owner captured-tensor belt gates on THIS flag, never the racy global
    # ``_state._logging_enabled`` (which the owner flips under ``pause_logging``),
    # closing the hon2_3 pause-race coin-flip.
    belt_armed: bool = False
    # (source_tensor, version_at_escape, byte_snapshot) for each mutable zero-copy alias
    # (``numpy`` / ``__array__``) handed to the host this forward. Checked for host write-back
    # at forward end. Strong refs keep the aliased storage alive until the comparison.
    writeback_watch: list[tuple[torch.Tensor, int | None, torch.Tensor]] = field(
        default_factory=list
    )
    # r67 C3: capture-scoped storage-origin map -- storage handle object ->
    # ("input", site) | ("state", frozenset[str]) | ("other", None). Populated by the
    # pre-index (known state/input storages) and by every bridge return; consulted by the
    # storage accessor wrappers so a handle acquired ANYWHERE this forward attributes its
    # actual reads. Uses weak keys when torch retains safe untyped-storage wrappers and
    # identity-keyed strong entries on older torch whose ephemeral storage weakrefs can dangle.
    # ``None`` until the wrappers arm.
    storage_origins: _StorageOriginRegistry | None = None
    # r67 C3: lazy ptr -> full state-name-set index (params + buffers, alias groups merged)
    # backing the origin resolver's pointer fallback. Built once per forward on first use.
    storage_state_ptr_names: dict[int, frozenset[str]] | None = None


class _StorageOriginRegistry:
    """Identity registry for capture-scoped storage-handle origins.

    Parameters
    ----------
    weak_keys:
        Use weak storage keys when the runtime retains a safe untyped-storage wrapper.
        Otherwise retain handles strongly for this forward so neither dangling weakrefs
        nor object-id reuse can corrupt attribution.
    """

    def __init__(self, *, weak_keys: bool) -> None:
        """Initialize an empty storage-origin registry.

        Parameters
        ----------
        weak_keys:
            Whether storage handles are safe to hold through weak references.
        """

        self._weak: weakref.WeakKeyDictionary[Any, tuple[str, Any]] | None = (
            weakref.WeakKeyDictionary() if weak_keys else None
        )
        self._strong: dict[int, tuple[Any, tuple[str, Any]]] = {}

    def get(self, handle: Any) -> tuple[str, Any] | None:
        """Return the origin registered for ``handle`` by object identity.

        Parameters
        ----------
        handle:
            Typed or untyped torch storage handle.

        Returns
        -------
        tuple[str, Any] | None
            Registered origin, or ``None`` when the handle is unknown.
        """

        if self._weak is not None:
            return self._weak.get(handle)
        entry = self._strong.get(id(handle))
        if entry is None or entry[0] is not handle:
            return None
        return entry[1]

    def register(self, handle: Any, origin: tuple[str, Any]) -> None:
        """Register ``handle`` once without weakening identity guarantees.

        Parameters
        ----------
        handle:
            Typed or untyped torch storage handle.
        origin:
            Storage origin classification for the active capture.
        """

        if self._weak is not None:
            if self._weak.get(handle) is None:
                self._weak[handle] = origin
            return
        self._strong.setdefault(id(handle), (handle, origin))


class _TensorOriginRegistry:
    """Identity-keyed weak map: live tensor object -> propagated origin set.

    ``WeakKeyDictionary`` is unusable for tensors (its ref-equality path invokes the
    tensor's elementwise ``__eq__``), so entries key on ``id(tensor)`` with a weakref
    finalizer removing the entry when the tensor dies, and a liveness identity check
    guarding against id reuse.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        """Initialize the weak identity map backing store."""
        self._entries: dict[int, tuple[Any, frozenset[str], frozenset[str]]] = {}

    def get(self, tensor: torch.Tensor) -> tuple[frozenset[str], frozenset[str]] | None:
        """Return the live alias metadata for ``tensor`` when still registered.

        Parameters
        ----------
        tensor:
            Tensor whose registration should be resolved.

        Returns
        -------
        tuple[frozenset[str], frozenset[str]] | None
            Registered display and leaf address sets, or ``None`` when absent or stale.
        """
        entry = self._entries.get(id(tensor))
        if entry is None:
            return None
        ref, display, leaf = entry
        return (display, leaf) if ref() is tensor else None

    def set(self, tensor: torch.Tensor, display: frozenset[str], leaf: frozenset[str]) -> None:
        """Register alias metadata for ``tensor`` with weak cleanup.

        Parameters
        ----------
        tensor:
            Tensor to register.
        display:
            Display-address set for ``tensor``.
        leaf:
            Leaf-address set for ``tensor``.
        """
        key = id(tensor)
        entries = self._entries

        def _cleanup(dead_ref: Any, key: int = key) -> None:
            """Delete the dead entry when the weakref target is reclaimed.

            Parameters
            ----------
            dead_ref:
                Weak reference whose referent just died.
            key:
                Identity-map key to clear when it still points at ``dead_ref``.
            """
            entry = entries.get(key)
            if entry is not None and entry[0] is dead_ref:
                del entries[key]

        try:
            ref = weakref.ref(tensor, _cleanup)
        except TypeError:
            return  # non-weakref-able exotic subclass: stays unregistered (-> unknown)
        entries[key] = (ref, display, leaf)


@dataclass
class _PlainScalarEscapeState:
    """Aggregate tensor-to-Python scalar escapes for one plain capture."""

    trace: Any
    owner_thread_id: int
    count: int = 0
    first_file: str | None = None
    first_line: int | None = None
