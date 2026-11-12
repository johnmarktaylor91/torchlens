"""TorchLens private metadata namespace helpers."""

from __future__ import annotations

import itertools
import weakref
from collections.abc import Iterable
from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.weak import WeakIdKeyDictionary

from .. import _state
from ._internal_read import IdKeyRegistry, internal_scalar_read
from ._torch_compat import get_tensorbase_class

__all__ = [
    "TorchLensMeta",
    "TensorMeta",
    "ParamMeta",
    "ModuleMeta",
    "DecorationTag",
    "TorchLensTLCollisionError",
    "get",
    "is_tracked",
    "clear_meta",
    "get_tensor_meta",
    "set_tensor_label",
    "get_tensor_label",
    "mark_tensor_data_alias",
    "is_tensor_data_alias",
    "raw_tensor_label",
    "get_live_tensor_label",
    "get_live_label_list",
    "begin_label_session",
    "end_label_session",
    "active_label_session_token",
    "session_meta_is_anchored",
    "session_label_storage_intact",
    "session_labeled_tensors",
    "session_storage_alias_candidates",
    "sweep_retired_label_stamps",
    "clear_tensor_label",
    "promote_label_to_buffer_source_and_clear_label",
    "set_buffer_address",
    "get_buffer_address",
    "get_label_list",
    "set_param_meta",
    "get_param_meta",
    "increment_param_call_index",
    "restore_param_requires_grad",
    "set_module_meta",
    "get_module_meta",
    "mark_decorated_function",
    "is_decorated_function",
    "mark_forward_call_decorated",
    "is_forward_call_decorated",
    "mark_tensor_replacement_wrapped",
    "is_tensor_replacement_wrapped",
    "copy_replacement_meta",
    "DescriptorCompatProperty",
]


class TorchLensMeta:
    """Branded base for TorchLens-owned ``._tl`` metadata."""


class DescriptorCompatProperty(property):
    """A ``property`` that can carry ``__objclass__``/``__name__`` like the C
    descriptor it replaces.

    A plain ``property`` has no ``__dict__`` and refuses an ``__objclass__``
    or ``__name__`` assignment (it is a slots-only builtin type), unlike the
    ``getset_descriptor`` / autograd-property it replaces on ``torch.Tensor``
    at several sites (wrappers.py's ``Tensor.real``/``imag`` rewrap, the
    completeness-witness ``requires_grad``/``grad_fn``/``is_leaf`` recording
    properties, the invisible-escape and structure-only-belt escalated
    properties). Third-party introspection over ``torch.Tensor``'s own
    attributes may assume every property-shaped member is a genuine
    descriptor with ``__objclass__``/``__name__`` and access them
    unconditionally:

    * torch 2.7.1's dynamo import-time ``populate_builtin_to_tensor_fn_map`` /
      ``is_tensor_base_attr_getter`` raises ``AttributeError: 'property'
      object has no attribute '__objclass__'`` the first time it runs after
      ANY of these replacements is installed.
    * torch 2.13+'s dynamo import-time ``variables/torch_function.py``
      (``banned_attrs`` list comprehension) walks every overridable
      function's bound ``__get__``, and for one whose ``__self__.__objclass__
      is torch._C.TensorBase`` (true once ``__objclass__`` is set above)
      unconditionally reads ``fn.__self__.__name__``, raising
      ``AttributeError: '...' object has no attribute '__name__'``.

    Every site that replaces a Tensor-level descriptor with a ``property``
    should use this subclass and set BOTH ``__objclass__`` (normally
    ``torch.Tensor`` or ``torch._C.TensorBase``) and ``__name__`` (the
    attribute name being replaced) so the replacement stays a faithful
    stand-in on every torch version.

    CONSTRUCTOR LANDMINE: always pass an explicit ``doc=`` keyword. CPython's
    ``property.__init__`` only stores an implicit ``fget.__doc__`` directly on
    the C struct for the EXACT ``property`` type; for any subclass it instead
    does ``self.__doc__ = fget.__doc__`` through the normal attribute-set
    protocol (even when ``fget.__doc__`` is ``None``), which raises
    ``AttributeError: '...' object attribute '__doc__' is read-only`` against
    this slots-only subclass's missing ``__dict__``. A caller that omits
    ``doc=`` gets that AttributeError on every construction -- indistinguishable
    from (and commonly swallowed by) the same ``except (TypeError,
    AttributeError)`` guards these replacements install under, silently
    degrading the capture to observer-install-failed instead of installing the
    replacement at all.
    """

    __slots__ = ("__objclass__", "__name__")
    # Declared for the type checker: mypy does not derive attribute types from
    # ``__slots__``, and every replacement site assigns both.
    __objclass__: type
    __name__: str


@dataclass
class TensorMeta(TorchLensMeta):
    """Metadata attached to non-Parameter tensors during a capture session."""

    label_raw: str | None = None
    address: str | None = None
    buffer_source: str | None = None
    # monotonic token of the capture session that issued ``label_raw``
    # (and, after promotion, ``buffer_source``). The per-object anchor that
    # makes label provenance current-session; see the label-session block below.
    label_session: int | None = None
    # STRONG reference to the ``UntypedStorage`` the object held when its
    # label was last stamped by ``set_tensor_label`` -- the label rung's
    # storage-integrity pin, the activation/label twin of the buffer-address
    # keeper (``_SessionBufferStamp.storage``). The IDENTITY anchor
    # (``label_session``) proves this is the current-session labeled object; this
    # pin proves the object's live storage was NOT ``.data=``/``set_``-rebound to
    # foreign/input-derived storage AFTER it was labeled. Stored on the object's
    # OWN metadata, so for an HONEST activation it is the SAME storage the tensor
    # already holds (zero net retention) and it is released the instant the
    # tensor dies -- it never pins the activation tensor alive, so sparse
    # ``save=`` memory is untouched (unlike the weak ``_LabelSession.stamped``
    # inventory, whose weakness is required for exactly that reason). Only a
    # genuinely rebound (adversarial) object leaves its superseded storage pinned
    # for the object's lifetime, which is what makes the ``data_ptr`` comparison a
    # true identity proof rather than a recyclable-pointer heuristic.
    label_storage: Any | None = None
    # Capture-local provenance for tensors returned by ``Tensor.data`` and
    # storage-sharing aliases/views derived from them. The getter is represented
    # as a canonical detach op, but a later write through this alias family must
    # still ceiling runnable faithfulness.
    data_alias: bool = False
    # Transient handoffs for a same-object return, stamped on the fresh copy it is
    # logged against and popped by activation logging (``_same_object_handoff``):
    # the call's mutation verdict, and for a non-mutating return the live object
    # the caller receives, which a by-reference save keeps instead of the copy.
    same_object_mutation: bool | None = None
    same_object_live_value: Any | None = None

    def __deepcopy__(self, memo: dict[int, Any]) -> TensorMeta:
        """Copy this metadata for a deep-copied tensor, minus the source's identity.

        ``Tensor.__deepcopy__`` deep-copies ``__dict__`` through the storage's
        memo, so a copy made in a captured forward would inherit the source's
        session anchor and a ``label_storage`` pin onto its own storage, read as
        the source, log no op and wire every consumer to the source node. The
        anchor, the pin and the alias/handoff flags describe one object, so the
        copy starts without them; the inert raw history (``label_raw``,
        ``address``, ``buffer_source``) is kept. Pickling strips the same fields.

        Parameters
        ----------
        memo:
            ``copy.deepcopy`` memo table.

        Returns
        -------
        TensorMeta
            Metadata for the copy.
        """

        copied = dataclass_replace(
            self,
            label_session=None,
            label_storage=None,
            data_alias=False,
            same_object_mutation=None,
            same_object_live_value=None,
        )
        memo[id(self)] = copied
        return copied

    def __getstate__(self) -> dict[str, Any]:
        """Pickle as ``__deepcopy__`` copies: a loaded object is never the labeled source."""
        return dict(vars(self.__deepcopy__({})))


@dataclass
class ParamMeta(TorchLensMeta):
    """Metadata attached to parameters during a capture session."""

    param_barcode: str | None = None
    param_address: str | None = None
    call_index: int = 0
    requires_grad_before_capture: bool | None = None


@dataclass
class ModuleMeta(TorchLensMeta):
    """Permanent metadata attached to modules after model preparation."""

    address: str | None = None
    module_type: str | None = None
    # Set (and kept) once a capture enters the module by a direct
    # ``module.forward(x)`` call, which no torch hook observes; the fast live
    # run routes exactly these modules around its hooks (``_fast_run``).
    direct_forward_called: bool = False


@dataclass
class DecorationTag(TorchLensMeta):
    """Sentinel metadata attached to decorated callables."""

    is_decorated_function: bool = False
    forward_call_is_decorated: bool = False
    tensor_replacement_wrapped: bool = False


class TorchLensTLCollisionError(AttributeError):
    """Raised when an existing ``._tl`` is foreign or the wrong metadata kind."""


# --------------------------------------------------------------------------- #
# Identity-keyed registries for USER-OWNED objects (modules + parameters).
#
# Modules and parameters are the user's objects; tagging them with a ``._tl``
# attribute pollutes them (state_dict / serialization / introspection leakage)
# and, for parameters, writes an attribute onto an ``nn.Parameter`` tensor. We
# instead store their TorchLens metadata in process-wide identity-keyed weak
# registries, leaving the user objects untouched. TorchLens-OWNED objects
# (plain activation tensors -> ``TensorMeta``; decorated callables ->
# ``DecorationTag``) keep using the ``._tl`` attribute -- no pollution concern,
# and that path is hot.
#
# ``WeakIdKeyDictionary`` (not stdlib ``WeakKeyDictionary``) is REQUIRED for the
# parameter registry: ``nn.Parameter`` is a tensor whose ``__eq__`` is
# elementwise, which breaks ``WeakKeyDictionary``'s internal ref equality. The
# id-keyed variant keys by object identity, is tensor-safe, and auto-removes
# dead entries via weakrefs (no manual finalizers, no id-reuse hazard). Entry
# lifetime == key-object lifetime, identical to the former attribute. Modules'
# metadata is permanent/cross-session; a process-wide registry preserves that.
# --------------------------------------------------------------------------- #
# Values: ModuleMeta / ParamMeta (documented, not annotated); reads cached by identity.
_MODULE_REGISTRY: WeakIdKeyDictionary = IdKeyRegistry()
_PARAM_REGISTRY: WeakIdKeyDictionary = IdKeyRegistry()


# --------------------------------------------------------------------------- #
# CURRENT-SESSION OBJECT ANCHORING FOR RAW LABELS.
#
# ``TensorMeta.label_raw`` (and its promoted ``buffer_source`` sibling) used to
# be validated by pure TEXT membership in the active capture's live event index.
# Label text is deterministic per op-kind + ordinal, so an ordinary op in a
# LATER, unrelated capture regenerates the same string: a tensor still carrying
# a label from an EARLIER capture was therefore accepted as current-session
# provenance and spliced into the new DAG as the same-named node (broken
# independently by three lanes -- a stock ``register_forward_hook`` activation
# collector sufficed, producing a SAME-INPUT wrong parent bind reported as
# ``VERIFIED``). The param rung has an object-identity belt and the
# buffer ``address`` rung has one; this is the label rung's.
#
# THE ANCHOR is ``TensorMeta.label_session``: the monotonic token of the capture
# that issued this object's label, written onto the object's OWN metadata by
# ``set_tensor_label`` -- the single choke point through which every label stamp
# in the torch backend flows (verified: no other site assigns ``label_raw``). A
# stamp therefore cannot come into existence without its anchor, and a future
# stamp site cannot silently escape the belt (the root-A failure mode).
# Tokens are monotonic and never reused, so a label issued by an earlier capture
# can never match the active one however the object re-enters, and whether or
# not cleanup managed to reach it. Being a field on metadata the caller already
# holds, the check is one integer compare -- no lookup and no allocation on the
# per-op hot path.
#
# WHERE THE GATE SITS: in ``get_tensor_label`` / ``get_label_list``, the two
# accessors every label consumer in the torch backend reads through, rather than
# at each consumer. That is what makes it exhaustive -- the graph-parent binder,
# the layout ancestry rooting rung, the dispatch-origin ladder, the host-escape
# attribution ladder, the buffer-write producer and the replay-template
# ``ParentRef`` builder are all closed at once. Gating the two provenance rungs
# in ``ops._tensor_has_known_provenance`` alone was empirically NOT sufficient:
# the free-lane launder rode the layout/origin rungs instead.
#
# ``_LabelSession.stamped`` is a WEAK inventory of the objects stamped this
# session, used only by the inventory-driven cleanup. It is weak because labels
# are stamped on every activation and strong refs would pin the whole activation
# graph, defeating sparse ``save=``; anything still alive at cleanup time is
# alive because something outside TorchLens holds it -- exactly the leak vehicles
# cleanup needs to reach. Weakness is safe here in a way it is NOT for the
# buffer stamp keeper: entries are only ever compared by dereferenced object
# identity, and ``WeakIdRef.__eq__`` returns False whenever either side is dead,
# so a recycled ``id()`` can never produce a false match.
# --------------------------------------------------------------------------- #
_LABEL_SESSION_COUNTER = itertools.count(1)


class _LabelSession:
    """One capture session's identity for issued raw labels."""

    __slots__ = ("token", "stamped", "by_storage_ptr")

    def __init__(self, token: int) -> None:
        """Initialize an empty anchor registry for one capture session.

        Parameters
        ----------
        token : int
            Monotonic session token, unique for the process lifetime.
        """
        self.token = token
        # Weak inventory of every object stamped this session, for the
        # inventory-driven cleanup. One entry per OBJECT, not
        # per stamp: relabeling an in-place receiver does not re-register.
        self.stamped: WeakIdKeyDictionary = WeakIdKeyDictionary()
        # Weak per-storage-base-address index of the stamped inventory, so an
        # in-place op can resolve OTHER live labeled tensors sharing its
        # target's storage (view-mediated mutation provenance) in
        # O(aliases) instead of scanning every stamped object. Keyed by the
        # stamp-time ``UntypedStorage.data_ptr()``; consumers re-validate the
        # LIVE storage before acting, so a stale (rebound/dead) entry is inert.
        # almost every storage has exactly ONE labeled tensor, and a
        # full ``WeakIdKeyDictionary`` per bucket cost ~6 marginal objects per
        # op. The single-alias case stores one bare ``weakref.ref``; a second
        # live distinct alias upgrades the bucket to a WeakIdKeyDictionary.
        self.by_storage_ptr: dict[int, weakref.ref | WeakIdKeyDictionary] = {}


_ACTIVE_LABEL_SESSION: _LabelSession | None = None
_RETIRED_LABEL_SESSION: _LabelSession | None = None

# True-original storage accessors, captured at import before any wrapper or per-forward
# escape patch exists (the completeness witness snapshots the same ones). Calling them runs no
# TorchLens wrapper, toggle or observer, so with the escape detector off and no fast collector
# a plain tensor's label read and stamp use them without ``pause_logging``.
_TENSORBASE = get_tensorbase_class()
_ORIG_UNTYPED_STORAGE: Any = None if _TENSORBASE is None else _TENSORBASE.untyped_storage
_ORIG_STORAGE_DATA_PTR: Any = torch.UntypedStorage.data_ptr
_ORIG_STORAGE_NBYTES: Any = torch.UntypedStorage.nbytes


def begin_label_session() -> int:
    """Install a fresh label-anchoring session and return its token.

    Called once per capture from per-session model preparation. Sweeping the
    RETIRED session's stamps happens here rather than at capture cleanup: the
    stamps are still needed after cleanup runs, because ``_cleanup_model_session``
    precedes ``_postprocess`` and the output-attribution fallback in
    ``postprocess.graph_traversal`` reads output-tensor labels. Sweeping at the
    next session's start is equally complete for the leak class -- a stale stamp
    can only ever matter to a SUBSEQUENT capture, and it is cleared before that
    capture stamps or reads anything.

    Returns
    -------
    int
        Monotonic token identifying the newly active session.
    """

    global _ACTIVE_LABEL_SESSION
    sweep_retired_label_stamps()
    token = next(_LABEL_SESSION_COUNTER)
    _ACTIVE_LABEL_SESSION = _LabelSession(token)
    return token


def end_label_session() -> None:
    """Retire the active label-anchoring session.

    The retired session's weak inventory is kept (not dropped) so the next
    capture can sweep the stamps that outlived it -- see
    :func:`sweep_retired_label_stamps`.

    Returns
    -------
    None
        Mutates module-level session state only.
    """

    global _ACTIVE_LABEL_SESSION, _RETIRED_LABEL_SESSION
    _RETIRED_LABEL_SESSION = _ACTIVE_LABEL_SESSION
    _ACTIVE_LABEL_SESSION = None


def sweep_retired_label_stamps() -> int:
    """Clear every still-live label stamp issued by the retired session (root A).

    The AUTHORITATIVE, inventory-driven counterpart to the reachability walk in
    ``model_prep._clear_session_tensor_metadata``, which has structural blind
    spots: it returns immediately for any ``nn.Module`` value outside the traced
    tree (a helper module or ``nn.Sequential`` used as an activation cache) and
    for any object with no ``__dict__`` (``__slots__``), and it reaches globals
    only through ``forward.__code__.co_names``, so a module-global appended to
    from a HOOK or a helper function, or a class attribute, is in none of its
    sets. Enumerating by REGISTRATION instead reaches all of them -- and the
    container-nested ``types.ModuleType`` stash the shallow buffer sweep could not.

    Defence-in-depth, NOT the correctness argument: the belt rejects an
    unanchored stamp whether or not this sweep ever reached the object.

    Returns
    -------
    int
        Number of still-live stamped objects cleared.
    """

    global _RETIRED_LABEL_SESSION
    retired = _RETIRED_LABEL_SESSION
    _RETIRED_LABEL_SESSION = None
    if retired is None:
        return 0
    cleared = 0
    for stamped_tensor in list(retired.stamped.keys()):
        clear_meta(stamped_tensor)
        cleared += 1
    return cleared


def active_label_session_token() -> int | None:
    """Return the active label session token, if a capture is in progress.

    Returns
    -------
    Optional[int]
        Session token, or ``None`` when no capture session is installed.
    """

    session = _ACTIVE_LABEL_SESSION
    return None if session is None else session.token


def session_labeled_tensors() -> list[Any]:
    """Return every still-live tensor stamped with a label this session.

    The registration-driven inventory that :func:`sweep_retired_label_stamps`
    consumes, and the observable form of root A. Dead entries have already been
    dropped by the weak registry, so this names exactly the stamped objects
    something is still holding.

    Returns
    -------
    List[Any]
        Live tensors that received at least one label this session.
    """

    session = _ACTIVE_LABEL_SESSION
    if session is None:
        return []
    return list(session.stamped.keys())


def storage_alias_index_key(storage: Any) -> int | None:
    """Return the alias-index key for one ``UntypedStorage`` (meta-safe).

    Every meta storage reports ``data_ptr() == 0``, so the pointer key would
    alias-collapse ALL meta tensors into one bucket — measured to relabel an
    unrelated op output with an in-place op's label on admitted weights-free
    captures (the weightsfree memo's latent ``data_ptr`` hazard class). On the
    meta substrate the key is the storage's ``_cdata`` object identity
    (distinguishes sibling allocations, shared by views); everywhere else the
    historical ``data_ptr()``. ``None`` = unreadable, callers fail closed.
    """

    try:
        ptr = int(storage.data_ptr())
        if (
            ptr == 0
            and getattr(storage, "device", None) is not None
            and storage.device.type == "meta"
        ):
            return int(storage._cdata)
        return ptr
    except Exception:  # noqa: BLE001 — feature detection over a private primitive; None fails closed
        return None


def _register_storage_alias(session: _LabelSession, storage_ptr: int, t: Any) -> None:
    """Register one stamped tensor in the per-storage alias index.

    Parameters
    ----------
    session : _LabelSession
        Active label session owning the index.
    storage_ptr : int
        Stamp-time ``UntypedStorage.data_ptr()``.
    t : Any
        Tensor being stamped.

    Notes
    -----
    Layout: the overwhelmingly common single-alias bucket is one bare
    ``weakref.ref`` (a full per-bucket ``WeakIdKeyDictionary`` cost ~6
    marginal objects per op); a second live distinct alias upgrades the
    bucket in place. Registration stays idempotent per object, and a
    non-weak-referenceable tensor is skipped exactly as the historical
    ``bucket[t] = True`` ``TypeError`` arm skipped it.
    """

    bucket = session.by_storage_ptr.get(storage_ptr)
    if bucket is None:
        try:
            session.by_storage_ptr[storage_ptr] = weakref.ref(t)
        except TypeError:
            pass
        return
    if isinstance(bucket, weakref.ref):
        existing = bucket()
        if existing is t:
            return
        if existing is None:
            try:
                session.by_storage_ptr[storage_ptr] = weakref.ref(t)
            except TypeError:
                del session.by_storage_ptr[storage_ptr]
            return
        upgraded: WeakIdKeyDictionary = WeakIdKeyDictionary()
        upgraded[existing] = True
        try:
            upgraded[t] = True
        except TypeError:
            pass
        session.by_storage_ptr[storage_ptr] = upgraded
        return
    try:
        bucket[t] = True
    except TypeError:
        pass


def session_storage_alias_candidates(storage_ptr: int) -> list[Any]:
    """Return live tensors this session stamped whose stamp-time storage base matches.

    Parameters
    ----------
    storage_ptr : int
        ``UntypedStorage.data_ptr()`` of the storage of interest.

    Returns
    -------
    List[Any]
        Still-live labeled tensors indexed under that base address. Callers
        must re-validate the LIVE storage (and label, via the gated
        :func:`get_tensor_label`) before treating an entry as a current alias:
        the index is keyed at stamp time, so a rebound or superseded entry is
        possible and must stay inert.
    """

    session = _ACTIVE_LABEL_SESSION
    if session is None:
        return []
    bucket = session.by_storage_ptr.get(storage_ptr)
    if bucket is None:
        return []
    if isinstance(bucket, weakref.ref):
        entry = bucket()
        return [] if entry is None else [entry]
    return list(bucket.keys())


def _session_gate_blocks(meta: TensorMeta) -> bool:
    """Return whether a capture is active that did NOT issue this meta's labels.

    The read gate shared by :func:`get_tensor_label` and :func:`get_label_list`.
    Outside a capture (postprocess and every read after ``end_label_session``)
    nothing is gated, so post-capture behaviour is exactly as before.

    Parameters
    ----------
    meta : TensorMeta
        Tensor metadata being read.

    Returns
    -------
    bool
        True when the label belongs to some OTHER session than the active one.
    """

    session = _ACTIVE_LABEL_SESSION
    return session is not None and meta.label_session != session.token


def session_meta_is_anchored(meta: TensorMeta | None) -> bool:
    """Return whether a tensor's label metadata was issued by the ACTIVE session.

    The belt, in its cheapest form: the anchor is an integer stamped onto the
    object's OWN metadata by :func:`set_tensor_label`, so the check is a field
    compare on metadata the caller already holds -- no lookup, no allocation on
    the per-op hot path. A tensor labeled by an earlier capture carries that
    capture's token and can never match the active one (tokens are monotonic
    and never reused). With no session installed nothing is anchored.

    Parameters
    ----------
    meta : Optional[TensorMeta]
        Tensor metadata whose label components are being validated.

    Returns
    -------
    bool
        True only when this object's labels were issued during the currently
        active capture session.
    """

    session = _ACTIVE_LABEL_SESSION
    if session is None or meta is None:
        return False
    return meta.label_session == session.token


def _fast_collector_live() -> bool:
    """Return whether a fast-run collector is live (it keeps the wrapped storage read)."""
    return _state._active_fast_run_collector is not None


def _plain_reads_allowed() -> bool:
    """Return whether the plain-tensor path may call the original storage accessors bare.

    True with no live fast collector and no armed wrapper-edge observer (so the escape
    detector is ``"off"``): the originals then reach no wrapper, toggle or patch, and the
    ``pause_logging`` the general path adds guards no observer.
    """
    return not _fast_collector_live() and not _state.diagnostic_observer_armed()


def _pinned_storage(t: Any) -> Any | None:
    """Return ``t``'s ``UntypedStorage`` for pinning/validation, else ``None``.

    Calls the ORIGINAL accessor (unwrap ledger), two frames less than the wrapped read;
    a subclass override or a live fast collector keeps the wrapped read. Both stay under
    ``pause_logging``: a raw ``untyped_storage`` call while logging is live is a torch
    callable outside its wrapper edge, which the shadow escape detector reports.
    """
    try:
        with _state.pause_logging():
            fast = _fast_collector_live()
            ledger = {} if fast else _state.wrap_epoch_ledgers()[0]
            original = ledger.get(id(getattr(type(t), "untyped_storage", None)))
            return t.untyped_storage() if original is None else original(t)
    except Exception:
        return None


def session_label_storage_intact(meta: TensorMeta | None, tensor: Any) -> bool:
    """Return whether a labeled object's LIVE storage is still its stamp-time storage.

    The STORAGE-INTEGRITY axis of label/activation provenance, the twin of
    The :func:`session_validated_buffer_address` storage check for the buffer
    ``address`` rung. A label anchor (:func:`session_meta_is_anchored`) proves an
    object is the current-session labeled producer; this proves that object's
    storage was not ``.data=``/``set_``-rebound to foreign or input-derived
    storage between when it was labeled and when it is consumed as an op argument.

    Keying on the storage OBJECT (``data_ptr`` + ``nbytes`` + device) against the
    STRONG keeper pinned at :func:`set_tensor_label` time is what draws the sharp
    zero-collateral line: an IN-PLACE write into the object's OWN storage
    (``copy_``, EMA ``mul_().add_()``, ``buf[:] = ...``) keeps the pointer and
    PASSES -- it is honest, tracked/journaled state mutation -- while a ``.data=``
    / ``set_`` rebind swaps the pointer to another storage and FAILS.

    Fail-closed, mirroring the buffer rung: a keeper or live storage inaccessible on exactly
    ONE side is NOT intact (``False``); only the symmetric-inaccessible case
    (both unreadable) falls back to pure object/anchor identity (``True``).

    Parameters
    ----------
    meta : Optional[TensorMeta]
        Tensor metadata carrying the ``label_storage`` keeper.
    tensor : Any
        The live tensor whose storage is being validated.

    Returns
    -------
    bool
        True only when the live storage still matches the stamp-time keeper (or
        both are symmetrically inaccessible).
    """

    if meta is None:
        return False
    keeper = meta.label_storage
    live = _pinned_storage(tensor)
    if keeper is None or live is None:
        return keeper is None and live is None
    try:
        with _state.pause_logging():
            return bool(
                live.data_ptr() == keeper.data_ptr()
                and live.nbytes() == keeper.nbytes()
                and str(live.device) == str(keeper.device)
            )
    except Exception:
        return False


def _session_storage_gate_blocks(meta: TensorMeta, t: Any) -> bool:
    """Return whether an active capture must reject this label for a STORAGE rebind.

    The STORAGE twin of:func:`_session_gate_blocks`, sharing the same two
    accessor choke points so the belt stays exhaustive: even a CURRENT-session
    anchored label is not trusted -- so its object never binds as a graph parent
    or roots a layout/origin chain -- when the object's LIVE storage was
    ``.data=``/``set_``-rebound away from the storage it held when labeled. A
    rebound state-derived activation that kept its label would otherwise be
    spliced back in as the same-named parent, replaying its pre-rebind value as a
    false VERIFIED (SOL-1); orphaning it here surfaces the existing break marker.

    Only applies during an active capture: outside one (postprocess, and every
    read after ``end_label_session``) nothing is gated, so post-capture behaviour
    is byte-identical to before. A foreign-session label is already rejected by
    :func:`_session_gate_blocks`, so only the current-session anchored label
    reaches the storage validation here. Fail-closed: an anchored label whose
    storage integrity cannot be proven (asymmetric-inaccessible) is rejected.

    Parameters
    ----------
    meta : TensorMeta
        Tensor metadata carrying the label anchor and storage pin.
    t : Any
        The live tensor whose storage is being validated.

    Returns
    -------
    bool
        True when the active capture must not trust this object's label.
    """

    session = _ACTIVE_LABEL_SESSION
    if session is None or meta.label_session != session.token:
        return False
    return not session_label_storage_intact(meta, t)


def get(obj: Any) -> TorchLensMeta | None:
    """Return TorchLens metadata attached to an object.

    Parameters
    ----------
    obj : Any
        Object that may carry a ``._tl`` namespace.

    Returns
    -------
    Optional[TorchLensMeta]
        TorchLens metadata if present, otherwise ``None``.

    Raises
    ------
    TorchLensTLCollisionError
        If ``obj._tl`` exists but is not TorchLens-owned metadata.
    """
    meta = getattr(obj, "_tl", None)
    if meta is None:
        return None
    if not isinstance(meta, TorchLensMeta):
        raise TorchLensTLCollisionError(
            f"Foreign _tl attribute on {type(obj).__name__}: {type(meta).__name__}"
        )
    return meta


def is_tracked(obj: Any) -> bool:
    """Return whether an object has TorchLens-owned ``._tl`` metadata.

    Parameters
    ----------
    obj : Any
        Object to inspect.

    Returns
    -------
    bool
        True when TorchLens metadata is present.
    """
    if isinstance(obj, nn.Module):
        return obj in _MODULE_REGISTRY
    if isinstance(obj, nn.Parameter):
        return obj in _PARAM_REGISTRY
    return get(obj) is not None


def clear_meta(obj: Any) -> None:
    """Remove TorchLens-owned ``._tl`` metadata from an object.

    Parameters
    ----------
    obj : Any
        Object whose TorchLens metadata should be cleared.

    Notes
    -----
    Foreign ``._tl`` values are preserved. Module/parameter metadata lives in the
    identity-keyed registries and is removed there.
    """
    existing = getattr(obj, "_tl", None)
    if isinstance(existing, TorchLensMeta):
        # TorchLens-owned tensor/callable: metadata is on the attribute.
        try:
            delattr(obj, "_tl")
        except AttributeError:
            pass
        return
    # No TorchLens attribute (any foreign ``_tl`` is left untouched). Registry-stored
    # module/param metadata, if any, is removed here. ``isinstance(_, nn.Parameter)``
    # is the expensive check, so it is reached only for attribute-less non-module
    # objects -- never the hot plain-tensor cleanup path.
    if isinstance(obj, nn.Module):
        _MODULE_REGISTRY.pop(obj, None)
    elif isinstance(obj, nn.Parameter):
        _PARAM_REGISTRY.pop(obj, None)


def clear_param_meta(param: Any) -> None:
    """Remove both TorchLens namespaces a prepared Parameter can carry.

    A Parameter's session metadata lives in the parameter registry, but a
    Parameter mutated in place during a capture also carries a tensor label on
    its ``._tl`` attribute (``wrappers._label_mutated_prepared_parameter``).
    :func:`clear_meta` stops at the attribute, so this clears both.

    Parameters
    ----------
    param : Any
        Parameter whose TorchLens metadata should be cleared.
    """
    clear_meta(param)
    _PARAM_REGISTRY.pop(param, None)


def get_tensor_meta(t: Any) -> TensorMeta | None:
    """Return tensor metadata, raising on foreign or wrong-kind metadata.

    Parameters
    ----------
    t : Any
        Tensor-like object to inspect.

    Returns
    -------
    Optional[TensorMeta]
        Tensor metadata if present.
    """
    meta = get(t)
    if meta is None:
        return None
    if not isinstance(meta, TensorMeta):
        raise TorchLensTLCollisionError(
            f"Expected TensorMeta on {type(t).__name__}, found {type(meta).__name__}"
        )
    return meta


def _ensure_tensor_meta(t: Any) -> TensorMeta:
    """Return existing tensor metadata or attach a new tensor namespace.

    Parameters
    ----------
    t : Any
        Tensor-like object to mutate.

    Returns
    -------
    TensorMeta
        Tensor metadata namespace.
    """
    meta = get_tensor_meta(t)
    if meta is None:
        meta = TensorMeta()
        t._tl = meta
    return meta


def set_tensor_label(t: Any, label: str) -> None:
    """Set the raw capture label on a tensor.

    A plain tensor (exact ``torch.Tensor`` carrying no metadata or TorchLens
    ``TensorMeta``) stamped while the escape detector is off and no fast collector
    runs takes :func:`_stamp_plain_tensor`, which writes the same fields through
    the true-original storage accessors. Everything else takes the general stamp
    below, unchanged.

    Parameters
    ----------
    t : Any
        Tensor-like object to tag.
    label : str
        Raw TorchLens label.

    Notes
    -----
    the stamp and its current-session anchor are written together
    here, the single choke point every torch-backend label stamp flows
    through (verified: no other site assigns ``TensorMeta.label_raw``), so a
    label can never come into existence without its anchor and no future
    stamp site can silently escape the belt.

    the storage-integrity pin (``label_storage``) is re-established here on
    EVERY label stamp -- the same choke point -- so a tracked op that (re)labels
    the object also re-affirms the storage it legitimately produced. Because
    ``_unattributed_tensor_arg_positions`` reads an op's INPUT provenance BEFORE
    the output relabel, an untraced ``.data=`` rebind between two tracked ops is
    still detected on the consuming op (against the PRIOR stamp) before this call
    re-pins to the post-op storage. A tracked in-place op keeps the pointer, so
    re-pinning to the same storage is a no-op for honest mutation.
    """
    meta = getattr(t, "_tl", None)
    # The plain-tensor conditions: an exact tensor with no foreign metadata, and storage
    # reads that ``_pinned_storage`` would take through the original accessor anyway (no
    # armed escape detector with its typed accounting, no fast collector keeping the wrapped
    # read), where the pause it adds guards no observer because the originals reach no wrapper.
    if (
        t.__class__ is Tensor
        and (meta is None or meta.__class__ is TensorMeta)
        and _ORIG_UNTYPED_STORAGE is not None
        and _plain_reads_allowed()
    ):
        _stamp_plain_tensor(t, label, meta)
        return
    _set_tensor_label_general(t, label)


def _stamp_plain_tensor(t: Any, label: str, meta: TensorMeta | None) -> None:
    """Stamp a plain tensor: :func:`_set_tensor_label_general`'s writes, fewer frames.

    Parameters
    ----------
    t : Any
        Exact ``torch.Tensor`` being stamped.
    label : str
        Raw TorchLens label.
    meta : Optional[TensorMeta]
        Its current metadata, already known to be TorchLens-owned or absent.
    """

    if meta is None:
        meta = TensorMeta()
        t._tl = meta
    meta.label_raw = label
    try:
        storage = _ORIG_UNTYPED_STORAGE(t)
    except Exception:  # noqa: BLE001 -- as _pinned_storage: an unreadable storage pins None
        storage = None
    meta.label_storage = storage
    session = _ACTIVE_LABEL_SESSION
    if session is None:
        meta.label_session = None
        return
    if meta.label_session != session.token:
        meta.label_session = session.token
        try:
            session.stamped[t] = True
        except TypeError:
            pass
    if storage is None:
        return
    try:
        storage_ptr: int | None = int(_ORIG_STORAGE_DATA_PTR(storage))
    except Exception:  # noqa: BLE001 -- as storage_alias_index_key: unreadable registers nothing
        storage_ptr = None
    if storage_ptr == 0:
        # Meta and empty storages: the general key (``_cdata`` on the meta substrate).
        storage_ptr = _general_alias_key(storage)
    if storage_ptr is None:
        return
    index = session.by_storage_ptr
    if storage_ptr not in index:
        # ``_register_storage_alias``'s empty-bucket arm, inline.
        try:
            index[storage_ptr] = weakref.ref(t)
        except TypeError:
            pass
        return
    _register_storage_alias(session, storage_ptr, t)


def _general_alias_key(storage: Any) -> int | None:
    """Return :func:`storage_alias_index_key` read under the general stamp's guards."""

    try:
        with _state.pause_logging(), internal_scalar_read():
            return storage_alias_index_key(storage)
    except Exception:  # noqa: BLE001 -- as the general stamp: unreadable registers nothing
        return None


def _set_tensor_label_general(t: Any, label: str) -> None:
    """Stamp any tensor-like object: the anchor, storage pin and alias index together.

    Parameters
    ----------
    t : Any
        Tensor-like object to tag.
    label : str
        Raw TorchLens label.
    """
    meta = _ensure_tensor_meta(t)
    meta.label_raw = label
    meta.label_storage = _pinned_storage(t)
    session = _ACTIVE_LABEL_SESSION
    if session is None:
        # Stamped outside any capture (only reachable from test/tooling code):
        # deliberately left unanchored, so it is never mistaken for provenance.
        meta.label_session = None
        return
    if meta.label_session != session.token:
        meta.label_session = session.token
        try:
            session.stamped[t] = True
        except TypeError:
            # Not weak-referenceable: the anchor still holds; only the
            # inventory-driven cleanup skips it.
            pass
    # Storage-alias index: registered on EVERY stamp (a relabel keeps
    # the same storage, so re-insertion is idempotent) so in-place ops can
    # resolve overlapping live labeled aliases of their target. The
    # ``data_ptr()`` read is TorchLens bookkeeping on an already-pinned
    # storage handle: it MUST run under ``internal_scalar_read`` plus
    # ``pause_logging``, or the runnable host-escape patches record it as a
    # user raw-pointer escape and fail-close EVERY runnable capture to
    # UNVERIFIABLE (``_HOST_ESCAPE_RAW_POINTER``).
    if meta.label_storage is not None:
        try:
            with _state.pause_logging(), internal_scalar_read():
                storage_ptr = storage_alias_index_key(meta.label_storage)
        except Exception:
            storage_ptr = None
        if storage_ptr is not None:
            _register_storage_alias(session, storage_ptr, t)


def get_tensor_label(t: Any) -> str | None:
    """Return a tensor's raw capture label.

    Parameters
    ----------
    t : Any
        Tensor-like object to inspect.

    Returns
    -------
    Optional[str]
        Raw label if present AND issued by the active capture session.

    Notes
    -----
    this is the single gate through which every label consumer in the
    torch backend reads provenance -- the graph-parent binder, the layout
    ancestry rooting rung, the dispatch-origin ladder, the host-escape
    attribution ladder, the buffer-write producer, and the replay-template
    ``ParentRef`` builder. Gating HERE rather than at each of them is what
    makes the belt exhaustive: a label issued by an EARLIER capture is
    invisible to all of them at once, so a foreign tensor can never be
    accepted as current-session state however it re-enters. With no session
    installed (postprocess, and any read after ``end_label_session``) the raw
    label is returned unchanged, so post-capture behaviour is untouched.

    the same choke point also rejects a CURRENT-session anchored label
    whose object was ``.data=``/``set_``-rebound to foreign/input-derived
    storage after labeling (:func:`_session_storage_gate_blocks`), so the
    rebound activation never re-binds as its same-named graph parent and the
    break marker on the consuming op stands.

    An exact ``torch.Tensor`` skips the AsyncCollectiveTensor delegation (a strict
    subclass) and, when its live storage is the one pinned at stamp time (same
    ``_cdata``), the pointer, size and device comparison.
    """
    plain_meta = getattr(t, "_tl", None)
    if t.__class__ is Tensor and (plain_meta is None or plain_meta.__class__ is TensorMeta):
        # Hot path (CC above 10 on purpose): one frame per plain-tensor read.
        if plain_meta is None or plain_meta.label_raw is None:
            return None
        session = _ACTIVE_LABEL_SESSION
        if session is None:
            return plain_meta.label_raw
        if plain_meta.label_session != session.token:
            return None
        keeper = plain_meta.label_storage
        if keeper is not None and _plain_reads_allowed():  # the stamp's read conditions
            try:  # a missing original raises too
                same_storage = _ORIG_UNTYPED_STORAGE(t)._cdata == keeper._cdata
                # One storage object equals itself; the two reads keep the general path's raises.
                same_storage = same_storage and _ORIG_STORAGE_DATA_PTR(keeper) >= 0
                same_storage = same_storage and _ORIG_STORAGE_NBYTES(keeper) >= 0
            except Exception:  # noqa: BLE001 -- unreadable: the general comparison decides
                same_storage = False
            if same_storage:
                return plain_meta.label_raw
        return plain_meta.label_raw if session_label_storage_intact(plain_meta, t) else None
    return _general_tensor_label(t)


def _general_tensor_label(t: Any) -> str | None:
    """Return :func:`get_tensor_label` for any object other than a plain tensor."""
    meta = get_tensor_meta(t)
    if meta is None or meta.label_raw is None:
        # AsyncCollectiveTensor is a transparent async view of its inner ``.elem``
        # (merge-ranks recording): the funcol boundary labels the inner tensor
        # through the ordinary relabel dance, while user code holds the ACT wrapper.
        # Delegating the read here -- the ONE label chokepoint -- lets every consumer
        # parent on the boundary without stamping the wrapper (whose storage pin would
        # not validate). The ``.elem`` read is a wrapper attribute access and never
        # triggers the ACT's wait.
        from ._tl_collective import _async_collective_elem

        inner = _async_collective_elem(t)
        if inner is not None:
            return get_tensor_label(inner)
        return None
    if _session_gate_blocks(meta) or _session_storage_gate_blocks(meta, t):
        return None
    return meta.label_raw


def mark_tensor_data_alias(t: Any) -> None:
    """Mark a tensor as originating from the unsafe ``Tensor.data`` surface.

    Parameters
    ----------
    t : Any
        Tensor-like object returned by ``Tensor.data`` or a storage-sharing
        alias/view derived from it.
    """

    _ensure_tensor_meta(t).data_alias = True


def is_tensor_data_alias(t: Any) -> bool:
    """Return whether a tensor is a current-capture ``Tensor.data`` alias.

    Parameters
    ----------
    t : Any
        Tensor-like object to inspect.

    Returns
    -------
    bool
        ``True`` only when the object carries data-alias provenance and a
        current-session capture label.
    """

    meta = get_tensor_meta(t)
    return bool(meta is not None and meta.data_alias and get_tensor_label(t) is not None)


def raw_tensor_label(t: Any) -> str | None:
    """Return a tensor's raw capture label WITHOUT the session-anchor gate.

    For the few callers that must observe a stamp irrespective of which
    session issued it -- notably cleanup, which clears foreign stamps, and
    diagnostics. Never use this to decide provenance.

    Parameters
    ----------
    t : Any
        Tensor-like object to inspect.

    Returns
    -------
    Optional[str]
        Raw label if present, from any session.
    """
    meta = get_tensor_meta(t)
    return None if meta is None else meta.label_raw


def get_live_tensor_label(t: Any, live_labels: Iterable[str]) -> str | None:
    """Return a tensor label only when it belongs to the active trace.

    Parameters
    ----------
    t : Any
        Tensor-like object to inspect.
    live_labels : Iterable[str]
        Raw labels present in the active trace live index.

    Returns
    -------
    Optional[str]
        Raw label if it resolves in the active trace, otherwise ``None``.

    Notes
    -----
    resolution requires BOTH that the label text is live in this
    capture AND that this object's stamp was issued by this session (the
    ``get_tensor_label`` gate). Text alone let a tensor carrying a colliding
    label from an earlier capture become a graph PARENT of the same-named
    live event -- a same-input wrong value bind reported as ``VERIFIED``.
    A rejected stamp is cleared off the object either way, so a foreign
    stamp does not linger once the capture has seen it.
    """

    label = get_tensor_label(t)
    if label is not None and label in live_labels:
        return label
    if raw_tensor_label(t) is not None:
        clear_tensor_label(t)
    return None


def get_live_label_list(tensor_list: Iterable[Any], live_labels: Iterable[str]) -> list[str]:
    """Return active-trace labels for tensors, clearing stale labels.

    Parameters
    ----------
    tensor_list : Iterable[Any]
        Tensor-like objects to inspect.
    live_labels : Iterable[str]
        Raw labels present in the active trace live index.

    Returns
    -------
    List[str]
        Labels that resolve in the active trace live index.
    """

    labels: list[str] = []
    for tensor in tensor_list:
        label = get_live_tensor_label(tensor, live_labels)
        if label is not None:
            labels.append(label)
    return labels


def mutated_parameter_label(param: Any) -> str | None:
    """Return the current-session mutation label a Parameter carries, if any.

    Only ``wrappers._label_mutated_prepared_parameter`` labels a Parameter: after
    an in-place op mutated a prepared Parameter, its reads bind to that op. The
    ``._tl`` precheck keeps the common unlabeled Parameter off the label gate.

    Parameters
    ----------
    param : Any
        Parameter to inspect.

    Returns
    -------
    str | None
        The live mutation-op label, or ``None`` for an unmutated Parameter.
    """
    if getattr(param, "_tl", None) is None:
        return None
    return get_tensor_label(param)


def promote_mutated_parameters(
    tensors: list[Any], params: list[Any]
) -> tuple[list[Any], list[Any]]:
    """Move Parameters that carry a current-session label into the tensor list.

    A prepared Parameter is labeled only after an in-place op mutated it during
    the active capture (``wrappers._label_mutated_prepared_parameter``). From then
    on a read of it consumes that op's output, so it binds as a graph parent
    instead of a parameter edge. Unlabeled Parameters, and Parameters whose label
    belongs to an earlier capture session, stay parameter edges.

    Parameters
    ----------
    tensors : list[Any]
        Non-Parameter tensors extracted from an op's arguments.
    params : list[Any]
        Parameters extracted from the same arguments.

    Returns
    -------
    tuple[list[Any], list[Any]]
        ``(tensors, params)`` with mutated Parameters moved to ``tensors``; the
        inputs are returned unchanged when no Parameter is labeled.
    """

    if not params or all(getattr(param, "_tl", None) is None for param in params):
        return tensors, params
    promoted = [param for param in params if mutated_parameter_label(param) is not None]
    if not promoted:
        return tensors, params
    promoted_ids = {id(param) for param in promoted}
    kept = [param for param in params if id(param) not in promoted_ids]
    return [*tensors, *promoted], kept


def clear_tensor_label(t: Any) -> None:
    """Clear only the raw capture label on a tensor.

    Parameters
    ----------
    t : Any
        Tensor-like object to update.
    """
    meta = get_tensor_meta(t)
    if meta is not None:
        meta.label_raw = None
        meta.data_alias = False


def promote_label_to_buffer_source_and_clear_label(t: Any) -> None:
    """Move a tensor label into ``buffer_source`` and clear the raw label.

    Parameters
    ----------
    t : Any
        Tensor-like object to update.
    """
    meta = get_tensor_meta(t)
    if meta is not None and meta.label_raw is not None:
        meta.buffer_source = meta.label_raw
        meta.label_raw = None
        meta.data_alias = False


def set_buffer_address(t: Any, address: str) -> None:
    """Set a tensor's buffer address.

    Parameters
    ----------
    t : Any
        Tensor-like object to tag.
    address : str
        Dotted buffer address.
    """
    _ensure_tensor_meta(t).address = address


def get_buffer_address(t: Any) -> str | None:
    """Return a tensor's buffer address.

    Parameters
    ----------
    t : Any
        Tensor-like object to inspect.

    Returns
    -------
    Optional[str]
        Dotted buffer address if present.
    """
    meta = get_tensor_meta(t)
    return None if meta is None else meta.address


def get_label_list(tensors: Iterable[Any]) -> list[str]:
    """Return sparse raw labels from a tensor iterable.

    Parameters
    ----------
    tensors : Iterable[Any]
        Tensor-like objects to scan.

    Returns
    -------
    List[str]
        Labels for tensors with ``TensorMeta.label_raw`` set.

    Raises
    ------
    TorchLensTLCollisionError
        If a tensor has a foreign non-TorchLens ``._tl`` value.

    Notes
    -----
    this is the fastlog/predicate recorder's graph-parent binder, so it
    carries the same current-session anchor gate as :func:`get_tensor_label` --
    a foreign tensor holding a colliding label from an earlier capture must not
    become a parent on this path either.
    """
    out: list[str] = []
    for t in tensors:
        meta = getattr(t, "_tl", None)
        if meta is None:
            continue
        if not isinstance(meta, TorchLensMeta):
            raise TorchLensTLCollisionError(f"Foreign _tl on tensor: {type(meta).__name__}")
        if (
            isinstance(meta, TensorMeta)
            and meta.label_raw is not None
            and not _session_gate_blocks(meta)
            and not _session_storage_gate_blocks(meta, t)
        ):
            out.append(meta.label_raw)
    return out


def get_param_meta(p: Any) -> ParamMeta | None:
    """Return parameter metadata, raising on foreign or wrong-kind metadata.

    Parameters
    ----------
    p : Any
        Parameter-like object to inspect.

    Returns
    -------
    Optional[ParamMeta]
        Parameter metadata if present.
    """
    return _PARAM_REGISTRY.get(p)


def _ensure_param_meta(p: Any) -> ParamMeta:
    """Return existing parameter metadata or attach a new parameter namespace.

    Parameters
    ----------
    p : Any
        Parameter-like object to mutate.

    Returns
    -------
    ParamMeta
        Parameter metadata namespace.
    """
    meta = _PARAM_REGISTRY.get(p)
    if meta is None:
        meta = ParamMeta()
        _PARAM_REGISTRY[p] = meta
    return meta


def set_param_meta(p: Any, *, barcode: str, address: str, requires_grad_before: bool) -> None:
    """Set all session metadata on a parameter.

    Parameters
    ----------
    p : Any
        Parameter-like object to tag.
    barcode : str
        Parameter-sharing barcode.
    address : str
        Dotted parameter address.
    requires_grad_before : bool
        ``requires_grad`` value before TorchLens changed it.
    """
    meta = _ensure_param_meta(p)
    meta.param_barcode = barcode
    meta.param_address = address
    meta.call_index = 0
    meta.requires_grad_before_capture = requires_grad_before


def increment_param_call_index(p: Any) -> int:
    """Increment and return a parameter's call index.

    Parameters
    ----------
    p : Any
        Parameter-like object to update.

    Returns
    -------
    int
        New call index.
    """
    meta = _ensure_param_meta(p)
    meta.call_index += 1
    return meta.call_index


def restore_param_requires_grad(p: Any) -> None:
    """Restore a parameter's pre-capture ``requires_grad`` flag (leaf parameters only).

    Parameters
    ----------
    p : Any
        Parameter-like object to restore.
    """
    meta = get_param_meta(p)
    if meta is None or meta.requires_grad_before_capture is None:
        return
    # A frozen Parameter mutated in place with a grad-requiring operand is a
    # non-leaf after the pass, exactly as eager leaves it; torch refuses to set
    # the flag of a non-leaf, and its flag is already the eager one.
    if getattr(p, "is_leaf", True):
        p.requires_grad = meta.requires_grad_before_capture


def get_module_meta(m: Any) -> ModuleMeta | None:
    """Return module metadata, raising on foreign or wrong-kind metadata.

    Parameters
    ----------
    m : Any
        Module-like object to inspect.

    Returns
    -------
    Optional[ModuleMeta]
        Module metadata if present.
    """
    return _MODULE_REGISTRY.get(m)


def _ensure_module_meta(m: Any) -> ModuleMeta:
    """Return existing module metadata or attach a new module namespace.

    Parameters
    ----------
    m : Any
        Module-like object to mutate.

    Returns
    -------
    ModuleMeta
        Module metadata namespace.
    """
    meta = _MODULE_REGISTRY.get(m)
    if meta is None:
        meta = ModuleMeta()
        _MODULE_REGISTRY[m] = meta
    return meta


def set_module_meta(m: Any, *, address: str, module_type: str) -> None:
    """Set permanent module metadata.

    Parameters
    ----------
    m : Any
        Module-like object to tag.
    address : str
        Dotted module address.
    module_type : str
        Module class name.
    """
    meta = _ensure_module_meta(m)
    meta.address = address
    meta.module_type = module_type


def _ensure_decoration_tag(fn: Any) -> DecorationTag:
    """Return existing callable metadata or attach a new decoration namespace.

    Parameters
    ----------
    fn : Any
        Callable-like object to tag.

    Returns
    -------
    DecorationTag
        Decoration metadata namespace.
    """
    meta = get(fn)
    if meta is None:
        meta = DecorationTag()
        fn._tl = meta
        return meta
    if not isinstance(meta, DecorationTag):
        raise TorchLensTLCollisionError(
            f"Expected DecorationTag on {type(fn).__name__}, found {type(meta).__name__}"
        )
    return meta


def _get_decoration_tag(fn: Any) -> DecorationTag | None:
    """Return callable decoration metadata if present.

    Parameters
    ----------
    fn : Any
        Callable-like object to inspect.

    Returns
    -------
    Optional[DecorationTag]
        Decoration metadata if present.
    """
    meta = get(fn)
    if meta is None:
        return None
    if not isinstance(meta, DecorationTag):
        raise TorchLensTLCollisionError(
            f"Expected DecorationTag on {type(fn).__name__}, found {type(meta).__name__}"
        )
    return meta


def mark_decorated_function(fn: Any) -> None:
    """Mark a wrapped torch function as decorated.

    Parameters
    ----------
    fn : Any
        Callable-like object to tag.
    """
    _ensure_decoration_tag(fn).is_decorated_function = True


def is_decorated_function(fn: Any) -> bool:
    """Return whether a callable is a decorated torch function.

    Parameters
    ----------
    fn : Any
        Callable-like object to inspect.

    Returns
    -------
    bool
        True when marked as a decorated torch function.
    """
    meta = _get_decoration_tag(fn)
    return False if meta is None else meta.is_decorated_function


def mark_forward_call_decorated(fwd: Any) -> None:
    """Mark a module ``forward`` replacement as decorated.

    Parameters
    ----------
    fwd : Any
        Callable-like object to tag.
    """
    _ensure_decoration_tag(fwd).forward_call_is_decorated = True


def is_forward_call_decorated(fwd: Any) -> bool:
    """Return whether a module ``forward`` replacement is decorated.

    Parameters
    ----------
    fwd : Any
        Callable-like object to inspect.

    Returns
    -------
    bool
        True when the callable is a decorated forward replacement.
    """
    meta = _get_decoration_tag(fwd)
    return False if meta is None else meta.forward_call_is_decorated


def mark_tensor_replacement_wrapped(hook: Any) -> None:
    """Mark an intervention hook as wrapped for tensor replacement.

    Parameters
    ----------
    hook : Any
        Callable-like object to tag.
    """
    _ensure_decoration_tag(hook).tensor_replacement_wrapped = True


def is_tensor_replacement_wrapped(hook: Any) -> bool:
    """Return whether an intervention hook is wrapped for tensor replacement.

    Parameters
    ----------
    hook : Any
        Callable-like object to inspect.

    Returns
    -------
    bool
        True when the hook has the tensor replacement wrapper sentinel.
    """
    meta = _get_decoration_tag(hook)
    return False if meta is None else meta.tensor_replacement_wrapped


def copy_replacement_meta(src: Any, dst: Any) -> None:
    """Copy TorchLens metadata from one replacement tensor to another.

    Parameters
    ----------
    src : Any
        Source object whose metadata should be copied.
    dst : Any
        Destination object that should receive a shallow dataclass copy.

    Notes
    -----
    the copy carries ``label_session`` with the rest of the dataclass,
    so an intervention replacement inherits the source's anchor exactly -- a
    live source's replacement keeps provenance, and a STALE source's label
    stays stale rather than laundering into a fresh object. The replacement is
    joined to the session inventory only when it is genuinely current-session,
    so cleanup can reach it.

    the STORAGE pin (``label_storage``) is re-established to ``dst``'s OWN
    storage rather than inheriting the source's keeper -- ``dst`` is a distinct
    current-session object taking over the label, so its integrity pin must
    reference the storage it actually holds, or the label would be spuriously
    suppressed at every consumer (a distinct object never matches the source's
    storage). This cannot launder a STALE label: the session-token anchor
    (unchanged by re-pinning) still governs, so a foreign-session source stays
    rejected regardless of storage.
    """
    src_meta = get(src)
    if src_meta is not None:
        dst._tl = dataclass_replace(cast(Any, src_meta))
        if isinstance(dst._tl, TensorMeta):
            dst._tl.label_storage = _pinned_storage(dst)
        session = _ACTIVE_LABEL_SESSION
        if (
            session is not None
            and isinstance(src_meta, TensorMeta)
            and src_meta.label_session == session.token
        ):
            try:
                session.stamped[dst] = True
            except TypeError:
                pass
