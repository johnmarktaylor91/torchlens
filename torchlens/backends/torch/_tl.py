"""TorchLens private metadata namespace helpers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, Iterable, List, Optional, cast

from torch import nn
from torch.utils.weak import WeakIdKeyDictionary

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
    "raw_tensor_label",
    "get_live_tensor_label",
    "get_live_label_list",
    "begin_label_session",
    "end_label_session",
    "active_label_session_token",
    "session_meta_is_anchored",
    "session_labeled_tensors",
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
]


class TorchLensMeta:
    """Branded base for TorchLens-owned ``._tl`` metadata."""


@dataclass
class TensorMeta(TorchLensMeta):
    """Metadata attached to non-Parameter tensors during a capture session."""

    label_raw: Optional[str] = None
    address: Optional[str] = None
    buffer_source: Optional[str] = None
    # r83 C1: monotonic token of the capture session that issued ``label_raw``
    # (and, after promotion, ``buffer_source``). The per-object anchor that
    # makes label provenance current-session; see the label-session block below.
    label_session: Optional[int] = None


@dataclass
class ParamMeta(TorchLensMeta):
    """Metadata attached to parameters during a capture session."""

    param_barcode: Optional[str] = None
    param_address: Optional[str] = None
    call_index: int = 0
    requires_grad_before_capture: Optional[bool] = None


@dataclass
class ModuleMeta(TorchLensMeta):
    """Permanent metadata attached to modules after model preparation."""

    address: Optional[str] = None
    module_type: Optional[str] = None


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
# Values are ModuleMeta / ParamMeta respectively (WeakIdKeyDictionary is not
# typing-subscriptable, so the value type is documented rather than annotated).
_MODULE_REGISTRY: "WeakIdKeyDictionary" = WeakIdKeyDictionary()
_PARAM_REGISTRY: "WeakIdKeyDictionary" = WeakIdKeyDictionary()


# --------------------------------------------------------------------------- #
# r83 C1 -- CURRENT-SESSION OBJECT ANCHORING FOR RAW LABELS.
#
# ``TensorMeta.label_raw`` (and its promoted ``buffer_source`` sibling) used to
# be validated by pure TEXT membership in the active capture's live event index.
# Label text is deterministic per op-kind + ordinal, so an ordinary op in a
# LATER, unrelated capture regenerates the same string: a tensor still carrying
# a label from an EARLIER capture was therefore accepted as current-session
# provenance and spliced into the new DAG as the same-named node (r82, broken
# independently by three lanes -- a stock ``register_forward_hook`` activation
# collector sufficed, producing a SAME-INPUT wrong parent bind reported as
# ``VERIFIED``). r79 gave the param rung an object-identity belt and r81 gave
# the buffer ``address`` rung one; this is the label rung's.
#
# THE ANCHOR is ``TensorMeta.label_session``: the monotonic token of the capture
# that issued this object's label, written onto the object's OWN metadata by
# ``set_tensor_label`` -- the single choke point through which every label stamp
# in the torch backend flows (verified: no other site assigns ``label_raw``). A
# stamp therefore cannot come into existence without its anchor, and a future
# stamp site cannot silently escape the belt (the r80 F1 root-A failure mode).
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
# the r82 free-lane launder rode the layout/origin rungs instead.
#
# ``_LabelSession.stamped`` is a WEAK inventory of the objects stamped this
# session, used only by the inventory-driven cleanup. It is weak because labels
# are stamped on every activation and strong refs would pin the whole activation
# graph, defeating sparse ``save=``; anything still alive at cleanup time is
# alive because something outside TorchLens holds it -- exactly the leak vehicles
# cleanup needs to reach. Weakness is safe here in a way it is NOT for the r81
# buffer stamp keeper: entries are only ever compared by dereferenced object
# identity, and ``WeakIdRef.__eq__`` returns False whenever either side is dead,
# so a recycled ``id()`` can never produce a false match.
# --------------------------------------------------------------------------- #
_LABEL_SESSION_COUNTER = itertools.count(1)


class _LabelSession:
    """One capture session's identity for issued raw labels."""

    __slots__ = ("token", "stamped")

    def __init__(self, token: int) -> None:
        """Initialize an empty anchor registry for one capture session.

        Parameters
        ----------
        token : int
            Monotonic session token, unique for the process lifetime.
        """
        self.token = token
        # Weak inventory of every object stamped this session, for the
        # inventory-driven cleanup (r83 C1 root A). One entry per OBJECT, not
        # per stamp: relabeling an in-place receiver does not re-register.
        self.stamped: "WeakIdKeyDictionary" = WeakIdKeyDictionary()


_ACTIVE_LABEL_SESSION: Optional[_LabelSession] = None
_RETIRED_LABEL_SESSION: Optional[_LabelSession] = None


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
    container-nested ``types.ModuleType`` stash r81's shallow sweep could not.

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


def active_label_session_token() -> Optional[int]:
    """Return the active label session token, if a capture is in progress.

    Returns
    -------
    Optional[int]
        Session token, or ``None`` when no capture session is installed.
    """

    session = _ACTIVE_LABEL_SESSION
    return None if session is None else session.token


def session_labeled_tensors() -> List[Any]:
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


def _session_gate_blocks(meta: TensorMeta) -> bool:
    """Return whether a capture is active that did NOT issue this meta's labels.

    The read gate shared by :func:`get_tensor_label` and :func:`get_label_list`.
    Outside a capture (postprocess and every read after ``end_label_session``)
    nothing is gated, so post-capture behaviour is exactly as before r83.

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


def session_meta_is_anchored(meta: Optional[TensorMeta]) -> bool:
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


def get(obj: Any) -> Optional[TorchLensMeta]:
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


def get_tensor_meta(t: Any) -> Optional[TensorMeta]:
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

    Parameters
    ----------
    t : Any
        Tensor-like object to tag.
    label : str
        Raw TorchLens label.

    Notes
    -----
    r83 C1: the stamp and its current-session anchor are written together
    here, the single choke point every torch-backend label stamp flows
    through (verified: no other site assigns ``TensorMeta.label_raw``), so a
    label can never come into existence without its anchor and no future
    stamp site can silently escape the belt.
    """
    meta = _ensure_tensor_meta(t)
    meta.label_raw = label
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


def get_tensor_label(t: Any) -> Optional[str]:
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
    r83 C1: this is the single gate through which every label consumer in the
    torch backend reads provenance -- the graph-parent binder, the layout
    ancestry rooting rung, the dispatch-origin ladder, the host-escape
    attribution ladder, the buffer-write producer, and the replay-template
    ``ParentRef`` builder. Gating HERE rather than at each of them is what
    makes the belt exhaustive: a label issued by an EARLIER capture is
    invisible to all of them at once, so a foreign tensor can never be
    accepted as current-session state however it re-enters. With no session
    installed (postprocess, and any read after ``end_label_session``) the raw
    label is returned unchanged, so post-capture behaviour is untouched.
    """
    meta = get_tensor_meta(t)
    if meta is None or meta.label_raw is None:
        return None
    if _session_gate_blocks(meta):
        return None
    return meta.label_raw


def raw_tensor_label(t: Any) -> Optional[str]:
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


def get_live_tensor_label(t: Any, live_labels: Iterable[str]) -> Optional[str]:
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
    r83 C1: resolution requires BOTH that the label text is live in this
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


def get_live_label_list(tensor_list: Iterable[Any], live_labels: Iterable[str]) -> List[str]:
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

    labels: List[str] = []
    for tensor in tensor_list:
        label = get_live_tensor_label(tensor, live_labels)
        if label is not None:
            labels.append(label)
    return labels


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


def get_buffer_address(t: Any) -> Optional[str]:
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


def get_label_list(tensors: Iterable[Any]) -> List[str]:
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
    r83 C1: this is the fastlog/predicate recorder's graph-parent binder, so it
    carries the same current-session anchor gate as :func:`get_tensor_label` --
    a foreign tensor holding a colliding label from an earlier capture must not
    become a parent on this path either.
    """
    out: List[str] = []
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
        ):
            out.append(meta.label_raw)
    return out


def get_param_meta(p: Any) -> Optional[ParamMeta]:
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
    """Restore a parameter's pre-capture ``requires_grad`` flag.

    Parameters
    ----------
    p : Any
        Parameter-like object to restore.
    """
    meta = get_param_meta(p)
    if meta is not None and meta.requires_grad_before_capture is not None:
        p.requires_grad = meta.requires_grad_before_capture


def get_module_meta(m: Any) -> Optional[ModuleMeta]:
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


def _get_decoration_tag(fn: Any) -> Optional[DecorationTag]:
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
    r83 C1: the copy carries ``label_session`` with the rest of the dataclass,
    so an intervention replacement inherits the source's anchor exactly -- a
    live source's replacement keeps provenance, and a STALE source's label
    stays stale rather than laundering into a fresh object. The replacement is
    joined to the session inventory only when it is genuinely current-session,
    so cleanup can reach it.
    """
    src_meta = get(src)
    if src_meta is not None:
        dst._tl = dataclass_replace(cast(Any, src_meta))
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
