"""Dispatch operator naming and callsite helpers."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)

from ... import _state
from ..._utils.tensor_utils import _is_uninitialized_lazy_tensor
from ._tl import (
    get_tensor_label,
    get_tensor_meta,
    is_tensor_data_alias,
)
from .buffer_writes import session_validated_buffer_address
from .escape_detection import (
    ExpectedOriginalToken,
)

__all__ = (
    "_resolved_dispatch_origins",
    "_record_escape_source_tensor",
    "_operator_name",
    "_operator_base_name",
    "_is_aten_operator",
    "_is_mutating_operator",
    "_is_buffer_state_view_dispatch",
    "_dispatch_callsite",
    "_first_scalar_escape_source",
    "_external_warning_stacklevel",
    "_iter_dispatch_tensors",
)


def _resolved_dispatch_origins(
    trace: Any,
    source: torch.Tensor,
    *,
    prefer_ledger: bool = False,
) -> tuple[set[str], set[str]] | None:
    """Resolve an unlabelled escape source to positive (labels, state names), or ``None``.

    Parameters
    ----------
    trace:
        Active capture trace owning the dispatch-origin ledger.
    source:
        Tensor whose value origins must be resolved.
    prefer_ledger:
        Whether to consult the dispatch ledger before the tensor's own label.
        ``Tensor.data`` aliases use this route because their canonical detach
        label represents the getter, while the ledger names the semantic base
        producer required by the host-escape witness.

    Returns ``None`` -- the caller MUST fail closed -- when the source's propagated
    origin set contains ``unknown`` (an operand the census could not attribute),
    ``rng`` (raw seeded-RNG output; the torch-RNG nets own that class, and value
    attribution through it would launder nondeterminism), or ``uninit`` (r53
    hon_2: uninitialized allocator bytes are not a function of the recorded
    computation, so attributing through them would launder nondeterminism as a
    deterministic chain). An empty origin pair is a positive result: the value
    derives from a literal-only deterministic chain that replays identically, so
    it needs no witness.
    """

    origins: frozenset[str] | None = None
    if prefer_ledger:
        registry = _DISPATCH_TENSOR_ORIGINS.get(trace)
        entry = registry.get(source) if registry is not None else None
        if entry is not None:
            origins = entry[0]
    if origins is None:
        with _state.pause_logging(), internal_scalar_read():
            origins = _operand_origins(trace, source)
    if _ORIGIN_UNKNOWN in origins or _ORIGIN_RNG in origins or _ORIGIN_UNINIT in origins:
        return None
    labels = {
        origin[len(_ORIGIN_LABEL_PREFIX) :]
        for origin in origins
        if origin.startswith(_ORIGIN_LABEL_PREFIX)
    }
    states = {
        origin[len(_ORIGIN_STATE_PREFIX) :]
        for origin in origins
        if origin.startswith(_ORIGIN_STATE_PREFIX)
    }
    return labels, states


def _record_escape_source_tensor(
    trace: Any,
    source: torch.Tensor,
    *,
    fail_closed: bool = True,
    resolve_origins: bool = True,
) -> None:
    """Record ONE tensor->host escape source, visible or census-invisible, uniformly.

    Both observation mechanisms — the scoped method patch (``.tolist()`` /
    ``.numpy()`` / ``__array__`` conversions) and the aten census
    (``aten._local_scalar_dense`` scalar escapes) — feed the SAME per-trace side
    tables so the runnable descriptor witnesses every source class -- input, internal op, bound/unbound param, bound/unbound buffer --
    by its capture-time digest through one uniform pass. Side tables are mutated via
    GIL-atomic ``set.add``/``dict`` writes (CPython), so cross-thread recording (r41)
    needs no extra locking.

    ``fail_closed`` (r41 hon2_1/F): ``False`` -- the FOREIGN (pre-existing) thread
    posture -- skips exactly the two unattributable rungs (the fail-closed bool/opaque
    records), so an unattributable foreign-thread read never ceilings the capture while
    every POSITIVE rung (label, registered-state alias, dispatch origin) still records.

    ``resolve_origins`` (r41): ``False`` skips the dispatch-origin resolution rung and
    the leaf-origin fallback recording, both of which take ``pause_logging`` (a GLOBAL
    toggle a non-owner thread must never flip mid-forward). An absent fallback entry is
    consumed by the producer exactly like the fail-closed ``None`` marker (an
    orphan-pruned label without a basis stays INCOMPLETE), so skipping never weakens.

    An escape dispatched from TorchLens's own op-logging internals (a metadata read of a
    freshly-produced op output) is NOT a user escape and is skipped, so the fail-closed
    INCOMPLETE gates never fire on TorchLens's own reads.
    """

    if _escape_source_is_torchlens_internal():
        return
    is_bool = source.dtype is torch.bool
    # ``Tensor.data`` is captured as a canonical detach node so ordinary tensor
    # replay retains a graph edge. For a host escape, however, the alias is not
    # the semantic value source: resolve its dispatch origins exactly like the
    # historical unlabelled ``.data`` object so the witness attributes the base
    # producer rather than the synthetic getter node.
    data_alias = is_tensor_data_alias(source)
    label = None if data_alias else get_tensor_label(source)
    if not isinstance(label, str):
        # An UNLABELLED escape source (a ``.data`` alias, a raw-dispatch product):
        # r37 INV-1 single-exit attribution ladder. Every rung is a POSITIVE
        # attribution to a witnessable source; the fallthrough IS the fail-closed
        # record. Banned forever as discharge mechanisms: scalar value equality
        # (hon2_2), ``.item()`` re-extraction on unknown-arity operands (hon2_1),
        # and any autograd-graph structural purity argument (hon2_3 / exp1).
        if is_bool:
            # A pruned, unlabelled bool predicate is covered by NO net -> fail closed
            # (skipped for a foreign thread: no attribution, no ceiling).
            if fail_closed:
                _HOST_ESCAPE_UNATTRIBUTABLE_BOOL.add(trace)
            return
        # Rung 1 (r18): direct registered-param storage alias -- witnessed by the
        # param state slot (``self.w.tolist()`` directly on a param carries no label).
        param_addresses = _param_derived_addresses(trace, source)
        if param_addresses:
            state_names = _HOST_ESCAPE_STATE_SOURCE_NAMES.get(trace)
            if state_names is None:
                state_names = set()
                _HOST_ESCAPE_STATE_SOURCE_NAMES[trace] = state_names
            state_names |= param_addresses
            return
        # Rung 2 (r37 mechanism A): positive dispatch-origin propagation. The census
        # registered this tensor's value origins at its producing dispatch; resolve
        # them to witnessable raw labels (tensor-op/input sources -> PASS B digest)
        # and state names (param/buffer sources -> PASS A digest). Multi-element and
        # scalar sources resolve identically -- no arity assumption anywhere.
        # Owner-thread only (resolution flips the global logging toggle).
        resolved = (
            _resolved_dispatch_origins(trace, source, prefer_ledger=data_alias)
            if resolve_origins
            else None
        )
        if resolved is not None:
            origin_labels, origin_states = resolved
            if origin_labels:
                sources = _HOST_ESCAPE_SOURCE_LABELS.get(trace)
                if sources is None:
                    sources = set()
                    _HOST_ESCAPE_SOURCE_LABELS[trace] = sources
                sources |= origin_labels
                # An origin label can itself be an interior (later orphan-pruned)
                # op label; give each the same leaf fallback basis.
                for origin_label in origin_labels:
                    _record_escape_label_fallback(trace, origin_label, source)
            if origin_states:
                state_names = _HOST_ESCAPE_STATE_SOURCE_NAMES.get(trace)
                if state_names is None:
                    state_names = set()
                    _HOST_ESCAPE_STATE_SOURCE_NAMES[trace] = state_names
                state_names |= origin_states
            # Empty label+state origins: a literal-only deterministic chain whose
            # baked value replays identically -- positively attributed, witness-free.
            return
        # Fallthrough: no positive attribution -> fail closed (INCOMPLETE). This is
        # the ONLY other exit; there is no third state (INV-1). A foreign thread
        # (``fail_closed=False``) skips the record: no attribution, no ceiling.
        if fail_closed:
            _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE.add(trace)
        return
    sources = _HOST_ESCAPE_SOURCE_LABELS.get(trace)
    if sources is None:
        sources = set()
        _HOST_ESCAPE_SOURCE_LABELS[trace] = sources
    sources.add(label)
    # r37 mechanism A: record the leaf-origin fallback basis NOW (the live tensor and
    # its propagated origins exist only during capture). Consumed by the producer only
    # if this label turns out orphan-pruned. Skipped on non-owner threads (r41,
    # ``resolve_origins=False``): an absent entry reads exactly like the fail-closed
    # marker if the label is later orphan-pruned -- never weaker.
    if resolve_origins:
        _record_escape_label_fallback(trace, label, source)
    # A BOOL predicate source stays in the main set (the pruned-RNG control-flow
    # detector consumes it) but is tracked here so the runnable producer excludes an
    # unresolved bool predicate from the tensor-op INCOMPLETE gate (it is the
    # control-witness / conditional / loop / pruned-RNG net's domain).
    if is_bool:
        bool_sources = _HOST_ESCAPE_BOOL_SOURCE_LABELS.get(trace)
        if bool_sources is None:
            bool_sources = set()
            _HOST_ESCAPE_BOOL_SOURCE_LABELS[trace] = bool_sources
        bool_sources.add(label)
    # A registered buffer/parameter source (identified by a non-None state address) is
    # witnessed by its state slot digest. Record BOTH the raw label (so the producer
    # does not close an unresolved orphan-pruned STATE label as a pruned tensor-op
    # chain) AND the state_dict name/address (so the producer witnesses the state slot
    # even when the state ALSO feeds a traced graph op -- bound-ness never exempts the
    # escape witness).
    meta = get_tensor_meta(source)
    address = getattr(meta, "address", None) if meta is not None else None
    # r18 + r19-C: a PARAMETER host escape resolves by state slot exactly like a buffer. A buffer
    # keeps a graph SOURCE node, so its ``.detach()`` host read survives orphan-pruning and witnesses
    # by its kept op; a parameter carries NO source node, so a param-rooted read op is orphan-pruned
    # and its raw label resolves to no final op -> the escape would fail closed (INCOMPLETE_SCALAR_
    # ESCAPE -> a spurious UNVERIFIABLE) even for a purely READ-ONLY stat log. Resolve the param
    # state slot(s) so the read-only escape is witnessed by the param's capture-time digest (value-
    # correct -- an unchanged param re-digests identically -> VERIFIED; a changed param -> UNVERIFIABLE),
    # the same honest read/write distinction the buffer path draws. ``_param_derived_addresses`` covers
    # both a DIRECT param alias (r18: ``self.w.detach()``, ``self.w[0]``) and a DERIVED pruned read
    # rooted purely in params (r19-C: ``self.w.sum()``, ``float(self.w.max())``). A genuine host WRITE
    # is caught independently by the parameter whole-storage byte tripwire
    # (``buffer_writes._reconcile_params`` -> ``_HOST_ESCAPE_MUTABLE_WRITEBACK``), so read resolution
    # never blesses a mutated param.
    state_addresses: set[str] = set()
    if address is not None:
        state_addresses.add(str(address))
    else:
        state_addresses |= _param_derived_addresses(trace, source)
    if state_addresses:
        state_sources = _HOST_ESCAPE_STATE_SOURCE_LABELS.get(trace)
        if state_sources is None:
            state_sources = set()
            _HOST_ESCAPE_STATE_SOURCE_LABELS[trace] = state_sources
        state_sources.add(label)
        state_names = _HOST_ESCAPE_STATE_SOURCE_NAMES.get(trace)
        if state_names is None:
            state_names = set()
            _HOST_ESCAPE_STATE_SOURCE_NAMES[trace] = state_names
        state_names |= state_addresses


def _operator_name(func: Any) -> str:
    """Return a stable dispatcher operator and overload name.

    Parameters
    ----------
    func:
        Dispatcher callable received by ``__torch_dispatch__``.

    Returns
    -------
    str
        Best-effort qualified operator name such as ``aten.relu.default``.
    """

    try:
        return str(func)
    except Exception:
        return type(func).__name__


def _operator_base_name(func: Any) -> str:
    """Return a dispatcher operator's namespace+base name with the overload stripped.

    ``_operator_name`` yields the fully-qualified ``aten.<op>.<overload>`` (e.g.
    ``aten.equal.default``); this drops the trailing ``.<overload>`` so a value-escape
    op is matched by its overload-independent base (``aten.equal``) against
    ``HOST_ESCAPE_OPERATORS``. A name with no overload segment is returned unchanged.
    """

    name = _operator_name(func)
    if name.count(".") >= 2:
        return name.rsplit(".", 1)[0]
    return name


def _is_aten_operator(func: Any) -> bool:
    """Return whether a dispatcher callable belongs to the aten namespace.

    Parameters
    ----------
    func:
        Dispatcher callable received by ``__torch_dispatch__``.

    Returns
    -------
    bool
        ``True`` only for the aten census domain.
    """

    namespace = getattr(func, "namespace", None)
    if isinstance(namespace, str):
        return namespace == "aten"
    return _operator_name(func).startswith("aten.")


def _is_mutating_operator(func: Any) -> bool:
    """Return whether a dispatcher operator writes to any of its arguments.

    Mutation is read from the operator's own ``FunctionSchema`` (torch ground
    truth), never a name-string heuristic: ``schema.is_mutable`` covers every
    in-place operator (trailing-underscore names such as ``mul_``/``copy_``) as
    well as ``out=`` overloads whose name does NOT end in an underscore. Per-arg
    ``alias_info.is_write`` is used as a robust fallback when the schema flag is
    unavailable. Pure reads such as ``aten.equal`` / ``aten.allclose`` return
    ``False``, which is exactly why benign ``owner_not_captured`` control-flow
    comparisons are never mistaken for value-affecting drops.

    Parameters
    ----------
    func:
        Dispatcher callable received by ``__torch_dispatch__``.

    Returns
    -------
    bool
        ``True`` when the operator mutates (writes) at least one argument.
    """

    schema = getattr(func, "_schema", None)
    if schema is None:
        return False
    is_mutable = getattr(schema, "is_mutable", None)
    if isinstance(is_mutable, bool):
        return is_mutable
    arguments = getattr(schema, "arguments", ()) or ()
    for argument in arguments:
        alias_info = getattr(argument, "alias_info", None)
        if alias_info is not None and getattr(alias_info, "is_write", False):
            return True
    return False


def _is_buffer_state_view_dispatch(
    trace: Any,
    func: Any,
    owner: ExpectedOriginalToken | None,
    mutates: bool,
    args: tuple[Any, ...],
) -> bool:
    """Return whether an aten dispatch is a ``.data``-accessor view on module state.

    This flags the intrinsic, legitimately-uncaptured ``aten.detach`` a registered buffer's
    ``.data`` property emits during a buffer WRITE (``self.b.data.copy_(x)``), and the same
    accessor view on a PENDING lazy slot: ``UninitializedParameter.materialize`` reads
    ``self.data`` (device, dtype) through the wrapper's uninitialized pass-through, which
    logs nothing because the slot has no storage to log (the materialized parameter
    registers at the module-entry gate). The predicate is intentionally strict on every
    axis so it can never mask a genuine untraced dispatch:

    * ``owner is None`` -- the dispatch has NO python-wrapper owner (a wrapped ``.detach()``
      call would be an accounted owner, not a gap; only the property-accessor path is unowned).
    * ``not mutates`` -- the operator writes to no argument (ground-truth schema flag). A
      value-affecting in-place drop can never be credited here.
    * ``aten.detach`` / ``aten.alias`` only -- pure aliasing views. A dropped value-producing
      op (``aten.add``/``aten.mul``/...) on the buffer is NOT in this set and stays unaccounted.
    * ``args[0]`` is a REGISTERED BUFFER whose stamp is SESSION-VALIDATED (r81:
      current-session object + live storage identity -- a stale or input-rebound
      stamp is never credited as a benign state view), or an UNINITIALIZED lazy
      parameter/buffer (``torch.nn.parameter.is_lazy``): a slot with no storage, whose
      view can carry no model value. Once materialized it is an ordinary parameter and
      no longer qualifies.

    Parameters
    ----------
    trace:
        Active capture trace owning the session buffer identity registry.
    func:
        Dispatcher operator overload.
    owner:
        The live wrapper owner of the dispatch, or ``None`` when unowned.
    mutates:
        Whether the operator writes to any argument (schema ground truth).
    args:
        Positional dispatcher arguments; ``args[0]`` is the view source.

    Returns
    -------
    bool
        ``True`` only for a ``.data``-accessor view dispatch on a registered buffer or a
        pending lazy slot.
    """

    if owner is not None or mutates:
        return False
    if _operator_base_name(func) not in _BUFFER_STATE_VIEW_OPERATORS:
        return False
    if not args:
        return False
    source = args[0]
    if not isinstance(source, torch.Tensor):
        return False
    if _is_uninitialized_lazy_tensor(source):
        return True
    return session_validated_buffer_address(trace, source) is not None


def _dispatch_callsite() -> _DispatchCallsite:
    """Return the first non-framework frame above the dispatch callback.

    Returns
    -------
    _DispatchCallsite
        Best-effort source location for an unowned dispatcher event.
    """

    frame: Any = sys._getframe(2)
    fallback = frame
    while frame is not None:
        filename = frame.f_code.co_filename
        framework_frame = _FRAMEWORK_FILENAME_VERDICTS.get(filename)
        if framework_frame is None:
            try:
                resolved = Path(filename).resolve()
                framework_frame = resolved.is_relative_to(_TORCH_ROOT) or resolved.is_relative_to(
                    _TORCHLENS_ROOT
                )
            except (OSError, RuntimeError, ValueError):
                framework_frame = False
            # Bounded: generated-code / notebook / plugin workloads mint fresh
            # co_filename values without limit; FIFO-drop the oldest verdict at
            # the cap rather than growing process-global forever.
            while len(_FRAMEWORK_FILENAME_VERDICTS) >= _FRAMEWORK_FILENAME_VERDICTS_MAX_ENTRIES:
                _FRAMEWORK_FILENAME_VERDICTS.pop(next(iter(_FRAMEWORK_FILENAME_VERDICTS)))
            _FRAMEWORK_FILENAME_VERDICTS[filename] = framework_frame
        if not framework_frame:
            return _DispatchCallsite(filename, frame.f_lineno, frame.f_code.co_name)
        fallback = frame
        frame = frame.f_back
    return _DispatchCallsite(
        fallback.f_code.co_filename,
        fallback.f_lineno,
        fallback.f_code.co_name,
    )


def _first_scalar_escape_source() -> tuple[str | None, int | None]:
    """Return the first non-TorchLens frame for a scalar escape call.

    Returns
    -------
    tuple[str | None, int | None]
        Source filename and line, or ``(None, None)`` if no user frame is visible.
    """
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            filename = Path(frame.f_code.co_filename).resolve()
            try:
                filename.relative_to(_TORCHLENS_ROOT)
            except ValueError:
                return str(filename), frame.f_lineno
            frame = frame.f_back
    finally:
        del frame
    return None, None


def _external_warning_stacklevel() -> int:
    """Return a warning stack level that resolves outside the TorchLens package.

    Returns
    -------
    int
        Stack level suitable for :func:`warnings.warn`.
    """
    level = 1
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            try:
                Path(frame.f_code.co_filename).resolve().relative_to(_TORCHLENS_ROOT)
            except ValueError:
                return level
            level += 1
            frame = frame.f_back
    finally:
        del frame
    return level


def _iter_dispatch_tensors(
    args: tuple[Any, ...], kwargs: dict[str, Any] | None
) -> Iterator[torch.Tensor]:
    """Yield every ``torch.Tensor`` operand of a dispatch, flattening list/tuple containers.

    Aten operands are tensors, scalars, or (for ``cat`` / ``stack`` / ``_foreach_*``) lists/tuples
    of tensors; this walks those one-deep-or-more without importing a pytree so the per-consumption
    watch can see every tensor an op reads.
    """

    stack: list[Any] = list(args)
    if kwargs:
        stack.extend(kwargs.values())
    while stack:
        item = stack.pop()
        if isinstance(item, torch.Tensor):
            yield item
        elif isinstance(item, (list, tuple)):
            stack.extend(item)


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from ._completeness_escape_state import (  # noqa: E402
    _record_escape_label_fallback,
)
from ._completeness_origins import (  # noqa: E402
    _escape_source_is_torchlens_internal,
    _operand_origins,
    _param_derived_addresses,
    internal_scalar_read,
)
from .completeness_witness import (  # noqa: E402
    _BUFFER_STATE_VIEW_OPERATORS,
    _DISPATCH_TENSOR_ORIGINS,
    _FRAMEWORK_FILENAME_VERDICTS,
    _FRAMEWORK_FILENAME_VERDICTS_MAX_ENTRIES,
    _HOST_ESCAPE_BOOL_SOURCE_LABELS,
    _HOST_ESCAPE_SOURCE_LABELS,
    _HOST_ESCAPE_STATE_SOURCE_LABELS,
    _HOST_ESCAPE_STATE_SOURCE_NAMES,
    _HOST_ESCAPE_UNATTRIBUTABLE_BOOL,
    _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE,
    _ORIGIN_LABEL_PREFIX,
    _ORIGIN_RNG,
    _ORIGIN_STATE_PREFIX,
    _ORIGIN_UNINIT,
    _ORIGIN_UNKNOWN,
    _TORCH_ROOT,
    _TORCHLENS_ROOT,
    _DispatchCallsite,
)
