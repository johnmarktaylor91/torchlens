"""Input and state metadata witness bookkeeping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)

from ... import _state
from ._completeness_types import _WitnessState
from ._tl import (
    get_tensor_label,
)
from .buffer_writes import session_validated_buffer_address

__all__ = (
    "_observe_input_derived_layout_read",
    "_resolve_layout_rooting_labels",
    "_layout_storage_rooting_labels",
    "record_storage_rebind_barrier",
    "storage_rebind_barrier_labels",
    "_layout_ancestry_tainted",
    "_state_derived_addresses",
    "_observe_state_metadata_read",
    "host_escape_state_metadata_reads",
    "_state_direct_address",
    "_observe_state_metadata_read_direct",
    "_expand_state_alias_addresses",
    "_record_state_metadata_read",
    "_record_state_metadata_observation",
    "host_escape_state_metadata_observations",
    "_observe_state_placement_read",
    "_placement_read_witnessed",
    "_discharge_placement_dispatch",
    "_observe_state_metadata_fact",
    "host_escape_state_metadata_facts",
    "_observe_state_property_read",
    "_tensor_receiver_origin",
    "_register_storage_handle_origin",
    "_register_storage_origin",
    "_lazy_storage_state_ptr_names",
    "_resolve_storage_origin",
    "_record_input_storage_nbytes",
)


def _observe_input_derived_layout_read(trace: Any, source: torch.Tensor) -> None:
    """Attribute a layout read on an INPUT-DERIVED activation to its rooting input(s) (r73 F1).

    Called only from the layout-trio fall-through of :func:`_observe_input_metadata_read`
    (receiver already proven NOT an input leaf, storage alias, or ``_base``-linked view).
    Resolution is bookkeeping only -- label/ledger/live-index lookups plus a wrapper-free
    storage-pointer read under the internal marker -- so it can never recurse back into the
    metadata patches.

    COVERAGE NOTE (r75 F1, LOCKED): any unlabeled-or-unresolvable-receiver layout consumer
    in this fall-through MUST FAIL CLOSED -- a silent no-record here is exactly the r74
    ``.data``-alias false-VERIFIED reopening. The r73 version fail-OPENED on three rungs
    (label ``None``, event ``None``, empty ``input_ancestors``); each now either resolves
    POSITIVELY or downgrades completeness through ``_INPUT_METADATA_VIEW_READ`` (the same
    presence-only weak set the sibling escape nets use for this receiver class).

    Resolution ladder (:func:`_resolve_layout_rooting_labels`), then per rooting label:

    * ANCESTRY-INTEGRITY check (:func:`_layout_ancestry_tainted`): the rooting event's
      transitive parent chain must contain NO op with ``unattributed_tensor_args`` -- an
      unattributed tensor arg (a ``.data``-style unlabeled alias consumed by a logged op)
      is precisely where traced ancestry BREAKS, so recorded ``input_ancestors`` are a
      lower bound, not the truth. A tainted labeled receiver is re-resolved through the
      dispatch-origin ledger's LEAF origins (value-DAG-true through the break); an
      unresolvable taint fails closed.
    * Non-empty resolved input ancestry -> record :data:`INPUT_DERIVED_LAYOUT_FACT_NAME`
      -- carrying the rooting LEAF's capture-time stride tuple -- on each ancestor input's
      boundary site. An ancestor missing from the capture-layout map cannot pin a
      comparison basis, so the fact is recorded on EVERY mapped site instead (fail closed:
      the ceiling then keys on any input layout change).
    * Empty resolved input ancestry with a CLEAN chain (or positive state/literal-only
      ledger origins) -> genuinely state/internal-rooted: record nothing (contract
      residual (3)'s STATE side stays untouched). Empty because ancestry ORPHANED is
      unreachable here -- orphaned receivers are re-resolved or fail closed above.
    """

    layouts = trace._runnable.input_label_layouts
    capture_events = getattr(trace, "capture_events", None)
    live_index = getattr(capture_events, "live_index", None)
    by_raw_label = getattr(live_index, "by_raw_label", None)
    if not layouts or by_raw_label is None:
        # Inputs exist (the object-identity map gated the caller) but no layout basis /
        # live index is available to attribute against: fail closed, never silent.
        _INPUT_METADATA_VIEW_READ.add(trace)
        return
    rooting_labels = _resolve_layout_rooting_labels(trace, by_raw_label, source)
    if rooting_labels is None:
        _INPUT_METADATA_VIEW_READ.add(trace)
        return
    ancestors: set[str] = set()
    for rooting_label in rooting_labels:
        event = by_raw_label.get(rooting_label)
        if event is None or _layout_ancestry_tainted(trace, by_raw_label, rooting_label):
            _INPUT_METADATA_VIEW_READ.add(trace)
            return
        ancestors.update(getattr(event, "input_ancestors", None) or ())
    if not ancestors:
        # Every rooting event is chain-clean with no input ancestry: positively
        # state/internal-rooted (residual (3)) or a literal-only deterministic chain
        # whose layout replays identically. Record nothing.
        return
    if all(ancestor in layouts for ancestor in ancestors):
        targets = [layouts[ancestor] for ancestor in ancestors]
    else:
        targets = list(layouts.values())
    for site, leaf_stride in targets:
        _record_input_metadata_read_at_site(
            trace, site, INPUT_DERIVED_LAYOUT_FACT_NAME, tuple(int(v) for v in leaf_stride)
        )


def _resolve_layout_rooting_labels(
    trace: Any, by_raw_label: Mapping[str, Any], source: torch.Tensor
) -> set[str] | None:
    """Resolve a layout-read receiver to the raw labels its VALUE roots through (r75 F1).

    Ladder, first positive resolution wins; ``None`` means the caller MUST fail closed:

    * OWN LABEL (or the ``_base`` fallback for an unlogged view of a logged activation),
      accepted only when its traced ancestry is INTACT (:func:`_layout_ancestry_tainted`).
    * DISPATCH-ORIGIN LEDGER leaf origins (r37 mechanism A): an unlabeled ``.data``-style
      alias -- or a labeled receiver whose traced ancestry broke at one -- carries a
      positive record of the terminal leaves its value derives from. ``rng`` / ``uninit``
      taints fail closed (layout attribution through nondeterminism would launder it);
      ``unknown`` falls through to the storage rung. A pure ``state:``/empty leaf set
      resolves to ZERO labels -- a positive state-rooted/literal-only signal the caller
      maps to "record nothing" (residual (3)), never a silent fail-open. r29 F1: that
      positive reading is honored only for an UNLABELED receiver; a labeled receiver
      that reached this rung through a TAINTED rung 1 fails closed instead (its
      identity-keyed ledger entry may predate the taint event), and a receiver whose
      current label is a storage-rebind BARRIER resolves ``unknown`` at the ledger
      itself (:func:`_operand_leaf_origins`).
    * STORAGE IDENTITY (r31 leaf / r63 state precedent): the receiver's true-original
      storage pointer matched against live captured producer tensors
      (``_CAPTURED_STORAGE_PTRS``) -- an alias mechanism the dispatch interpose never saw
      still resolves to the logged activation whose bytes it shares.
    """

    label = get_tensor_label(source)
    if label is None:
        base = _input_base_tensor(source)
        label = get_tensor_label(base) if base is not None else None
    if label is not None and not _layout_ancestry_tainted(trace, by_raw_label, label):
        return {label}
    with _state.pause_logging(), internal_scalar_read():
        leaf_origins = _operand_leaf_origins(trace, source)
    if _ORIGIN_RNG in leaf_origins or _ORIGIN_UNINIT in leaf_origins:
        return None
    if _ORIGIN_UNKNOWN not in leaf_origins:
        rooting = {
            origin[len(_ORIGIN_LABEL_PREFIX) :]
            for origin in leaf_origins
            if origin.startswith(_ORIGIN_LABEL_PREFIX)
        }
        if not rooting and label is not None:
            # r29 F1 (defense in depth): the receiver HAS a label but its traced
            # ancestry is TAINTED, and the ledger resolved a pure-``state:``/
            # literal (empty-label) basis. The identity-keyed ledger entry may
            # predate the taint event (a non-dispatch mutation of the same
            # object), so "positively state-rooted" cannot be trusted here --
            # returning the empty set would fail OPEN in the caller (record
            # nothing). Only an UNLABELED receiver's ledger resolution, or a
            # clean-chain rung-1 label, may positively claim state rooting.
            return None
        return rooting
    if label is None:
        return _layout_storage_rooting_labels(trace, source)
    return None


def _layout_storage_rooting_labels(trace: Any, source: torch.Tensor) -> set[str] | None:
    """Resolve an unlabeled receiver to live captured producers by STORAGE IDENTITY (r75 F1).

    Liveness-verified exactly like the r43 cross-thread belt: a pointer matches only while
    a captured producing tensor is still alive AND still occupies that address, so a freed-
    then-reused allocation can never misattribute. Pointer reads use the wrapper-free
    true-original accessors under the internal marker (no observer recursion). ``None`` --
    no live labeled producer shares the receiver's storage -- means the caller fails closed.
    """

    captured = _CAPTURED_STORAGE_PTRS.get(trace)
    if not captured:
        return None
    with _state.pause_logging(), internal_scalar_read():
        ptr = _raw_storage_ptr_no_observe(source)
    if ptr is None:
        return None
    labels: set[str] = set()
    for producer_ref in captured.get(ptr, ()):
        producer = producer_ref()
        if producer is None or _raw_storage_ptr_no_observe(producer) != ptr:
            continue
        producer_label = get_tensor_label(producer)
        if isinstance(producer_label, str):
            labels.add(producer_label)
    return labels or None


def record_storage_rebind_barrier(trace: Any, raw_label: str) -> None:
    """Register one storage-swapping rebind op label as an ancestry barrier.

    Parameters
    ----------
    trace:
        Active capture Trace.
    raw_label:
        The rebind op's raw capture label (e.g. ``"data_1_3_raw"``).
    """

    labels = _STORAGE_REBIND_BARRIER_LABELS.get(trace)
    if labels is None:
        labels = set()
        _STORAGE_REBIND_BARRIER_LABELS[trace] = labels
    labels.add(raw_label)


def storage_rebind_barrier_labels(trace: Any) -> frozenset[str]:
    """Return the storage-swapping rebind barrier labels recorded for one trace."""

    labels = _STORAGE_REBIND_BARRIER_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def _layout_ancestry_tainted(trace: Any, by_raw_label: Mapping[str, Any], label: str) -> bool:
    """Return whether a logged event's transitive traced ancestry is BROKEN (r75 F1).

    ``OpEvent.input_ancestors`` unions only LABELED parents, so an op that consumed an
    unlabeled tensor arg (``unattributed_tensor_args`` non-empty -- the ``.data`` alias, a
    laundered raw-dispatch product) truncates ancestry SILENTLY: everything downstream
    inherits the truncated set. Any such op anywhere in the parent DAG -- including the
    event itself, an orphaned parentless root (``(y.data * 1.0)``), or a MIXED op with both
    labeled parents and an unattributed arg (``torch.cat([y1, y.data])``) -- makes the
    recorded ``input_ancestors`` a lower bound, so a layout consumer must not trust them.
    An unresolvable parent label also counts as tainted (fail closed on observer
    uncertainty). Pure event-field bookkeeping: no tensor method calls, no recursion into
    the metadata patches.
    """

    clean = _LAYOUT_ANCESTRY_CLEAN.get(trace)
    if clean is None:
        clean = set()
        _LAYOUT_ANCESTRY_CLEAN[trace] = clean
    if label in clean:
        return False
    rebind_barriers = _STORAGE_REBIND_BARRIER_LABELS.get(trace)
    stack = [label]
    visited: set[str] = set()
    while stack:
        current = stack.pop()
        if current in visited or current in clean:
            continue
        visited.add(current)
        event = by_raw_label.get(current)
        if event is None:
            return True
        if getattr(event, "unattributed_tensor_args", None):
            return True
        # r28 reconcile: a storage-SWAPPING ``.data=`` rebind op is an ancestry
        # barrier -- the r79/r81 belt posture never attributes verdict-steering
        # facts across a storage-pointer swap, even though the capture graph
        # honestly threads the rebind's consumers to its RHS producer.
        if rebind_barriers and current in rebind_barriers:
            return True
        stack.extend(edge.parent_label_raw for edge in (getattr(event, "parents", None) or ()))
    clean.update(visited)
    return False


def _state_derived_addresses(trace: Any, source: torch.Tensor) -> set[str]:
    """Resolve ``source`` to the registered-state addresses whose storage it aliases (r63 C1).

    Positive-attribution ladder, first hit wins: the buffer meta address stamped at forward
    start (a DIRECT registered-buffer receiver; model inputs carry a label but never an
    address, so an input can never resolve here); the forward-start param storage index
    (``self.w`` and any ``.data`` / view / detach alias of it); the forward-start buffer
    storage index (the ``.data`` / view alias twin for buffers). A miss returns empty --
    an activation receiver records nothing (its geometry is recomputed by the replayed DAG).

    r81: the direct-stamp rung requires the SESSION-VALIDATED stamp (current-session
    object + live storage identity); a stale or ``.data``-rebound stamp falls through
    to the storage-index rungs, which are anchored on live registered storage.
    """

    address = session_validated_buffer_address(trace, source)
    if address is not None:
        return {str(address)}
    addresses = _param_derived_addresses(trace, source)
    if addresses:
        return addresses
    buffer_storage_addresses = getattr(trace, "_buffer_storage_addresses", None)
    if buffer_storage_addresses:
        ptr = _escape_storage_ptr(source)
        if ptr is not None and ptr in buffer_storage_addresses:
            return {str(buffer_storage_addresses[ptr])}
    return set()


def _observe_state_metadata_read(trace: Any, source: torch.Tensor, read_kind: str) -> None:
    """Attribute one PHYSICAL-metadata read on registered state as a state escape (r63 C1).

    Closes the four r62/r63 attribution gaps: ``is_contiguous`` / ``stride`` /
    ``storage_offset`` / ``is_conj`` (+ the ``is_neg`` lazy-bit sibling) on a registered
    param/buffer previously routed ONLY through the model-input observer, which ignores
    state -- so a model branching on ``self.weight.is_contiguous()`` produced no witness and
    the transport-normalized replay reported a false ``verified``. A resolved read now:

    * joins ``_HOST_ESCAPE_STATE_SOURCE_NAMES`` -- the slot is digest-witnessed by PASS A
      (``unbound_state_escape:<name>`` fact; changed staged state -> ``unverifiable``),
      exactly like a ``self.threshold.item()`` value read; and
    * records its READ KIND in the per-slot metadata ledger consumed by the escape-gated
      producer preflight (a read dim that was non-canonical at capture refuses the save;
      an unread non-canonical slot -- the channels-last population -- stays saveable).

    A model-input leaf receiver is the input nets' domain and is skipped; an unresolvable
    receiver (an activation) records nothing here.
    """

    sites = trace._runnable.input_tensor_sites
    if sites and id(source) in sites:
        return
    addresses = _state_derived_addresses(trace, source)
    if not addresses:
        return
    _record_state_metadata_read(trace, addresses, read_kind)


def host_escape_state_metadata_reads(trace: Any) -> dict[str, frozenset[str]]:
    """Return the per-state-name PHYSICAL-metadata read kinds witnessed for one trace."""

    reads = _HOST_ESCAPE_STATE_METADATA_READS.get(trace)
    if not reads:
        return {}
    return {name: frozenset(kinds) for name, kinds in reads.items()}


def _state_direct_address(trace: Any, source: torch.Tensor) -> str | None:
    """Resolve ``source`` to a state address ONLY when it IS the registered object (r65).

    The DIRECT-receiver discriminator for the autograd/structural family
    (``_STATE_METADATA_DIRECT_ONLY_NAMES``): a registered buffer carries the buffer meta
    address stamped at forward start (a ``.data``/view alias carries none), and a registered
    parameter is an exact-type ``nn.Parameter`` object (op outputs and ``.data``/``detach()``
    aliases are plain ``Tensor``s -- torch ops never construct ``nn.Parameter`` results, so
    exact-type + param-storage membership identifies the registered object; an op that
    returns the parameter ITSELF, e.g. an already-contiguous ``w.contiguous()``, is the same
    object and attributes correctly). A miss returns ``None`` -- the read is the documented
    alias/view residual, never misattributed.

    r81: "IS the registered object" is enforced with the session belt (current-session
    stamp + live storage identity), never the raw static stamp.
    """

    address = session_validated_buffer_address(trace, source)
    if address is not None:
        return str(address)
    if type(source) is torch.nn.Parameter:
        addresses = _param_derived_addresses(trace, source)
        if addresses:
            # r67 C6: the former ``len(addresses) == 1`` restriction silently DROPPED a
            # tied parameter's direct read. Any resolved membership attributes; the
            # recording layer fans out to the complete r37 alias group.
            return next(iter(sorted(addresses)))
    return None


def _observe_state_metadata_read_direct(trace: Any, source: torch.Tensor, read_kind: str) -> None:
    """Attribute one DIRECT-receiver-only metadata read on registered state (r65).

    The autograd/structural twin of :func:`_observe_state_metadata_read`: joins the same
    escape-source and read-kind ledgers, but ONLY when the receiver is the registered object
    itself (see :func:`_state_direct_address`). An alias/view receiver records nothing (the
    documented residual); an input-leaf receiver is the input nets' domain and is skipped.
    """

    sites = trace._runnable.input_tensor_sites
    if sites and id(source) in sites:
        return
    address = _state_direct_address(trace, source)
    if address is None:
        return
    _record_state_metadata_read(trace, {address}, read_kind)


def _expand_state_alias_addresses(trace: Any, addresses: set[str]) -> set[str]:
    """Fan resolved state addresses out to their COMPLETE r37 alias groups (r67 C6).

    A direct read on ONE canonical name of a tied parameter / double-registered buffer is a
    read of the shared allocation: every name in the identity-tied alias group carries the
    fact. The r37 topology snapshot (``groups``: name -> group id) is the authority; a trace
    without one keeps the resolved set unchanged (fail-safe: no expansion, the resolved
    address still records).
    """

    if not addresses:
        return addresses
    topology = trace._runnable.state_alias_topology
    groups = topology.get("groups") if isinstance(topology, Mapping) else None
    if not isinstance(groups, Mapping) or not groups:
        return addresses
    group_ids = {groups[name] for name in addresses if name in groups}
    if not group_ids:
        return addresses
    return addresses | {str(name) for name, group_id in groups.items() if group_id in group_ids}


def _record_state_metadata_read(trace: Any, addresses: set[str], read_kind: str) -> None:
    """Join resolved state addresses into the escape-source + read-kind ledgers (r63/r65).

    r67 C6: fans out to the complete alias group -- a tied-parameter direct read marks
    every canonical name sharing the allocation.
    """

    addresses = _expand_state_alias_addresses(trace, addresses)
    state_names = _HOST_ESCAPE_STATE_SOURCE_NAMES.get(trace)
    if state_names is None:
        state_names = set()
        _HOST_ESCAPE_STATE_SOURCE_NAMES[trace] = state_names
    state_names |= addresses
    reads = _HOST_ESCAPE_STATE_METADATA_READS.get(trace)
    if reads is None:
        reads = {}
        _HOST_ESCAPE_STATE_METADATA_READS[trace] = reads
    for address in addresses:
        reads.setdefault(address, set()).add(read_kind)


def _record_state_metadata_observation(
    trace: Any, addresses: set[str], read_kind: str, observed: bool | None
) -> None:
    """Record one placement accessor's ACTUAL return against its state alias group (r67 C3)."""

    _record_state_metadata_read(trace, addresses, read_kind)
    addresses = _expand_state_alias_addresses(trace, addresses)
    observations = _HOST_ESCAPE_STATE_METADATA_OBSERVATIONS.get(trace)
    if observations is None:
        observations = {}
        _HOST_ESCAPE_STATE_METADATA_OBSERVATIONS[trace] = observations
    for address in addresses:
        slot = observations.setdefault(address, {})
        if read_kind in slot and slot[read_kind] != observed:
            slot[read_kind] = None  # disagreeing observations: unknown, fail closed
        else:
            slot[read_kind] = observed


def host_escape_state_metadata_observations(trace: Any) -> dict[str, dict[str, bool | None]]:
    """Return the per-state-name OBSERVED placement accessor returns for one trace (r67 C3)."""

    observations = _HOST_ESCAPE_STATE_METADATA_OBSERVATIONS.get(trace)
    if not observations:
        return {}
    return {name: dict(values) for name, values in observations.items()}


def _observe_state_placement_read(
    trace: Any, source: torch.Tensor, name: str, observed: bool | None
) -> None:
    """Attribute one placement accessor's ACTUAL return on a state receiver (r67 C3).

    Same storage-identity attribution family as :func:`_observe_state_metadata_read`
    (placement is a pure function of the slot's storage), with the observed value carried
    into the observation ledger. ``None`` means the accessor raised or was arg-directed:
    unknown, the producer refuses. Input-leaf receivers are the input nets' domain;
    unrelated receivers record nothing.
    """

    sites = trace._runnable.input_tensor_sites
    if sites and id(source) in sites:
        return
    addresses = _state_derived_addresses(trace, source)
    if not addresses:
        return
    _record_state_metadata_observation(trace, addresses, STATE_METADATA_MIRROR[name][1], observed)


def _placement_read_witnessed(trace: Any, source: torch.Tensor) -> bool:
    """Return whether a placement accessor's receiver is positively attributed (r67 C3).

    True for a model-input leaf (fact recorded / alias-safe path), an input storage alias
    (fact recorded, or derived-view fail-closed downgrade -- both honest), or a resolved
    state receiver (observation recorded). False for an unrelated/unattributable receiver:
    the census ledger fact then stands and the capture stays fail-closed.
    """

    sites = trace._runnable.input_tensor_sites or {}
    if id(source) in sites:
        return True
    if _state_derived_addresses(trace, source):
        return True
    kind, _site, _leaf_conj_neg = _classify_input_storage_alias(trace, source)
    return kind is not None


def _discharge_placement_dispatch(state: _WitnessState, base_operator: str) -> None:
    """Mark the most recent matching host-return census event as metadata-witnessed."""

    for event in reversed(state.events):
        if (
            event.outcome == "returned_host_or_none"
            and not event.metadata_witnessed
            and (event.operator == base_operator or event.operator.startswith(base_operator + "."))
        ):
            event.metadata_witnessed = True
            return


def _observe_state_metadata_fact(
    trace: Any, source: torch.Tensor, fact_name: str, fact_value: bool
) -> None:
    """Record one DECLARED-STATE fact for a DIRECT registered param/buffer receiver (r65 F-1).

    ``requires_grad`` records its bool value; ``grad_fn`` records presence. The fact ledger is
    separate from the escape machinery by design (see ``_STATE_METADATA_FACTS``): recording is
    idempotent-by-value for a stable bit, a repeated read overwrites with the latest observed
    value, and a spurious TorchLens-internal read (should one survive the source markers)
    records the true current bit, which staging reproduces -- harmless by construction.
    """

    sites = trace._runnable.input_tensor_sites
    if sites and id(source) in sites:
        return
    address = _state_direct_address(trace, source)
    if address is None:
        return
    facts = _STATE_METADATA_FACTS.get(trace)
    if facts is None:
        facts = {}
        _STATE_METADATA_FACTS[trace] = facts
    # r67 C6: a tied slot's declared fact belongs to every canonical name sharing the
    # allocation (one allocation, one autograd bit).
    for name in _expand_state_alias_addresses(trace, {address}):
        facts.setdefault(name, {})[fact_name] = bool(fact_value)


def host_escape_state_metadata_facts(trace: Any) -> dict[str, dict[str, bool]]:
    """Return the per-state-name DECLARED-STATE metadata facts witnessed for one trace."""

    facts = _STATE_METADATA_FACTS.get(trace)
    if not facts:
        return {}
    return {name: dict(values) for name, values in facts.items()}


def _observe_state_property_read(trace: Any, source: torch.Tensor, name: str, value: Any) -> None:
    """Dispatch one getset-PROPERTY read on a state receiver through the mirror (r65).

    The property wrapper's state branch (the r64 gap: it had NONE): ``requires_grad`` /
    ``grad_fn`` route to the declared-fact ledger; every other property routes to the
    escape-gated read-kind ledger. All property names are autograd-family, so attribution is
    DIRECT-receiver-only throughout; a non-state receiver records nothing.
    """

    route = STATE_METADATA_MIRROR.get(name)
    if route is None:
        return
    route_kind, detail = route
    if route_kind == _STATE_ROUTE_DECLARED_FACT:
        if name in _INPUT_METADATA_PRESENCE_PROPERTY_NAMES:
            fact_value = value is not None
        else:
            fact_value = bool(value)
        _observe_state_metadata_fact(trace, source, detail, fact_value)
    elif route_kind == _STATE_ROUTE_READ_KIND:
        _observe_state_metadata_read_direct(trace, source, detail)


def _tensor_receiver_origin(trace: Any, source: torch.Tensor) -> tuple[str, Any]:
    """Classify a storage-bridge RECEIVER tensor as input-site / state-group / other (r67 C3).

    Storage-level facts (byte count, sharing, pinning) are pure functions of the BASE storage,
    invariant across every view/alias, so ANY input-storage-aliasing receiver (the leaf, a
    ``.data``/``.detach()`` alias, a derived view) attributes to the leaf site, and any
    state-storage-aliasing receiver attributes to the COMPLETE r37 alias group. An unrelated
    receiver (a genuine activation) is ``("other", None)`` -- its storage geometry is
    re-derived by the replayed DAG, so reads on it record nothing.
    """

    sites = trace._runnable.input_tensor_sites
    if sites:
        site = sites.get(id(source))
        if site is not None:
            return ("input", site)
        kind, aliased_site, _leaf_conj_neg = _classify_input_storage_alias(trace, source)
        if kind is not None:
            return ("input", aliased_site)
    addresses = _state_derived_addresses(trace, source)
    if addresses:
        return ("state", frozenset(_expand_state_alias_addresses(trace, addresses)))
    return ("other", None)


def _register_storage_handle_origin(
    state: _WitnessState, storage: Any, origin: tuple[str, Any]
) -> None:
    """Register one storage handle (and a typed handle's untyped backing) in the origin map."""

    origins = state.storage_origins
    if origins is None or storage is None:
        return
    for handle in (storage, getattr(storage, "_untyped_storage", None)):
        if handle is None:
            continue
        try:
            origins.register(handle, origin)
        except TypeError:
            # Unhashable/unweakrefable exotic handle: the pointer fallback still resolves
            # it, and an unresolvable accessor fails closed -- never silently unrecorded.
            continue


def _register_storage_origin(state: _WitnessState, source: torch.Tensor, storage: Any) -> None:
    """Attribute one bridge-returned storage handle at ACQUISITION time (r67 C3/C6).

    Acquisition records ORIGIN (+ the caller's existing writeback watch) ONLY -- no read
    kind, no geometry fact, no input fact: a discarded handle is not an observation
    (corr1-4). The actual accessor call on the handle records through
    ``STORAGE_METADATA_ACCESSOR_DISPOSITIONS``.
    """

    with _state.pause_logging(), internal_scalar_read():
        _register_storage_handle_origin(
            state, storage, _tensor_receiver_origin(state.trace, source)
        )


def _lazy_storage_state_ptr_names(state: _WitnessState) -> dict[int, frozenset[str]]:
    """Build (once per forward) the ptr -> full state-name-set fallback index (r67 C3).

    Covers handles acquired BEFORE the accessor wrappers armed (a pre-forward
    ``model.w.untyped_storage()`` held by the caller): live registered param/buffer storage
    pointers are stable for the forward's duration (the model holds them), so pointer
    identity is a sound attribution key here; a miss falls through to the input pointer
    index and then fails closed as unattributable.
    """

    cached = state.storage_state_ptr_names
    if cached is not None:
        return cached
    trace = state.trace
    merged: dict[int, set[str]] = {}
    for attribute in ("_param_storage_addresses", "_buffer_storage_addresses"):
        table = getattr(trace, attribute, None)
        if isinstance(table, dict):
            for ptr, address in table.items():
                try:
                    merged.setdefault(int(ptr), set()).add(str(address))
                except (TypeError, ValueError):
                    continue
    index = {
        ptr: frozenset(_expand_state_alias_addresses(trace, names)) for ptr, names in merged.items()
    }
    state.storage_state_ptr_names = index
    return index


def _resolve_storage_origin(state: _WitnessState, storage: Any) -> tuple[str, Any] | None:
    """Resolve a storage RECEIVER to its origin, or ``None`` when unattributable (r67 C3).

    Ladder: the capture-scoped weak origin map (the handle itself, then a typed handle's
    untyped backing), then the pointer fallback (state index, input index). ``None`` means
    an owner-thread accessor on a storage TorchLens cannot attribute -- the caller MUST
    fail closed (observer uncertainty), never record nothing.
    """

    origins = state.storage_origins
    if origins is not None:
        for handle in (storage, getattr(storage, "_untyped_storage", None)):
            if handle is None:
                continue
            try:
                origin = origins.get(handle)
            except TypeError:
                origin = None
            if origin is not None:
                return origin
    with _state.pause_logging(), internal_scalar_read():
        try:
            backing = (
                storage
                if isinstance(storage, torch.UntypedStorage)
                else getattr(storage, "_untyped_storage", None)
            )
            ptr = int(_ORIG_UNTYPED_STORAGE_DATA_PTR(backing)) if backing is not None else None
        except (RuntimeError, TypeError, AttributeError, NotImplementedError):
            ptr = None
        if not ptr:
            return None
        names = _lazy_storage_state_ptr_names(state).get(ptr)
        if names:
            return ("state", names)
        input_sites = _RUNNABLE_INPUT_STORAGE_SITES.get(state.trace)
        if input_sites:
            candidates = input_sites.get(ptr)
            if candidates:
                return ("input", candidates[0][0])
    return None


def _record_input_storage_nbytes(trace: Any, site: Any, storage: Any) -> None:
    """Record the BASE-storage byte-count fact for a resolved input site at ACTUAL read time.

    ``x.untyped_storage().nbytes()`` / ``.size()`` / ``len(...)`` return the input's BASE
    storage byte count, which the shape+dtype input contract does NOT pin (r29-C1 F4): a
    same-shape input that is a slice of a larger buffer differs from a freshly-allocated
    contiguous twin. The recorded fact is ALWAYS the base untyped byte count (a typed
    handle's element count is derived from the same base), re-checked against the RAW
    runtime leaf. An unreadable base count fails closed (opaque ceiling), never records a
    wrong literal.
    """

    with _state.pause_logging(), internal_scalar_read():
        try:
            backing = (
                storage
                if isinstance(storage, torch.UntypedStorage)
                else getattr(storage, "_untyped_storage", None)
            )
            nbytes = int(_ORIG_UNTYPED_STORAGE_NBYTES(backing)) if backing is not None else None
        except (RuntimeError, TypeError, AttributeError, NotImplementedError, ValueError):
            nbytes = None
    if nbytes is None:
        _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE.add(trace)
        return
    _record_input_metadata_read_at_site(trace, site, "storage_nbytes", nbytes)


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from ._completeness_boundaries import (  # noqa: E402
    _classify_input_storage_alias,
    _input_base_tensor,
    _record_input_metadata_read_at_site,
)
from ._completeness_escape_state import (  # noqa: E402
    _raw_storage_ptr_no_observe,
)
from ._completeness_origins import (  # noqa: E402
    _escape_storage_ptr,
    _operand_leaf_origins,
    _param_derived_addresses,
    internal_scalar_read,
)
from .completeness_witness import (  # noqa: E402
    _CAPTURED_STORAGE_PTRS,
    _HOST_ESCAPE_STATE_METADATA_OBSERVATIONS,
    _HOST_ESCAPE_STATE_METADATA_READS,
    _HOST_ESCAPE_STATE_SOURCE_NAMES,
    _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE,
    _INPUT_METADATA_PRESENCE_PROPERTY_NAMES,
    _INPUT_METADATA_VIEW_READ,
    _LAYOUT_ANCESTRY_CLEAN,
    _ORIG_UNTYPED_STORAGE_DATA_PTR,
    _ORIG_UNTYPED_STORAGE_NBYTES,
    _ORIGIN_LABEL_PREFIX,
    _ORIGIN_RNG,
    _ORIGIN_UNINIT,
    _ORIGIN_UNKNOWN,
    _RUNNABLE_INPUT_STORAGE_SITES,
    _STATE_METADATA_FACTS,
    _STATE_ROUTE_DECLARED_FACT,
    _STATE_ROUTE_READ_KIND,
    _STORAGE_REBIND_BARRIER_LABELS,
    INPUT_DERIVED_LAYOUT_FACT_NAME,
    STATE_METADATA_MIRROR,
)
