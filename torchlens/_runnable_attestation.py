"""Numeric attestation and nondeterminism checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from ._runnable import (
    ActivationPayloadLayerDescriptor,
    ActivationPayloadMember,
    ControlWitnessKind,
    RunnableCallDescriptor,
    SparseRunDescriptor,
    TensorSlotRole,
    is_mode_sensitive_qualname,
)
from ._utils.rng import (
    deterministic_fill_governs,
    qualname_is_uninit_growth_resize,
    qualname_is_uninit_size_gated_alloc,
    qualname_is_uninit_total_writer,
    qualname_is_uninitialized_alloc,
    uninit_new_call_is_size_form,
)

if TYPE_CHECKING:
    from ._runnable_execution import (
        _HOST_RNG_SOURCE_KIND,
        _MODULE_TRAINING_MODE_SITE_PREFIX,
        _SEEDED_SOURCE_KIND,
        _UNINIT_SANITIZER_NAMESPACES,
        _UNINIT_SOURCE_KIND,
        _call_consumes_seeded_rng,
        _decode_literal,
        _tensor_derived_scalar_witness_slot_ids,
        _within_layout_reduction_tolerance,
    )

__all__ = (
    "_is_benign_downstream_nonreproducible",
    "_has_journaled_buffer_activation_member",
    "_has_out_mutated_activation_member",
    "_raw_activation_slot_ids",
    "_descriptor_has_seeded_rng",
    "_descriptor_has_nondeterministic_rng",
    "_decoded_positional_literals",
    "_nondeterministic_value_sources",
    "_uninit_taint_reaches",
    "_control_witness_source_slot_ids",
    "_declared_nondeterministic_sources",
    "_is_mode_sensitive_qualname",
    "_descriptor_has_mode_sensitive_op",
    "_descriptor_declares_training_mode",
    "_mode_sensitive_op_unwitnessed",
)


def _is_benign_downstream_nonreproducible(
    descriptor: SparseRunDescriptor,
    member: ActivationPayloadMember,
    recomputed: Any,
    archived_value: Any,
    *,
    benign_layout_slot_ids: set[str],
) -> bool:
    """Return whether a mismatch is tight-ULP fallout from a benign BLAS slot.

    Parameters
    ----------
    descriptor:
        Sparse descriptor naming slot producers and tensor arguments.
    member:
        Archived activation member under attestation.
    recomputed:
        Fresh replay value for the slot.
    archived_value:
        Capture-time archived value for the slot (byte-verified intact by caller).
    benign_layout_slot_ids:
        Slots already proven to be benign layout/reduction-order mismatches.

    Returns
    -------
    bool
        Whether this member is fed by a previously proven benign slot and remains
        within the same tight ULP band.
    """

    if not benign_layout_slot_ids:
        return False
    if not (isinstance(recomputed, torch.Tensor) and isinstance(archived_value, torch.Tensor)):
        return False
    if not _within_layout_reduction_tolerance(recomputed, archived_value):
        return False
    if member.slot_id in benign_layout_slot_ids:
        return True
    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    slot = slots.get(member.slot_id)
    if slot is not None and (
        slot.producer_slot_id in benign_layout_slot_ids or slot.version_of in benign_layout_slot_ids
    ):
        return True
    for call in descriptor.calls:
        if member.slot_id not in call.output_slot_ids:
            continue
        return any(argument.slot_id in benign_layout_slot_ids for argument in call.tensor_arguments)
    return False


def _has_journaled_buffer_activation_member(
    descriptor: SparseRunDescriptor,
    layer: Any,
    members: Sequence[ActivationPayloadMember],
) -> bool:
    """Return whether selected activation payloads include JOURNALED buffer slots.

    A buffer state entry with multiple selected activation members is journaled only when at
    least one member's ARCHIVED bytes DIFFER from the capture-time state digest -- i.e. the
    buffer was actually WRITTEN mid-forward (a norm layer updating its running stats). When
    every repeated member equals the capture state digest, the buffer is a STABLE read-only
    source (running stats under eval, or an unwritten buffer read at several points), which is
    fully byte-attestable and must NOT be suppressed (r29-C4, codex-F2). When the capture state
    digest is unavailable for a repeated buffer name, fail closed (treat as journaled) rather
    than risk a false mismatch.

    Parameters
    ----------
    descriptor:
        Sparse descriptor declaring tensor slot roles.
    layer:
        Activation payload layer descriptor carrying ``capture_state_digests``.
    members:
        Raw activation payload members selected for byte attestation.

    Returns
    -------
    bool
        ``True`` when a repeated same-state buffer entry was actually journaled (written).
    """

    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    state_digests = {
        item.state_dict_name: item.byte_digest
        for item in getattr(layer, "capture_state_digests", ()) or ()
    }
    members_by_state: dict[str, list[ActivationPayloadMember]] = {}
    for member in members:
        slot = slots.get(member.slot_id)
        if slot is None or slot.role is not TensorSlotRole.BUFFER:
            continue
        binding = slot.state_binding
        if binding is None:
            continue
        members_by_state.setdefault(binding.state_dict_name, []).append(member)
    for state_name, state_members in members_by_state.items():
        if len(state_members) <= 1:
            continue
        if state_name not in state_digests:
            return True
        if any(member.byte_digest != state_digests[state_name] for member in state_members):
            return True
    return False


def _has_out_mutated_activation_member(
    descriptor: SparseRunDescriptor,
    members: Sequence[ActivationPayloadMember],
) -> bool:
    """Return whether activation attestation includes an ``out=`` destination.

    Parameters
    ----------
    descriptor:
        Sparse recipe whose mutating calls are being inspected.
    members:
        Archived raw activation records selected at capture time.

    Returns
    -------
    bool
        Whether any archived slot is later written through an explicit ``out=``
        argument and therefore has no stable pre-write byte contract.
    """

    out_slots = {
        argument.slot_id
        for call in descriptor.calls
        if call.is_inplace
        for argument in call.tensor_arguments
        if argument.argument_path == ("kwargs", "out")
    }
    return any(member.slot_id in out_slots for member in members)


def _raw_activation_slot_ids(descriptor: SparseRunDescriptor) -> frozenset[str]:
    """Return raw selected-activation slots that require production snapshots.

    Parameters
    ----------
    descriptor:
        Sparse runnable descriptor carrying optional activation archive metadata.

    Returns
    -------
    frozenset[str]
        Slot IDs whose raw output values must be copied at production time.
    """

    layer = descriptor.payload_layers.activations
    if not isinstance(layer, ActivationPayloadLayerDescriptor):
        return frozenset()
    return frozenset(member.slot_id for member in layer.members if member.field == "out")


def _descriptor_has_seeded_rng(descriptor: SparseRunDescriptor) -> bool:
    """Return whether replay actually consumes non-reproducible seeded RNG.

    A captured RNG source slot always taints the replay. For seeded-RNG ATen
    ops the answer is keyed off *actual RNG consumption*, not the op name: a
    ``dropout`` family call in eval mode (``training=False``) or with ``p == 0``
    draws nothing from the generator and replays byte-exact, so it must NOT be
    treated as seeded (that over-triggered ``not_applicable`` on eval-mode
    transformers). Every other seeded-RNG op (``rand``/``randn``/``bernoulli``/
    ``multinomial`` and a genuinely training dropout with ``p > 0``) still taints.

    Parameters
    ----------
    descriptor:
        Sparse runnable descriptor whose registry entries and calls identify
        replayed PyTorch operations.

    Returns
    -------
    bool
        Whether replay includes a captured RNG source or a call that actually
        draws from a PyTorch seeded generator during replay.
    """

    if any(slot.role is TensorSlotRole.RNG_SOURCE for slot in descriptor.tensor_slots):
        return True
    registry = {entry.registry_id: entry for entry in descriptor.callable_registry}
    return any(_call_consumes_seeded_rng(call, registry) for call in descriptor.calls)


def _descriptor_has_nondeterministic_rng(descriptor: SparseRunDescriptor) -> bool:
    """Return whether declared nondeterminism reaches the byte-exact attestation claim.

    Seeded-RNG consumption anywhere in the replay keeps its historical
    presence-based ineligibility (``_descriptor_has_seeded_rng``). r53 hon_2
    additionally consults the uninitialized-memory value-source classifier
    (``_nondeterministic_value_sources``): surviving ``uninitialized_alloc``
    taint that reaches an archived-activation member slot or a model output
    slot makes the artifact declared-nondeterministic, so numeric attestation
    is ``not_applicable`` UPFRONT -- never the r52 contradictory
    ``replay_value_mismatch`` raise on bytes the recorded computation
    never determined. Fully sanitized flows (``empty`` then a total write) and
    dead tainted intermediates keep the run attestation-eligible, and a byte
    mismatch on an UNTAINTED slot still raises (the tripwire is untouched).
    """

    if _descriptor_has_seeded_rng(descriptor):
        return True
    taint = _nondeterministic_value_sources(descriptor)
    if not taint:
        return False
    reach: set[str] = {
        slot.slot_id
        for slot in descriptor.tensor_slots
        if slot.role is TensorSlotRole.OUTPUT or slot.output_path is not None
    }
    layer = descriptor.payload_layers.activations
    if isinstance(layer, ActivationPayloadLayerDescriptor):
        reach.update(member.slot_id for member in layer.members)
    return _uninit_taint_reaches(taint, reach)


def _decoded_positional_literals(call: RunnableCallDescriptor) -> tuple[Any, ...]:
    """Return a call's positional LITERAL arguments, decoded, in index order.

    Tensor operands (a method receiver at ``args[0]``, other tensor arguments)
    are naturally excluded -- they are ``tensor_arguments``, not literals -- so
    the result is exactly the size-argument tuple ``uninit_new_call_is_size_form``
    classifies for ``Tensor.new(...)``. Index gaps are collapsed (order is
    preserved), which is all the arg-form predicate needs.
    """

    indexed: dict[int, Any] = {}
    for literal in call.literal_arguments:
        path = tuple(literal.argument_path)
        if len(path) == 2 and path[0] == "args" and isinstance(path[1], int):
            indexed[path[1]] = _decode_literal(literal.value)
    return tuple(indexed[index] for index in sorted(indexed))


def _nondeterministic_value_sources(
    descriptor: SparseRunDescriptor,
) -> dict[str, frozenset[str]]:
    """Classify every tensor slot's declared nondeterministic VALUE sources (r53 hon_2).

    THE single load-side choke point for nondeterministic-value recognition: no
    call site outside this function may re-derive nondeterminism from qualnames
    (source-scan meta-test). Sources are the uninitialized-memory op family
    (``empty`` factories; a GROWING ``resize_``/``resize_as_`` decided from the
    recorded receiver-vs-product element counts) and seeded-RNG products
    (``RNG_SOURCE`` slots plus actually-consuming seeded calls). Taint
    propagates along the recorded call schedule: a product inherits the union
    of its tensor-argument taints, EXCEPT at a total write -- an ``out=``
    destination or an in-place ``copy_``/``zero_``/``fill_``/RNG-fill receiver
    -- where the destination's prior taint is dropped and the written version
    carries only the value-source operands' taint (exact value semantics; the
    RNG fills replace uninit taint with the seeded classification through the
    ordinary seeded-consumption rule). A partial or unprovable in-place write
    propagates (fail closed; r35 unknown-alias precedent). The whole family is
    clean when the recorded ambient context proves deterministic fill
    (``deterministic_algorithms`` true and ``fill_uninitialized_memory`` not
    false) and per-product when the product has zero elements.

    Returns
    -------
    dict[str, frozenset[str]]
        Mapping from slot id to its non-empty declared source-kind set; slots
        with no declared nondeterministic source are absent.
    """

    registry = {entry.registry_id: entry for entry in descriptor.callable_registry}
    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    ambient = descriptor.ambient_context
    fill_deterministic = deterministic_fill_governs(
        ambient.deterministic_algorithms, ambient.fill_uninitialized_memory
    )

    def _slot_numel(slot_id: str) -> int | None:
        """Element count of ``slot_id`` from its recorded shape, or ``None`` if unknown.

        ``None`` means the slot is absent from the descriptor, never an empty
        tensor: a zero-element slot reports ``0``.
        """

        slot = slots.get(slot_id)
        if slot is None:
            return None
        numel = 1
        for dim in slot.shape:
            numel *= int(dim)
        return numel

    taint: dict[str, frozenset[str]] = {
        slot.slot_id: frozenset({_SEEDED_SOURCE_KIND})
        for slot in descriptor.tensor_slots
        if slot.role is TensorSlotRole.RNG_SOURCE
    }
    for call in descriptor.calls:
        entry = registry.get(call.registry_id)
        namespace = entry.key.namespace if entry is not None else None
        qualname = entry.key.qualname if entry is not None else None
        consumes_seeded = _call_consumes_seeded_rng(call, registry)
        # Total-write destination: exact first-level receiver/out leaf only; a
        # nested or absent destination path never sanitizes, and an ``out=``
        # kwarg sanitizes only for trusted torch-namespace callables whose
        # ``out=`` convention IS a total overwrite (fail closed for customs).
        dest_slot_id: str | None = None
        if namespace in _UNINIT_SANITIZER_NAMESPACES:
            for ref in call.tensor_arguments:
                if tuple(ref.argument_path) == ("kwargs", "out"):
                    dest_slot_id = ref.slot_id
                    break
            # r55 hon_2: an ``out=`` that ALIASES a value operand
            # (``torch.add(a, x, out=a)``) computes a result that READS the
            # destination's prior (uninitialized) bytes -- it is NOT a pure total
            # overwrite. "All elements written" is not "result independent of
            # prior content". Preserve the destination's taint (fail closed, r35
            # unknown-alias precedent) by declining to treat it as a total-write
            # destination; only a genuine total write to a destination that is
            # not itself read (below, or the ``copy_``/``zero_``/``fill_`` receiver
            # branch) drops taint.
            if dest_slot_id is not None and any(
                ref.slot_id == dest_slot_id and tuple(ref.argument_path) != ("kwargs", "out")
                for ref in call.tensor_arguments
            ):
                dest_slot_id = None
        if dest_slot_id is None and qualname_is_uninit_total_writer(namespace, qualname):
            for ref in call.tensor_arguments:
                if tuple(ref.argument_path) == ("args", 0):
                    dest_slot_id = ref.slot_id
                    break
        operand_union: set[str] = set()
        sanitized_union: set[str] = set()
        for ref in call.tensor_arguments:
            ref_taint = taint.get(ref.slot_id)
            if not ref_taint:
                continue
            operand_union |= ref_taint
            if ref.slot_id != dest_slot_id:
                sanitized_union |= ref_taint
        produced = sanitized_union if dest_slot_id is not None else operand_union
        if consumes_seeded:
            produced = produced | {_SEEDED_SOURCE_KIND}
        is_uninit_factory = qualname_is_uninitialized_alloc(namespace, qualname)
        # r55 hon_1 consumer: ``Tensor.new`` is uninitialized-memory ONLY in its
        # SIZE-argument form (``new(*sizes)`` redispatches ``aten.empty``); the
        # DATA form (``new([values])``/``new(tensor)``) is a deterministic copy
        # constructor. Decode the call's positional literal sizes and consult
        # W1's arg-form predicate; the data form (``False``) stays clean, and an
        # undecidable form (``None`` -- a plain int-tuple, since the portable
        # grammar erases ``torch.Size``) fails CLOSED to tainted (grow-gate
        # posture). ``new`` has no aten spelling, so this Python-method surface
        # is the only place its uninit taint is recognized.
        is_size_gated_uninit = False
        if qualname_is_uninit_size_gated_alloc(namespace, qualname):
            try:
                new_sizes = _decoded_positional_literals(call)
            except Exception:
                new_sizes = None
            if new_sizes is None:
                is_size_gated_uninit = True  # undecodable form -> fail closed
            else:
                is_size_gated_uninit = uninit_new_call_is_size_form(new_sizes) is not False
        is_growth_resize = qualname_is_uninit_growth_resize(namespace, qualname)
        receiver_numel: int | None = None
        if is_growth_resize:
            receiver_numel = next(
                (
                    _slot_numel(ref.slot_id)
                    for ref in call.tensor_arguments
                    if tuple(ref.argument_path) == ("args", 0)
                ),
                None,
            )
        for out_slot_id in call.output_slot_ids:
            slot_taint = set(produced)
            if is_uninit_factory or is_size_gated_uninit:
                # A factory product's value derives from NO operand: it is pure
                # allocator garbage (tainted) unless deterministically filled
                # or empty of elements (both value-constant).
                out_numel = _slot_numel(out_slot_id)
                slot_taint = set()
                if not fill_deterministic and (out_numel is None or out_numel > 0):
                    slot_taint = {_UNINIT_SOURCE_KIND}
            elif is_growth_resize and not fill_deterministic:
                # Shrink/same-size preserves the element prefix (probed clean);
                # a GROW exposes stale allocator tail bytes. An undecidable
                # grow fact fails closed to tainted.
                out_numel = _slot_numel(out_slot_id)
                if receiver_numel is None or out_numel is None or out_numel > receiver_numel:
                    slot_taint.add(_UNINIT_SOURCE_KIND)
            if slot_taint:
                taint[out_slot_id] = frozenset(slot_taint)
            else:
                taint.pop(out_slot_id, None)
    # Value-identity closure: an OUTPUT (or other alias) slot that no call
    # produces is wired to its source through ``producer_slot_id``/``version_of``
    # -- same value, same taint. ONLY call-orphan slots inherit here: a slot a
    # call produces already carries that call's (possibly deliberately
    # SANITIZED) taint, and re-unioning its ``version_of`` base would resurrect
    # exactly the taint a total write removed. Iterate to fixpoint (chains are
    # short and acyclic by construction).
    call_produced: set[str] = {
        out_slot_id for call in descriptor.calls for out_slot_id in call.output_slot_ids
    }
    changed = True
    while changed:
        changed = False
        for slot in descriptor.tensor_slots:
            if slot.slot_id in call_produced:
                continue
            inherited: set[str] = set(taint.get(slot.slot_id, frozenset()))
            before = len(inherited)
            for link in (slot.producer_slot_id, slot.version_of):
                if link is not None:
                    inherited |= taint.get(link, frozenset())
            if len(inherited) > before:
                taint[slot.slot_id] = frozenset(inherited)
                changed = True
    return taint


def _uninit_taint_reaches(taint: Mapping[str, frozenset[str]], slot_ids: Iterable[str]) -> bool:
    """Return whether surviving uninitialized-memory taint reaches any listed slot."""

    return any(_UNINIT_SOURCE_KIND in taint.get(slot_id, frozenset()) for slot_id in slot_ids)


def _control_witness_source_slot_ids(descriptor: SparseRunDescriptor) -> frozenset[str]:
    """Return every slot whose VALUE decided a recorded control fact.

    The set is the tensor->host escape-source witnesses
    (``TENSOR_DERIVED_SCALAR_LITERAL`` site labels are their source slot ids)
    plus the predicate-producing call outputs of ``SCALAR_BOOL`` /
    ``LOOP_PREDICATE`` witnesses. ``CONDITIONAL_ARM_ENTRY`` witnesses carry
    ``call_id=None`` (they record arm-edge structure, not a predicate source)
    and contribute nothing here -- a tainted value computed INSIDE an arm never
    ceilings the branch decision itself (over-trigger guard).
    """

    sources: set[str] = set(_tensor_derived_scalar_witness_slot_ids(descriptor))
    calls_by_id = {call.call_id: call for call in descriptor.calls}
    for witness in descriptor.control_witnesses:
        if witness.kind not in {
            ControlWitnessKind.SCALAR_BOOL,
            ControlWitnessKind.LOOP_PREDICATE,
            ControlWitnessKind.CONDITIONAL_ARM_ENTRY,
        }:
            continue
        if witness.call_id is None:
            continue
        call = calls_by_id.get(witness.call_id)
        if call is not None:
            sources.update(call.output_slot_ids)
    return frozenset(sources)


def _declared_nondeterministic_sources(
    descriptor: SparseRunDescriptor,
    taint: Mapping[str, frozenset[str]],
) -> frozenset[str]:
    """Derive the run's declared nondeterministic sources (closed vocabulary, r53 F4).

    ``seeded_rng`` and ``host_rng`` are the descriptor's existing presence
    facts; ``uninitialized_alloc`` is declared exactly when surviving family
    taint reaches an observable surface (a model output slot, an archived
    activation member, or a control-fact source) -- a fully sanitized scratch
    or a dead tainted intermediate declares nothing.
    """

    sources: set[str] = set()
    if _descriptor_has_seeded_rng(descriptor):
        sources.add(_SEEDED_SOURCE_KIND)
    if descriptor.rng_profile.host_rng_consumed:
        sources.add(_HOST_RNG_SOURCE_KIND)
    if taint:
        reach: set[str] = {
            slot.slot_id
            for slot in descriptor.tensor_slots
            if slot.role is TensorSlotRole.OUTPUT or slot.output_path is not None
        }
        layer = descriptor.payload_layers.activations
        if isinstance(layer, ActivationPayloadLayerDescriptor):
            reach.update(member.slot_id for member in layer.members)
        reach.update(_control_witness_source_slot_ids(descriptor))
        if _uninit_taint_reaches(taint, reach):
            sources.add(_UNINIT_SOURCE_KIND)
    return frozenset(sources)


def _is_mode_sensitive_qualname(qualname: str | None) -> bool:
    """Delegate to the ONE shared mode-sensitivity classifier (r71 A).

    Single-sourced in ``torchlens._runnable.is_mode_sensitive_qualname`` so the
    parse-time ``module_training_mode`` domain derivation and this runtime belt can
    never disagree on which ops are train/eval mode-sensitive.
    """

    return is_mode_sensitive_qualname(qualname)


def _descriptor_has_mode_sensitive_op(descriptor: SparseRunDescriptor) -> bool:
    """Return whether the taken path contains a train/eval mode-sensitive op."""

    registry = {entry.registry_id: entry for entry in descriptor.callable_registry}
    for call in descriptor.calls:
        entry = registry.get(call.registry_id)
        if entry is not None and _is_mode_sensitive_qualname(entry.key.qualname):
            return True
    return False


def _descriptor_declares_training_mode(descriptor: SparseRunDescriptor) -> bool:
    """Return whether the descriptor declares the capture-time per-module train/eval mode."""

    return any(
        witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT
        and witness.site_label.startswith(_MODULE_TRAINING_MODE_SITE_PREFIX)
        for witness in descriptor.control_witnesses
    )


def _mode_sensitive_op_unwitnessed(descriptor: SparseRunDescriptor) -> bool:
    """Return whether a mode-sensitive op replays without a declared train/eval mode.

    ``self.training`` is declared state the VERIFIED oracle (a fresh instance in the
    captured mode on the given inputs) reproduces. A BatchNorm / InstanceNorm op in the
    taken path whose capture-time mode is NOT recorded has no witnessed proof of the mode
    it corresponds to (eval running-stats vs train batch-stats), so the honest ceiling is
    UNVERIFIABLE. Captures with no mode-sensitive op, or that declare the mode (every new
    intervention-ready capture does), stay VERIFIED -- no over-trigger.
    """

    return _descriptor_has_mode_sensitive_op(descriptor) and not _descriptor_declares_training_mode(
        descriptor
    )
