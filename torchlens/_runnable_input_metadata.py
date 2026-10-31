"""Input structure, literal, and metadata witness helpers."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

import torch

from torchlens.runnable import ContractCheck, RunnableDiagnostic

from . import _state
from ._runnable_state import (
    RunResourceCeiling,
    _guarded_defensive_materialize,
)
from .errors import (
    PathDivergenceError,
)
from .runnable import (
    PathFaithfulness,
    RunnableErrorCode,
    SparseRunDescriptor,
    TensorSlotDescriptor,
    TensorSlotRole,
)
from .utils._torch_compat import tensor_has_named_dims
from .utils.alias_footprint import touched_bytes_relation

if TYPE_CHECKING:
    from ._runnable_execution import (
        _VIEW_OP_QUALNAMES,
        _contract_check,
        _fact_path_tuple_or_none,
        _input_alias_topology_checks,
        _input_error,
        _input_metadata_contract_checks,
        _input_nontensor_tree_contract_checks,
        _input_site_value,
        _input_tree_contract_checks,
        _model_input_arity_positions,
        _model_input_literal_facts,
        _runtime_top_level_positions,
        _value_at_path,
    )

__all__ = (
    "_top_level_input_site_contract_checks",
    "_literal_leaf_equal",
    "_input_literal_contract_checks",
    "_bind_runtime_inputs",
    "_runtime_mirror_clone",
    "_model_input_version_closure",
    "_model_input_storage_closure",
    "_touched_bytes_relation",
)


def _top_level_input_site_contract_checks(
    descriptor: SparseRunDescriptor, inputs: Any, positions: set[Any]
) -> tuple[ContractCheck, ...]:
    """Reject a runtime call carrying MORE top-level input sites than capture (r42 corr1_1).

    The sparse descriptor records CONCRETE boundary positions and encodes no open-ended variadic
    contract: a capture of a ``*args`` / ``**kwargs`` model that took two args recorded exactly
    two concrete sites, so a third runtime arg (or an unrecorded keyword) is outside the recorded
    taken path -- ignored input, not valid replay, and MUST NOT report VERIFIED. This mirrors the
    tensor/non-tensor leaf-set contracts at the TOP level: any extra positional index or keyword
    diverges. Missing sites stay the existing tree/binder contracts' domain.
    """

    expected = {p for p in positions if isinstance(p, tuple) and len(p) == 2}
    if not expected:
        return ()
    actual = _runtime_top_level_positions(inputs, positions)
    if actual is None:
        return ()
    extra = actual - expected
    if not extra:
        return ()
    return (
        _contract_check(
            "input_arity_extra",
            False,
            RunnableErrorCode.INPUT_ARITY_EXTRA,
            "Runtime call carries top-level model-input sites not present at capture; the "
            "recorded taken path is not valid for extra positional/keyword arguments (including "
            "for Python variadic signatures whose recorded taken path is a finite concrete site "
            "set).",
            details=(
                ("expected_positions", repr(sorted(expected, key=repr))),
                ("actual_positions", repr(sorted(actual, key=repr))),
                ("extra_positions", repr(sorted(extra, key=repr))),
            ),
        ),
    )


def _literal_leaf_equal(recorded: Any, runtime: Any) -> bool:
    """Type-strict, bit-exact equality for recorded vs runtime non-tensor input leaves.

    ``bool`` is a subclass of ``int`` and floats are distinct from ints, so a
    changed control input like ``True`` -> ``1`` or ``2`` -> ``2.0`` must count as
    a divergence rather than silently comparing equal.

    Floats are compared by their IEEE-754 bit pattern, never ``==``. Ordinary
    ``==`` is dishonest for control witnesses at two values: ``-0.0 == +0.0`` is
    ``True`` (a changed sign bit that steers ``math.copysign``/``1/x`` control flow
    would falsely pass), while ``nan == nan`` is ``False`` (an unchanged ``nan``
    would falsely diverge). Bit-pattern identity makes ``-0.0`` differ from
    ``+0.0`` and a ``nan`` equal to a ``nan`` with the same bits.
    """

    from torchlens._input_walk import classify_scalar

    # r71 B: a ``slice`` is a COMPOSITE literal leaf. ``slice.__eq__`` compares
    # component VALUES only, so ``slice(5) == slice(MyIntEnum.FAST)`` is True and a
    # type-steered branch would falsely VERIFY. Compare component-by-component through
    # the SAME scalar type/value rules (never ``slice.__eq__`` as authority).
    if isinstance(recorded, slice) or isinstance(runtime, slice):
        if not (isinstance(recorded, slice) and isinstance(runtime, slice)):
            return False
        return all(
            _literal_leaf_equal(getattr(recorded, field), getattr(runtime, field))
            for field in ("start", "stop", "step")
        )

    recorded_kind, recorded_norm = classify_scalar(recorded)
    runtime_kind, runtime_norm = classify_scalar(runtime)
    # r69 B: a semantic-typed scalar (enum member, builtin/np-scalar subclass) never
    # collapses into the builtin atom families -- type identity is application
    # semantics, so a same-value cross-class replacement DIVERGES in both directions
    # (a semantic capture cannot exist in a saved artifact: producer refuses typed).
    if recorded_kind == "semantic" or runtime_kind == "semantic":
        return False
    # RATIFIED stock-wrapper VALUE lane: exact stock NumPy numeric/bool wrappers
    # normalize to their builtin atoms (``np.float64(2.0) <-> 2.0`` verifies).
    if recorded_kind == "numpy":
        recorded = recorded_norm
    if runtime_kind == "numpy":
        runtime = runtime_norm
    if isinstance(recorded, bool) or isinstance(runtime, bool):
        return isinstance(recorded, bool) and isinstance(runtime, bool) and recorded == runtime
    # Float family (exact builtins after wrapper normalization). ``int`` stays
    # distinct from ``float`` (``2`` vs ``2.0`` must diverge). Compare by IEEE-754
    # bit pattern of the float VALUE so the r4 signed-zero/NaN honesty is preserved.
    rec_is_float = isinstance(recorded, float)
    run_is_float = isinstance(runtime, float)
    if rec_is_float or run_is_float:
        if not (rec_is_float and run_is_float):
            return False
        return struct.pack(">d", float(recorded)) == struct.pack(">d", float(runtime))
    # Integer family, excluding bool. ``int`` vs a non-int type diverges; two
    # integers compare by value.
    rec_is_int = isinstance(recorded, int)
    run_is_int = isinstance(runtime, int)
    if rec_is_int or run_is_int:
        if not (rec_is_int and run_is_int):
            return False
        return int(recorded) == int(runtime)
    if type(recorded) is not type(runtime):
        return False
    return bool(recorded == runtime)


def _input_literal_contract_checks(
    descriptor: SparseRunDescriptor,
    inputs: Any,
    positions: set[Any],
) -> tuple[ContractCheck, ...]:
    """Compare runtime non-tensor input leaves with recorded capture-time values.

    A differing non-tensor leaf means the recorded taken-path DAG may be wrong
    for this input, so the check fails and the run diverges instead of silently
    replaying a numerically wrong result.
    """

    checks: list[ContractCheck] = []
    for witness, fact in _model_input_literal_facts(descriptor):
        raw_position = fact.get("position")
        position = tuple(raw_position) if isinstance(raw_position, (list, tuple)) else raw_position
        path = _fact_path_tuple_or_none(fact)
        if path is None:
            checks.append(
                _contract_check(
                    "input_literal_path_malformed",
                    False,
                    RunnableErrorCode.CONTEXT_FIELD_INVALID,
                    "A literal fact carries a non-sequence container path; the "
                    "fact cannot be resolved against the runtime input tree.",
                    affected_op_labels=(witness.site_label,),
                    details=(("model_site_position", repr(position)),),
                )
            )
            continue
        recorded = fact.get("value")
        if not bool(fact.get("encodable", False)):
            # Opaque capture-time leaf: not representable in the frozen literal
            # grammar, so it cannot be compared across save/load. Its position
            # still constrains arity; a value claim would be dishonest.
            continue
        try:
            root = _input_site_value(inputs, position, positions)
            runtime_leaf = _value_at_path(root, path)
            resolved = True
        except (KeyError, IndexError, TypeError, AttributeError):
            runtime_leaf = None
            resolved = False
        passed = resolved and _literal_leaf_equal(recorded, runtime_leaf)
        checks.append(
            _contract_check(
                f"input_literal:{position!r}:{path!r}",
                passed,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                "Runtime non-tensor input leaf differs from the recorded "
                "capture-time value; the recorded taken path may not be valid "
                "for this input.",
                affected_op_labels=(witness.site_label,),
                details=(
                    ("model_site_position", repr(position)),
                    ("container_path", repr(path)),
                    ("recorded_value", repr(recorded)),
                    ("runtime_value", repr(runtime_leaf) if resolved else "<unresolved>"),
                ),
            )
        )
    return tuple(checks)


def _bind_runtime_inputs(
    descriptor: SparseRunDescriptor,
    inputs: Any,
    ceiling: RunResourceCeiling,
) -> tuple[dict[str, torch.Tensor], tuple[ContractCheck, ...], bool]:
    """Bind and defensively clone public input leaves by persisted model sites.

    Returns the bound clone map, the ordered input contract checks, and the
    ``input_alias_topology_unresolved`` ceiling flag (r35 decision D). Accepted
    leaves are mirrored through the transaction ``ceiling``'s byte guard (r61
    corr_2): a tampered input slot shape plus a runtime expanded view refuses
    typed at ``clone_allocation_preflight`` instead of dying in the allocator.
    """

    input_slots = tuple(
        slot for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.MODEL_INPUT
    )
    values: dict[str, torch.Tensor] = {}
    checks: list[ContractCheck] = []
    positions = _model_input_arity_positions(descriptor)
    # r42 corr1_1: reject EXTRA top-level args/kwargs (site-set superset) BEFORE any per-site
    # tensor/literal/metadata check, so an ignored extra input can never be blessed VERIFIED.
    checks.extend(_top_level_input_site_contract_checks(descriptor, inputs, positions))
    # Torch capture DE-ALIASES model inputs (each input leaf is cloned before the forward,
    # so ``forward(a, b)`` with ``a is b`` is captured as two DISTINCT tensors). The recorded
    # DAG and activation archive therefore reflect distinct-input semantics, and each runtime
    # input slot is likewise cloned independently. A runtime call that passes ALIASED inputs
    # (same object, or distinct views sharing storage) is NOT reproducible against that
    # de-aliased capture when an in-place op mutates an input -- see
    # ``_input_alias_topology_checks``, which fails such a run closed instead of a false VERIFIED.
    raw_values: dict[str, torch.Tensor] = {}
    for slot in input_slots:
        binding = slot.input_binding
        if binding is None:
            raise _input_error(
                RunnableErrorCode.MISSING_INPUT_CONTAINER_CONTRACT,
                slot,
                "Input slot has no persisted model-site binding.",
            )
        try:
            root = _input_site_value(inputs, binding.model_site_position, positions)
            value = _value_at_path(root, binding.container_path)
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise _input_error(
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                slot,
                f"Runtime input tree does not contain the recorded input path: {exc}",
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise _input_error(
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                slot,
                f"Runtime input leaf is {type(value).__name__}, expected torch.Tensor.",
            )
        # r37 corr2-6 phase 0: EXACT-type admission precedes every dispatchable
        # property read. A tensor SUBCLASS routes ``shape``/``dtype``/``device``/
        # ``names`` through ``__torch_function__``, so reading any of them before
        # this gate would execute user subclass code and leak its raw exception in
        # place of the typed hard-precondition refusal. Diagnostics use ONLY
        # ``type(value).__qualname__`` -- never a property.
        if type(value) not in {torch.Tensor, torch.nn.Parameter}:
            subclass_check = _contract_check(
                f"input_layout:{slot.slot_id}",
                False,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                "Runtime input is a tensor SUBCLASS the sparse replay cannot "
                "faithfully reproduce; runnable capture records only plain "
                "strided torch.Tensor/Parameter leaves, so such an input fails "
                "closed (typed) before any property dispatch, under every "
                "divergence policy.",
                affected_op_labels=(slot.slot_id.removeprefix("slot:"),),
                details=(
                    ("slot_id", slot.slot_id),
                    ("tensor_class", type(value).__qualname__),
                ),
            )
            diagnostic = subclass_check.diagnostic
            raise PathDivergenceError(
                diagnostic.message
                if diagnostic is not None
                else "Unsupported tensor-subclass input.",
                code=RunnableErrorCode.INPUT_TREE_MISMATCH.value,
                path_faithfulness=PathFaithfulness.DIVERGED,
                first_mismatch=diagnostic,
                contract_check=subclass_check,
            )
        raw_values[slot.slot_id] = value
        # Phase-1 facts are exception-safe: exotic layouts (nested tensors) can
        # refuse even a ``sizes()`` read, which must surface as a failed check,
        # never a raw backend error.
        try:
            actual_shape: tuple[int, ...] | None = tuple(value.shape)
        except (RuntimeError, TypeError, NotImplementedError):
            actual_shape = None
        shape_ok = actual_shape == slot.shape
        dtype_ok = str(value.dtype) == slot.dtype
        checks.append(
            _contract_check(
                f"input_shape:{slot.slot_id}",
                shape_ok,
                RunnableErrorCode.INPUT_SHAPE_MISMATCH,
                f"Runtime input shape {actual_shape} does not match {slot.shape}.",
                affected_op_labels=(slot.slot_id.removeprefix("slot:"),),
                details=(
                    ("slot_id", slot.slot_id),
                    ("expected_shape", repr(slot.shape)),
                    ("actual_shape", repr(actual_shape)),
                ),
            )
        )
        checks.append(
            _contract_check(
                f"input_dtype:{slot.slot_id}",
                dtype_ok,
                RunnableErrorCode.INPUT_DTYPE_MISMATCH,
                f"Runtime input dtype {value.dtype} does not match {slot.dtype}.",
                affected_op_labels=(slot.slot_id.removeprefix("slot:"),),
                details=(
                    ("slot_id", slot.slot_id),
                    ("expected_dtype", slot.dtype),
                    ("actual_dtype", str(value.dtype)),
                ),
            )
        )
        # r33 F7: pin DEVICE and LAYOUT next to shape+dtype. The shape+dtype contract does NOT
        # pin either, yet the state lives on the capture DEVICE and the recorded DAG assumes a
        # STRIDED DENSE input, so a same-shape+dtype runtime input on a different device or with
        # an exotic layout (sparse/meta/nested/named) cannot be faithfully reproduced by the
        # sparse replay -- today only an INCIDENTAL torch device/layout error guards them. Device
        # TYPE is compared strictly; the INDEX only when both are concrete (a capture recorded as
        # a bare ``cuda`` has index ``None`` and must not falsely diverge a ``cuda:0`` runtime).
        device_ok = value.device.type == slot.device_type and (
            slot.device_index is None
            or value.device.index is None
            or value.device.index == slot.device_index
        )
        checks.append(
            _contract_check(
                f"input_device:{slot.slot_id}",
                device_ok,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                f"Runtime input device {value.device} does not match the capture device "
                f"{slot.device_type}"
                + (f":{slot.device_index}" if slot.device_index is not None else "")
                + "; the recorded state and DAG cannot be replayed against a different device.",
                affected_op_labels=(slot.slot_id.removeprefix("slot:"),),
                details=(
                    ("slot_id", slot.slot_id),
                    ("expected_device_type", slot.device_type),
                    ("expected_device_index", repr(slot.device_index)),
                    ("actual_device", str(value.device)),
                ),
            )
        )
        has_named_dims = tensor_has_named_dims(value)
        layout_ok = (
            value.layout == torch.strided
            and not value.is_nested
            and not bool(getattr(value, "is_meta", False))
            and not bool(getattr(value, "is_quantized", False))
            and not has_named_dims
            and type(value) in {torch.Tensor, torch.nn.Parameter}
        )
        checks.append(
            _contract_check(
                f"input_layout:{slot.slot_id}",
                layout_ok,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                "Runtime input has a non-strided/meta/nested/named/quantized/subclass "
                "layout the sparse replay cannot faithfully reproduce; runnable capture "
                "records only plain strided dense tensors, so such an input must fail "
                "closed rather than replay the recorded DAG.",
                affected_op_labels=(slot.slot_id.removeprefix("slot:"),),
                details=(
                    ("slot_id", slot.slot_id),
                    ("actual_layout", str(value.layout)),
                    ("is_nested", repr(bool(value.is_nested))),
                    ("is_meta", repr(bool(getattr(value, "is_meta", False)))),
                    ("is_quantized", repr(bool(getattr(value, "is_quantized", False)))),
                    ("named", repr(has_named_dims)),
                    ("tensor_class", type(value).__qualname__),
                ),
            )
        )
    checks.extend(_input_tree_contract_checks(descriptor, inputs))
    checks.extend(_input_literal_contract_checks(descriptor, inputs, positions))
    checks.extend(_input_metadata_contract_checks(descriptor, inputs, positions))
    checks.extend(_input_nontensor_tree_contract_checks(descriptor, inputs, positions))
    # ------------------------------------------------------------------
    # r35 I5 (corr2_6) -- CONTRACT-BEFORE-TOUCH admission choke point.
    # Everything ABOVE this comment reads only non-materializing, exception-
    # safe facts from the RAW bound leaves. Everything BELOW may clone,
    # transfer, view, digest, or otherwise materialize input bytes. Hard
    # executability preconditions (meta/sparse/nested/named/quantized/
    # subclass layouts) are enforced HERE, before any byte operation, and
    # raise regardless of the divergence policy: ``return_diverged`` may
    # continue only with an EXECUTABLE poisoned input, never past a hard
    # precondition. Any future preamble addition that touches input bytes
    # MUST be placed below this point.
    # ------------------------------------------------------------------
    hard_failure = next(
        (check for check in checks if not check.passed and check.name.startswith("input_layout:")),
        None,
    )
    if hard_failure is not None:
        diagnostic = hard_failure.diagnostic
        raise PathDivergenceError(
            diagnostic.message if diagnostic is not None else "Unsupported input layout.",
            code=(
                diagnostic.code.value
                if diagnostic is not None
                else RunnableErrorCode.INPUT_TREE_MISMATCH.value
            ),
            path_faithfulness=PathFaithfulness.DIVERGED,
            first_mismatch=diagnostic,
            contract_check=hard_failure,
        )
    # r37 3-ADJ-6 (R10 ordering): alias-topology math runs BELOW the hard layout
    # precondition raise, so the shared byte engine's domain is admitted plain
    # strided tensors BY CONSTRUCTION -- it can never see a meta/nested/named/
    # quantized layout the gate is about to reject.
    alias_checks, input_alias_unresolved = _input_alias_topology_checks(
        descriptor, input_slots, raw_values
    )
    checks.extend(alias_checks)
    if input_alias_unresolved:
        # r25-4: the unresolved-topology ceiling was previously an unnamed boolean
        # threaded to the faithfulness classifier, so a caller told to branch on the
        # documented ``input_alias_topology_unresolved`` code could never observe it.
        # Surface the code on a PASSED disclosure row in ``report.contract_checks``:
        # ``passed=True`` because unresolved is a ceiling (UNVERIFIABLE), never an
        # observed contradiction -- a failed check would wrongly classify DIVERGED.
        checks.append(
            ContractCheck(
                name=RunnableErrorCode.INPUT_ALIAS_TOPOLOGY_UNRESOLVED.value,
                passed=True,
                diagnostic=RunnableDiagnostic(
                    code=RunnableErrorCode.INPUT_ALIAS_TOPOLOGY_UNRESOLVED,
                    message=(
                        "The three-valued alias engine could prove neither overlap "
                        "nor disjointness for a same-storage model-input pair; the "
                        "run verdict is capped at UNVERIFIABLE "
                        "(input_alias_topology_unresolved), never DIVERGED by "
                        "assumption and never VERIFIED."
                    ),
                    registry_id=None,
                    affected_op_labels=(),
                    recorded_runtime=None,
                    current_runtime=str(torch.__version__),
                    detection_stage="run_honesty_contract",
                    resolver_provenance=None,
                    is_analysis_load_available=True,
                    details=(),
                ),
            )
        )
    # Phase 4: clone only ACCEPTED (executable) tensors.
    for slot in input_slots:
        raw = raw_values.get(slot.slot_id)
        if isinstance(raw, torch.Tensor):
            values[slot.slot_id] = _runtime_mirror_clone(raw, ceiling, slot)
    return values, tuple(checks), input_alias_unresolved


def _runtime_mirror_clone(
    raw: torch.Tensor,
    ceiling: RunResourceCeiling,
    slot: TensorSlotDescriptor,
) -> torch.Tensor:
    """Defensively clone one leaf while MIRRORING its runtime autograd metadata.

    r37 corr2-7 (R13): ``detach().clone()`` strips a leaf's ``requires_grad``, so
    the recorded ``grad_enabled=True`` call context became semantically inert (no
    autograd history on replay) and the exact original input was needlessly
    attestation-ineligible (fingerprint flag mismatch). The runtime mirror restores
    ``raw.requires_grad`` on the clone where legal; fingerprints stay on the
    executed-clone basis, so an intentionally changed-flag input remains a physical
    input change (attestation ``not_applicable``), never normalized away. This is
    the ONE second-clone helper -- every input-bind/state-clone/staging site routes
    through it or the staging helper's recorded-trainable rule.

    r61 corr_2: the mirror routes through the transaction ceiling's byte guard --
    the clone's ``numel()`` is the LOGICAL numel of an expanded runtime view, so a
    tampered input slot (`[1]` -> huge) with a matching runtime ``expand`` refuses
    typed at ``clone_allocation_preflight`` BEFORE the materializing clone, never
    an allocator death. Honest inputs clone with ``requires_grad`` mirrored, so
    attestation eligibility is unchanged.

    r67 C5 (corr1-2): the mirror is a PRE-EXECUTION defensive materialization, so it
    runs under the neutral ambient -- a caller inside ``torch.inference_mode()`` no
    longer mints an inference-mode mirror (which lost attestation on an otherwise
    exact run and could not carry the mirrored ``requires_grad``).
    """

    with _guarded_defensive_materialize():
        return ceiling.guarded_clone(
            raw,
            call_id=None,
            slot_id=slot.slot_id,
            affected_op_labels=(slot.slot_id.removeprefix("slot:"),),
            mirror_requires_grad=True,
        )


def _model_input_version_closure(descriptor: SparseRunDescriptor) -> set[str]:
    """Return model-input slot ids plus every slot transitively versioned from one."""

    closure = {
        slot.slot_id for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.MODEL_INPUT
    }
    changed = True
    while changed:
        changed = False
        for slot in descriptor.tensor_slots:
            if slot.version_of in closure and slot.slot_id not in closure:
                closure.add(slot.slot_id)
                changed = True
    return closure


def _model_input_storage_closure(descriptor: SparseRunDescriptor) -> set[str]:
    """Model-input slots plus every slot reachable by version chains AND view-op lineage.

    A view op (``a[0]``, ``a.t()``, ``a.narrow(...)``) produces an output that SHARES STORAGE
    with its tensor input, so an in-place op on that view mutates the input's storage even
    though the view slot has no ``version_of`` link to the input (r29-C3, F2-hon). The gate
    that guards the input-aliasing fail-closed therefore follows both version chains and
    view-producing lineage from the model-input slots: a slot enters the closure if it is a
    model input, is versioned from a closure slot, or is the output of a view op whose tensor
    argument is already in the closure.
    """

    reg_qualname = {
        entry.registry_id: getattr(entry.key, "qualname", None)
        for entry in descriptor.callable_registry
    }
    closure = _model_input_version_closure(descriptor)
    changed = True
    while changed:
        changed = False
        for call in descriptor.calls:
            if reg_qualname.get(call.registry_id) not in _VIEW_OP_QUALNAMES:
                continue
            if not any(arg.slot_id in closure for arg in call.tensor_arguments):
                continue
            for output_slot_id in call.output_slot_ids:
                if output_slot_id not in closure:
                    closure.add(output_slot_id)
                    changed = True
        # Re-follow version chains from any newly-added view outputs.
        for slot in descriptor.tensor_slots:
            if slot.version_of in closure and slot.slot_id not in closure:
                closure.add(slot.slot_id)
                changed = True
    return closure


def _touched_bytes_relation(left: torch.Tensor, right: torch.Tensor) -> str:
    """Shared-engine adapter (see :func:`torchlens.utils.alias_footprint.touched_bytes_relation`)."""

    with _state.pause_logging():
        return touched_bytes_relation(left, right)
