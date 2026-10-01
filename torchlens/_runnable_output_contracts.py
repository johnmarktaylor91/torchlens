"""Output reconstruction and post-execution contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from .errors import (
    RunPreconditionError,
)
from .runnable import (
    ContractCheck,
    ControlWitness,
    ControlWitnessKind,
    RunnableCallDescriptor,
    RunnableErrorCode,
    SparseRunDescriptor,
    TensorSlotRole,
    WitnessGapKind,
)

if TYPE_CHECKING:
    from ._runnable_execution import (
        _INPUT_STRUCTURE_SITE_PREFIX,
        _MODULE_TRAINING_MODE_SITE_PREFIX,
        _container_kind,
        _container_leaf_paths,
        _contract_check,
        _decode_literal,
        _input_site_value,
        _inventory_site_positions,
        _is_model_input_literal_witness,
        _is_model_input_metadata_witness,
        _is_state_metadata_fact_witness,
        _is_unbound_state_escape_witness,
        _op_for_label,
    )

__all__ = (
    "LiveOutputDiagnosis",
    "_diagnose_live_output",
    "_reconstruct_live_output",
    "_container_from_paths",
    "_write_output_path",
    "_call_witness_checks",
    "_post_execution_contract_checks",
    "_conditional_arm_check",
    "_input_structure_positions",
    "_input_structure_witness_check",
    "_structure_witness_check",
    "_runtime_input_for_structure_witness",
    "_raw_runtime_output",
    "_registered_flatten_children",
    "_codec_component",
)


#: Closed reason vocabulary for a live output the provider could not rebuild
#: faithfully (AUD-HONESTY H1/H2). Every reason pairs with a remedy naming the
#: user action; consumers branch on ``RunReport.first_mismatch.code`` +
#: ``details["reason"]``, never on message text.
_LIVE_OUTPUT_REASON_REMEDIES: dict[str, str] = {
    "opaque_leaf": (
        "the declared output container carries a slot TorchLens cannot rebuild (an "
        "opaque tensor-holding object such as a HuggingFace DynamicCache under "
        "past_key_values); for HuggingFace models pass use_cache=False "
        "(model.config.use_cache = False) or return_dict=False, or wrap the model to "
        "return the logits tensor, then re-capture and run again"
    ),
    "container_contract_unrecorded": (
        "this capture recorded the output leaf paths but no output-container contract "
        "(the default capture does not persist it); re-capture with "
        "capture=CaptureOptions(capture_container_structure=True) -- or "
        "intervention_ready=True, which implies it -- so the live run can rebuild the "
        "exact tuple/dict/ModelOutput, or wrap the model to return a bare tensor"
    ),
    "opaque_root": (
        "return a tensor or a supported container (tuple, list, dict, namedtuple, "
        "dataclass, HuggingFace ModelOutput) instead of an unordered set / custom object"
    ),
    "lossy_reconstruction": (
        "the output container type carries instance state (computed non-field/non-key "
        "attributes, __slots__, or a data-descriptor field) the non-invoking rebuild "
        "cannot restore; return a plain container or a bare tensor"
    ),
    "reconstruction_failed": (
        "the recorded output-container contract could not be applied to the refreshed "
        "leaves; re-capture with capture_container_structure=True and, for HuggingFace "
        "models, use_cache=False or return_dict=False"
    ),
}


@dataclass(frozen=True)
class LiveOutputDiagnosis:
    """The live provider's output-reconstruction verdict with its teaching remedy.

    ``faithful`` is ``True`` only when ``output`` is the exact object a live forward
    returns (correct container type, non-tensor literal leaves preserved) or the
    genuine single bare tensor. Otherwise ``output`` is a best-effort approximation,
    ``reason`` is one closed :data:`_LIVE_OUTPUT_REASON_REMEDIES` key, ``detail``
    names the concrete evidence, and ``remedy`` the user action -- the typed
    settlement the audit's H1/H2 findings demanded in place of a bare
    ``ValueError`` or a remedy-less ``output_structure_mismatch``.
    """

    output: Any
    faithful: bool
    reason: str | None = None
    detail: str = ""

    @property
    def remedy(self) -> str | None:
        """Return the closed-vocabulary remedy for ``reason`` (``None`` when faithful)."""

        return None if self.reason is None else _LIVE_OUTPUT_REASON_REMEDIES[self.reason]


def _reconstruct_live_output(trace: Any) -> tuple[Any, bool]:
    """Reconstruct refreshed live output faithfully and report reconstruction fidelity.

    Returns
    -------
    tuple[Any, bool]
        ``(output, faithful)``. ``output`` is the exact model-output object rebuilt
        from the captured :class:`ContainerSpec` (correct container type, non-tensor
        literal leaves preserved) when a reconstructable final-output container was
        recorded, or the genuine single bare-tensor output. ``faithful`` is ``False``
        only when the output could merely be approximated from naive leaf paths (no
        faithful container contract, e.g. an opaque/BFS-fallback container); the
        caller then downgrades ``path_faithfulness`` to ``UNVERIFIABLE`` instead of
        blessing a lossy substitution with ``VERIFIED``. The diagnosed spelling is
        :func:`_diagnose_live_output`, which also carries the reason and remedy.
    """

    diagnosis = _diagnose_live_output(trace)
    return diagnosis.output, diagnosis.faithful


def _diagnose_live_output(trace: Any) -> LiveOutputDiagnosis:
    """Rebuild the refreshed live output and diagnose any loss of fidelity (H1/H2).

    Arms, in order:

    1. A recorded reconstructable final-output container whose declared leaf-slot
       count equals the captured tensor-leaf count rebuilds the exact object
       (``faithful=True``). A slot-count excess is the OPAQUE-LEAF arm: the capture
       admitted a tensor-holding non-tensor object (HuggingFace ``DynamicCache``) as
       a leaf the contract cannot represent -- settled typed with the
       ``use_cache=False`` / ``return_dict=False`` remedy instead of the bare
       "Not enough leaves" ``ValueError`` the rebuild would raise.
    2. A genuine single bare-tensor root (gated on the FRESH refresh proof, r39
       corr2_5) is returned as-is.
    3. Output leaves carrying container PATHS but no recorded spec are a declared
       tuple/dict/ModelOutput whose contract the default capture did not persist:
       best-effort ``_container_from_paths`` approximation, reason
       ``container_contract_unrecorded`` naming ``capture_container_structure=True``.
    4. Anything else is an opaque root (set / custom object): best-effort value,
       reason ``opaque_root``.

    A lossy container type (``_container_spec_reconstruction_lossy``) downgrades any
    arm to ``lossy_reconstruction``.
    """

    # The REBOUND spellings: the raw ``_runnable_call_outputs`` functions resolve
    # their own helpers (``_op_for_label``) only through the execution module's
    # globals.
    from ._runnable_execution import (
        _container_spec_reconstruction_lossy,
        _fresh_bare_tensor_root,
        _output_container_spec,
    )
    from ._runnable_output_contracts import LiveOutputDiagnosis, _unfaithful_root_diagnosis
    from .data_classes.container import container_from_op
    from .ir.container import declared_leaf_slots

    output_labels = tuple(getattr(trace, "output_layers", ()) or ())
    lossy = _container_spec_reconstruction_lossy(_output_container_spec(trace))

    def _approximation() -> Any:
        """Best-effort container rebuilt from the retained output leaf paths."""

        values = [
            (
                tuple(getattr(trace.ops[label], "container_path", ()) or ()),
                trace.ops[label].out,
            )
            for label in output_labels
        ]
        return _container_from_paths(values)

    for label in output_labels:
        op = trace.ops[label]
        container = container_from_op(op)
        # A reconstructable final-output view carries the captured ContainerSpec, so
        # ``reconstruct`` rebuilds the SAME object a live forward returns (container
        # kind + literal leaves + fields).
        if (
            container is None
            or container.root_kind != "final_output"
            or not container.supports_reconstruct
            or container.spec is None
        ):
            continue
        spec = container.spec
        slots = declared_leaf_slots(spec)
        leaves = len(container.leaves)
        if slots != leaves:
            return LiveOutputDiagnosis(
                _approximation(),
                False,
                "opaque_leaf",
                f"declared {spec.kind} output {spec.type_qualname or ''} has {slots} leaf "
                f"slot(s) but the capture holds {leaves} tensor leaf/leaves",
            )
        try:
            rebuilt = container.reconstruct(values="out")
        except ValueError as exc:
            return LiveOutputDiagnosis(_approximation(), False, "reconstruction_failed", str(exc))
        if lossy:
            return LiveOutputDiagnosis(
                rebuilt,
                False,
                "lossy_reconstruction",
                f"{spec.kind} output {spec.type_qualname or ''} reconstructs lossily",
            )
        return LiveOutputDiagnosis(rebuilt, True)
    if len(output_labels) == 1:
        op = trace.ops[output_labels[0]]
        has_spec = getattr(op, "container_spec", None) is not None
        has_path = bool(getattr(op, "container_path", ()) or ())
        if not has_spec and not has_path and _fresh_bare_tensor_root(trace):
            # Genuine single bare-tensor model output: no container to reconstruct. r39
            # corr2_5: gated on the FRESH refresh proof's ``bare_tensor_root`` fact -- an
            # opaque set/frozenset/custom container the traversal fell back on produces the
            # SAME "one leaf, no spec, no path" signature, so without this positive proof a
            # wrong bare-tensor object would be blessed faithful (and a multi-tensor set would
            # silently drop a leaf). A missing/opaque proof falls through to faithful=False.
            return LiveOutputDiagnosis(op.out, not lossy, "lossy_reconstruction" if lossy else None)
    # Multi-leaf output lacking a faithful reconstructable container contract, or a
    # single leaf that was actually a non-reconstructable (opaque) container. Return
    # a best-effort approximation but report it as NOT faithful, naming WHY.
    return _unfaithful_root_diagnosis(trace, output_labels, lossy, _approximation())


def _unfaithful_root_diagnosis(
    trace: Any, output_labels: Sequence[str], lossy: bool, approximation: Any
) -> LiveOutputDiagnosis:
    """Classify a non-faithful live output root into its closed reason + evidence.

    Reads the fresh refresh's ``output_losslessness`` proof and the recorded leaf
    paths to pick ONE :data:`_LIVE_OUTPUT_REASON_REMEDIES` key: a lossy container
    type, the H1 opaque-leaf proof refusal, the H2 unrecorded container contract,
    or the opaque root fallback.
    """

    proof = getattr(getattr(trace, "__dict__", {}).get("_runnable"), "output_losslessness", None)
    root_type = proof.get("root_type") if isinstance(proof, Mapping) else None
    root_kind = proof.get("root_kind") if isinstance(proof, Mapping) else None
    proof_reason = str(proof.get("reason") or "") if isinstance(proof, Mapping) else ""
    pathed = any(getattr(trace.ops[label], "container_path", ()) for label in output_labels)
    detail = f"fresh output root {root_type!r} ({root_kind or 'unproven'})"
    if lossy:
        reason = "lossy_reconstruction"
    elif proof_reason.startswith("opaque_leaf:"):
        # H1: the fresh forward's losslessness proof names the exact opaque leaf
        # (``opaque_leaf:DynamicCache``) inside an otherwise declared container.
        reason = "opaque_leaf"
        detail = f"{detail}; proof refused on {proof_reason}"
    elif pathed and root_kind != "opaque":
        # H2: declared tuple/dict/ModelOutput leaves with recorded paths -- the
        # contract exists in the forward but the default capture did not persist it.
        reason = "container_contract_unrecorded"
    else:
        reason = "opaque_root"
        if proof_reason:
            detail = f"{detail}; proof refused on {proof_reason}"
    return LiveOutputDiagnosis(approximation, False, reason, detail)


def _container_from_paths(values: Sequence[tuple[tuple[str | int, ...], Any]]) -> Any:
    """Build a conservative tuple/dict output container from leaf paths."""

    if len(values) == 1 and not values[0][0]:
        return values[0][1]
    if not values:
        return None
    paths = [path for path, _ in values]
    root: Any = [] if all(path and isinstance(path[0], int) for path in paths) else {}
    if isinstance(root, list):
        root.extend([None] * (max(cast(int, path[0]) for path in paths) + 1))
    for path, value in values:
        _write_output_path(root, path, value)
    return tuple(root) if isinstance(root, list) else root


def _write_output_path(root: Any, path: tuple[str | int, ...], value: Any) -> None:
    """Write one output leaf, growing positional containers as needed."""

    if not path:
        raise RunPreconditionError(
            "Multiple output leaves cannot share an empty container path.",
            code=RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH.value,
            remedy=(
                "re-capture and re-save the runnable artifact with the current "
                "torchlens producer so every output leaf records a distinct "
                "container path"
            ),
        )
    current = root
    for index, component in enumerate(path):
        last = index == len(path) - 1
        if isinstance(current, list):
            position = cast(int, component)
            while len(current) <= position:
                current.append(None)
            if last:
                current[position] = value
                return
            if current[position] is None:
                current[position] = [] if isinstance(path[index + 1], int) else {}
            current = current[position]
        else:
            if last:
                current[component] = value
                return
            current = current.setdefault(component, [] if isinstance(path[index + 1], int) else {})


def _call_witness_checks(
    descriptor: SparseRunDescriptor,
    call: RunnableCallDescriptor,
    slot_values: Mapping[str, torch.Tensor],
) -> tuple[ContractCheck, ...]:
    """Compare scalar-bool and loop witnesses immediately after their call.

    r71 A2: the call's OWNER-RECORD ``control_obligations`` are the required check
    domain (replay-consumed structure, never the witness stream). Every obligation
    demands its exact same-kind witness; a missing witness whose obligation carries
    no typed predicate gap fails closed as a check (belt behind the parse-time
    discharge equality). Surviving witnesses without an obligation cannot exist
    post-parse.
    """

    checks: list[ContractCheck] = []
    witnesses_by_identity = {
        (witness.call_id, witness.kind, witness.site_label): witness
        for witness in descriptor.control_witnesses
        if witness.kind in {ControlWitnessKind.SCALAR_BOOL, ControlWitnessKind.LOOP_PREDICATE}
    }
    gap_members = {
        gap.source_member
        for gap in descriptor.coverage_gaps
        if gap.gap_kind
        in {WitnessGapKind.UNOBSERVED_PREDICATE, WitnessGapKind.UNCLASSIFIED_TERMINAL_BOOL}
    }
    for obligation in call.control_obligations:
        witness = witnesses_by_identity.get((call.call_id, obligation.kind, obligation.site_label))
        code = (
            RunnableErrorCode.LOOP_PREDICATE_DIVERGENCE
            if obligation.kind is ControlWitnessKind.LOOP_PREDICATE
            else RunnableErrorCode.SCALAR_BOOL_DIVERGENCE
        )
        if witness is None:
            if obligation.output_slot_id in gap_members:
                # Gap-discharged obligation: the derived completeness floor already
                # ceilings this run at UNVERIFIABLE; there is no witness to compare.
                continue
            checks.append(
                _contract_check(
                    f"control_obligation:{call.call_id}:{obligation.site_label}",
                    False,
                    RunnableErrorCode.CONTEXT_FIELD_INVALID,
                    "Control obligation has no discharging witness or typed gap.",
                    affected_op_labels=(obligation.site_label,),
                )
            )
            continue
        scalar = slot_values.get(obligation.output_slot_id)
        expected = bool(_decode_literal(witness.observed_value))
        actual: bool | None = None
        if isinstance(scalar, torch.Tensor) and scalar.numel() == 1:
            actual = bool(scalar.item())
        checks.append(
            _contract_check(
                f"control_witness:{witness.witness_id}",
                actual is not None and actual == expected,
                code,
                f"Control witness {witness.witness_id!r} disagreed with the recorded path.",
                affected_op_labels=(witness.site_label,),
                details=(
                    ("witness_id", witness.witness_id),
                    ("expected", repr(expected)),
                    ("actual", repr(actual)),
                    ("order", str(witness.order)),
                ),
            )
        )
    return tuple(checks)


def _post_execution_contract_checks(
    descriptor: SparseRunDescriptor,
    *,
    inputs: Any,
    output: Any,
    slot_values: Mapping[str, torch.Tensor],
    fork: Any,
) -> tuple[ContractCheck, ...]:
    """Validate final slot production, arm identity, and structure witnesses."""

    checks: list[ContractCheck] = []
    # The frozen input-site inventory is a pure function of the descriptor; resolve
    # it at most once per transaction (at the FIRST input-structure witness, so the
    # inventory belt still raises at exactly the point the per-witness computation
    # did) instead of rescanning and re-decoding every witness per witness.
    input_structure_positions: set[Any] | None = None
    for call in descriptor.calls:
        missing = tuple(slot_id for slot_id in call.output_slot_ids if slot_id not in slot_values)
        checks.append(
            _contract_check(
                f"call_slot_production:{call.call_id}",
                not missing,
                RunnableErrorCode.SLOT_PRODUCTION_MISMATCH,
                f"Call {call.call_id!r} did not produce every recorded output slot.",
                affected_op_labels=call.op_labels,
                details=(("missing_slot_ids", repr(missing)),),
            )
        )
    for witness in sorted(descriptor.control_witnesses, key=lambda item: item.order):
        if _is_model_input_literal_witness(witness):
            # Non-tensor input leaves are compared in the input contract before
            # execution; they are not runtime container-structure facts.
            continue
        if _is_model_input_metadata_witness(witness):
            # Model-input metadata-predicate facts are compared against the RAW
            # runtime inputs in the input contract before execution; they are not
            # runtime container-structure facts.
            continue
        if _is_unbound_state_escape_witness(witness):
            # Unbound state escapes are compared by capture-digest in the dedicated
            # staleness check, not against runtime container structure.
            continue
        if _is_state_metadata_fact_witness(witness):
            # r65 F-1: declared capture-time state-metadata facts (requires_grad /
            # grad_fn presence) are REPRODUCED by run preparation (the recorded bit is
            # applied to the staged slot), not compared against runtime container
            # structure.
            continue
        if witness.site_label.startswith(_MODULE_TRAINING_MODE_SITE_PREFIX):
            # The declared per-module train/eval mode is a capture-time state fact anchoring
            # VERIFIED (see ``_mode_sensitive_op_unwitnessed``), not a runtime container
            # structure fact; it must not be compared against the runtime container.
            continue
        if witness.kind is ControlWitnessKind.CONDITIONAL_ARM_ENTRY:
            checks.append(_conditional_arm_check(witness, fork))
        elif witness.site_label.startswith(_INPUT_STRUCTURE_SITE_PREFIX):
            # r67 C2: per-site input-boundary structure facts compare against a runtime
            # snapshot built by the SAME spine function -- kind, exact class, child
            # schema, ordered codec keys, arity, and the symmetric instance-state proof.
            if input_structure_positions is None:
                input_structure_positions = _input_structure_positions(descriptor)
            checks.append(
                _input_structure_witness_check(
                    witness,
                    inputs=inputs,
                    all_positions=input_structure_positions,
                )
            )
        elif witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            checks.append(
                _structure_witness_check(
                    witness,
                    descriptor,
                    inputs=inputs,
                    output=output,
                )
            )
        elif (
            witness.kind
            in {
                ControlWitnessKind.SCALAR_BOOL,
                ControlWitnessKind.LOOP_PREDICATE,
            }
            and witness.call_id is None
        ):
            checks.append(
                _contract_check(
                    f"control_witness:{witness.witness_id}",
                    False,
                    RunnableErrorCode.SLOT_PRODUCTION_MISMATCH,
                    f"Control witness {witness.witness_id!r} has no recomputable call.",
                    affected_op_labels=(witness.site_label,),
                )
            )
    # r71 A2 belt: every owner-record arm-entry dependency edge must own its witness
    # (parse enforces the equality; a descriptor that somehow reaches execution with
    # an unwitnessed edge fails closed as a check, never a silently skipped arm).
    from .runnable import control_dependency_site_label

    arm_witness_labels = {
        witness.site_label
        for witness in descriptor.control_witnesses
        if witness.kind is ControlWitnessKind.CONDITIONAL_ARM_ENTRY
    }
    for call in descriptor.calls:
        for edge in call.control_dependencies:
            edge_label = control_dependency_site_label(edge)
            if edge_label not in arm_witness_labels:
                checks.append(
                    _contract_check(
                        f"control_dependency:{edge_label}",
                        False,
                        RunnableErrorCode.CONTEXT_FIELD_INVALID,
                        "Arm-entry control dependency has no discharging witness.",
                        affected_op_labels=(edge_label,),
                    )
                )
    return tuple(checks)


def _conditional_arm_check(witness: ControlWitness, fork: Any) -> ContractCheck:
    """Validate that one recorded conditional arm-entry edge was produced."""

    edge_text = witness.site_label.rsplit(":", 1)[-1]
    parent, separator, child = edge_text.partition("->")
    parent_op = _op_for_label(fork, parent) if separator else None
    child_op = _op_for_label(fork, child) if separator else None
    passed = (
        parent_op is not None
        and child_op is not None
        and isinstance(getattr(parent_op, "out", None), torch.Tensor)
        and isinstance(getattr(child_op, "out", None), torch.Tensor)
    )
    affected = tuple(label for label in (parent, child) if label)
    return _contract_check(
        f"control_witness:{witness.witness_id}",
        passed,
        RunnableErrorCode.CONDITIONAL_ARM_DIVERGENCE,
        f"Conditional arm witness {witness.witness_id!r} did not enter its recorded edge.",
        affected_op_labels=affected or (witness.site_label,),
        details=(("recorded_edge", edge_text), ("order", str(witness.order))),
    )


def _input_structure_positions(descriptor: SparseRunDescriptor) -> set[Any]:
    """Return the required input-site set for structure-check site selection (r69 A).

    Consumes the parse-validated descriptor-native inventory -- NEVER the surviving
    ``input_structure`` witnesses (the r68 secA-F1 lane re-derived positions from the
    stream being validated, so a stripped family restored weaker semantics). A belt
    cross-checks the surviving facts against the inventory and fails closed typed on
    any deficit that somehow reached execution.
    """

    inventory_positions = _inventory_site_positions(descriptor)
    witness_positions: set[Any] = set()
    for witness in descriptor.control_witnesses:
        if not witness.site_label.startswith(_INPUT_STRUCTURE_SITE_PREFIX):
            continue
        try:
            fact = _decode_literal(witness.observed_value)
        except Exception:
            continue
        position = fact.get("position") if isinstance(fact, Mapping) else None
        if isinstance(position, list) and len(position) == 2:
            witness_positions.add(tuple(position))
    if witness_positions != inventory_positions:
        raise RunPreconditionError(
            "Surviving input-structure facts do not equal the parse-validated "
            f"required inventory (facts {sorted(witness_positions, key=repr)!r}, "
            f"required {sorted(inventory_positions, key=repr)!r}).",
            code=RunnableErrorCode.CONTEXT_FIELD_INVALID.value,
            remedy=(
                "re-save the artifact with the current torchlens producer instead of "
                "stripping or editing its input-structure witness facts"
            ),
        )
    return inventory_positions


def _input_structure_witness_check(
    witness: ControlWitness, *, inputs: Any, all_positions: set[Any]
) -> ContractCheck:
    """Compare one persisted input-boundary structure fact with the runtime site (r67 C2).

    The runtime snapshot comes from the SAME spine function that produced the persisted
    fact, so capture and runtime can never diverge in vocabulary. Any node mismatch --
    kind swap at any depth (hon1-F2a), exact-class swap (free-F3/hon1-F2b), empty-
    dataclass arity (free-F2/hon1-F2c), ordered-key or grammar-key drift (hon1-F1),
    registered schema drift (corr1-3) -- fails the check (``input_tree_mismatch``
    class), as does RUNTIME-ADDED undeclared instance state (the symmetric bind-side
    proof: a runtime snapshot refusal can never pass).
    """

    from torchlens._input_walk import snapshot_input_boundary

    expected = _decode_literal(witness.observed_value)
    position_raw = expected.get("position")
    position = tuple(position_raw) if isinstance(position_raw, list) else position_raw
    try:
        runtime_value = _input_site_value(inputs, position, all_positions or {position})
    except (KeyError, IndexError, TypeError, AttributeError):
        return _contract_check(
            f"control_witness:{witness.witness_id}",
            False,
            RunnableErrorCode.INPUT_TREE_MISMATCH,
            f"Model input site {position!r} is missing from the runtime inputs.",
            affected_op_labels=(witness.site_label,),
            details=(("position", repr(position)),),
        )
    runtime_snapshot = snapshot_input_boundary(runtime_value)
    expected_nodes = expected.get("nodes", [])
    actual_nodes = runtime_snapshot.get("nodes", [])
    refusals = runtime_snapshot.get("refusals", [])
    passed = expected_nodes == actual_nodes and not refusals
    return _contract_check(
        f"control_witness:{witness.witness_id}",
        passed,
        RunnableErrorCode.INPUT_TREE_MISMATCH,
        f"Runtime input structure at site {position!r} differs from the captured "
        "boundary snapshot (container kind, exact class, child schema, mapping keys, "
        "or undeclared instance state).",
        affected_op_labels=(witness.site_label,),
        details=(
            ("position", repr(position)),
            ("expected_nodes", repr(expected_nodes)[:2000]),
            ("actual_nodes", repr(actual_nodes)[:2000]),
            ("runtime_refusals", repr(refusals)),
        ),
    )


def _structure_witness_check(
    witness: ControlWitness,
    descriptor: SparseRunDescriptor,
    *,
    inputs: Any,
    output: Any,
) -> ContractCheck:
    """Compare a model-boundary container witness with runtime structure."""

    expected = _decode_literal(witness.observed_value)
    role = expected.get("role")
    if role == "model_input":
        runtime_value = _runtime_input_for_structure_witness(descriptor, expected, inputs)
        expected_paths = tuple(tuple(path) for path in expected.get("leaf_paths", ()))
        expected_kind = str(expected.get("kind", "unknown"))
    else:
        runtime_value = output
        expected_paths = tuple(tuple(path) for path in expected.get("leaf_paths", ()))
        expected_kind = str(expected.get("kind", "unknown"))
    actual_paths = tuple(_container_leaf_paths(runtime_value))
    actual_kind = _container_kind(runtime_value)
    kind_matches = expected_kind in {"unknown", actual_kind}
    passed = expected_paths == actual_paths and kind_matches
    return _contract_check(
        f"control_witness:{witness.witness_id}",
        passed,
        RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
        f"Structure witness {witness.witness_id!r} disagreed with the runtime container.",
        affected_op_labels=(witness.site_label,),
        details=(
            ("role", repr(role)),
            ("expected_kind", expected_kind),
            ("actual_kind", actual_kind),
            ("expected_paths", repr(expected_paths)),
            ("actual_paths", repr(actual_paths)),
        ),
    )


def _runtime_input_for_structure_witness(
    descriptor: SparseRunDescriptor,
    expected: Mapping[str, Any],
    inputs: Any,
) -> Any:
    """Select the runtime input site named by a container structure witness."""

    record_id = expected.get("record_id")
    bindings = [
        slot.input_binding
        for slot in descriptor.tensor_slots
        if slot.role is TensorSlotRole.MODEL_INPUT
        and slot.input_binding is not None
        and slot.input_binding.container_record_id == record_id
    ]
    positions = {binding.model_site_position for binding in bindings}
    if len(positions) == 1:
        return _input_site_value(inputs, next(iter(positions)), positions)
    return inputs


def _raw_runtime_output(
    descriptor: SparseRunDescriptor,
    reconstructed_output: Any,
    call_outputs: Mapping[str, Any],
) -> Any:
    """Return the raw final call container when one call owns every model output."""

    output_sources = {
        slot.producer_slot_id or slot.version_of
        for slot in descriptor.tensor_slots
        if slot.role is TensorSlotRole.OUTPUT
    }
    owner_ids = {
        call.call_id
        for call in descriptor.calls
        if output_sources and output_sources.issubset(set(call.output_slot_ids))
    }
    if len(owner_ids) == 1:
        return call_outputs.get(next(iter(owner_ids)), reconstructed_output)
    return reconstructed_output


def _registered_flatten_children(value: Any) -> list[Any] | None:
    """Return a registered container's flatten children, else ``None`` (r67 C2, corr1-3)."""

    from torchlens.ir.container import get_registered_container

    if isinstance(value, (torch.Tensor, Mapping, list, tuple, str, bytes)) or value is None:
        return None
    registration = get_registered_container(type(value))
    if registration is None:
        return None
    try:
        return list(registration.flatten(value)[0])
    except Exception:
        return None


def _codec_component(key: Any) -> str | int | None:
    """Encode one runtime mapping key through the canonical codec, or ``None`` (r67 C2)."""

    from torchlens._input_walk import encode_mapping_key

    try:
        return encode_mapping_key(key)
    except ValueError:
        return None
