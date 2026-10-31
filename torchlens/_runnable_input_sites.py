"""Live input sites and metadata contract checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from torchlens._runnable import ContractCheck, ReadinessReport

from ._runnable import (
    ControlWitness,
    ControlWitnessKind,
    ReadinessStatus,
    RunnableErrorCode,
    SparseRunDescriptor,
    TensorSlotRole,
)
from .errors import (
    ReattachError,
    RunCapabilityUnavailableError,
    RunPreconditionError,
)

if TYPE_CHECKING:
    from ._runnable_execution import (
        _INPUT_DERIVED_LAYOUT_FACT_NAME,
        _MODEL_INPUT_LITERAL_FACT_KEY,
        _MODEL_INPUT_LITERAL_SITE_PREFIX,
        _MODEL_INPUT_METADATA_FACT_KEY,
        _MODEL_INPUT_METADATA_SITE_PREFIX,
        _contract_check,
        _decode_literal,
        _input_site_value,
        _live_runtime_input_leaves,
        _positions_are_mixed,
        _split_mixed_inputs,
        _value_at_path,
    )

__all__ = (
    "_INPUT_CHECK_UNAVAILABLE",
    "_first_failed_live_input_check",
    "raise_analysis_run_unavailable",
    "_require_loaded_sparse_provider",
    "_model_input_literal_facts",
    "_is_model_input_literal_witness",
    "_model_input_metadata_facts",
    "_is_model_input_metadata_witness",
    "_runtime_input_metadata_value",
    "_fact_path_tuple_or_none",
    "_input_metadata_contract_checks",
    "_input_derived_layout_stale",
    "_inventory_site_positions",
    "_model_input_arity_positions",
    "_runtime_top_level_positions",
)


#: Distinguished "the classifier itself failed" result (R22-2): callers with
#: ADMISSION power (the fast-run pre-forward gate) must fail CLOSED on it —
#: guard-machinery failure is otherwise indistinguishable from inputs-match.
#: The soft native-failure-time consumer treats it as unclassifiable and lets
#: the native error re-raise raw, exactly the old ``None`` behavior there.
_INPUT_CHECK_UNAVAILABLE: Any = object()


def _first_failed_live_input_check(
    trace: Any, input_args: Any, input_kwargs: Any
) -> ContractCheck | None | Any:
    """Return the first failed input-contract check for a live refresh.

    r41 hon1_3 (corr2_4 parity): reads the capture-recorded input ops' shape/dtype off
    the live Trace and compares the runtime leaves positionally, mirroring the sparse
    ``_bind_runtime_inputs`` check names (``input_shape:slot:<label>`` /
    ``input_dtype:slot:<label>``) and message text so both providers speak identically.

    Sentinel-or-raise contract (R22-2): ``None`` means the checks RAN and
    passed (or there was legitimately nothing to check); the module-level
    ``_INPUT_CHECK_UNAVAILABLE`` sentinel means the classifier machinery
    itself failed and NOTHING was verified. The two used to collapse into
    ``None``, which let the ``fast=True`` admission gate treat a broken
    guard as inputs-match and run the forward unguarded.
    """

    try:
        input_labels = list(getattr(trace, "input_layers", ()) or ())
        if not input_labels:
            return None
        layer_dict = getattr(trace, "_layer_dict_all_keys", None) or {}
        recorded: list[tuple[str, tuple[int, ...] | None, str | None]] = []
        for label in input_labels:
            op = layer_dict.get(label)
            if op is None:
                return _INPUT_CHECK_UNAVAILABLE
            shape = getattr(op, "shape", None)
            dtype = getattr(op, "dtype", None)
            recorded.append(
                (
                    str(label),
                    tuple(shape) if shape is not None else None,
                    str(dtype) if dtype is not None else None,
                )
            )
        leaves = _live_runtime_input_leaves(input_args, input_kwargs)
        if leaves is None:
            return _INPUT_CHECK_UNAVAILABLE
        if len(leaves) != len(recorded):
            return _contract_check(
                "input_arity",
                False,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                f"Runtime input tree carries {len(leaves)} tensor leaves; "
                f"the capture recorded {len(recorded)}.",
                details=(
                    ("expected_leaves", str(len(recorded))),
                    ("actual_leaves", str(len(leaves))),
                ),
            )
        for (label, expected_shape, expected_dtype), value in zip(recorded, leaves, strict=True):
            try:
                actual_shape: tuple[int, ...] | None = tuple(value.shape)
            except (RuntimeError, TypeError, NotImplementedError):
                actual_shape = None
            if expected_shape is not None and actual_shape != expected_shape:
                return _contract_check(
                    f"input_shape:slot:{label}",
                    False,
                    RunnableErrorCode.INPUT_SHAPE_MISMATCH,
                    f"Runtime input shape {actual_shape} does not match {expected_shape}.",
                    affected_op_labels=(label,),
                    details=(
                        ("slot_id", f"slot:{label}"),
                        ("expected_shape", repr(expected_shape)),
                        ("actual_shape", repr(actual_shape)),
                    ),
                )
            if expected_dtype is not None and str(value.dtype) != expected_dtype:
                return _contract_check(
                    f"input_dtype:slot:{label}",
                    False,
                    RunnableErrorCode.INPUT_DTYPE_MISMATCH,
                    f"Runtime input dtype {value.dtype} does not match {expected_dtype}.",
                    affected_op_labels=(label,),
                    details=(
                        ("slot_id", f"slot:{label}"),
                        ("expected_dtype", expected_dtype),
                        ("actual_dtype", str(value.dtype)),
                    ),
                )
        return None
    except Exception:
        return _INPUT_CHECK_UNAVAILABLE


def raise_analysis_run_unavailable(trace: Any) -> None:
    """Raise the typed capability error for an analysis-only loaded Trace.

    Parameters
    ----------
    trace:
        Analysis-only loaded Trace.

    Raises
    ------
    RunCapabilityUnavailableError
        Always, with the load-time readiness report attached.
    """

    readiness = trace._runnable.readiness
    diagnostics = () if readiness is None else readiness.diagnostics
    raise RunCapabilityUnavailableError(
        "This loaded Trace is analysis-only and has no sparse run descriptor. "
        "Remedy: save a runnable artifact at capture time with "
        "trace.save(path, level='runnable') and load that instead.",
        code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
        readiness=readiness,
        diagnostics=diagnostics,
    )


def _require_loaded_sparse_provider(
    trace: Any,
) -> tuple[SparseRunDescriptor, ReadinessReport, Mapping[str, Callable[..., Any]]]:
    """Return ready descriptor state or raise one aggregate typed error."""

    readiness = trace._runnable.readiness
    descriptor = trace._runnable.descriptor
    callables = trace._runnable.callables_by_call_id
    if not isinstance(readiness, ReadinessReport) or not isinstance(
        descriptor, SparseRunDescriptor
    ):
        raise_analysis_run_unavailable(trace)
    if readiness.status is not ReadinessStatus.READY or not isinstance(callables, Mapping):
        # R65-2: this is the main torch-drift refusal a loaded-sparse run()
        # hits, and it used to carry no code, no summary, and no remedy while
        # the per-callable causes sat unsummarized in fields["diagnostics"].
        # Mirror _preflight_failure_message: fold the first diagnostic into
        # the message and stamp ITS code (the true branchable cause).
        diagnostics = readiness.diagnostics
        first = diagnostics[0] if diagnostics else None
        if first is not None:
            detail = " ".join(first.message.split())
            if len(detail) > 300:
                detail = detail[:297] + "..."
            summary = f": [{first.code.value}] {detail}"
            remainder = len(diagnostics) - 1
            if remainder:
                summary += f" (+{remainder} more on exc.fields['diagnostics'])"
            first_code = first.code.value
        else:
            summary = f" (readiness status {readiness.status.value!r}, no diagnostics)"
            first_code = RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value
        raise ReattachError(
            "Sparse callable reattachment did not produce a ready atomic "
            f"attachment{summary}. Remedy: inspect exc.fields['diagnostics'] "
            "for the per-callable drift causes, or re-save the artifact from "
            "a live capture on this torch runtime to rebind its callables.",
            code=first_code,
            readiness=readiness,
            diagnostics=diagnostics,
        )
    return descriptor, readiness, cast(Mapping[str, Callable[..., Any]], callables)


def _model_input_literal_facts(
    descriptor: SparseRunDescriptor,
) -> list[tuple[ControlWitness, Mapping[str, Any]]]:
    """Decode every witnessed non-tensor model-input leaf fact."""

    facts: list[tuple[ControlWitness, Mapping[str, Any]]] = []
    for witness in descriptor.control_witnesses:
        if witness.kind is not ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            continue
        if not witness.site_label.startswith(_MODEL_INPUT_LITERAL_SITE_PREFIX):
            continue
        decoded = _decode_literal(witness.observed_value)
        if isinstance(decoded, Mapping) and decoded.get(_MODEL_INPUT_LITERAL_FACT_KEY) is True:
            facts.append((witness, decoded))
    return facts


def _is_model_input_literal_witness(witness: ControlWitness) -> bool:
    """Return whether a structure witness records a non-tensor input leaf."""

    return (
        witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT
        and witness.site_label.startswith(_MODEL_INPUT_LITERAL_SITE_PREFIX)
    )


def _model_input_metadata_facts(
    descriptor: SparseRunDescriptor,
) -> list[tuple[ControlWitness, Mapping[str, Any]]]:
    """Decode every witnessed model-input metadata-predicate fact (r27-H2)."""

    facts: list[tuple[ControlWitness, Mapping[str, Any]]] = []
    for witness in descriptor.control_witnesses:
        if witness.kind is not ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            continue
        if not witness.site_label.startswith(_MODEL_INPUT_METADATA_SITE_PREFIX):
            continue
        decoded = _decode_literal(witness.observed_value)
        if isinstance(decoded, Mapping) and decoded.get(_MODEL_INPUT_METADATA_FACT_KEY) is True:
            facts.append((witness, decoded))
    return facts


def _is_model_input_metadata_witness(witness: ControlWitness) -> bool:
    """Return whether a structure witness records a model-input metadata-predicate read."""

    return (
        witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT
        and witness.site_label.startswith(_MODEL_INPUT_METADATA_SITE_PREFIX)
    )


def _runtime_input_metadata_value(value: torch.Tensor, name: str) -> Any:
    """Evaluate one recorded metadata predicate on the RAW bound runtime input.

    Evaluated on the user-provided tensor BEFORE the defensive detach-clone, which
    erases the autograd state (``requires_grad`` / ``grad_fn`` / ``is_leaf``) and resets
    ``storage_offset`` -- the capture-time read saw the forward's real input, so the
    comparison must too. ``grad_fn`` is compared as a PRESENCE boolean (the exact backward
    object is not comparable across runs), mirroring the capture-time recording.
    """

    try:
        if name == "is_contiguous":
            return bool(value.is_contiguous())
        if name == "stride":
            return [int(v) for v in value.stride()]
        if name == "storage_offset":
            return int(value.storage_offset())
        if name == "requires_grad":
            return bool(value.requires_grad)
        if name == "grad_fn":
            return bool(value.grad_fn is not None)
        if name == "is_leaf":
            return bool(value.is_leaf)
        if name == "retains_grad":
            return bool(value.retains_grad)
        if name == "_base":
            return bool(value._base is not None)
        if name == "_is_view":
            return bool(value._is_view())
        if name == "is_conj":
            return bool(value.is_conj())
        if name == "is_neg":
            return bool(value.is_neg())
        if name == "is_inference":
            return bool(value.is_inference())
        if name == "is_pinned":
            return bool(value.is_pinned())
        if name == "is_shared":
            return bool(value.is_shared())
        if name == "is_coalesced":
            return bool(value.is_coalesced())
        if name == "grad":
            return bool(value.grad is not None)
        if name == "_grad":
            return bool(value._grad is not None)
        if name == "_version":
            return int(value._version)
        if name == "output_nr":
            return int(value.output_nr)
        if name == "storage_nbytes":
            return int(value.untyped_storage().nbytes())
    except (RuntimeError, AttributeError, TypeError, ValueError, NotImplementedError):
        return None
    return None


def _fact_path_tuple_or_none(fact: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Belt twin of the parse-side path validator (r75 L1): ``None`` means fail closed.

    Parse refuses a non-sequence fact ``path`` as ``context_field_invalid`` before any
    execution-side consumer runs; this belt keeps the execution readers total anyway so a
    malformed path can never crash ``tuple(...)`` into an untyped lane. r77 L1 extends
    the belt one nesting level down to match the parse contract: a non-``str``/``int``
    COMPONENT (a nested list/mapping/slice decoded literal) also fails closed here
    instead of riding into the runtime-tree resolvers as an unresolvable key.
    """

    raw_path = fact.get("path", ()) or ()
    if not isinstance(raw_path, (list, tuple)):
        return None
    if any(not isinstance(component, (str, int)) for component in raw_path):
        return None
    return tuple(raw_path)


def _input_metadata_contract_checks(
    descriptor: SparseRunDescriptor,
    inputs: Any,
    positions: set[Any],
) -> tuple[ContractCheck, ...]:
    """Compare runtime input metadata predicates with capture-time observed facts (r27-H2).

    The capture-time forward READ these predicates (``is_contiguous`` / ``stride`` /
    ``requires_grad``) on a model-input leaf, so their values can have steered the
    recorded taken path. The input contract checks only shape+dtype; a same-shape
    runtime input differing in layout or grad flag would replay the captured arm a
    fresh model would not take -- a false VERIFIED+ATTESTED. Witnesses exist ONLY for
    captures that performed such a read, so an ordinary layout-oblivious model has no
    metadata witnesses and can never over-trigger here.
    """

    checks: list[ContractCheck] = []
    # r71 A2 belt: the totalized envelope domain is the boundary record's tensor-site
    # set (one envelope per bound tensor leaf, empty read sets explicit). Parse
    # enforces the equality; a deficit that somehow reached execution fails closed.
    boundary_site_count = sum(len(site.tensor_sites) for site in descriptor.input_boundary)
    envelope_count = sum(1 for _witness, _fact in _model_input_metadata_facts(descriptor))
    if envelope_count != boundary_site_count:
        checks.append(
            _contract_check(
                "input_metadata_envelope_totality",
                False,
                RunnableErrorCode.CONTEXT_FIELD_INVALID,
                "Metadata envelope count does not equal the input-boundary tensor "
                "site count; the totalized envelope domain is incomplete.",
                details=(
                    ("boundary_tensor_sites", str(boundary_site_count)),
                    ("envelopes", str(envelope_count)),
                ),
            )
        )
    for witness, fact in _model_input_metadata_facts(descriptor):
        raw_position = fact.get("position")
        position = tuple(raw_position) if isinstance(raw_position, (list, tuple)) else raw_position
        path = _fact_path_tuple_or_none(fact)
        if path is None:
            checks.append(
                _contract_check(
                    "input_metadata_path_malformed",
                    False,
                    RunnableErrorCode.CONTEXT_FIELD_INVALID,
                    "A metadata envelope carries a non-sequence container path; the "
                    "fact cannot be resolved against the runtime input tree.",
                    affected_op_labels=(witness.site_label,),
                    details=(("model_site_position", repr(position)),),
                )
            )
            continue
        recorded_facts = fact.get("facts")
        if not isinstance(recorded_facts, Mapping):
            continue
        try:
            root = _input_site_value(inputs, position, positions)
            runtime_leaf = _value_at_path(root, path)
            resolved = isinstance(runtime_leaf, torch.Tensor)
        except (KeyError, IndexError, TypeError, AttributeError):
            runtime_leaf = None
            resolved = False
        for name in sorted(recorded_facts):
            if str(name) == _INPUT_DERIVED_LAYOUT_FACT_NAME:
                # r73 F1: the derived-layout fact is a run-time UNVERIFIABLE ceiling
                # (``_input_derived_layout_stale``), never a DIVERGED compare -- a
                # changed input layout does not prove the derived intermediate's
                # layout differs, so it must not fail a contract check here.
                continue
            recorded_value = recorded_facts[name]
            runtime_value = (
                _runtime_input_metadata_value(runtime_leaf, str(name)) if resolved else None
            )
            passed = resolved and runtime_value == recorded_value
            checks.append(
                _contract_check(
                    f"input_metadata:{name}:{position!r}:{path!r}",
                    passed,
                    RunnableErrorCode.INPUT_TREE_MISMATCH,
                    f"Runtime input {name} differs from the capture-time value the "
                    "forward read; the recorded taken path may not be valid for "
                    "this input.",
                    affected_op_labels=(witness.site_label,),
                    details=(
                        ("model_site_position", repr(position)),
                        ("container_path", repr(path)),
                        ("predicate", str(name)),
                        ("recorded_value", repr(recorded_value)),
                        ("runtime_value", repr(runtime_value) if resolved else "<unresolved>"),
                    ),
                )
            )
    return tuple(checks)


def _input_derived_layout_stale(descriptor: SparseRunDescriptor, inputs: Any) -> bool:
    """Return whether an input-derived layout escape's rooting input changed layout (r73 F1).

    The capture-time forward read a layout predicate (``is_contiguous`` / ``stride`` /
    ``storage_offset``) on an ACTIVATION whose value DAG roots at a model input; memory
    format propagates through elementwise ops, so the recorded taken path may depend on
    that input's layout -- which the input contract does NOT pin (shape+dtype+device+
    strided only). The envelope fact carries the rooting leaf's capture-time stride
    tuple; this check compares it against the RAW runtime leaf's strides (shape equality
    is already contract-enforced, so the tuples are commensurable; ``storage_offset``
    never propagates to a fresh-storage intermediate, and input VIEWS are the r31
    view-read gap's domain, so strides are the complete comparison basis).

    Equal strides -> ``False``: a same-layout runtime input reproduces every propagated
    layout bit, so an honest channels_last-on-channels_last model stays VERIFIED (zero
    collateral). Any difference -- or an unresolvable leaf, an unreadable runtime
    stride, or a malformed recorded tuple (foreign-authored artifact) -- returns
    ``True`` and the run ceilings UNVERIFIABLE, never a false VERIFIED and never an
    over-claimed DIVERGED (the derived layout is not proven different, merely
    unverifiable). Captures that never read intermediate layout carry no such fact and
    can never trigger here.
    """

    positions = _model_input_arity_positions(descriptor)
    for _witness, fact in _model_input_metadata_facts(descriptor):
        recorded_facts = fact.get("facts")
        if not isinstance(recorded_facts, Mapping):
            continue
        if _INPUT_DERIVED_LAYOUT_FACT_NAME not in recorded_facts:
            continue
        recorded = recorded_facts[_INPUT_DERIVED_LAYOUT_FACT_NAME]
        if not isinstance(recorded, Sequence) or isinstance(recorded, (str, bytes)):
            return True
        try:
            recorded_stride = [int(v) for v in recorded]
        except (TypeError, ValueError):
            return True
        raw_position = fact.get("position")
        position = tuple(raw_position) if isinstance(raw_position, (list, tuple)) else raw_position
        path = _fact_path_tuple_or_none(fact)
        if path is None:
            return True
        try:
            root = _input_site_value(inputs, position, positions)
            runtime_leaf = _value_at_path(root, path)
        except (KeyError, IndexError, TypeError, AttributeError):
            return True
        if not isinstance(runtime_leaf, torch.Tensor):
            return True
        try:
            runtime_stride = [int(v) for v in runtime_leaf.stride()]
        except (RuntimeError, TypeError, ValueError, NotImplementedError):
            return True
        if runtime_stride != recorded_stride:
            return True
    return False


def _inventory_site_positions(descriptor: SparseRunDescriptor) -> set[Any]:
    """Decode the required input-site set from the REQUIRED boundary record (r71 A2).

    THE run-side site authority is the witness-free ``input_boundary`` record --
    replay-consumed structure, exact-validated at parse against the MODEL_INPUT slot
    bindings and dense positional roots. The required-witness inventory is a
    redundant mirror; a disagreement here (impossible post-parse) fails closed typed
    (belt against parser regressions), and runtime site selection never re-derives
    arity from surviving witnesses (the secA-F1 self-referential lane).
    """

    from torchlens._runnable import decode_input_site_position

    try:
        positions = {
            decode_input_site_position(site.position) for site in descriptor.input_boundary
        }
    except ValueError as exc:
        raise RunPreconditionError(
            f"Malformed input-boundary site position: {exc}",
            code=RunnableErrorCode.CONTEXT_FIELD_INVALID.value,
        ) from exc
    row = next(
        (
            family
            for family in descriptor.required_witness_inventory.families
            if family.family == "input_structure"
        ),
        None,
    )
    mirror: set[Any] | None = None
    if row is not None:
        try:
            mirror = {decode_input_site_position(member) for member in row.members}
        except ValueError:
            mirror = None
    if mirror != positions:
        raise RunPreconditionError(
            "Input-boundary site record disagrees with the inventory mirror; the "
            "descriptor is internally contradictory.",
            code=RunnableErrorCode.CONTEXT_FIELD_INVALID.value,
        )
    return positions


def _model_input_arity_positions(descriptor: SparseRunDescriptor) -> set[Any]:
    """Return every distinct model-input site position, tensor and non-tensor.

    The tensor-slot positions alone undercount arity when a model site carries a
    non-tensor Python argument (e.g. ``forward(x, flag)``): the single tensor
    slot would falsely trigger the "single bare input" shortcut and bind the
    whole runtime argument list as one tensor. Including witnessed non-tensor
    leaf positions makes the shortcut fire only for a genuinely single-argument
    model, so a mixed ``[tensor, python_arg]`` call binds each site correctly.

    r69 A: the parse-validated required inventory is the site AUTHORITY (union
    belt below); a descriptor whose slot/literal-derived positions escape the
    inventory set fails closed typed instead of silently widening arity.
    """

    positions = {
        slot.input_binding.model_site_position
        for slot in descriptor.tensor_slots
        if slot.role is TensorSlotRole.MODEL_INPUT and slot.input_binding is not None
    }
    for _witness, fact in _model_input_literal_facts(descriptor):
        position = fact.get("position")
        if isinstance(position, (list, tuple)):
            positions.add(tuple(position))
        elif position is not None:
            positions.add(position)
    inventory_positions = _inventory_site_positions(descriptor)
    stray = {
        position for position in positions if isinstance(position, tuple) and len(position) == 2
    } - inventory_positions
    if stray:
        raise RunPreconditionError(
            f"Descriptor input positions {sorted(stray, key=repr)!r} escape the "
            "parse-validated required-witness inventory.",
            code=RunnableErrorCode.CONTEXT_FIELD_INVALID.value,
        )
    return positions | inventory_positions


def _runtime_top_level_positions(inputs: Any, positions: set[Any]) -> set[Any] | None:
    """Enumerate the runtime call's TOP-LEVEL model-input site positions (r42 corr1_1).

    Mirrors :func:`_input_site_value`'s dispatch so the actual top-level site SET aligns with
    the recorded ``("arg", i)`` / ``("kwarg", key)`` positions. A single bare positional site is
    NOT exploded (a list/dataclass/dict root is one site, its inner leaves are checked by the
    tree/literal contracts). Returns ``None`` for an input spelling the standard binder handles
    (and raises on) itself, so the arity check never double-reports a shape the binder rejects.
    """

    if _positions_are_mixed(positions):
        try:
            args, kwargs = _split_mixed_inputs(inputs)
        except TypeError:
            return None
        return {("arg", index) for index in range(len(args))} | {("kwarg", key) for key in kwargs}
    kinds = {p[0] for p in positions if isinstance(p, tuple) and len(p) == 2}
    if kinds == {"arg"}:
        if len(positions) == 1 and ("arg", 0) in positions:
            # Single bare positional: the whole ``inputs`` IS the one site (never exploded).
            return {("arg", 0)}
        if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
            return None
        return {("arg", index) for index in range(len(inputs))}
    if kinds == {"kwarg"}:
        if not isinstance(inputs, Mapping):
            return None
        return {("kwarg", key) for key in inputs}
    return None
