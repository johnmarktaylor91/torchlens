"""Seed, RNG, attestation, and fork utilities."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from . import _state
from ._io._torch_symbols import torch_attr
from ._runnable_state import (
    PreparedRunnableState,
    runnable_tensor_byte_digest,
)
from .errors import (
    NumericAttestationError,
    PathDivergenceError,
    RunPreconditionError,
)
from .runnable import (
    CallableRegistryEntry,
    ContractCheck,
    DivergencePolicy,
    InputAttestationFingerprint,
    LiteralAtom,
    LiteralAtomKind,
    LiteralMapping,
    LiteralSequence,
    LiteralSequenceKind,
    LiteralSlice,
    LiteralTorchSymbol,
    LiteralTupleKey,
    NonTensorLiteral,
    NumericAttestationStatus,
    PathFaithfulness,
    RunnableCallDescriptor,
    RunnableDiagnostic,
    RunnableErrorCode,
    SparseRunDescriptor,
    StateSource,
    TensorSlotDescriptor,
    TensorSlotRole,
)
from .utils.rng import (
    aten_qualname_is_seeded_rng,
)

if TYPE_CHECKING:
    from ._runnable_execution import (
        _ALLOWED_TORCH_SYMBOL_TYPES,
        _MAX_DECODE_NESTING_DEPTH,
        _RUN_FORK_COUNTER,
        _container_field_names,
    )

__all__ = (
    "_call_consumes_seeded_rng",
    "_is_dropout_qualname",
    "_dropout_call_draws_rng",
    "_named_literal_values",
    "_lacks_recorded_original_input_eligibility",
    "_attestation_inputs_match",
    "_attestation_state_matches",
    "_raise_numeric_attestation_failure",
    "_contract_check",
    "_first_failed_contract",
    "_raise_failed_contract_as_divergence",
    "_raise_first_divergence",
    "_raise_monotonic_divergence",
    "_decode_literal",
    "_decode_nonfinite_float_literal",
    "_decode_torch_symbol",
    "_field_getattr",
    "_value_at_path",
    "_input_error",
    "_op_for_label",
    "_op_for_slot",
    "_run_fork_name",
)


def _call_consumes_seeded_rng(
    call: RunnableCallDescriptor,
    registry: Mapping[str, CallableRegistryEntry],
) -> bool:
    """Return whether one replayed call actually draws from a seeded generator.

    Parameters
    ----------
    call:
        Replayed sparse call descriptor.
    registry:
        Registry-id to callable-entry map for the descriptor.

    Returns
    -------
    bool
        Whether this specific call consumes non-reproducible seeded RNG at
        replay time (op-name seeding refined by dropout ``training``/``p``).
    """

    entry = registry.get(call.registry_id)
    if entry is None:
        return False
    namespace = entry.key.namespace
    qualname = entry.key.qualname
    if not aten_qualname_is_seeded_rng(namespace, qualname):
        return False
    if _is_dropout_qualname(qualname):
        # A dropout family op draws from the RNG only when it is genuinely
        # active (training=True and p>0); eval/identity dropout is RNG-inert.
        return _dropout_call_draws_rng(call)
    return True


def _is_dropout_qualname(qualname: str | None) -> bool:
    """Return whether a captured qualname belongs to the dropout op family.

    Covers ``dropout``/``dropout_``/``feature_dropout``/``alpha_dropout``/
    ``feature_alpha_dropout`` under any namespace spelling. All members share
    the ``training``+``p`` RNG-consumption contract.
    """

    if not qualname:
        return False
    tail = qualname.rsplit(".", 1)[-1]
    if tail.endswith("_"):
        tail = tail[:-1]
    return tail.endswith("dropout")


def _dropout_call_draws_rng(call: RunnableCallDescriptor) -> bool:
    """Return whether a dropout call consumes RNG given its recorded literals.

    Dropout draws from the generator only when ``training is True`` AND ``p > 0``.
    The decision is keyed off the recorded ``training`` and ``p`` literals; when
    a value cannot be proven RNG-inert the call stays conservatively tagged as
    seeded (fail-closed: never a false ``attested``).

    Parameters
    ----------
    call:
        The dropout-family sparse call descriptor.

    Returns
    -------
    bool
        Whether the recorded dropout call actually draws from the RNG.
    """

    named = _named_literal_values(call)
    training = named.get("training", named.get("train"))
    p_value = named.get("p")
    if training is False:
        return False
    return not (
        isinstance(p_value, (int, float)) and not isinstance(p_value, bool) and p_value == 0
    )


def _named_literal_values(call: RunnableCallDescriptor) -> dict[str, Any]:
    """Map recorded literal arguments to their parameter names.

    Positional literals are named through ``call.argument_names``; keyword
    literals use their stored key. Only the safe literal grammar is decoded.

    Parameters
    ----------
    call:
        Sparse call descriptor whose literal leaves are being named.

    Returns
    -------
    dict[str, Any]
        Parameter name to decoded literal value mapping.
    """

    values: dict[str, Any] = {}
    for literal_argument in call.literal_arguments:
        path = literal_argument.argument_path
        if len(path) != 2:
            continue
        root, key = path
        if root == "args" and isinstance(key, int) and 0 <= key < len(call.argument_names):
            values[call.argument_names[key]] = _decode_literal(literal_argument.value)
        elif root == "kwargs":
            values[str(key)] = _decode_literal(literal_argument.value)
    return values


def _lacks_recorded_original_input_eligibility(
    descriptor: SparseRunDescriptor,
    layer: Any,
) -> bool:
    """Return whether the ARCHIVE ITSELF can never support numeric attestation.

    Distinguishes a SAVE-time eligibility gap from a run-time input change. With a
    ``save=`` selector that does not select the model input, ``_capture_activation_blob_specs``
    records no ``original_input_digests`` and no ``input_fingerprints`` at all, so
    ``_attestation_inputs_match`` can never be satisfied by ANY caller input. Naming that
    case keeps ``NOT_APPLICABLE`` honest instead of implicitly blaming the caller.

    Parameters
    ----------
    descriptor:
        Runnable descriptor whose model-input slots define what must be recorded.
    layer:
        Present activation payload layer descriptor.

    Returns
    -------
    bool
        ``True`` when the descriptor declares model-input slots but the archive records
        no original-input digests or no input fingerprints for them.
    """

    expected_slot_ids = {
        slot.slot_id for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.MODEL_INPUT
    }
    if not expected_slot_ids:
        return False
    recorded_digests = {digest.slot_id for digest in layer.original_input_digests}
    recorded_fingerprints = {
        fingerprint.slot_id for fingerprint in (getattr(layer, "input_fingerprints", ()) or ())
    }
    return not recorded_digests or not recorded_fingerprints


def _attestation_inputs_match(
    descriptor: SparseRunDescriptor,
    layer: Any,
    input_byte_digests: Mapping[str, str],
    input_fingerprints: Mapping[str, InputAttestationFingerprint],
) -> bool:
    """Return whether the run's inputs are LOGICALLY and PHYSICALLY original.

    r35 hon1_3 (H-a): eligibility is layout-strict. Alongside the logical byte
    digests, every recorded ``InputAttestationFingerprint`` must equal the
    fingerprint of the value that actually seeds execution -- sizes, strides,
    storage offset, memory-format flags, conj/neg bits, device, subclass class,
    grad/inference metadata, and data-pointer alignment class. A physical twin
    (byte-identical values, different layout) is changed-input-for-attestation
    ONLY: ``not_applicable``, with path faithfulness untouched.
    """

    expected_slot_ids = {
        slot.slot_id for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.MODEL_INPUT
    }
    observed_slot_ids = {digest.slot_id for digest in layer.original_input_digests}
    if not expected_slot_ids or observed_slot_ids != expected_slot_ids:
        return False
    if not all(
        input_byte_digests.get(digest.slot_id) == digest.byte_digest
        for digest in layer.original_input_digests
    ):
        return False
    recorded_fingerprints = tuple(getattr(layer, "input_fingerprints", ()) or ())
    if {fingerprint.slot_id for fingerprint in recorded_fingerprints} != expected_slot_ids:
        # v2 requires a fingerprint per input slot; anything else is ineligible
        # (fail-safe: never attest without the physical identity proof).
        return False
    return all(
        input_fingerprints.get(fingerprint.slot_id) == fingerprint
        for fingerprint in recorded_fingerprints
    )


def _attestation_state_matches(
    descriptor: SparseRunDescriptor,
    layer: Any,
    state: PreparedRunnableState,
    state_byte_digests: Mapping[str, str],
    trace: Any,
) -> bool:
    """Return whether runtime state is capture-equivalent rather than random.

    r35 corr2_5: eligibility is PARTITIONED by persistence. PERSISTENT slots
    (the canonical ``state_dict``) compare against the activation layer's
    ``capture_state_digests`` (which by construction contain only canonical
    entries). USED NON-PERSISTENT buffer slots are separately required to
    originate from the present, schema-valid, load-validated capture-embedded
    ``runnable_nonpersistent_buffer_v1`` family and to match its byte digests;
    staged user state can never supply them. Comparison uses PRE-EXECUTION
    digests so a slot a call mutates mid-run is judged by its capture-start
    state.
    """

    persistent_slots: dict[str, TensorSlotDescriptor] = {}
    nonpersistent_slots: dict[str, TensorSlotDescriptor] = {}
    for slot in descriptor.tensor_slots:
        binding = slot.state_binding
        if binding is None:
            continue
        if binding.persistent:
            persistent_slots[binding.state_dict_name] = slot
        else:
            nonpersistent_slots[binding.state_dict_name] = slot
    if not persistent_slots and not nonpersistent_slots:
        return True
    if persistent_slots:
        if state.state_source not in {
            StateSource.EMBEDDED_CAPTURE_STATE,
            StateSource.USER_STATE_DICT,
        }:
            return False
        expected = {item.state_dict_name: item.byte_digest for item in layer.capture_state_digests}
        if set(expected) != set(persistent_slots):
            return False
        if not all(
            state_byte_digests.get(slot.slot_id) == expected[name]
            for name, slot in persistent_slots.items()
        ):
            return False
    if nonpersistent_slots:
        embedded = trace._runnable.embedded_nonpersistent_buffers
        if not isinstance(embedded, Mapping):
            return False
        for name, slot in nonpersistent_slots.items():
            recorded = embedded.get(name)
            if not isinstance(recorded, torch.Tensor):
                return False
            try:
                recorded_digest = runnable_tensor_byte_digest(recorded)
            except Exception:
                return False
            if state_byte_digests.get(slot.slot_id) != recorded_digest:
                return False
    return True


def _raise_numeric_attestation_failure(fork: Any, check: ContractCheck) -> None:
    """Rollback and raise the mandatory saved-activation mismatch tripwire."""

    _state.unregister_log(fork)
    diagnostic = check.diagnostic
    raise NumericAttestationError(
        diagnostic.message if diagnostic is not None else "Numeric attestation failed.",
        code=RunnableErrorCode.NUMERIC_ATTESTATION_FAILED.value,
        path_faithfulness=PathFaithfulness.DIVERGED,
        numeric_attestation=NumericAttestationStatus.NUMERIC_ATTESTATION_FAILED,
        first_mismatch=diagnostic,
        contract_check=check,
    )


def _contract_check(
    name: str,
    passed: bool,
    code: RunnableErrorCode,
    message: str,
    *,
    affected_op_labels: tuple[str, ...] = (),
    details: tuple[tuple[str, str], ...] = (),
) -> ContractCheck:
    """Build one honesty check with a diagnostic only on contradiction."""

    diagnostic = None
    if not passed:
        diagnostic = RunnableDiagnostic(
            code=code,
            message=message,
            registry_id=None,
            affected_op_labels=affected_op_labels,
            recorded_runtime=None,
            current_runtime=str(torch.__version__),
            detection_stage="run_honesty_contract",
            resolver_provenance=None,
            analysis_load_available=True,
            details=details,
        )
    return ContractCheck(name=name, passed=passed, diagnostic=diagnostic)


def _first_failed_contract(checks: Sequence[ContractCheck]) -> ContractCheck | None:
    """Return the first failed contract check, or ``None`` when all passed (r39 corr2_4)."""

    return next((check for check in checks if not check.passed), None)


def _raise_failed_contract_as_divergence(
    failed: ContractCheck, *, fork: Any | None, cause: BaseException | None = None
) -> None:
    """Roll back and raise a first-failed contract check as a typed ``PathDivergenceError``.

    The single typed-divergence raise shared by hard admission (:func:`_raise_first_divergence`),
    the call loop (r39 corr2_4), and the live provider's classify-at-native-failure fold (r41
    hon1_3): a call that throws AFTER an input contract check already failed is INPUT
    DIVERGENCE, not resolved-callable ``RuntimeSignatureDriftError``, so it must surface as
    ``PathDivergenceError`` carrying that first failed check under either policy. ``cause``
    (live provider) chains the native error as ``__cause__`` so the underlying torch failure
    stays inspectable.
    """

    if fork is not None:
        _state.unregister_log(fork)
    diagnostic = failed.diagnostic
    error = PathDivergenceError(
        diagnostic.message if diagnostic is not None else "Sparse run path diverged.",
        code=(
            diagnostic.code.value
            if diagnostic is not None
            else RunnableErrorCode.CALL_STRUCTURE_MISMATCH.value
        ),
        path_faithfulness=PathFaithfulness.DIVERGED,
        first_mismatch=diagnostic,
        contract_check=failed,
    )
    if cause is not None:
        raise error from cause
    raise error


def _raise_first_divergence(
    checks: Sequence[ContractCheck],
    policy: DivergencePolicy,
    *,
    fork: Any | None,
) -> None:
    """Raise and discard transactional state at the first observed contradiction."""

    failed = _first_failed_contract(checks)
    if failed is None or policy is DivergencePolicy.RETURN_DIVERGED:
        return
    _raise_failed_contract_as_divergence(failed, fork=fork)


def _raise_monotonic_divergence(
    fork: Any,
    status: PathFaithfulness,
    mismatch: RunnableDiagnostic | None,
    policy: DivergencePolicy,
    *,
    unregister_fork: bool = True,
) -> None:
    """Enforce strict policy for an inherited monotonic divergence mark.

    ``unregister_fork=False`` is the fast-live provider's spelling: its "fork"
    is the USER'S live Trace, which a refusal must never evict from the process
    registry. Every other provider passes a throwaway transactional fork and
    keeps the default discard.
    """

    if status is not PathFaithfulness.DIVERGED or policy is DivergencePolicy.RETURN_DIVERGED:
        return
    if unregister_fork:
        _state.unregister_log(fork)
    raise PathDivergenceError(
        "Sparse run Trace retains a prior path divergence and cannot become faithful.",
        code=(
            mismatch.code.value
            if mismatch is not None
            else RunnableErrorCode.POISONED_RUN_REFUSED.value
        ),
        path_faithfulness=status,
        first_mismatch=mismatch,
    )


def _decode_literal(value: NonTensorLiteral | LiteralTupleKey, _depth: int = 0) -> Any:
    """Decode one safe sparse literal without importing artifact-selected code."""

    if _depth > _MAX_DECODE_NESTING_DEPTH:
        raise ValueError(
            f"Runnable literal nesting exceeds the maximum decode depth of "
            f"{_MAX_DECODE_NESTING_DEPTH}."
        )
    if isinstance(value, LiteralAtom):
        # ``ELLIPSIS`` and ``NONE`` both carry ``value is None`` on the wire (``...`` has
        # no JSON-native representation), so the atom KIND -- not the stored value -- is
        # what disambiguates a real ``None`` index from a ``...`` index at decode time.
        if value.kind is LiteralAtomKind.ELLIPSIS:
            return Ellipsis
        if value.kind is LiteralAtomKind.NONFINITE_FLOAT:
            return _decode_nonfinite_float_literal(value.value)
        return value.value
    if isinstance(value, LiteralSlice):
        return slice(
            _decode_literal(value.start, _depth + 1),
            _decode_literal(value.stop, _depth + 1),
            _decode_literal(value.step, _depth + 1),
        )
    if isinstance(value, LiteralTupleKey):
        return tuple(_decode_literal(item, _depth + 1) for item in value.items)
    if isinstance(value, LiteralSequence):
        items = [_decode_literal(item, _depth + 1) for item in value.items]
        return tuple(items) if value.kind is LiteralSequenceKind.TUPLE else items
    if isinstance(value, LiteralMapping):
        return {
            _decode_literal(entry.key, _depth + 1): _decode_literal(entry.value, _depth + 1)
            for entry in value.entries
        }
    if isinstance(value, LiteralTorchSymbol):
        return _decode_torch_symbol(value.qualname)
    raise TypeError(f"Unknown sparse literal type {type(value).__name__}.")


def _decode_nonfinite_float_literal(value: Any) -> float:
    """Decode one non-finite float atom payload.

    Parameters
    ----------
    value:
        Serialized non-finite float payload.

    Returns
    -------
    float
        ``nan``, ``inf``, or ``-inf``.
    """

    if value == "nan":
        return float("nan")
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    raise RunPreconditionError(
        f"Unsupported non-finite float literal payload {value!r}.",
        code=RunnableErrorCode.UNSUPPORTED_LITERAL.value,
    )


def _decode_torch_symbol(qualname: str) -> Any:
    """Decode one allowlisted non-callable torch symbolic literal.

    ``torch.device(...)`` round-trips through the device constructor. Every other
    accepted symbol must be a bare ``torch.<name>`` lookup resolving to a
    dtype / layout / memory_format / qscheme instance or the ``torch.Size`` type;
    dotted attribute traversal, modules, callables (other than ``torch.Size``),
    and any other attribute are rejected as unsupported literals.
    """

    if qualname.startswith("torch.device(") and qualname.endswith(")"):
        # r41 secC: the device constructor is inert value construction but rejects a
        # malformed payload with a raw torch error; an attacker-edited qualname must
        # surface as the SAME typed literal refusal as every other branch, never a
        # bare ``RuntimeError`` outside the ``RunnableErrorCode`` vocabulary.
        try:
            return torch.device(qualname[13:-1])
        except (RuntimeError, ValueError, TypeError) as exc:
            raise RunPreconditionError(
                f"Unsupported torch literal symbol {qualname!r}.",
                code=RunnableErrorCode.UNSUPPORTED_LITERAL.value,
            ) from exc
    name = qualname.removeprefix("torch.")
    if name == qualname or "." in name or not name.isidentifier():
        raise RunPreconditionError(
            f"Unsupported torch literal symbol {qualname!r}.",
            code=RunnableErrorCode.UNSUPPORTED_LITERAL.value,
        )
    # r42 secC_1 / r45: the shared ``torch_attr`` helper reads ``torch.__dict__`` WITHOUT firing
    # ``torch.__getattr__``, so an attacker qualname (``torch._inductor`` / ``_dynamo`` /
    # ``_export`` / ``onnx`` / deprecated ``has_cuda``) triggers NO lazy submodule import and
    # invokes NO deprecated ``replacement()`` -- both unrequested side effects the old
    # ``getattr(torch, name, None)`` ran, one of which could also leak a raw ``ImportError``
    # outside the typed vocabulary. Every allowlisted dtype/layout/memory_format/qscheme/``Size``
    # symbol is a real ``torch.__dict__`` entry and still resolves; everything else returns
    # ``None`` -> the typed refusal below.
    symbol = torch_attr(name)
    if symbol is torch.Size or isinstance(symbol, _ALLOWED_TORCH_SYMBOL_TYPES):
        return symbol
    raise RunPreconditionError(
        f"Unsupported torch literal symbol {qualname!r}.",
        code=RunnableErrorCode.UNSUPPORTED_LITERAL.value,
    )


def _field_getattr(current: Any, component: Any) -> Any:
    """Read one attribute-path component against a structurally-known field only.

    Attacker-controlled path strings from an untrusted bundle reach this function
    (``input_binding.container_path``, the recorded literal-witness ``fact["path"]``,
    and the reconstructed output ``slot.output_path``). An unconstrained ``getattr``
    would fire arbitrary victim-object descriptor getters and could walk dunder chains
    (``__class__.__init__.__globals__`` ...), so the component must be a string that is
    STRUCTURALLY PRESENT as a declared field on the current object's type -- a dataclass
    field, a namedtuple ``_fields`` entry, or a ``torch.return_types`` structseq field.
    Dunder / descriptor attributes (``__class__``, ``__dict__``, ...) are never declared
    fields, so this check inherently excludes the escape chains; anything else raises
    ``AttributeError`` and every caller fails closed.
    """

    if not isinstance(component, str):
        raise AttributeError(f"Non-string attribute component {component!r}.")
    allowed: set[str] = set()
    if dataclasses.is_dataclass(current) and not isinstance(current, type):
        allowed.update(field.name for field in dataclasses.fields(current))
    allowed.update(_container_field_names(current))
    if component not in allowed:
        raise AttributeError(
            f"Attribute component {component!r} is not a structurally-known field of "
            f"{type(current).__name__}."
        )
    return getattr(current, component)


def _value_at_path(value: Any, path: Sequence[str | int]) -> Any:
    """Read one list/tuple/mapping/object path from a runtime value.

    Attribute traversal is constrained to structurally-known container fields via
    :func:`_field_getattr`; string components are never fed to an unconstrained
    ``getattr`` (untrusted-bundle path strings could otherwise walk descriptor / dunder
    chains).

    One synthetic component form (r29-C2) is decoded: a terminal
    ``EMPTY_CONTAINER_PATH_MARKER`` resolves to the KIND string of the empty container at
    the parent path (so a runtime empty container of a different kind, or a non-empty/scalar
    value, diverges). Mapping lookups bind by CANONICAL ENCODED TOKEN for every key
    (r69 D): each runtime candidate is encoded through the one
    ``encode_mapping_key`` authority and compared by exact token type/value -- the
    r29 ``(BOOL_KEY_PATH_TAG, bool)`` tuple spelling is retired (no producer emits
    it; a decoded-value lookup lane would reopen hash-equality conflation).
    """

    from torchlens._input_walk import encode_mapping_key
    from torchlens._io.runnable import (
        EMPTY_CONTAINER_PATH_MARKER,
        empty_container_kind,
    )
    from torchlens.ir.container import get_registered_container

    current = value
    for component in path:
        if component == EMPTY_CONTAINER_PATH_MARKER:
            kind = empty_container_kind(current)
            if kind is None:
                raise KeyError(EMPTY_CONTAINER_PATH_MARKER)
            return kind
        # r67 C2 (corr1-3): a registered container resolves indexed components by
        # RE-FLATTENING through its registration -- never ``obj[index]`` (the type may
        # not support indexing at all). A missing/throwing registration raises typed
        # KeyError (bind-side fence), never an AttributeError crash.
        if not isinstance(current, (Mapping, list, tuple)) and not isinstance(component, str):
            registration = get_registered_container(type(current))
            if registration is not None:
                try:
                    children = list(registration.flatten(current)[0])
                except Exception as exc:
                    raise KeyError(f"registered flatten failed: {exc!r}") from exc
                if not isinstance(component, int) or not (0 <= component < len(children)):
                    raise KeyError(component)
                current = children[component]
                continue
        if isinstance(current, Mapping):
            # r69 D: ONE canonical key identity end to end -- EVERY mapping lookup
            # (including raw-looking str/int components) encodes each runtime
            # candidate through the same ``encode_mapping_key`` authority and
            # compares exact canonical token type/value. A persisted component is
            # never decoded for a runtime value-equality scan (NaN could not bind
            # itself and Python hash/equality conflated bool/int/float twins).
            # Candidates the grammar refuses are skipped (they can never have been
            # recorded); multiple token matches fail closed as
            # ``ambiguous_mapping_key``.
            matches = []
            for candidate in current.keys():
                try:
                    token = encode_mapping_key(candidate)
                except ValueError:
                    continue
                if type(token) is type(component) and token == component:
                    matches.append(candidate)
            if not matches:
                raise KeyError(component)
            if len(matches) > 1:
                raise KeyError(
                    f"ambiguous_mapping_key: {len(matches)} runtime keys encode to "
                    f"the canonical token {component!r}"
                )
            current = current[matches[0]]
        elif isinstance(component, int):
            current = current[component]
        else:
            current = _field_getattr(current, component)
    return current


def _input_error(
    code: RunnableErrorCode,
    slot: TensorSlotDescriptor,
    message: str,
) -> RunPreconditionError:
    """Build a typed input-binding precondition error."""

    return RunPreconditionError(message, code=code.value, slot_id=slot.slot_id)


def _op_for_label(trace: Any, label: str) -> Any | None:
    """Resolve a descriptor op label against a fork's lookup aliases."""

    layer_dict = getattr(trace, "layer_dict_all_keys", {}) or {}
    if label in layer_dict:
        return layer_dict[label]
    return next(
        (op for op in getattr(trace, "layer_list", ()) if str(getattr(op, "label", "")) == label),
        None,
    )


def _op_for_slot(trace: Any, slot_id: str) -> Any | None:
    """Resolve the cooked Op named by a ``slot:<label>`` descriptor ID."""

    return _op_for_label(trace, slot_id.removeprefix("slot:"))


def _run_fork_name(trace: Any) -> str:
    """Return a process-unique monotonic Trace label for a run transaction."""

    base_name = trace.trace_label or "trace"
    return f"{base_name}_fork_{next(_RUN_FORK_COUNTER)}"
