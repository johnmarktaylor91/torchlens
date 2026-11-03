"""Conformance cells that drive a backend through its ``BackendProtocol``.

The c0 cells here run for every adapter whose name is a registered backend
with a protocol factory (torch and the five previews). The reserved ``fake``
plant substrate and external providers have no registered spec, so they
define none of these slots; their c0 pack stays the provider-API trio.

Cells (SPEC section 4.2):

- ``c0_protocol_complete``: the factory returns an object with every member.
- ``c0_framework_requirement_consistent``: the declared range, tested
  version and pin agree with each other.
- ``c0_refusal_codes_registered``: every code the backend says it raises is
  in the closed registry.
- ``c0_capability_table_complete``: every capability field holds a value of
  its declared kind.

The c1 and c2 cells here run for every adapter that exposes a protocol view
(``adapter.protocol()``): the torch reference, the five previews and the
reserved ``fake`` substrate, whose view is torch's protocol behind a plant
layer. Per roster row: graph integrity, shape metadata against the payload, no
silent gap, intermediates against the row's NumPy oracle, replay validation
passes, a tampered payload is named with a reason, constants are recorded
parents, RNG is consumed once (lazy backends); once per backend: a typed
refusal for every False capability flag and for every public trace option the
backend does not implement; per row at c2: payloads survive the round trip bit
for bit and the reloaded trace still validates.

Every request recipe, probe value and plant-to-cell mapping lives in one table
here so a later change is one edit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._adapters import RosterModel
from ._c0_protocol_cases import (
    C0_PROTOCOL_CASES as C0_PROTOCOL_CASES,
    registered_protocol_spec as registered_protocol_spec,
    run_c0_protocol_cases as run_c0_protocol_cases,
)
from ._cases import (
    C2_MODEL_CASES,
    TIER_C1,
    TIER_C2,
    CaseResult,
    _failed,
    _failure_detail,
    _passed,
)
from ._validation_cells import (
    _is_input,
    capture_for_validation,
    names_reason_problems,
    tamper_target,
    tampered_payload,
    validation_status_problems,
)

__tl_layer__ = "L9"


def graph_integrity_problems(trace: Any) -> tuple[str, ...]:
    """Structural problems in a captured graph, each as one line.

    Parents resolve to recorded ops, parent and child lists are symmetric and
    input ops have no parents. The orphan rule (a parentless non-input op
    must be a constant producer or buffer) joins once the shared constants
    rule records ``is_constant_layer``.
    """

    rows = list(trace._layer_list)
    by_key: dict[str, Any] = {}
    for op in rows:
        for key in (getattr(op, "_label_raw", None), op.label, getattr(op, "layer_label", None)):
            if key:
                by_key.setdefault(key, op)
    problems: list[str] = []
    for op in rows:
        for parent_key in tuple(op.parents or ()):
            parent = by_key.get(parent_key)
            if parent is None:
                problems.append(f"{op.label}: parent {parent_key!r} unresolved")
                continue
            kids = set(parent.children or ())
            own_keys = {getattr(op, "_label_raw", None), op.label, getattr(op, "layer_label", None)}
            if not kids & own_keys:
                problems.append(f"{op.label}: not listed among children of {parent_key}")
        for child_key in tuple(op.children or ()):
            if child_key not in by_key:
                problems.append(f"{op.label}: child {child_key!r} unresolved")
    inputs = list(trace.input_ops)
    if not inputs:
        problems.append("no input ops")
    if not list(trace.output_ops):
        problems.append("no output ops")
    problems.extend(f"input op {op.label} has parents" for op in inputs if tuple(op.parents or ()))
    return tuple(problems)


def shape_metadata_problems(trace: Any, host_copy: Any) -> tuple[str, ...]:
    """Ops whose recorded ``shape`` disagrees with the saved payload's shape."""

    problems: list[str] = []
    for op in trace._layer_list:
        shape = getattr(op, "shape", getattr(op, "tensor_shape", None))
        payload = getattr(op, "out", None)
        if shape is None or payload is None:
            continue
        got = _payload_shape(payload, host_copy)
        if got is None:
            continue
        if tuple(shape) != got:
            problems.append(f"{op.label}: recorded shape {tuple(shape)} != payload {got}")
    return tuple(problems)


def _payload_shape(payload: Any, host_copy: Any) -> tuple[int, ...] | None:
    """The host-copied payload's shape; ``None`` for a non-array payload (outside this check)."""

    try:
        return tuple(host_copy(payload).shape)
    except Exception:  # noqa: BLE001 - a non-array payload is outside this check
        return None


def silent_gap_reason(trace: Any) -> str | None:
    """The recorded verification ceiling when the capture flagged itself, else ``None``."""

    if trace._provenance_verified is False:
        return str(trace._provenance_verification_reason or "unrecorded reason")
    return None


# ----------------------------------------------------------------------------
# Case ids (SPEC 4.2) and the protocol view of an adapter.

#: Per-row c1 cells every protocol-bearing adapter defines, in report order
#: (``c1_intermediates_match`` joins after the first three when the row declares
#: an oracle; the constants and RNG cells join at the end when the row is flagged).
C1_PROTOCOL_ROW_CASES: tuple[str, ...] = (
    "c1_graph_integrity",
    "c1_shape_metadata_matches_payload",
    "c1_no_silent_gap",
    "c1_validation_passes",
    "c1_validation_names_reason",
)
C1_INTERMEDIATES_CASE = "c1_intermediates_match"
C1_CONSTANTS_CASE = "c1_constant_parents_recorded"
C1_RNG_CASE = "c1_rng_consumed_once"
C1_REFUSAL_PREFIX = "c1_refusal_"
C1_REFUSAL_OPTION_PREFIX = "c1_refusal_trace_option_"
#: Per-row c2 cells every protocol-bearing adapter defines, after the v1 trio.
C2_PROTOCOL_ROW_CASES: tuple[str, ...] = ("c2_payload_values_equal", "c2_validation_after_load")

#: The fake provider's protocol-era plants and the one cell each must fail.
PLANT_INTENDED_CELL: dict[str, str] = {
    "untyped_refusal": f"{C1_REFUSAL_PREFIX}intermediate_derived_grads",
    "bare_bool_validation": "c1_validation_names_reason",
    "missing_constant_payload": C1_CONSTANTS_CASE,
    "shape_drift_after_load": "c2_payload_values_equal",
    "flagged_trace": "c1_no_silent_gap",
    "shape_metadata_drift": "c1_shape_metadata_matches_payload",
}


def adapter_protocol(adapter: Any) -> Any | None:
    """The adapter's protocol view, or ``None`` when it exposes none.

    Parameters
    ----------
    adapter:
        A resolved conformance adapter.

    Returns
    -------
    Any | None
        The ``BackendProtocol`` object (or the fake's planted view); ``None``
        for providers without one, whose protocol cells then do not exist.
    """

    probe = getattr(adapter, "protocol", None)
    if not callable(probe):
        return None
    try:
        return probe()
    except Exception:  # noqa: BLE001 - a failing factory is c0's red cell, not a slot here
        return None


def false_capability_flags(protocol: Any) -> tuple[str, ...]:
    """Refusable capability flags the backend declares False, in table order."""

    from torchlens.backends._backend_protocol import REFUSABLE_CAPABILITY_FLAGS

    table = protocol.capabilities()
    return tuple(flag for flag in REFUSABLE_CAPABILITY_FLAGS if getattr(table, flag, None) is False)


def foreign_trace_options(protocol: Any) -> tuple[str, ...]:
    """Public trace options the backend does not implement, in spine order."""

    from torchlens.backends.registry import PUBLIC_OPTION_SPINE_TRACE_OPTIONS

    implemented = set(protocol.trace_options())
    return tuple(
        option for option in PUBLIC_OPTION_SPINE_TRACE_OPTIONS if option not in implemented
    )


def c1_row_case_ids(adapter: Any, model: RosterModel) -> tuple[str, ...]:
    """Every c1 case id one roster row defines for ``adapter``, in report order."""

    ids = ["c1_capture_completes"]
    if model.expected_op_kinds is not None:
        ids.append("c1_op_inventory_complete")
    ids += ["c1_reference_parity", "c1_deterministic_repeat"]
    protocol = adapter_protocol(adapter)
    if protocol is None:
        return tuple(ids)
    ids += list(C1_PROTOCOL_ROW_CASES[:3])
    if model.expected_intermediates:
        ids.append(C1_INTERMEDIATES_CASE)
    ids += list(C1_PROTOCOL_ROW_CASES[3:])
    if model.has_constants:
        ids.append(C1_CONSTANTS_CASE)
    if model.random_init and getattr(protocol, "materialization", "eager") == "lazy":
        ids.append(C1_RNG_CASE)
    return tuple(ids)


def c1_tail_case_ids(adapter: Any) -> tuple[str, ...]:
    """The once-per-backend c1 case ids (consumption, then every refusal cell)."""

    ids = ["c1_capabilities_consumed"]
    protocol = adapter_protocol(adapter)
    if protocol is None:
        return tuple(ids)
    ids += [f"{C1_REFUSAL_PREFIX}{flag}" for flag in false_capability_flags(protocol)]
    ids += [f"{C1_REFUSAL_OPTION_PREFIX}{name}" for name in foreign_trace_options(protocol)]
    return tuple(ids)


def c2_row_case_ids(adapter: Any, model: RosterModel) -> tuple[str, ...]:
    """Every c2 case id one roster row defines for ``adapter``, in report order."""

    ids = list(C2_MODEL_CASES)
    if adapter_protocol(adapter) is not None:
        ids += list(C2_PROTOCOL_ROW_CASES)
    return tuple(ids)


# ----------------------------------------------------------------------------
# Shared probes.


def _as_output_list(reference: Any) -> list[Any]:
    """Ground-truth outputs as the list ``validate`` takes."""

    if isinstance(reference, (tuple, list)):
        return list(reference)
    return [reference]


def constant_like_ops(trace: Any) -> list[Any]:
    """Parentless non-input ops: closure constants, literal producers, buffers.

    The shared constants rule marks them ``is_constant_layer`` (F-E); until
    that field lands, a parentless non-input op is constant-like by position.
    This is the one place the test for a constant op lives.
    """

    rows = []
    for op in trace._layer_list:
        if _is_input(op) or tuple(getattr(op, "parents", None) or ()):
            continue
        if getattr(op, "is_buffer", getattr(op, "is_buffer_layer", False)):
            continue
        rows.append(op)
    return rows


def _is_marked_constant(op: Any) -> bool:
    """Whether the record itself says the op is a constant (field or ``func_name``)."""

    if getattr(op, "is_constant_layer", getattr(op, "is_constant", None)) is True:
        return True
    return str(getattr(op, "func_name", "")) == "constant"


def _tolerance_for(array: Any) -> tuple[float, float] | None:
    """The dtype-honest replay band for a float payload; ``None`` means exact."""

    import numpy as np

    from torchlens.backends._validation_shared import float_replay_tolerances

    kind = np.asarray(array).dtype.kind
    if kind == "f":
        return float_replay_tolerances(np.finfo(np.asarray(array).dtype))
    if kind == "c":
        component = np.finfo(np.asarray(array).real.dtype)
        return float_replay_tolerances(component)
    return None


def _arrays_match(got: Any, want: Any) -> str | None:
    """One-line mismatch description between a captured and an expected array, else ``None``."""

    import numpy as np

    got_arr, want_arr = np.asarray(got), np.asarray(want)
    if got_arr.shape != want_arr.shape:
        return f"shape {got_arr.shape} != expected {want_arr.shape}"
    band = _tolerance_for(got_arr)
    if band is None:
        return None if np.array_equal(got_arr, want_arr) else "values differ (exact compare)"
    rtol, atol = band
    if np.allclose(got_arr.astype(np.float64), want_arr.astype(np.float64), rtol=rtol, atol=atol):
        return None
    gap = float(np.max(np.abs(got_arr.astype(np.float64) - want_arr.astype(np.float64))))
    return f"max abs difference {gap:.3e} outside rtol {rtol:.3e} atol {atol:.3e}"


def intermediates_problems(trace: Any, expected: dict[str, Any], host_copy: Any) -> tuple[str, ...]:
    """Captured ``out`` values of each declared op kind against the row's NumPy oracle."""

    problems: list[str] = []
    for kind, want in expected.items():
        wants = list(want) if isinstance(want, (list, tuple)) else [want]
        ops = [op for op in trace._layer_list if str(getattr(op, "func_name", "")) == kind]
        if len(ops) != len(wants):
            problems.append(f"{kind}: {len(ops)} captured ops, {len(wants)} expected values")
            continue
        for op, value in zip(ops, wants, strict=True):
            payload = getattr(op, "out", None)
            if payload is None:
                problems.append(f"{op.label}: no saved payload")
                continue
            try:
                got = host_copy(payload)
            except Exception as exc:  # noqa: BLE001 - an unreadable payload is a defect
                problems.append(f"{op.label}: host_copy raised {type(exc).__name__}: {exc}")
                continue
            mismatch = _arrays_match(got, value)
            if mismatch:
                problems.append(f"{op.label}: {mismatch}")
    return tuple(problems)


def constant_parents_problems(trace: Any, host_copy: Any) -> tuple[str, ...]:
    """Every constant-like op has a readable payload and a consumer; a flagged row has one."""

    rows = constant_like_ops(trace)
    if not rows:
        return ("no constant op recorded although the row declares has_constants",)
    problems: list[str] = []
    for op in rows:
        payload = getattr(op, "out", None)
        if payload is None:
            problems.append(f"{op.label}: constant recorded without a payload")
            continue
        try:
            host_copy(payload)
        except Exception as exc:  # noqa: BLE001 - an unreadable constant is a defect
            problems.append(f"{op.label}: payload unreadable ({type(exc).__name__}: {exc})")
        if not tuple(getattr(op, "children", None) or ()):
            problems.append(f"{op.label}: constant has no consuming op")
        marked = getattr(op, "is_constant_layer", getattr(op, "is_constant", None))
        if marked is False and not _is_marked_constant(op):
            problems.append(f"{op.label}: parentless op not marked is_constant_layer")
    return tuple(problems)


def payload_equality_problems(live: Any, loaded: Any, host_copy: Any) -> tuple[str, ...]:
    """Saved ``out`` payloads equal the reloaded ones bit for bit, shapes included."""

    import numpy as np

    loaded_by_label = {op.label: op for op in loaded._layer_list}
    problems: list[str] = []
    for op in live._layer_list:
        payload = getattr(op, "out", None)
        if payload is None:
            continue
        twin = loaded_by_label.get(op.label)
        if twin is None:
            problems.append(f"{op.label}: missing after reload")
            continue
        twin_payload = getattr(twin, "out", None)
        if twin_payload is None:
            problems.append(f"{op.label}: payload missing after reload")
            continue
        try:
            left, right = host_copy(payload), host_copy(twin_payload)
        except Exception as exc:  # noqa: BLE001 - an unreadable payload is a defect
            problems.append(f"{op.label}: host_copy raised {type(exc).__name__}: {exc}")
            continue
        if left.shape != right.shape:
            problems.append(f"{op.label}: shape {left.shape} != reloaded {right.shape}")
        elif left.dtype != right.dtype:
            problems.append(f"{op.label}: dtype {left.dtype} != reloaded {right.dtype}")
        elif not np.array_equal(left, right, equal_nan=True):
            problems.append(f"{op.label}: values differ after reload")
    return tuple(problems)


# ----------------------------------------------------------------------------
# Running the cells.


def _cell(case_id: str, tier: str, problems: tuple[str, ...], model: RosterModel) -> CaseResult:
    """A passed row when ``problems`` is empty, else a failed row with the joined detail."""

    if problems:
        return _failed(case_id, tier, "; ".join(problems)[:300], model)
    return _passed(case_id, tier, model)


def _guarded(
    case_id: str, tier: str, model: RosterModel | None, body: Callable[[], tuple[str, ...]]
) -> CaseResult:
    """Run one cell body; a typed refusal or any raise becomes a red row, never a crash."""

    try:
        problems = body()
    except Exception as exc:  # noqa: BLE001 - provider failures become red cells, never crashes
        return _failed(case_id, tier, _failure_detail(exc), model)
    if model is None:
        return (
            _failed(case_id, tier, "; ".join(problems)[:300])
            if problems
            else _passed(case_id, tier)
        )
    return _cell(case_id, tier, problems, model)


def run_c1_protocol_row_cases(
    adapter: Any, model: RosterModel, trace: Any, reference: Any
) -> list[CaseResult]:
    """The per-row protocol c1 cells for one captured roster row.

    Parameters
    ----------
    adapter:
        The resolved adapter (must expose a protocol view).
    model:
        The roster row.
    trace:
        The row's captured trace.
    reference:
        The direct-run reference output.

    Returns
    -------
    list[CaseResult]
        One row per protocol c1 case id the row defines, in report order.
    """

    protocol = adapter_protocol(adapter)
    if protocol is None:
        return []
    host_copy = protocol.host_copy
    results = [
        _guarded("c1_graph_integrity", TIER_C1, model, lambda: graph_integrity_problems(trace)),
        _guarded(
            "c1_shape_metadata_matches_payload",
            TIER_C1,
            model,
            lambda: shape_metadata_problems(trace, host_copy),
        ),
        _guarded("c1_no_silent_gap", TIER_C1, model, lambda: _silent_gap_problems(trace)),
    ]
    if model.expected_intermediates:
        expected = model.expected_intermediates
        results.append(
            _guarded(
                C1_INTERMEDIATES_CASE,
                TIER_C1,
                model,
                lambda: intermediates_problems(trace, expected, host_copy),
            )
        )
    results.extend(_c1_validation_and_tail_cells(adapter, protocol, model, trace, reference))
    return results


def _c1_validation_and_tail_cells(
    adapter: Any, protocol: Any, model: RosterModel, trace: Any, reference: Any
) -> list[CaseResult]:
    """The validation, constants and RNG cells that close a row's protocol c1 block."""

    host_copy = protocol.host_copy
    results = [
        _guarded(
            "c1_validation_passes",
            TIER_C1,
            model,
            lambda: validation_status_problems(
                protocol.validate(trace, _as_output_list(reference))
            ),
        ),
        _guarded(
            "c1_validation_names_reason",
            TIER_C1,
            model,
            lambda: _names_reason_body(adapter, protocol, model),
        ),
    ]
    if model.has_constants:
        results.append(
            _guarded(
                C1_CONSTANTS_CASE,
                TIER_C1,
                model,
                lambda: constant_parents_problems(trace, host_copy),
            )
        )
    if model.random_init and getattr(protocol, "materialization", "eager") == "lazy":
        results.append(
            _guarded(C1_RNG_CASE, TIER_C1, model, lambda: _rng_once_body(adapter, model))
        )
    return results


def _silent_gap_problems(trace: Any) -> tuple[str, ...]:
    """A flagged capture is a failed cell carrying its recorded reason."""

    reason = silent_gap_reason(trace)
    return (f"capture_verified=False: {reason}",) if reason is not None else ()


def _names_reason_body(adapter: Any, protocol: Any, model: RosterModel) -> tuple[str, ...]:
    """Fresh capture, one tampered payload, the verdict must name it with a reason."""

    import warnings

    net, example = model.builder()
    trace = capture_for_validation(adapter, net, example)
    reference = adapter.reference_output(net, example)
    target = tamper_target(trace)
    if target is None:
        return ("no saved payload to tamper",)
    with warnings.catch_warnings():
        # The direct payload write is the tamper itself, not a recipe edit.
        warnings.simplefilter("ignore")
        target.out = tampered_payload(target.out)
    status = protocol.validate(trace, _as_output_list(reference))
    return names_reason_problems(status, target, trace)


def _rng_once_body(adapter: Any, model: RosterModel) -> tuple[str, ...]:
    """Capture equals one eager forward in outputs and in RNG consumption."""

    import warnings

    import numpy as np

    rng_state = getattr(adapter, "rng_state", None)
    if not callable(rng_state):
        return ("adapter exposes no rng_state() probe",)
    as_array = getattr(adapter, "as_array", None)
    convert = as_array if callable(as_array) else np.asarray
    net, example = model.builder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = adapter.trace(net, example)
    after_capture = rng_state()
    eager_net, eager_example = model.builder()
    eager = adapter.reference_output(eager_net, eager_example)
    after_eager = rng_state()
    problems: list[str] = []
    captured = trace.output_ops[0].out if trace.output_ops else None
    if captured is None or not np.array_equal(convert(captured), convert(eager)):
        problems.append("captured output differs from the eager forward of the same seed")
    if not np.array_equal(np.asarray(after_capture), np.asarray(after_eager)):
        problems.append("RNG state after capture differs from the state after one eager forward")
    return tuple(problems)


def load_for_replay(adapter: Any, trace: Any, path: Path, loaded: Any) -> Any:
    """The reloaded trace ``c2_validation_after_load`` validates.

    An adapter whose replay validation re-executes from saved arguments
    declares ``replay_save_level`` (torch: ``"executable_with_callables"``, the
    level that ships re-execution payloads); the live trace is saved again at
    that level beside ``path`` and reloaded. Other adapters validate the
    default-level reload. A failure is returned as the exception, so only the
    after-load cell turns red.

    Parameters
    ----------
    adapter:
        The resolved adapter.
    trace:
        The live trace.
    path:
        The default-level artifact path (the replay artifact sits beside it).
    loaded:
        The default-level reload.

    Returns
    -------
    Any
        The trace to validate, or the exception that prevented producing it.
    """

    level = getattr(adapter, "replay_save_level", None)
    if level is None:
        return loaded
    import warnings

    replay_path = path.with_name(f"{path.stem}_replay{path.suffix}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter.save(trace, replay_path, level=level)
            return adapter.load(replay_path)
    except Exception as exc:  # noqa: BLE001 - a provider failure becomes this cell's red row
        return exc


def _validate_after_load_problems(protocol: Any, replay_loaded: Any, reference: Any) -> Any:
    """Problems of ``c2_validation_after_load``; a failed replay save or load is re-raised."""

    if isinstance(replay_loaded, BaseException):
        raise replay_loaded
    return validation_status_problems(protocol.validate(replay_loaded, _as_output_list(reference)))


@dataclass(frozen=True)
class C2Reload:
    """One row's saved-and-reloaded pair for the protocol c2 cells.

    Parameters
    ----------
    trace:
        The live trace that was saved.
    loaded:
        The reloaded trace.
    replay_loaded:
        The reload ``c2_validation_after_load`` validates (see
        :func:`load_for_replay`); ``None`` validates ``loaded``.
    """

    trace: Any
    loaded: Any
    replay_loaded: Any = None


def run_c2_protocol_row_cases(
    adapter: Any, model: RosterModel, reload: C2Reload, reference: Any
) -> list[CaseResult]:
    """The per-row protocol c2 cells: payload equality and validation after load.

    Parameters
    ----------
    adapter:
        The resolved adapter (must expose a protocol view).
    model:
        The roster row.
    reload:
        The live trace, its reload and the replay reload.
    reference:
        The direct-run reference output.

    Returns
    -------
    list[CaseResult]
        One row per :data:`C2_PROTOCOL_ROW_CASES`.
    """

    protocol = adapter_protocol(adapter)
    if protocol is None:
        return []
    return [
        _guarded(
            "c2_payload_values_equal",
            TIER_C2,
            model,
            lambda: payload_equality_problems(reload.trace, reload.loaded, protocol.host_copy),
        ),
        _guarded(
            "c2_validation_after_load",
            TIER_C2,
            model,
            lambda: _validate_after_load_problems(
                protocol,
                reload.loaded if reload.replay_loaded is None else reload.replay_loaded,
                reference,
            ),
        ),
    ]
