"""Conformance cells that drive a backend through its ``BackendProtocol``.

The C0 cells here run for every adapter whose name is a registered backend
with a protocol factory (torch and the five previews). The reserved ``fake``
plant substrate and external providers have no registered spec, so they
define none of these slots; their C0 pack stays the provider-API trio.

Cells (SPEC section 4.2):

- ``c0_protocol_complete``: the factory returns an object with every member.
- ``c0_framework_requirement_consistent``: the declared range, tested
  version and pin agree with each other.
- ``c0_refusal_codes_registered``: every code the backend says it raises is
  in the closed registry.
- ``c0_capability_table_complete``: every capability field holds a value of
  its declared kind.

Backend-neutral C1 helpers (graph integrity, shape metadata against the
payload, no silent gap) are exposed for the C1 pack; the pack wires them in
the suite-completion packet so slot counts move once.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

from ._cases import CaseResult, _failed, _passed

__tl_layer__ = "L9"

#: The protocol-level C0 case ids, in report order.
C0_PROTOCOL_CASES: tuple[str, ...] = (
    "c0_protocol_complete",
    "c0_framework_requirement_consistent",
    "c0_refusal_codes_registered",
    "c0_capability_table_complete",
)


def registered_protocol_spec(backend: str) -> Any | None:
    """The registered spec for ``backend`` when it carries a protocol factory, else ``None``."""

    from ..backends.registry import UnknownBackendError, get_backend_spec

    try:
        spec = get_backend_spec(backend)
    except UnknownBackendError:
        return None
    return spec if spec.protocol is not None else None


def run_c0_protocol_cases(backend: str) -> list[CaseResult]:
    """The protocol-level C0 cells for one registered backend.

    Parameters
    ----------
    backend:
        Registered backend name with a protocol factory.

    Returns
    -------
    list[CaseResult]
        One row per :data:`C0_PROTOCOL_CASES`; empty when the backend has no
        registered protocol (its slots do not exist).
    """

    from ..backends._backend_protocol import REFUSAL_CODES, protocol_problems

    spec = registered_protocol_spec(backend)
    if spec is None:
        return []
    try:
        candidate = spec.protocol()
    except Exception as exc:  # noqa: BLE001 - a factory crash is a red cell, never a crash
        detail = f"protocol factory raised {type(exc).__name__}: {exc}"[:300]
        return [_failed(case_id, "C0", detail) for case_id in C0_PROTOCOL_CASES]
    results: list[CaseResult] = []
    problems = protocol_problems(candidate, expected_name=backend)
    if problems:
        results.append(_failed("c0_protocol_complete", "C0", "; ".join(problems)[:300]))
    else:
        results.append(_passed("c0_protocol_complete", "C0"))
    results.append(_framework_requirement_cell(candidate))
    results.append(_refusal_codes_cell(candidate, REFUSAL_CODES))
    results.append(_capability_table_cell(candidate))
    return results


def _framework_requirement_cell(candidate: Any) -> CaseResult:
    """Range, tested version and pin agree (``FrameworkRequirement.consistency_problems``)."""

    requirement = getattr(candidate, "framework", None)
    check = getattr(requirement, "consistency_problems", None)
    if not callable(check):
        return _failed(
            "c0_framework_requirement_consistent", "C0", "framework is not a FrameworkRequirement"
        )
    problems = check()
    if problems:
        return _failed("c0_framework_requirement_consistent", "C0", "; ".join(problems)[:300])
    return _passed("c0_framework_requirement_consistent", "C0")


def _refusal_codes_cell(candidate: Any, registry: frozenset[str]) -> CaseResult:
    """Every declared refusal code is in the closed registry."""

    try:
        declared = set(candidate.refusal_codes())
    except Exception as exc:  # noqa: BLE001
        return _failed("c0_refusal_codes_registered", "C0", f"refusal_codes() raised {exc}"[:300])
    unknown = sorted(code for code in declared if code not in registry)
    if unknown:
        return _failed(
            "c0_refusal_codes_registered", "C0", f"codes outside REFUSAL_CODES: {unknown}"[:300]
        )
    return _passed("c0_refusal_codes_registered", "C0")


def _capability_table_cell(candidate: Any) -> CaseResult:
    """Every ``BackendCapabilities`` field holds a value of its declared kind."""

    from ..backends.registry import BackendCapabilities

    try:
        table = candidate.capabilities()
    except Exception as exc:  # noqa: BLE001
        return _failed("c0_capability_table_complete", "C0", f"capabilities() raised {exc}"[:300])
    if not isinstance(table, BackendCapabilities):
        return _failed(
            "c0_capability_table_complete", "C0", f"capabilities() returned {type(table).__name__}"
        )
    bad: list[str] = []
    for field_info in dataclass_fields(BackendCapabilities):
        value = getattr(table, field_info.name)
        kind = str(field_info.type)
        if "bool" in kind and not isinstance(value, bool):
            bad.append(f"{field_info.name} is not bool")
        elif "tuple" in kind and not (
            isinstance(value, tuple) and all(isinstance(item, str) for item in value)
        ):
            bad.append(f"{field_info.name} is not a tuple of str")
        elif "Literal" in kind and value not in ("none", "paths_only", "full_spec"):
            bad.append(f"{field_info.name} has value {value!r}")
    if bad:
        return _failed("c0_capability_table_complete", "C0", "; ".join(bad)[:300])
    return _passed("c0_capability_table_complete", "C0")


# ----------------------------------------------------------------------------
# Backend-neutral C1 helpers (wired into the C1 pack by the suite packet).


def graph_integrity_problems(trace: Any) -> tuple[str, ...]:
    """Structural problems in a captured graph, each as one line.

    Parents resolve to recorded ops, parent and child lists are symmetric and
    input ops have no parents. The orphan rule (a parentless non-input op
    must be a constant producer or buffer) joins once the shared constants
    rule records ``is_constant_layer``.
    """

    rows = list(trace.layer_list)
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
    """Ops whose ``tensor_shape`` disagrees with the saved payload's shape."""

    problems: list[str] = []
    for op in trace.layer_list:
        shape = getattr(op, "tensor_shape", None)
        payload = getattr(op, "out", None)
        if shape is None or payload is None:
            continue
        try:
            got = tuple(host_copy(payload).shape)
        except Exception:  # noqa: BLE001 - a non-array payload is outside this check
            continue
        if tuple(shape) != got:
            problems.append(f"{op.label}: tensor_shape {tuple(shape)} != payload {got}")
    return tuple(problems)


def silent_gap_reason(trace: Any) -> str | None:
    """The recorded verification ceiling when the capture flagged itself, else ``None``."""

    if getattr(trace, "capture_verified", None) is False:
        return str(getattr(trace, "capture_verification_reason", None) or "unrecorded reason")
    return None
