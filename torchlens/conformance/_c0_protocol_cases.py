"""The protocol-level tier-0 conformance cells for one registered backend.

Extracted from ``_protocol_cases`` so that module stays under the file cap; the
spellings stay importable from ``torchlens.conformance._protocol_cases``.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

from ..backends._backend_protocol import REFUSAL_CODES, protocol_problems
from ..backends.registry import BackendCapabilities
from ._cases import TIER_C0, CaseResult, _failed, _passed

__tl_layer__ = "L9"

#: The protocol-level c0 case ids, in report order.
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
    """The protocol-level c0 cells for one registered backend.

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

    spec = registered_protocol_spec(backend)
    if spec is None:
        return []
    try:
        candidate = spec.protocol()
    except Exception as exc:  # noqa: BLE001 - a factory crash is a red cell, never a crash
        detail = f"protocol factory raised {type(exc).__name__}: {exc}"[:300]
        return [_failed(case_id, TIER_C0, detail) for case_id in C0_PROTOCOL_CASES]
    results: list[CaseResult] = []
    problems = protocol_problems(candidate, expected_name=backend)
    if problems:
        results.append(_failed("c0_protocol_complete", TIER_C0, "; ".join(problems)[:300]))
    else:
        results.append(_passed("c0_protocol_complete", TIER_C0))
    results.append(_framework_requirement_cell(candidate))
    results.append(_refusal_codes_cell(candidate, REFUSAL_CODES))
    results.append(_capability_table_cell(candidate))
    return results


def _framework_requirement_cell(candidate: Any) -> CaseResult:
    """Range, tested version and pin agree (``FrameworkRequirement._consistency_problems``)."""

    requirement = getattr(candidate, "framework", None)
    check = getattr(requirement, "_consistency_problems", None)
    if not callable(check):
        return _failed(
            "c0_framework_requirement_consistent",
            TIER_C0,
            "framework is not a FrameworkRequirement",
        )
    problems = check()
    if problems:
        return _failed("c0_framework_requirement_consistent", TIER_C0, "; ".join(problems)[:300])
    return _passed("c0_framework_requirement_consistent", TIER_C0)


def _refusal_codes_cell(candidate: Any, registry: frozenset[str]) -> CaseResult:
    """Every declared refusal code is in the closed registry."""

    try:
        declared = set(candidate.refusal_codes())
    except Exception as exc:  # noqa: BLE001
        return _failed(
            "c0_refusal_codes_registered", TIER_C0, f"refusal_codes() raised {exc}"[:300]
        )
    unknown = sorted(code for code in declared if code not in registry)
    if unknown:
        return _failed(
            "c0_refusal_codes_registered", TIER_C0, f"codes outside REFUSAL_CODES: {unknown}"[:300]
        )
    return _passed("c0_refusal_codes_registered", TIER_C0)


def _capability_table_cell(candidate: Any) -> CaseResult:
    """Every ``BackendCapabilities`` field holds a value of its declared kind."""

    try:
        table = candidate.capabilities()
    except Exception as exc:  # noqa: BLE001
        return _failed(
            "c0_capability_table_complete", TIER_C0, f"capabilities() raised {exc}"[:300]
        )
    if not isinstance(table, BackendCapabilities):
        return _failed(
            "c0_capability_table_complete",
            TIER_C0,
            f"capabilities() returned {type(table).__name__}",
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
        return _failed("c0_capability_table_complete", TIER_C0, "; ".join(bad)[:300])
    return _passed("c0_capability_table_complete", TIER_C0)


# ----------------------------------------------------------------------------
# Backend-neutral c1 helpers (wired into the c1 pack by the suite packet).
