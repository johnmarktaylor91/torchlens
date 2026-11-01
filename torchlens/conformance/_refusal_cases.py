"""Refusal cells: every False capability flag and every foreign trace option refuses typed.

One table (:data:`CAPABILITY_REQUESTS`) says how the suite asks for each
capability through the public entry; one table (:data:`TRACE_OPTION_PROBES`)
holds a non-default probe value per public spine option. A cell passes only on
a ``BackendUnsupportedError`` whose code is the contracted one and whose raise
left the framework state (RNG, wrappers) unchanged; an entry that proceeds or
raises untyped fails (SPEC 1.3 and 4.2).

Expected codes come from the registry, never from a spelling written here: a
capability cell expects ``refusal_code_for_capability(flag)`` and an option cell
expects :data:`_TRACE_OPTION_CODE`, the one ``REFUSAL_CODES`` member for an
option outside ``trace_options()`` (a test keeps it registered).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from ._adapters import RosterModel
from ._cases import CaseResult, _failed, _passed
from ._protocol_cases import (
    C1_REFUSAL_OPTION_PREFIX,
    C1_REFUSAL_PREFIX,
    _as_output_list,
    adapter_protocol,
    false_capability_flags,
    foreign_trace_options,
)

__tl_layer__ = "L9"

#: SPEC 1.3's code for a public trace option outside ``trace_options()``.
_TRACE_OPTION_CODE = "trace_option_unsupported"


@dataclass(frozen=True)
class CapabilityRequest:
    """How the suite asks a backend for one capability through the public entry.

    Parameters
    ----------
    kind:
        ``"trace_option"`` (a ``tl.trace`` keyword), ``"capture_option"`` (a
        ``CaptureOptions`` field), ``"record"`` (``tl.record``), ``"validate"``
        (``protocol.validate`` on a captured trace) or ``"save"``
        (``trace.save`` of a captured trace).
    option:
        The keyword or field name for the option kinds, else ``None``.
    value:
        Zero-arg factory for the probe value (built lazily so no framework import
        happens at module import).
    companions:
        Options the request must also pass so the shared argument preconditions
        hold and the call reaches the backend's refusal (``chunk_paths`` needs
        ``chunk_size``); name -> zero-arg value factory.
    """

    kind: str
    option: str | None = None
    value: Callable[[], Any] | None = None
    companions: Mapping[str, Callable[[], Any]] | None = None


def _storage_probe() -> Any:
    """A streaming-save request with a throwaway destination."""

    import tempfile

    from ..options import StorageOptions

    return StorageOptions(path=Path(tempfile.mkdtemp(prefix="tl_conf_stream_")))


def _intervene_probe() -> Any:
    """A minimal intervention request (never reached on a refusing backend)."""

    return lambda _op: None


#: ``BackendCapabilities`` flag -> the request that must refuse when the flag is False.
CAPABILITY_REQUESTS: dict[str, CapabilityRequest] = {
    "backward_capture": CapabilityRequest("capture_option", "backward_ready", lambda: True),
    "fastlog": CapabilityRequest("record"),
    "interventions": CapabilityRequest("trace_option", "intervene", _intervene_probe),
    "rng_replay": CapabilityRequest("capture_option", "save_rng", lambda: True),
    "has_streaming": CapabilityRequest("trace_option", "streaming", _storage_probe),
    "structure_only_capture": CapabilityRequest("capture_option", "structure_only", lambda: True),
    "has_validation_replay": CapabilityRequest("validate"),
    "payload_materialization": CapabilityRequest("save"),
    "intermediate_derived_grads": CapabilityRequest(
        "trace_option", "grad_options", lambda: {"derived": True}
    ),
}

#: Public spine trace option -> a non-default probe value a refusing backend must reject.
TRACE_OPTION_PROBES: dict[str, Callable[[], Any]] = {
    "jax_control_flow": lambda: "region",
    "jax_max_control_flow_unroll": lambda: 2,
    "module_identity_mode": lambda: "object_identity",
    "payload_policy": lambda: "materialize",
    "save_preview": lambda: True,
    "capture_container_structure": lambda: True,
    "inference_only": lambda: True,
    "chunk_size": lambda: 1,
    "chunk_paths": lambda: (0,),
}

#: Spine option -> the options its probe also passes. ``chunk_paths`` alone trips the
#: shared ``ChunkedForwardConfigError`` precondition before any backend gate, which
#: would measure argument validation instead of the refusal.
TRACE_OPTION_COMPANIONS: dict[str, dict[str, Callable[[], Any]]] = {
    "chunk_paths": {"chunk_size": TRACE_OPTION_PROBES["chunk_size"]},
}

#: Public spine options that are ``CaptureOptions`` fields rather than ``tl.trace`` keywords.
CAPTURE_OPTION_SPINE_FIELDS: frozenset[str] = frozenset(
    {
        "jax_control_flow",
        "jax_max_control_flow_unroll",
        "module_identity_mode",
        "payload_policy",
        "save_preview",
        "capture_container_structure",
        "inference_only",
    }
)


def request_option_for_flag(flag: str) -> str | None:
    """The option name the suite requests a capability through, else ``None``."""

    request = CAPABILITY_REQUESTS.get(flag)
    return request.option if request is not None else None


def refusal_code_of(exc: BaseException) -> str:
    """The typed code a refusal carries (instance ``fields['code']`` first, then the class code)."""

    fields = getattr(exc, "fields", None)
    code = fields.get("code") if isinstance(fields, dict) else None
    return str(code or getattr(exc, "code", "") or type(exc).__name__)


def run_c1_refusal_cases(adapter: Any, roster: tuple[RosterModel, ...]) -> list[CaseResult]:
    """The once-per-backend refusal cells (capability flags, then foreign trace options).

    Parameters
    ----------
    adapter:
        The resolved adapter (must expose a protocol view).
    roster:
        The roster; the first row's model carries each request.

    Returns
    -------
    list[CaseResult]
        One row per refusal cell the backend defines.
    """

    from ..backends._backend_protocol import refusal_code_for_capability

    protocol = adapter_protocol(adapter)
    if protocol is None or not roster:
        return []
    target = _RefusalTarget(adapter, protocol, roster[0])
    results: list[CaseResult] = []
    for flag in false_capability_flags(protocol):
        request = CAPABILITY_REQUESTS.get(flag)
        case_id = f"{C1_REFUSAL_PREFIX}{flag}"
        if request is None:
            results.append(_failed(case_id, "C1", f"no request recipe for capability {flag}"))
            continue
        expected = refusal_code_for_capability(flag)
        results.append(_refusal_cell(case_id, target, request, expected))
    for option in foreign_trace_options(protocol):
        case_id = f"{C1_REFUSAL_OPTION_PREFIX}{option}"
        probe = TRACE_OPTION_PROBES.get(option)
        if probe is None:
            results.append(_failed(case_id, "C1", f"no probe value declared for option {option}"))
            continue
        kind = "capture_option" if option in CAPTURE_OPTION_SPINE_FIELDS else "trace_option"
        request = CapabilityRequest(kind, option, probe, TRACE_OPTION_COMPANIONS.get(option))
        results.append(_refusal_cell(case_id, target, request, _TRACE_OPTION_CODE))
    return results


class _RefusalTarget(NamedTuple):
    """What every refusal cell of one run issues its request against."""

    adapter: Any
    protocol: Any
    model: RosterModel


def _refusal_cell(
    case_id: str, target: _RefusalTarget, request: CapabilityRequest, expected_code: str
) -> CaseResult:
    """Run one request; passed only on a typed refusal with the contracted code and no side effect."""

    import warnings

    from ..backends.registry import BackendUnsupportedError

    adapter, protocol, model = target
    snapshot = getattr(adapter, "side_effect_snapshot", None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = _prepare_request(adapter, model, request)
    except Exception as exc:  # noqa: BLE001 - a broken prerequisite is a red cell, not a crash
        return _failed(
            case_id, "C1", f"request prerequisites failed: {type(exc).__name__}: {str(exc)[:200]}"
        )
    # Building the model (and a prerequisite capture) may consume framework
    # RNG; only the refusing call itself must leave the framework untouched.
    before = snapshot() if callable(snapshot) else None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _issue_request(adapter, protocol, request, prepared)
    except BackendUnsupportedError as exc:
        code = refusal_code_of(exc)
        if code != expected_code:
            return _failed(
                case_id, "C1", f"refused with code {code!r}, contract is {expected_code!r}"
            )
    except Exception as exc:  # noqa: BLE001 - an untyped raise is a failed refusal
        return _failed(case_id, "C1", f"untyped {type(exc).__name__}: {str(exc)[:200]}")
    else:
        return _failed(case_id, "C1", "entry proceeded instead of refusing")
    after = snapshot() if callable(snapshot) else None
    if before != after:
        return _failed(case_id, "C1", "framework state changed before the refusal")
    return _passed(case_id, "C1")


def _prepare_request(
    adapter: Any, model: RosterModel, request: CapabilityRequest
) -> tuple[Any, ...]:
    """Build the model and any prerequisite capture the request issues against."""

    net, example = model.builder()
    if request.kind in {"validate", "save"}:
        return net, example, adapter.trace(net, example)
    return net, example, None


def _issue_request(
    adapter: Any, protocol: Any, request: CapabilityRequest, prepared: tuple[Any, ...]
) -> None:
    """Issue one capability request through the public entry."""

    from ..options import CaptureOptions

    net, example, trace = prepared
    value = request.value() if request.value is not None else None
    options: dict[str, Any] = {
        name: factory() for name, factory in (request.companions or {}).items()
    }
    options[str(request.option)] = value
    if request.kind == "trace_option":
        adapter.trace_with(net, example, **options)
    elif request.kind == "capture_option":
        # Construct with only the probed fields: ``dataclasses.replace`` would pass
        # every field to ``__init__`` and mark them all explicit, so the option
        # filter would refuse an unrelated field before the capability gate.
        capture = CaptureOptions(**options)
        adapter.trace_with(net, example, capture=capture)
    elif request.kind == "record":
        import torchlens as tl

        tl.record(net, example, backend=protocol.name)
    elif request.kind == "validate":
        protocol.validate(trace, _as_output_list(adapter.reference_output(net, example)))
    elif request.kind == "save":
        import tempfile

        adapter.save(trace, Path(tempfile.mkdtemp(prefix="tl_conf_save_")) / "refusal.tlspec")
    else:
        raise ValueError(f"unknown request kind {request.kind!r}")
