"""Refusal cells: every False capability flag and every foreign trace option refuses typed.

One table (:data:`CAPABILITY_REQUESTS`) says how the suite asks for each
capability through the public entry; one table (:data:`TRACE_OPTION_PROBES`)
holds a non-default probe value per public spine option. A cell passes only on
a ``BackendUnsupportedError`` whose code is the contracted one and whose raise
left the framework state (RNG, wrappers) unchanged; an entry that proceeds or
raises untyped fails (SPEC 1.3 and 4.2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    """

    kind: str
    option: str | None = None
    value: Callable[[], Any] | None = None


def _storage_probe() -> Any:
    """A streaming-save request with a throwaway destination."""

    import tempfile

    from ..options import StorageOptions

    return StorageOptions(bundle_path=Path(tempfile.mkdtemp(prefix="tl_conf_stream_")))


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
    "output_style": lambda: "semantic",
    "output_head": lambda: "lm_head",
    "chunk_size": lambda: 1,
    "chunk_paths": lambda: (0,),
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
        "output_style",
        "output_head",
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

    protocol = adapter_protocol(adapter)
    if protocol is None or not roster:
        return []
    results: list[CaseResult] = []
    for flag in false_capability_flags(protocol):
        request = CAPABILITY_REQUESTS.get(flag)
        case_id = f"{C1_REFUSAL_PREFIX}{flag}"
        if request is None:
            results.append(_failed(case_id, "C1", f"no request recipe for capability {flag}"))
            continue
        expected = f"{flag}_unsupported"
        results.append(_refusal_cell(case_id, adapter, protocol, roster[0], request, expected))
    for option in foreign_trace_options(protocol):
        case_id = f"{C1_REFUSAL_OPTION_PREFIX}{option}"
        probe = TRACE_OPTION_PROBES.get(option)
        if probe is None:
            results.append(_failed(case_id, "C1", f"no probe value declared for option {option}"))
            continue
        kind = "capture_option" if option in CAPTURE_OPTION_SPINE_FIELDS else "trace_option"
        request = CapabilityRequest(kind, option, probe)
        results.append(
            _refusal_cell(
                case_id, adapter, protocol, roster[0], request, "trace_option_unsupported"
            )
        )
    return results


def _refusal_cell(
    case_id: str,
    adapter: Any,
    protocol: Any,
    model: RosterModel,
    request: CapabilityRequest,
    expected_code: str,
) -> CaseResult:
    """Run one request; passed only on a typed refusal with the contracted code and no side effect."""

    import warnings

    from ..backends.registry import BackendUnsupportedError

    snapshot = getattr(adapter, "side_effect_snapshot", None)
    before = snapshot() if callable(snapshot) else None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _perform_request(adapter, protocol, model, request)
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


def _perform_request(
    adapter: Any, protocol: Any, model: RosterModel, request: CapabilityRequest
) -> None:
    """Issue one capability request through the public entry."""

    import warnings
    from dataclasses import replace

    from ..options import CaptureOptions

    net, example = model.builder()
    value = request.value() if request.value is not None else None
    if request.kind == "trace_option":
        adapter.trace_with(net, example, **{str(request.option): value})
    elif request.kind == "capture_option":
        capture = replace(CaptureOptions(), **{str(request.option): value})
        adapter.trace_with(net, example, capture=capture)
    elif request.kind == "record":
        import torchlens as tl

        tl.record(net, example, backend=protocol.name)
    elif request.kind in {"validate", "save"}:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = adapter.trace(net, example)
        if request.kind == "validate":
            protocol.validate(trace, _as_output_list(adapter.reference_output(net, example)))
        else:
            import tempfile

            adapter.save(trace, Path(tempfile.mkdtemp(prefix="tl_conf_save_")) / "refusal.tlspec")
    else:  # pragma: no cover - the kinds are the closed set above
        raise ValueError(f"unknown request kind {request.kind!r}")
