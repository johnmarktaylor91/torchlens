"""The shared refusal gates raise the registered code on ``exc.code`` (SPEC 1.3).

Every shared gate a backend's refusal passes through declares its code where it
raises: a capability flag's gate raises ``refusal_code_for_capability(flag)``,
an option gate raises ``trace_option_unsupported``. The refusal cells read
their expected codes from the same registry, never from a spelling of their own.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.backends._backend_protocol import (
    REFUSABLE_CAPABILITY_FLAGS,
    REFUSAL_CODES,
    refusal_code_for_capability,
)
from torchlens.backends._options import (
    TRACE_OPTION_CAPABILITY_GATES,
    ExtraKwargPolicy,
    PreviewTraceOptionPolicy,
    reject_extra_trace_kwargs,
    reject_unsupported_trace_options,
)
from torchlens.backends.registry import BackendCapabilities, BackendUnsupportedError
from torchlens.conformance import _refusal_cases
from torchlens.conformance._refusal_cases import (
    CAPABILITY_REQUESTS,
    TRACE_OPTION_COMPANIONS,
    TRACE_OPTION_PROBES,
    CapabilityRequest,
)

_OPTION_CODE = "trace_option_unsupported"


def _assert_code(exc: BackendUnsupportedError, expected: str) -> None:
    """The instance code, the structured field and the registry agree."""

    assert expected in REFUSAL_CODES, expected
    assert exc.code == expected, (exc.code, expected)
    assert exc.fields["code"] == expected


def _spec(
    false_flag: str | None, trace_options: tuple[str, ...] = (), name: str = "fake_preview"
) -> Any:
    """A spec stand-in whose boolean capability flags are True except ``false_flag``."""

    flags = {
        field.name: field.name != false_flag
        for field in fields(BackendCapabilities)
        if field.type in ("bool", bool)
    }
    capabilities = SimpleNamespace(**flags, trace_options=trace_options)
    return SimpleNamespace(name=name, capabilities=capabilities)


@pytest.mark.parametrize(("option", "flag"), sorted(TRACE_OPTION_CAPABILITY_GATES.items()))
def test_trace_capability_gate_raises_the_flag_code(option: str, flag: str) -> None:
    """``tl.trace``'s capability gate refuses with the False flag's registered code."""

    from torchlens._capture.prepare.dispatch import _enforce_capability_option_gates

    with pytest.raises(BackendUnsupportedError) as excinfo:
        _enforce_capability_option_gates({option: object()}, _spec(flag))
    _assert_code(excinfo.value, refusal_code_for_capability(flag))
    assert excinfo.value.fields["option"] == option


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("jax_static_argnums", _OPTION_CODE),
        ("module_identity_mode", _OPTION_CODE),
        ("chunk_paths", _OPTION_CODE),
        ("grad_options", refusal_code_for_capability("intermediate_derived_grads")),
    ],
)
def test_trace_option_filter_raises_the_registered_code(option: str, expected: str) -> None:
    """``tl.trace``'s option filter refuses a foreign option with its registered code."""

    from torchlens._capture.prepare.dispatch import _filter_trace_kwargs_for_backend

    with pytest.raises(BackendUnsupportedError) as excinfo:
        _filter_trace_kwargs_for_backend({option: object()}, _spec(None))
    _assert_code(excinfo.value, expected)
    assert excinfo.value.fields["option"] == option


def test_default_backend_early_refusal_is_coded() -> None:
    """``backend=None`` with a jax-only option refuses with the option code before capture."""

    import torch

    import torchlens as tl

    calls: list[int] = []

    class _Counts(torch.nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            calls.append(1)
            return v * 2.0

    with pytest.raises(BackendUnsupportedError) as excinfo:
        tl.trace(_Counts(), torch.ones(2), jax_static_argnums=(0,))
    _assert_code(excinfo.value, _OPTION_CODE)
    assert not calls


@pytest.mark.parametrize("backend_name", ["torch", "fake_preview"])
def test_record_capability_gate_raises_the_fastlog_code(
    backend_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``tl.record`` refuses a backend whose ``fastlog`` flag is False with its code."""

    import torch

    import torchlens as tl
    from torchlens.recording import _record_one_shot

    monkeypatch.setattr(
        _record_one_shot, "get_backend_spec", lambda name: _spec("fastlog", name=backend_name)
    )
    with pytest.raises(BackendUnsupportedError) as excinfo:
        tl.record(torch.nn.Identity(), torch.ones(2), backend=backend_name)
    _assert_code(excinfo.value, refusal_code_for_capability("fastlog"))
    assert excinfo.value.fields["backend"] == backend_name


def _policy(**messages: Any) -> PreviewTraceOptionPolicy:
    """A preview policy refusing only what ``messages`` names."""

    return PreviewTraceOptionPolicy(backend_name="fake_preview", **messages)


_SUPPORTED_DEFAULTS: dict[str, Any] = {
    "layers_to_save": "all",
    "output_device": "same",
    "save_raw_activations": True,
    "lookback": 0,
    "lookback_payload_policy": "metadata_only",
}


@pytest.mark.parametrize(
    ("policy_field", "options"),
    [
        ("input_kwargs_message", {"input_kwargs": {"a": 1}}),
        ("full_save_message", {"layers_to_save": ["x"]}),
        ("output_device_message", {"output_device": "cpu"}),
        ("save_raw_activations_false_message", {"save_raw_activations": False}),
        ("save_window_message", {"lookback": 2}),
    ],
)
def test_option_policy_refusals_raise_the_option_code(
    policy_field: str, options: dict[str, Any]
) -> None:
    """Every fixed refusal of ``reject_unsupported_trace_options`` carries the option code."""

    with pytest.raises(BackendUnsupportedError) as excinfo:
        reject_unsupported_trace_options(
            {**_SUPPORTED_DEFAULTS, **options}, _policy(**{policy_field: "refused"})
        )
    _assert_code(excinfo.value, _OPTION_CODE)
    assert excinfo.value.fields["backend"] == "fake_preview"


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        *[
            (option, refusal_code_for_capability(flag))
            for option, flag in sorted(TRACE_OPTION_CAPABILITY_GATES.items())
        ],
        ("chunk_size", _OPTION_CODE),
        ("activation_transform", _OPTION_CODE),
    ],
)
def test_option_policy_truthy_refusal_raises_the_owning_code(option: str, expected: str) -> None:
    """A refused truthy option raises its capability's code, else the option code."""

    policy = _policy(rejected_truthy_messages={option: f"{option} refused"})
    with pytest.raises(BackendUnsupportedError) as excinfo:
        reject_unsupported_trace_options({**_SUPPORTED_DEFAULTS, option: 1}, policy)
    _assert_code(excinfo.value, expected)
    assert excinfo.value.fields["option"] == option


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"storage": object()}, refusal_code_for_capability("has_streaming")),
        ({"storage": object(), "streaming": object()}, "streaming_unsupported"),
        ({"chunk_size": 2}, _OPTION_CODE),
        ({"storage": object(), "chunk_size": 2}, _OPTION_CODE),
    ],
)
def test_extra_kwarg_refusal_raises_one_registered_code(
    kwargs: dict[str, Any], expected: str, runtime: bool
) -> None:
    """``reject_extra_trace_kwargs``: a shared capability code, else the option code."""

    policy = ExtraKwargPolicy(
        runtime_option_names=frozenset(),
        runtime_message="runtime {names}",
        fallback_message="fallback {names}",
        always_runtime=runtime,
    )
    with pytest.raises(BackendUnsupportedError) as excinfo:
        reject_extra_trace_kwargs(kwargs, policy)
    _assert_code(excinfo.value, expected)
    assert excinfo.value.fields["options"] == sorted(kwargs)


# ---------------------------------------------------------------------------
# The refusal cells take their codes from the registry.
# ---------------------------------------------------------------------------


def test_every_refusable_flag_has_a_request_recipe() -> None:
    """Each flag the registry maps has exactly one request recipe."""

    assert set(CAPABILITY_REQUESTS) == set(REFUSABLE_CAPABILITY_FLAGS)


def test_option_cell_code_is_registered() -> None:
    """The one option code the cells expect is a registry member."""

    assert _refusal_cases._TRACE_OPTION_CODE == _OPTION_CODE
    assert _OPTION_CODE in REFUSAL_CODES


def test_refusal_cells_hard_code_no_mapped_code() -> None:
    """No refusal-code spelling sits in the cell module beyond the option constant.

    Capability codes come from ``refusal_code_for_capability``; a literal or an
    f-string ending in ``_unsupported`` would bypass the registry the way the
    old ``f"{flag}_unsupported"`` read did (``has_streaming_unsupported``).
    """

    tree = ast.parse(inspect.getsource(_refusal_cases))
    spellings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.endswith("_unsupported")
    ]
    assert spellings == [_OPTION_CODE], spellings
    formatted = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        and any(
            isinstance(part, ast.Constant) and str(part.value).endswith("_unsupported")
            for part in node.values
        )
    ]
    assert not formatted, formatted


def test_streaming_cell_expects_the_registered_code() -> None:
    """The streaming cell reads ``streaming_unsupported``, the code the flag maps to."""

    assert refusal_code_for_capability("has_streaming") == "streaming_unsupported"


class _RecordingAdapter:
    """Records the options each refusal request passes to the public entry."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def trace_with(self, net: Any, example: Any, **options: Any) -> None:
        self.calls.append(options)


def test_chunk_paths_cell_passes_chunk_size_too() -> None:
    """The ``chunk_paths`` probe carries ``chunk_size`` so the precondition holds."""

    assert set(TRACE_OPTION_COMPANIONS["chunk_paths"]) == {"chunk_size"}
    request = CapabilityRequest(
        "trace_option",
        "chunk_paths",
        TRACE_OPTION_PROBES["chunk_paths"],
        TRACE_OPTION_COMPANIONS["chunk_paths"],
    )
    adapter = _RecordingAdapter()
    _refusal_cases._issue_request(adapter, None, request, (object(), object(), None))
    assert adapter.calls == [{"chunk_paths": (0,), "chunk_size": 1}]


def test_chunk_paths_and_size_pass_the_shared_precondition() -> None:
    """Both options together clear ``ChunkedForwardConfigError`` and reach a backend gate.

    The probe pair on a backend lacking chunked capture refuses typed with the
    option code; alone, ``chunk_paths`` stops at the argument precondition.
    """

    import torch

    import torchlens as tl
    from torchlens._capture.prepare.dispatch import _filter_trace_kwargs_for_backend
    from torchlens.intervention.errors import ChunkedForwardConfigError

    with pytest.raises(ChunkedForwardConfigError):
        tl.trace(torch.nn.Identity(), torch.ones(2), chunk_paths=(0,))
    with pytest.raises(BackendUnsupportedError) as excinfo:
        _filter_trace_kwargs_for_backend({"chunk_paths": (0,), "chunk_size": 1}, _spec(None))
    _assert_code(excinfo.value, _OPTION_CODE)


@pytest.mark.parametrize(
    "flag", sorted(f for f, r in CAPABILITY_REQUESTS.items() if r.kind == "capture_option")
)
def test_capture_option_probe_marks_only_its_own_fields(flag: str) -> None:
    """A capture-option request's ``CaptureOptions`` marks only the probed fields explicit.

    Every other field explicit would let the option filter refuse an unrelated
    spine field with ``trace_option_unsupported`` before the capability gate ran.
    """

    from dataclasses import fields as dataclass_fields

    from torchlens.options import CaptureOptions

    request = CAPABILITY_REQUESTS[flag]
    adapter = _RecordingAdapter()
    _refusal_cases._issue_request(adapter, None, request, (object(), object(), None))
    [call] = adapter.calls
    capture = call["capture"]
    explicit = {
        field.name
        for field in dataclass_fields(CaptureOptions)
        if not field.name.startswith("_") and capture.is_field_explicit(field.name)
    }
    assert explicit == {str(request.option)}, explicit


def test_capture_option_probe_reaches_the_capability_gate() -> None:
    """The probe's ``capture=`` passes the option filter and refuses with the flag's code."""

    from torchlens._capture.prepare.dispatch import (
        _enforce_capability_option_gates,
        _filter_trace_kwargs_for_backend,
    )
    from torchlens.options import CaptureOptions

    kwargs: dict[str, Any] = {"capture": CaptureOptions(backward_ready=True)}
    spec = _spec("backward_capture")
    _filter_trace_kwargs_for_backend(kwargs, spec)
    with pytest.raises(BackendUnsupportedError) as excinfo:
        _enforce_capability_option_gates(kwargs, spec)
    _assert_code(excinfo.value, refusal_code_for_capability("backward_capture"))
