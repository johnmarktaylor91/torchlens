"""The shared refusal gates raise the registered code on ``exc.code`` (SPEC 1.3).

Every shared gate a backend's refusal passes through declares its code where it
raises: a capability flag's gate raises ``refusal_code_for_capability(flag)``,
an option gate raises ``trace_option_unsupported``. The refusal cells read
their expected codes from the same registry, never from a spelling of their own.
"""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.backends._backend_protocol import (
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

_OPTION_CODE = "trace_option_unsupported"


def _assert_code(exc: BackendUnsupportedError, expected: str) -> None:
    """The instance code, the structured field and the registry agree."""

    assert expected in REFUSAL_CODES, expected
    assert exc.code == expected, (exc.code, expected)
    assert exc.fields["code"] == expected


def _spec(false_flag: str | None, trace_options: tuple[str, ...] = ()) -> Any:
    """A spec stand-in whose boolean capability flags are True except ``false_flag``."""

    flags = {
        field.name: field.name != false_flag
        for field in fields(BackendCapabilities)
        if field.type in ("bool", bool)
    }
    capabilities = SimpleNamespace(**flags, trace_options=trace_options)
    return SimpleNamespace(name="fake_preview", capabilities=capabilities)


@pytest.mark.parametrize(("option", "flag"), sorted(TRACE_OPTION_CAPABILITY_GATES.items()))
def test_trace_capability_gate_raises_the_flag_code(option: str, flag: str) -> None:
    """``tl.trace``'s capability gate refuses with the False flag's registered code."""

    from torchlens.user_funcs import _enforce_capability_option_gates

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

    from torchlens.user_funcs import _filter_trace_kwargs_for_backend

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
