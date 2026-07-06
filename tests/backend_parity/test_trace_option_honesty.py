"""Backend trace-option honesty checks."""

from __future__ import annotations

from typing import Any

import pytest

from torchlens.backends import BackendSpec, BackendUnsupportedError, registered_backend_specs
from torchlens.backends._options import (
    JAX_PREVIEW_TRACE_OPTION_POLICY,
    MLX_PREVIEW_TRACE_OPTION_POLICY,
    PADDLE_PREVIEW_TRACE_OPTION_POLICY,
    TF_PREVIEW_TRACE_OPTION_POLICY,
    TINYGRAD_PREVIEW_TRACE_OPTION_POLICY,
    PreviewTraceOptionPolicy,
    reject_unsupported_trace_options,
)
from torchlens.backends.registry import PUBLIC_OPTION_SPINE_TRACE_OPTIONS
from torchlens.user_funcs import _filter_trace_kwargs_for_backend

pytestmark = pytest.mark.backend_parity

_PUBLIC_SPINE_OPTION_VALUES: dict[str, Any] = {
    "jax_control_flow": "unroll",
    "jax_max_control_flow_unroll": 4,
    "module_identity_mode": "object_module",
    "payload_policy": "full",
    "save_preview": True,
    "capture_container_structure": True,
    "inference_only": True,
    "output_style": "label",
    "output_head": "default",
    "chunk_size": 2,
    "chunk_paths": [("args", 0)],
}

_PREVIEW_POLICIES: tuple[PreviewTraceOptionPolicy, ...] = (
    JAX_PREVIEW_TRACE_OPTION_POLICY,
    MLX_PREVIEW_TRACE_OPTION_POLICY,
    PADDLE_PREVIEW_TRACE_OPTION_POLICY,
    TF_PREVIEW_TRACE_OPTION_POLICY,
    TINYGRAD_PREVIEW_TRACE_OPTION_POLICY,
)

_PREVIEW_DEFAULT_OPTIONS: dict[str, Any] = {
    "layers_to_save": "all",
    "input_kwargs": {},
    "activation_transform": None,
    "detach_saved_activations": False,
    "save_grads": None,
    "save_arg_values": False,
    "save_code_context": False,
    "save_rng_states": False,
    "backward_ready": False,
    "module_filter": None,
    "transform": None,
    "output_device": "same",
    "layer_visualizers": None,
    "save_visualizations": False,
    "save_raw_activations": True,
    "lookback": 0,
    "lookback_payload_policy": "metadata_only",
}

_PREVIEW_UNSUPPORTED_VALUES: tuple[tuple[str, Any], ...] = (
    ("layers_to_save", ["relu_1_1"]),
    ("input_kwargs", {"bias": object()}),
    ("activation_transform", lambda value: value),
    ("detach_saved_activations", True),
    ("save_grads", True),
    ("save_arg_values", True),
    ("save_code_context", True),
    ("save_rng_states", True),
    ("backward_ready", True),
    ("module_filter", lambda module: True),
    ("transform", lambda value: value),
    ("output_device", "cpu"),
    ("layer_visualizers", {"relu": object()}),
    ("save_visualizations", True),
    ("save_raw_activations", False),
    ("lookback", 1),
    ("lookback_payload_policy", "detached_raw"),
)


@pytest.mark.parametrize("spec", registered_backend_specs(), ids=lambda spec: str(spec.name))
@pytest.mark.parametrize("option_name", PUBLIC_OPTION_SPINE_TRACE_OPTIONS)
def test_registered_backend_public_trace_options_are_declared_or_rejected(
    spec: BackendSpec,
    option_name: str,
) -> None:
    """Every registered backend either declares or rejects each public spine option."""

    kwargs = {option_name: _PUBLIC_SPINE_OPTION_VALUES[option_name]}
    if option_name in spec.capabilities.trace_options:
        _filter_trace_kwargs_for_backend(kwargs, spec)
        assert kwargs == {option_name: _PUBLIC_SPINE_OPTION_VALUES[option_name]}
        return

    with pytest.raises(BackendUnsupportedError, match=option_name):
        _filter_trace_kwargs_for_backend(kwargs, spec)


@pytest.mark.parametrize("policy", _PREVIEW_POLICIES, ids=lambda policy: policy.backend_name)
@pytest.mark.parametrize(("option_name", "value"), _PREVIEW_UNSUPPORTED_VALUES)
def test_preview_backend_options_are_centrally_rejected(
    policy: PreviewTraceOptionPolicy,
    option_name: str,
    value: Any,
) -> None:
    """Preview backend unsupported options are rejected by the shared policy helper."""

    options = dict(_PREVIEW_DEFAULT_OPTIONS)
    options[option_name] = value
    expects_rejection = (
        (option_name == "input_kwargs" and policy.input_kwargs_message is not None)
        or (option_name == "layers_to_save" and policy.full_save_message is not None)
        or option_name in (policy.rejected_truthy_messages or {})
        or (option_name == "output_device" and policy.output_device_message is not None)
        or (
            option_name == "save_raw_activations"
            and policy.save_raw_activations_false_message is not None
        )
        or (
            option_name in {"lookback", "lookback_payload_policy"}
            and policy.save_window_message is not None
        )
    )
    if not expects_rejection:
        reject_unsupported_trace_options(options, policy)
        return

    with pytest.raises(BackendUnsupportedError):
        reject_unsupported_trace_options(options, policy)
