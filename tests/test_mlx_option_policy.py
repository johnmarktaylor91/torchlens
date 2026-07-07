"""Policy-level tests for MLX preview option honesty."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from torchlens.backends import BackendUnsupportedError
from torchlens.backends._options import (
    MLX_EXTRA_KWARG_POLICY,
    MLX_PREVIEW_TRACE_OPTION_POLICY,
    reject_extra_trace_kwargs,
    reject_unsupported_trace_options,
)


def _valid_mlx_options() -> dict[str, Any]:
    """Return default MLX options accepted by the central policy helper."""

    return {
        "layers_to_save": "all",
        "activation_transform": None,
        "detach_saved_activations": False,
        "save_arg_values": False,
        "save_grads": None,
        "save_code_context": False,
        "save_rng_states": False,
        "backward_ready": False,
        "module_filter": None,
        "transform": None,
        "output_device": "same",
        "layer_visualizers": None,
        "save_visualizations": False,
    }


@pytest.mark.parametrize(
    ("option_name", "value"),
    [
        ("layers_to_save", ["linear_1_1"]),
        ("activation_transform", lambda value: value),
        ("detach_saved_activations", True),
        ("save_arg_values", True),
        ("save_grads", True),
        ("save_code_context", True),
        ("save_rng_states", True),
        ("backward_ready", True),
        ("module_filter", lambda module: True),
        ("transform", lambda value: value),
        ("output_device", "cpu"),
        ("layer_visualizers", {"linear": object()}),
        ("save_visualizations", True),
    ],
)
def test_mlx_preview_policy_rejects_unimplemented_options(
    option_name: str,
    value: Any,
) -> None:
    """MLX preview policy rejects each unsupported non-default option."""

    options = _valid_mlx_options()
    options[option_name] = value

    with pytest.raises(BackendUnsupportedError, match=option_name):
        reject_unsupported_trace_options(options, MLX_PREVIEW_TRACE_OPTION_POLICY)


def test_mlx_preview_policy_accepts_defaults() -> None:
    """MLX preview policy accepts default capture options."""

    reject_unsupported_trace_options(_valid_mlx_options(), MLX_PREVIEW_TRACE_OPTION_POLICY)


@pytest.mark.parametrize(
    ("option_name", "value"),
    [
        ("storage", object()),
        ("stop_after", object()),
        ("profile", True),
        ("cache", True),
        ("raise_on_nan", True),
        ("save_outs_to", Path("bundle.tl")),
        ("keep_outs_in_memory", False),
        ("out_sink", lambda label, tensor: None),
        ("cache_dir", Path("cache")),
        ("save_mode", "reference"),
        ("capture_tensor_grad_hooks", False),
        ("save_raw_gradients", False),
        ("mark_layer_depths", True),
        ("source_context_lines", 3),
        ("compute_input_output_distances", True),
        ("unwrap_when_done", True),
        ("reconstruction_ready", True),
    ],
)
def test_mlx_extra_kwarg_policy_rejects_inert_runtime_options(
    option_name: str,
    value: Any,
) -> None:
    """MLX extra-kwarg policy rejects accepted-but-inert public options."""

    kwargs: dict[str, Any] = {
        "storage": None,
        "stop_after": None,
        "profile": False,
        "cache": False,
        "raise_on_nan": False,
        "save_outs_to": None,
        "keep_outs_in_memory": True,
        "out_sink": None,
        "cache_dir": None,
        "save_mode": "copy",
        "capture_tensor_grad_hooks": True,
        "save_raw_gradients": True,
        "mark_layer_depths": False,
        "source_context_lines": 7,
        "compute_input_output_distances": False,
        "unwrap_when_done": False,
        "reconstruction_ready": False,
    }
    kwargs[option_name] = value

    with pytest.raises(BackendUnsupportedError, match=option_name):
        reject_extra_trace_kwargs(kwargs, MLX_EXTRA_KWARG_POLICY)
