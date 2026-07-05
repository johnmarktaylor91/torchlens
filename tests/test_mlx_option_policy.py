"""Policy-level tests for MLX preview option honesty."""

from __future__ import annotations

from typing import Any

import pytest

from torchlens.backends import BackendUnsupportedError
from torchlens.backends._options import (
    MLX_PREVIEW_TRACE_OPTION_POLICY,
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
