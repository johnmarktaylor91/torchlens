"""torch.compile rejection tests for train-mode capture APIs."""

from __future__ import annotations

from typing import Callable

import pytest
import torch

import torchlens as tl
from .conftest import TwoLayerMlp


def _compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """Compile a model with the lightweight eager backend.

    Parameters
    ----------
    model:
        Model to compile.

    Returns
    -------
    torch.nn.Module
        Compiled model wrapper.
    """

    compile_fn: Callable[..., torch.nn.Module] | None = getattr(torch, "compile", None)
    if compile_fn is None:
        pytest.skip("torch.compile is unavailable in this PyTorch version")
    return compile_fn(model, backend="eager")


def test_log_forward_pass_rejects_torch_compile(
    two_layer_mlp: TwoLayerMlp,
) -> None:
    """Slow train-mode capture rejects compiled model wrappers."""

    compiled_model = _compile_model(two_layer_mlp)

    with pytest.raises(RuntimeError, match="torch.compile"):
        tl.log_forward_pass(
            compiled_model,
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
        )


def test_save_new_activations_rejects_torch_compile(
    two_layer_mlp: TwoLayerMlp,
) -> None:
    """Replay train-mode capture rejects compiled model wrappers."""

    model_log = tl.log_forward_pass(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        random_seed=0,
    )
    compiled_model = _compile_model(two_layer_mlp)

    with pytest.raises(RuntimeError, match="torch.compile"):
        model_log.save_new_activations(
            compiled_model,
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
            random_seed=0,
        )
    model_log.cleanup()


def test_fastlog_record_rejects_torch_compile(
    two_layer_mlp: TwoLayerMlp,
) -> None:
    """Fastlog train-mode capture rejects compiled model wrappers."""

    compiled_model = _compile_model(two_layer_mlp)

    with pytest.raises(RuntimeError, match="torch.compile"):
        tl.fastlog.record(
            compiled_model,
            torch.randn(3, 4, requires_grad=True),
            train_mode=True,
        )
