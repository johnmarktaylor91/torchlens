"""torch.compile rejection tests for train-mode capture APIs."""

from __future__ import annotations

from typing import Callable
import warnings

import pytest
import torch

import torchlens as tl
from torchlens._capture_state_helpers import reset_compiled_model_unwrap_warning_state
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


def _op_structure(trace: tl.Trace) -> list[tuple[str, str | None]]:
    """Return the eager-comparable operation structure for a trace.

    Parameters
    ----------
    trace:
        Captured trace to summarize.

    Returns
    -------
    list[tuple[str, str | None]]
        Operation type and function-name pairs in execution order.
    """

    return [(op.layer_type, op.func_name) for op in trace.layer_list]


def test_trace_unwraps_torch_compile_once(
    two_layer_mlp: TwoLayerMlp,
) -> None:
    """Slow train-mode capture unwraps compiled modules to their eager source."""

    compiled_model = _compile_model(two_layer_mlp)
    inputs = torch.randn(3, 4, requires_grad=True)
    eager_trace = tl.trace(two_layer_mlp, inputs.clone(), backward_ready=True)
    reset_compiled_model_unwrap_warning_state()

    with pytest.warns(UserWarning, match="compiled model detected"):
        compiled_trace = tl.trace(
            compiled_model,
            inputs,
            backward_ready=True,
        )
    assert _op_structure(compiled_trace) == _op_structure(eager_trace)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        repeated_trace = tl.trace(compiled_model, inputs.clone(), backward_ready=True)
    assert not any("compiled model detected" in str(warning.message) for warning in caught)

    repeated_trace.cleanup()
    compiled_trace.cleanup()
    eager_trace.cleanup()


def test_save_new_outs_unwraps_torch_compile(
    two_layer_mlp: TwoLayerMlp,
) -> None:
    """Replay train-mode capture unwraps compiled model wrappers."""

    trace = tl.trace(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        random_seed=0,
    )
    compiled_model = _compile_model(two_layer_mlp)

    reset_compiled_model_unwrap_warning_state()
    with pytest.warns(UserWarning, match="compiled model detected"):
        trace.save_new_outs(
            compiled_model,
            torch.randn(3, 4, requires_grad=True),
            backward_ready=True,
            random_seed=0,
        )
    trace.cleanup()


def test_fastlog_record_unwraps_torch_compile(
    two_layer_mlp: TwoLayerMlp,
) -> None:
    """Fastlog train-mode capture unwraps compiled model wrappers."""

    compiled_model = _compile_model(two_layer_mlp)

    reset_compiled_model_unwrap_warning_state()
    with pytest.warns(UserWarning, match="compiled model detected"):
        recording = tl.fastlog.record(
            compiled_model,
            torch.randn(3, 4, requires_grad=True),
            backward_ready=True,
        )
    assert recording.to_trace().layer_list
