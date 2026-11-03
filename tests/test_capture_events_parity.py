"""Parity tests for M3 capture-event materialization."""

from __future__ import annotations

import time
import weakref
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch
from _pickle_compare import _tensor_equal
from _pickle_compare_allowlist import ALLOWED_PICKLE_DIFF_FIELDS
from torch import nn

import torchlens as tl
from torchlens._data_classes._state_adapter import state_items

_GZIP_MAGIC = b"\x1f\x8b"
_EXPECTED_ALLOWLIST = {
    "Trace": frozenset(
        {
            "backward_peak_memory",
            "cleanup_duration",
            "capture_end_time",
            "forward_duration",
            "forward_peak_memory",
            # forks.13: set iff a CPU forward delta read 0, so it varies exactly
            # with forward_peak_memory between two otherwise identical captures.
            "forward_peak_memory_reason",
            "func_calls_duration",
            "setup_duration",
            "capture_start_time",
        }
    ),
    "Op": frozenset(
        {
            "bytes_peak_at_call",
            "raw_index",
            "func_call_id",
            "func_duration",
        }
    ),
}


class TinyTransformerBlock(nn.Module):
    """Small GPT-style block used to avoid optional Hugging Face dependencies."""

    def __init__(self) -> None:
        """Initialize the attention and MLP submodules."""

        super().__init__()
        self.ln_1 = nn.LayerNorm(16)
        self.attn = nn.MultiheadAttention(16, 2, batch_first=True)
        self.ln_2 = nn.LayerNorm(16)
        self.mlp = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one residual attention block.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, seq, features)``.

        Returns
        -------
        torch.Tensor
            Block output with the same shape as ``x``.
        """

        attn_in = self.ln_1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.ln_2(x))


@pytest.fixture(
    params=[
        "tiny_mlp",
        "resnet50",
        "gpt2_small",
        "lstm_small",
    ]
)
def fixture_model_input(request: pytest.FixtureRequest) -> tuple[nn.Module, torch.Tensor, str]:
    """Build a deterministic model/input pair for capture parity.

    Parameters
    ----------
    request
        Parametrized pytest fixture request.

    Returns
    -------
    tuple[nn.Module, torch.Tensor, str]
        Model, input tensor, and stable model name.
    """

    model_class_name = str(request.param)
    torch.manual_seed(12345)
    model: nn.Module
    if model_class_name == "tiny_mlp":
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        x = torch.randn(2, 8)
    elif model_class_name == "resnet50":
        torchvision = pytest.importorskip("torchvision")
        model = torchvision.models.resnet50(weights=None)
        x = torch.randn(1, 3, 64, 64)
    elif model_class_name == "gpt2_small":
        model = TinyTransformerBlock()
        x = torch.randn(2, 4, 16)
    elif model_class_name == "lstm_small":
        model = nn.LSTM(input_size=16, hidden_size=32, num_layers=2)
        x = torch.randn(4, 2, 16)
    else:
        raise AssertionError(f"unknown fixture {model_class_name}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, x, model_class_name


@pytest.fixture(autouse=True)
def deterministic_time(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make trace timing fields deterministic across golden and current runs.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.

    Yields
    ------
    None
        Control returns to the test with patched time.
    """

    counter = {"value": 1_000.0}

    def fake_time() -> float:
        """Return a deterministic monotonically increasing timestamp."""

        counter["value"] += 0.001
        return counter["value"]

    monkeypatch.setattr(time, "time", fake_time)
    yield


def _first_state_difference(
    left: Any,
    right: Any,
    path: str = "trace",
    seen: set[tuple[int, int]] | None = None,
) -> str | None:
    """Return a concise path to the first visible state difference.

    Parameters
    ----------
    left
        Actual object.
    right
        Expected object.
    path
        Human-readable state path.

    Returns
    -------
    str | None
        First differing path, or ``None`` when no shallow difference is found.
    """

    if seen is None:
        seen = set()
    pair = (id(left), id(right))
    if pair in seen:
        return None
    seen.add(pair)
    if type(left) is not type(right):
        return f"{path}: type {type(left)!r} != {type(right)!r}"
    if isinstance(left, weakref.ReferenceType):
        return None
    if getattr(type(left), "_PORTABLE_STATE_SPEC", None) is not None:
        left_state = dict(state_items(left))
        right_state = dict(state_items(right))
        left_keys = set(left_state)
        right_keys = set(right_state)
        if left_keys != right_keys:
            return f"{path}: keys {sorted(left_keys ^ right_keys)!r}"
        for key in sorted(left_keys):
            diff = _first_state_difference(left_state[key], right_state[key], f"{path}.{key}", seen)
            if diff is not None:
                return diff
        return None
    if hasattr(left, "__dict__") and hasattr(right, "__dict__"):
        left_keys = set(left.__dict__)
        right_keys = set(right.__dict__)
        if left_keys != right_keys:
            return f"{path}: keys {sorted(left_keys ^ right_keys)!r}"
        for key in sorted(left_keys):
            diff = _first_state_difference(
                left.__dict__[key], right.__dict__[key], f"{path}.{key}", seen
            )
            if diff is not None:
                return diff
        return None
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: dict keys differ {sorted(set(left) ^ set(right))!r}"
        for key in left:
            diff = _first_state_difference(left[key], right[key], f"{path}[{key!r}]", seen)
            if diff is not None:
                return diff
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return f"{path}: len {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            diff = _first_state_difference(left_item, right_item, f"{path}[{index}]", seen)
            if diff is not None:
                return diff
        return None
    try:
        equal = left == right
    except Exception:
        equal = repr(left) == repr(right)
    if isinstance(equal, torch.Tensor):
        equal = bool(torch.all(equal).item())
    if isinstance(equal, np.ndarray):
        equal = bool(np.all(equal))
    if not bool(equal):
        return f"{path}: {left!r} != {right!r}"
    return None


def test_pickle_compare_allowlist_stable() -> None:
    """Assert the parity allow-list is an explicit committed contract."""

    assert ALLOWED_PICKLE_DIFF_FIELDS == _EXPECTED_ALLOWLIST


def test_tensor_equal_handles_bool_int_nan_dtypes() -> None:
    """Cover dtype-sensitive exact tensor comparisons used by parity tests."""

    assert _tensor_equal(torch.tensor([True, False]), torch.tensor([True, False]))
    assert not _tensor_equal(torch.tensor([True, False]), torch.tensor([False, False]))
    assert _tensor_equal(
        torch.tensor([1, 2], dtype=torch.int32), torch.tensor([1, 2], dtype=torch.int32)
    )
    assert _tensor_equal(
        torch.tensor([1, 2], dtype=torch.int64), torch.tensor([1, 2], dtype=torch.int64)
    )
    assert _tensor_equal(torch.tensor([float("nan"), 1.0]), torch.tensor([float("nan"), 1.0]))
    assert not _tensor_equal(
        torch.tensor([1], dtype=torch.int32), torch.tensor([1], dtype=torch.int64)
    )


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/param.py.
def test_released_param_grad_refetches_after_manual_backward() -> None:
    """Param grad metadata re-fetches the live model parameter after ref release."""

    model = nn.Linear(3, 1)
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(seed=123, backward_ready=True))
    param_log = trace.params["weight"]

    assert param_log._param_ref is None
    assert param_log.has_grad is False
    trace[trace.output_layers[0]].out.sum().backward()

    assert param_log.grad is model.weight.grad
    assert param_log.has_grad is True
    assert param_log.grad_shape == tuple(model.weight.grad.shape)


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_walk_detects_accumulategrad_via_type_name() -> None:
    """Backward walk records AccumulateGrad parameter attribution by grad-fn label."""

    model = nn.Linear(3, 1)
    x = torch.randn(2, 3)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(seed=123, save_grads=True))
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss)
    assert trace._grad_fn_param_refs
    assert set(trace._grad_fn_param_refs.values()) <= set(trace._param_logs.keys())
