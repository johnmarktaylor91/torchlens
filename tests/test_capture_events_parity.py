"""Parity tests for M3 capture-event materialization."""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import time
from typing import Any, Iterator
import weakref

import pytest
import torch
from torch import nn
import numpy as np

import torchlens as tl
from torchlens.user_funcs import _detach_nested_for_cache


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

    model_name = str(request.param)
    torch.manual_seed(12345)
    if model_name == "tiny_mlp":
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        x = torch.randn(2, 8)
    elif model_name == "resnet50":
        torchvision = pytest.importorskip("torchvision")
        model = torchvision.models.resnet50(weights=None)
        x = torch.randn(1, 3, 224, 224)
    elif model_name == "gpt2_small":
        model = TinyTransformerBlock()
        x = torch.randn(2, 4, 16)
    elif model_name == "lstm_small":
        model = nn.LSTM(input_size=16, hidden_size=32, num_layers=2)
        x = torch.randn(4, 2, 16)
    else:
        raise AssertionError(f"unknown fixture {model_name}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, x, model_name


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


def _make_trace_pickleable(trace: Any) -> None:
    """Strip non-picklable runtime objects from a trace in place.

    Parameters
    ----------
    trace
        Trace returned by ``tl.trace``.

    Returns
    -------
    None
        Mutates runtime-only fields consistently for golden/current parity.
    """

    for layer in getattr(trace, "layer_list", []):
        for field_name in ("out", "transformed_out", "grad", "transformed_grad"):
            value = getattr(layer, field_name, None)
            if isinstance(value, torch.Tensor):
                layer._internal_set(field_name, value.detach().cpu())
        layer.func = None
        layer.grad_fn = None
        layer.grad_fn_id = 0 if layer.grad_fn_id is not None else None
        layer.grad_fn_log = None
        layer.code_context = []
        layer._source_trace_ref = None
        layer._internal_set("saved_args", _detach_nested_for_cache(layer.saved_args))
        layer._internal_set("saved_kwargs", _detach_nested_for_cache(layer.saved_kwargs))
    for layer_log in getattr(trace, "layer_logs", {}).values():
        state = getattr(layer_log, "__dict__", {})
        state["func"] = None
        state["grad_fn_id"] = 0 if state.get("grad_fn_id") is not None else None
        state["code_context"] = []
        for field_name in ("transformed_out", "transformed_grad", "grad_fn", "grad_fn_log"):
            if field_name not in state:
                continue
            value = state[field_name]
            if isinstance(value, torch.Tensor):
                state[field_name] = value.detach().cpu()
            elif field_name in {"grad_fn", "grad_fn_log"}:
                state[field_name] = None
    for volatile_field in ("_mod_call_index", "_mod_call_labels", "_mod_entered", "_mod_exited"):
        if hasattr(trace, volatile_field):
            setattr(trace, volatile_field, {})
    if hasattr(trace, "input_id"):
        trace.input_id = 0
    if hasattr(trace, "model_id"):
        trace.model_id = 0
    if hasattr(trace, "_source_code_blob"):
        trace._source_code_blob = {}
    if hasattr(trace, "forward_source_file"):
        trace.forward_source_file = None
    if hasattr(trace, "model_source_file"):
        trace.model_source_file = None
    if hasattr(trace, "name"):
        trace.name = "capture_events_parity"


def test_pickle_byte_equal_pre_post_m3(
    fixture_model_input: tuple[nn.Module, torch.Tensor, str],
) -> None:
    """Compare current M3 trace pickle bytes against M2 pre-M3 goldens.

    Parameters
    ----------
    fixture_model_input
        Parametrized model/input/name fixture.
    """

    model, x, model_name = fixture_model_input
    trace = tl.trace(
        model,
        x,
        vis_opt="none",
        random_seed=123,
        layers_to_save=None,
        save_rng_states=False,
        save_code_context=False,
    )
    _make_trace_pickleable(trace)
    golden_path = Path("tests/golden/m2_pre_m3") / f"{model_name}.pkl"

    actual = pickle.dumps(trace, protocol=4)
    if os.environ.get("TL_REGEN_GOLDEN") == "1":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_bytes(actual)
        pytest.skip(f"regenerated golden at {golden_path}")

    expected = golden_path.read_bytes()
    if actual != expected:
        actual_trace = pickle.loads(actual)
        expected_trace = pickle.loads(expected)
        diff = _first_state_difference(actual_trace, expected_trace)
        if diff is not None:
            pytest.fail(f"Trace pickle bytes differ for {model_name}: {diff}")
