"""Parity tests for M3 capture-event materialization."""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import time
from typing import Any, Iterator
import weakref
import gzip

import pytest
import torch
from torch import nn
import numpy as np

import torchlens as tl
from torchlens.user_funcs import _detach_nested_for_cache

from _pickle_compare import _canonical_pickle_diff, _tensor_equal
from _pickle_compare_allowlist import ALLOWED_PICKLE_DIFF_FIELDS

_GZIP_MAGIC = b"\x1f\x8b"
_EXPECTED_ALLOWLIST = {
    "Trace": frozenset(
        {
            "backward_peak_memory",
            "cleanup_duration",
            "capture_end_time",
            "forward_duration",
            "forward_peak_memory",
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
        layer.grad_fn_handle = None
        layer.grad_fn_object_id = 0 if layer.grad_fn_object_id is not None else None
        layer.grad_fn_handle = None
        layer.code_context = []
        layer._source_trace_ref = None
        layer._internal_set("saved_args", _detach_nested_for_cache(layer.saved_args))
        layer._internal_set("saved_kwargs", _detach_nested_for_cache(layer.saved_kwargs))
    for layer_log in getattr(trace, "layer_logs", {}).values():
        state = getattr(layer_log, "__dict__", {})
        state["func"] = None
        state["grad_fn_object_id"] = 0 if state.get("grad_fn_object_id") is not None else None
        state["code_context"] = []
        for field_name in (
            "transformed_out",
            "transformed_grad",
            "grad_fn_handle",
            "grad_fn_handle",
        ):
            if field_name not in state:
                continue
            value = state[field_name]
            if isinstance(value, torch.Tensor):
                state[field_name] = value.detach().cpu()
            elif field_name in {"grad_fn_handle", "grad_fn_handle"}:
                state[field_name] = None
    for volatile_field in ("_mod_call_index", "_mod_call_labels", "_mod_entered", "_mod_exited"):
        if hasattr(trace, volatile_field):
            setattr(trace, volatile_field, {})
    if hasattr(trace, "input_object_id"):
        trace.input_object_id = 0
    if hasattr(trace, "model_object_id"):
        trace.model_object_id = 0
    if hasattr(trace, "_source_code_blob"):
        trace._source_code_blob = {}
    if hasattr(trace, "forward_source_file"):
        trace.forward_source_file = None
    if hasattr(trace, "forward_source_line"):
        trace.forward_source_line = None
    if hasattr(trace, "class_source_file"):
        trace.class_source_file = None
    if hasattr(trace, "class_source_line"):
        trace.class_source_line = None
    if hasattr(trace, "name"):
        trace.trace_label = "capture_events_parity"
    _drop_capture_scratch(trace)


def _drop_capture_scratch(trace: Any) -> None:
    """Remove transient capture-only fields before parity serialization.

    Parameters
    ----------
    trace
        Trace-like object to canonicalize in place.

    Returns
    -------
    None
        Mutates ``trace`` by deleting capture-only scratch fields.
    """

    for field_name in (
        "_build_state",
        "_raw_layer_dict",
        "_raw_layer_labels_list",
        "_layer_counter",
        "_raw_layer_type_counter",
        "_unsaved_layers_lookup_keys",
        "_current_func_barcode",
        "_mod_entered",
        "_mod_exited",
        "_mod_call_index",
        "_mod_call_labels",
        "_module_build_data",
        "_module_metadata",
        "_module_forward_args",
        "_module_containment_engine",
        "_exhaustive_module_stack",
        "_grad_fn_strong_refs",
        "_grad_fn_param_refs",
        "_param_log_by_pid",
        "_in_exhaustive_pass",
        "_pending_live_fire_records",
        "_output_container_specs_by_raw_label",
        "_out_writer",
        "_out_sink",
        "_keep_outs_in_memory",
        "_keep_grads_in_memory",
        "_defer_streaming_bundle_finalization",
    ):
        trace.__dict__.pop(field_name, None)
    if trace.__dict__.get("orphan_logs") is None:
        trace.__dict__["orphan_logs"] = ()
    if "forward_source_line" in trace.__dict__:
        trace.__dict__["forward_source_line"] = None
    if "class_source_line" in trace.__dict__:
        trace.__dict__["class_source_line"] = None


def _read_golden_pickle_bytes(path: Path) -> bytes:
    """Read raw pickle bytes, transparently handling compressed large goldens.

    Parameters
    ----------
    path
        Golden pickle path.

    Returns
    -------
    bytes
        Uncompressed pickle bytes.
    """

    data = path.read_bytes()
    if data.startswith(_GZIP_MAGIC):
        return gzip.decompress(data)
    return data


def _write_golden_pickle_bytes(path: Path, data: bytes, model_class_name: str) -> None:
    """Write pickle bytes, compressing oversized ResNet goldens for pre-commit.

    Parameters
    ----------
    path
        Golden pickle path.
    data
        Raw pickle bytes to persist.
    model_class_name
        Fixture model name.

    Returns
    -------
    None
        Writes the golden artifact in place.
    """

    payload = gzip.compress(data, compresslevel=9) if model_class_name == "resnet50" else data
    path.write_bytes(payload)


def test_pickle_byte_equal_pre_m6(
    fixture_model_input: tuple[nn.Module, torch.Tensor, str],
) -> None:
    """Compare current trace pickle bytes against M5 pre-M6 goldens.

    Parameters
    ----------
    fixture_model_input
        Parametrized model/input/name fixture.
    """

    model, x, model_class_name = fixture_model_input
    trace = tl.trace(
        model,
        x,
        random_seed=123,
        layers_to_save=None,
        save_rng_states=False,
        save_code_context=False,
    )
    _make_trace_pickleable(trace)
    golden_path = Path("tests/golden/m5_pre_m6") / f"{model_class_name}.pkl"

    actual = pickle.dumps(trace, protocol=4)
    if os.environ.get("TL_REGEN_GOLDEN") == "1":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        _write_golden_pickle_bytes(golden_path, actual, model_class_name)
        pytest.skip(f"regenerated golden at {golden_path}")

    expected_trace = pickle.loads(_read_golden_pickle_bytes(golden_path))
    _drop_capture_scratch(expected_trace)
    expected = pickle.dumps(expected_trace, protocol=4)
    if actual != expected:
        actual_trace = pickle.loads(actual)
        expected_trace = pickle.loads(expected)
        diffs = _canonical_pickle_diff(actual_trace, expected_trace)
        if diffs:
            diff = _first_state_difference(actual_trace, expected_trace)
            if diff is not None:
                diffs.insert(0, diff)
            pytest.fail(f"Trace pickle differs for {model_class_name}: {diffs[:5]!r}")


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


def test_param_log_by_pid_populated_in_create_session_param_logs() -> None:
    """Param pid lookup values point at real Param addresses."""

    model = nn.Linear(3, 2)
    x = torch.randn(1, 3)
    trace = tl.trace(model, x, random_seed=123)
    assert trace._param_log_by_pid
    assert set(trace._param_log_by_pid.values()) <= set(trace.param_logs.keys())


def test_walk_detects_accumulategrad_via_type_name() -> None:
    """Backward walk records AccumulateGrad parameter attribution by grad-fn label."""

    model = nn.Linear(3, 1)
    x = torch.randn(2, 3)
    trace = tl.trace(model, x, random_seed=123, save_gradients=True)
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss)
    assert trace._grad_fn_param_refs
    assert set(trace._grad_fn_param_refs.values()) <= set(trace.param_logs.keys())
