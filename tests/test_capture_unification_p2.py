"""Phase 2 capture-unification regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch._tl import get_tensor_label
from torchlens.visualization._summary_internal._builder import _live_op_count, _live_op_rows


class _MidForwardProbe(nn.Module):
    """Model that inspects TorchLens state during an active forward pass."""

    def __init__(self) -> None:
        """Initialize the probe model."""

        super().__init__()
        self.linear = nn.Linear(3, 4)
        self.observations: dict[str, Any] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and inspect the active trace after one op."""

        y = self.linear(x)
        trace = _state._active_trace
        assert trace is not None
        events = trace.capture_events
        label = get_tensor_label(y)
        assert isinstance(label, str)
        live_view = trace[label]
        rows = _live_op_rows(trace)
        self.observations = {
            "raw_dict_len": len(trace._raw_layer_dict),
            "raw_labels_len": len(trace._raw_layer_labels_list),
            "live_by_raw_label_len": len(events.live_by_raw_label),
            "event_count": len(events.op_events),
            "live_index_has_label": label in events.live_index.by_raw_label,
            "getitem_label": live_view._label_raw,
            "getitem_shape": live_view.shape,
            "summary_count": _live_op_count(trace),
            "summary_rows": rows,
        }
        return torch.relu(y)


class _AtomicBlockModel(nn.Module):
    """Nested module model with stable atomic-module classification."""

    def __init__(self) -> None:
        """Initialize nested block layers."""

        super().__init__()
        self.block = nn.Sequential(nn.Linear(3, 4), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested block."""

        return self.block(x)


def test_negative_gate_no_live_mutation_during_forward() -> None:
    """A real forward pass should not populate mutable live-op structures."""

    model = _MidForwardProbe()
    tl.trace(model, torch.randn(1, 3), layers_to_save="all")

    assert model.observations["raw_dict_len"] == 0
    assert model.observations["raw_labels_len"] == 0
    assert model.observations["live_by_raw_label_len"] == 0
    assert model.observations["event_count"] > 0
    assert model.observations["live_index_has_label"] is True


def test_no_hot_path_liveoprecord_construction_or_field_writes() -> None:
    """Hot capture files should not construct or mutate live operation records."""

    root = Path(__file__).resolve().parents[1]
    hot_files = [
        root / "torchlens/backends/torch/ops.py",
        root / "torchlens/backends/torch/model_prep.py",
        root / "torchlens/backends/torch/buffer_writes.py",
        root / "torchlens/backends/torch/tensor_tracking.py",
        root / "torchlens/capture/trace.py",
        root / "torchlens/user_funcs.py",
    ]
    hot_source = "\n".join(path.read_text() for path in hot_files)

    assert "LiveOpRecord(" not in hot_source
    assert "live_record_for_label(" not in hot_source
    assert ".fields[" not in hot_source


def test_in_pass_getitem_returns_event_backed_op() -> None:
    """In-pass raw-label lookup should return the current event-backed op."""

    model = _MidForwardProbe()
    tl.trace(model, torch.randn(1, 3), layers_to_save="all")

    assert model.observations["getitem_label"].startswith("linear_")
    assert model.observations["getitem_shape"] == (1, 4)


def test_live_preview_summary_reads_events_during_capture() -> None:
    """Live preview rows should be rendered from emitted events."""

    model = _MidForwardProbe()
    tl.trace(model, torch.randn(1, 3), layers_to_save="all")

    rows = model.observations["summary_rows"]
    assert model.observations["summary_count"] == model.observations["event_count"]
    assert rows
    assert rows[-1]["name"].startswith("linear_")
    assert rows[-1]["shape"] == "[1,4]"
    assert rows[-1]["dtype"] == "float32"


def test_atomic_module_classification_matches_phase1_expectation() -> None:
    """A nested ``Sequential(Linear, ReLU)`` exposes two atomic leaf modules.

    ``block.0`` (a bare ``nn.Linear``) and ``block.1`` (a bare ``nn.ReLU``) are each
    single-op leaf modules, so they classify as atomic module outputs. The parent
    ``block`` is multi-op and is not atomic. (Atomic detection was dormant before
    the detector was restored, which is why this once expected ``False``.)
    """

    log = tl.trace(_AtomicBlockModel(), torch.randn(1, 3), layers_to_save="all")
    by_func = {op.func_name: op for op in log.ops if op.func_name in {"linear", "relu"}}

    assert by_func["linear"].is_module_output is True
    assert by_func["linear"].is_atomic_module is True
    assert by_func["linear"].atomic_module_call == "block.0:1"
    assert by_func["relu"].is_module_output is True
    assert by_func["relu"].is_atomic_module is True
    assert by_func["relu"].atomic_module_call == "block.1:1"
