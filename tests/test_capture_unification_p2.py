"""Phase 2 capture-unification regression tests."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends._protocol import CaptureBackend
from torchlens.backends.torch._tl import get_tensor_label
from torchlens.data_classes.trace import Trace
from torchlens.ir.workspaces import (
    LEGACY_TRACE_BUILD_STATE_KEYS,
    ModuleCaptureWorkspace,
    RawGraphWorkspace,
    WrapperRuntimeWorkspace,
)
from torchlens.visualization._summary_internal._builder import _live_op_count


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
        self.observations = {
            "raw_dict_len": len(trace._raw_graph_ws.raw_layer_dict),
            "raw_labels_len": len(trace._raw_graph_ws.raw_layer_labels_list),
            "event_count": len(events.op_events),
            "live_index_has_label": label in events.live_index.by_raw_label,
            "getitem_label": live_view._label_raw,
            "getitem_shape": live_view.shape,
            "summary_count": _live_op_count(trace),
            "live_repr": repr(trace),
        }
        try:
            trace.summary(level="op")
        except Exception as exc:  # noqa: BLE001 - recorded for the test to assert on
            self.observations["summary_error"] = exc
        return torch.relu(y)


def test_trace_build_state_has_one_eager_owner_and_no_flat_alias_shim() -> None:
    """Keep transient capture state explicit and the legacy routing dunders absent.

    M10: the flat TraceBuildState dissolved into three named per-phase
    workspaces; each is eagerly owned by the Trace during capture, no flat
    alias shim exists, and no legacy flat scratch key resolves.
    """

    trace = Trace(model_class_name="BuildStateProbe")
    try:
        assert isinstance(trace._raw_graph_ws, RawGraphWorkspace)
        assert isinstance(trace._module_capture_ws, ModuleCaptureWorkspace)
        assert isinstance(trace._wrapper_runtime_ws, WrapperRuntimeWorkspace)
        assert "_build_state" not in trace.__dict__
        assert "__setattr__" not in Trace.__dict__
        assert "_ensure_build_state" not in Trace.__dict__
        assert "_build_state_attr_map" not in Trace.__dict__
        for field_name in LEGACY_TRACE_BUILD_STATE_KEYS:
            with pytest.raises(AttributeError):
                getattr(trace, field_name)
    finally:
        trace.cleanup()


def test_backend_finalization_requires_explicit_trace_build_state() -> None:
    """Keep the backend protocol's build-state ownership argument mandatory."""

    parameter = inspect.signature(CaptureBackend.finalize_forward_session).parameters["trace_state"]
    assert parameter.default is inspect.Parameter.empty


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
    tl.trace(model, torch.randn(1, 3), capture=tl.options.CaptureOptions(layers_to_save="all"))

    assert model.observations["raw_dict_len"] == 0
    assert model.observations["raw_labels_len"] == 0
    assert model.observations["event_count"] > 0
    assert model.observations["live_index_has_label"] is True


def test_no_hot_path_liveoprecord_construction_or_field_writes() -> None:
    """Hot capture files should not construct or mutate live operation records."""

    root = Path(__file__).resolve().parents[1]
    hot_files = [
        root / "torchlens/backends/torch/ops.py",
        *sorted((root / "torchlens/backends/torch").glob("_ops_*.py")),
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
    tl.trace(model, torch.randn(1, 3), capture=tl.options.CaptureOptions(layers_to_save="all"))

    assert model.observations["getitem_label"].startswith("linear_")
    assert model.observations["getitem_shape"] == (1, 4)


def test_live_repr_reads_events_during_capture() -> None:
    """The mid-capture repr counts ops from emitted events."""

    model = _MidForwardProbe()
    tl.trace(model, torch.randn(1, 3), capture=tl.options.CaptureOptions(layers_to_save="all"))

    count = model.observations["summary_count"]
    assert count == model.observations["event_count"]
    assert f"layers={count}" in model.observations["live_repr"]


def test_mid_forward_summary_refuses_typed() -> None:
    """summary() during an active forward refuses with trace_not_finished.

    The removed legacy renderer printed a partial live table; the rebuilt
    summary has no partial form, so the call must refuse typed instead of
    running over a half-built trace.
    """

    from torchlens._errors import CaptureContextError

    model = _MidForwardProbe()
    tl.trace(model, torch.randn(1, 3), capture=tl.options.CaptureOptions(layers_to_save="all"))

    error = model.observations.get("summary_error")
    assert isinstance(error, CaptureContextError)
    assert error.fields["code"] == "trace_not_finished"
    assert "repr(trace)" in str(error)


def test_atomic_module_classification_matches_phase1_expectation() -> None:
    """A nested ``Sequential(Linear, ReLU)`` exposes two atomic leaf modules.

    ``block.0`` (a bare ``nn.Linear``) and ``block.1`` (a bare ``nn.ReLU``) are each
    single-op leaf modules, so they classify as atomic module outputs. The parent
    ``block`` is multi-op and is not atomic. (Atomic detection was dormant before
    the detector was restored, which is why this once expected ``False``.)
    """

    log = tl.trace(
        _AtomicBlockModel(),
        torch.randn(1, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    by_func = {op.func_name: op for op in log.ops if op.func_name in {"linear", "relu"}}

    assert by_func["linear"].is_module_output is True
    assert by_func["linear"].is_atomic_module is True
    assert by_func["linear"].atomic_module_call == "block.0:1"
    assert by_func["relu"].is_module_output is True
    assert by_func["relu"].is_atomic_module is True
    assert by_func["relu"].atomic_module_call == "block.1:1"
