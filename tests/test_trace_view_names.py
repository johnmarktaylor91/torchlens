"""Public spellings of the Trace concern views and their text blocks."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.types import TraceHealth, TraceProvenance
from torchlens.validation import validate_forward_pass


def _tiny_trace() -> tl.Trace:
    """Capture a one-layer model."""

    torch.manual_seed(0)
    return tl.trace(nn.Sequential(nn.Linear(3, 2), nn.ReLU()), torch.randn(2, 3))


def test_trace_provenance_summary_is_public() -> None:
    """``trace.provenance.summary()`` returns the capture-provenance text block."""

    trace = _tiny_trace()
    try:
        assert isinstance(trace.provenance, TraceProvenance)
        assert isinstance(trace.health, TraceHealth)
        text = trace.provenance.summary()
        assert isinstance(text, str)
        assert "TorchLens Discoverability Summary" in text
        assert not hasattr(trace.provenance, "_summary")
    finally:
        trace.cleanup()


class _WithBuffer(nn.Module):
    """A linear layer, a batch norm (running-stat buffers) and a scalar output."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)
        self.norm = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc(x)).sum()


#: The label property names JMT confirmed through the record refactor (decision walk,
#: 1 Nov 2026): owner type, then the names that must still resolve on it.
_CONFIRMED_LABEL_NAMES: dict[str, tuple[str, ...]] = {
    "Op": ("label", "layer_label", "short_label", "short_layer_label", "raw_label"),
    "Layer": ("label", "layer_label", "short_label", "short_layer_label"),
    "Trace": ("op_labels", "layer_labels", "validate_forward_pass"),
    "GradFnCall": ("grad_fn_label",),
    "Buffer": ("writer_label",),
    "ModuleCall": ("label",),
    "Module": ("call_labels",),
}


def test_confirmed_label_names_still_resolve() -> None:
    """Every confirmed label spelling resolves on a live instance of its owner type."""

    torch.manual_seed(0)
    trace = tl.trace(
        _WithBuffer(),
        torch.randn(4, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads=True),
        save="all",
    )
    try:
        trace.backward(trace[trace.output_layers[0]].out)
        instances = {
            "Op": trace.ops[trace.op_labels[-1]],
            "Layer": trace.layers[trace.layer_labels[-1]],
            "Trace": trace,
            "GradFnCall": trace.autograd.grad_fn_calls[0],
            "Buffer": trace.buffers[0],
            "ModuleCall": trace.module_calls[0],
            "Module": trace.modules["fc"],
        }
        missing = []
        for owner, names in _CONFIRMED_LABEL_NAMES.items():
            instance = instances[owner]
            home = tl if owner in {"Op", "Layer", "Trace"} else tl.types
            assert isinstance(instance, getattr(home, owner)), owner
            missing.extend(f"{owner}.{name}" for name in names if not hasattr(instance, name))
        assert missing == []
        op = instances["Op"]
        assert op.layer_label in trace.layer_labels
        assert op.label in trace.op_labels
        assert callable(trace.validate_forward_pass)
        assert callable(validate_forward_pass)
    finally:
        trace.cleanup()
