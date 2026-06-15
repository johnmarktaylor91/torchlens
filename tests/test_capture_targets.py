"""Phase 4 capture target tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.backends.torch.ops import _save_activation_fields
from torchlens.options import CaptureOptions


class _KpiModel(torch.nn.Module):
    """Tiny model that records a KPI during capture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a small forward pass."""

        y = torch.relu(x)
        tl.record_kpi_in_graph("loss", float(y.sum().detach()))
        return y


class _ConfiguredClassifier(torch.nn.Module):
    """Tiny classifier carrying HF-style output metadata on ``config``."""

    def __init__(self) -> None:
        """Initialize the classifier and its portable config metadata."""

        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.config = SimpleNamespace(id2label={"0": "negative", 1: "positive"}, num_labels=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear classifier.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.linear(x)


def _minimal_activation_fields(label: str) -> dict[str, Any]:
    """Return the field subset required by ``_save_activation_fields``."""

    return {
        "_label_raw": f"{label}_raw",
        "_layer_label_raw": label,
        "annotations": {},
        "backward_ready": False,
        "detach_saved_activations": False,
        "func_name": "manual_test",
        "output_device": "same",
    }


def test_capture_memory_fields_and_forward_source_line() -> None:
    """Populate Phase 4 Op memory fields and forward line number."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    log = tl.trace(model, torch.ones(1, 2))
    assert isinstance(log.forward_source_line, int)
    assert all(hasattr(layer, "bytes_delta_at_call") for layer in log.layer_list)
    assert all(hasattr(layer, "bytes_peak_at_call") for layer in log.layer_list)


def test_record_kpi_in_graph() -> None:
    """Attach arbitrary KPI metadata during capture."""

    log = tl.trace(_KpiModel(), torch.ones(1, 2))
    assert "loss" in log.annotations


def test_content_hash_cache_hit_and_miss(tmp_path: Path) -> None:
    """Reuse a content-hash capture cache entry."""

    model = torch.nn.Linear(2, 2)
    x = torch.ones(1, 2)
    capture = CaptureOptions(cache=True, cache_dir=tmp_path)
    first = tl.trace(model, x, capture=capture)
    second = tl.trace(model, x, capture=capture)
    assert first.capture_cache_hit is False
    assert second.capture_cache_hit is True
    assert first.capture_cache_key == second.capture_cache_key


def test_config_output_metadata_pickles_into_capture_cache(tmp_path: Path) -> None:
    """HF-style config metadata is captured before cache serialization."""

    model = _ConfiguredClassifier()
    x = torch.ones(1, 2)
    capture = CaptureOptions(cache=True, cache_dir=tmp_path)

    first = tl.trace(model, x, capture=capture)
    second = tl.trace(model, x, capture=capture)

    assert first.capture_cache_hit is False
    assert first.output_id2label == {0: "negative", 1: "positive"}
    assert first.output_num_classes == 2
    assert second.capture_cache_hit is True
    assert second.output_id2label == {0: "negative", 1: "positive"}
    assert second.output_num_classes == 2


def test_public_option_spine_changes_capture_cache_key(tmp_path: Path) -> None:
    """Declared public options participate in the content-hash cache key."""

    model = torch.nn.Linear(2, 2)
    x = torch.ones(1, 2)
    first = tl.trace(
        model,
        x,
        capture=CaptureOptions(
            cache=True,
            cache_dir=tmp_path,
            jax_max_control_flow_unroll=8,
        ),
    )
    second = tl.trace(
        model,
        x,
        capture=CaptureOptions(
            cache=True,
            cache_dir=tmp_path,
            jax_max_control_flow_unroll=16,
        ),
    )

    assert first.capture_cache_key != second.capture_cache_key


def test_saved_activation_identity_dedup_reuses_same_source_live_fields() -> None:
    """Live saved activations dedup only when the source tensor object is identical."""

    trace = tl.trace(torch.nn.Identity(), torch.randn(1, 2))
    source = torch.randn(2, 2)
    first_fields = _minimal_activation_fields("manual_1")
    second_fields = _minimal_activation_fields("manual_2")

    _save_activation_fields(trace, first_fields, source, (), {}, None)
    _save_activation_fields(trace, second_fields, source, (), {}, None)

    assert first_fields["out"] is second_fields["out"]
    assert first_fields["out"] is not source
    assert second_fields["annotations"]["dedup_reference_label"] == "manual_1"
    cached_source = trace._out_identity_cache[id(source)][0]
    assert cached_source is source


def test_trace_releases_identity_dedup_cache_after_pass() -> None:
    """Public traces release source pins after the capture pass ends."""

    trace = tl.trace(torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU()), torch.ones(1, 2))

    assert trace._out_identity_cache == {}


def test_saved_activation_identity_dedup_ignores_equal_different_sources() -> None:
    """Equal tensor contents do not dedup by default when source objects differ."""

    trace = tl.trace(torch.nn.Identity(), torch.randn(1, 2))
    first_source = torch.ones(2, 2)
    second_source = torch.ones(2, 2)
    first_fields = _minimal_activation_fields("manual_1")
    second_fields = _minimal_activation_fields("manual_2")

    _save_activation_fields(trace, first_fields, first_source, (), {}, None)
    _save_activation_fields(trace, second_fields, second_source, (), {}, None)

    assert first_fields["out"] is not second_fields["out"]
    assert "dedup_reference_label" not in second_fields["annotations"]


def test_saved_activation_identity_dedup_rejects_key_collision() -> None:
    """A mismatched source under an id key does not false-merge cached payloads."""

    trace = tl.trace(torch.nn.Identity(), torch.randn(1, 2))
    old_source = torch.zeros(2, 2)
    new_source = torch.ones(2, 2)
    stale_out = torch.full((2, 2), 9.0)
    trace._out_identity_cache[id(new_source)] = (old_source, "old", stale_out)
    fields = _minimal_activation_fields("manual_new")

    _save_activation_fields(trace, fields, new_source, (), {}, None)

    assert fields["out"] is not stale_out
    assert "dedup_reference_label" not in fields["annotations"]
    cached_source = trace._out_identity_cache[id(new_source)][0]
    assert cached_source is new_source


def test_op_save_activation_identity_dedup_reuses_same_source() -> None:
    """Replay/slow ``Op.save_activation`` uses identity dedup for the same source."""

    trace = tl.trace(torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU()), torch.ones(1, 2))
    trace._out_identity_cache.clear()
    source = torch.randn(1, 2)
    first_op, second_op = trace.layer_list[:2]

    first_op.save_activation(source, (), {}, save_arg_values=False)
    second_op.save_activation(source, (), {}, save_arg_values=False)

    assert first_op.out is second_op.out
    assert second_op.annotations["dedup_reference_label"] == first_op._layer_label_raw


def test_trace_reference_save_mode_raises_if_saved_out_mutates() -> None:
    """Reference-mode saved outputs fail loudly after mutation."""

    log = tl.trace(torch.nn.ReLU(), torch.ones(1, 2), save_mode="reference")
    op = next(layer for layer in log.layer_list if layer.func_name == "relu")
    saved_out = op._slot("out")

    saved_out.add_(1)

    with pytest.raises(tl.MutatedReferenceError, match="mutated after capture"):
        _ = op.out


def test_trace_reference_save_mode_copies_known_inplace_ops() -> None:
    """Known in-place outputs fall back to copy under reference save mode."""

    class InplaceRelu(torch.nn.Module):
        """Tiny module with a named in-place op."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run an in-place relu on a non-leaf tensor."""

            return x.clone().relu_()

    log = tl.trace(InplaceRelu(), torch.ones(1, 2), save_mode="reference")
    op = next(layer for layer in log.layer_list if layer.func_name == "relu_")
    saved_out = op._slot("out")

    saved_out.add_(1)

    assert "save_mode" not in op.annotations
    assert torch.equal(op.out, torch.full((1, 2), 2.0))


def test_trace_view_save_mode_keeps_saved_out_grad_connected() -> None:
    """View mode keeps saved activation payloads on the autograd graph."""

    model = torch.nn.Linear(2, 2)
    log = tl.trace(model, torch.ones(1, 2, requires_grad=True), save_mode="view")
    op = next(layer for layer in log.layer_list if layer.func_name == "linear")

    assert op.out.requires_grad is True
    assert op.out.grad_fn is not None


def test_tied_parameter_notation_smoke() -> None:
    """Expose tied-parameter notation when shared storage is detected."""

    class Tied(torch.nn.Module):
        """Tiny tied-parameter model."""

        def __init__(self) -> None:
            """Initialize tied layers."""

            super().__init__()
            self.a = torch.nn.Linear(2, 2, bias=False)
            self.b = torch.nn.Linear(2, 2, bias=False)
            self.b.weight = self.a.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run tied layers."""

            return self.b(self.a(x))

    log = tl.trace(Tied(), torch.ones(1, 2))
    tied = [
        layer.annotations.get("tied_parameter_notation")
        for layer in log.layer_list
        if layer.annotations.get("tied_parameter_notation")
    ]
    assert tied
