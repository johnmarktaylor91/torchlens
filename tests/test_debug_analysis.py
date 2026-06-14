"""Tests for TorchLens debug analysis utilities."""

from __future__ import annotations

from inspect import signature
from typing import Any

import torch
from torch import nn

import torchlens as tl


class TinyMlp(nn.Module):
    """Small MLP used by debug analysis tests."""

    def __init__(self) -> None:
        """Initialize layers with deterministic dimensions."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.net(x)


class TinyCnn(nn.Module):
    """Small CNN used by debug analysis tests."""

    def __init__(self) -> None:
        """Initialize convolutional layers."""

        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3), nn.ReLU(), nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CNN forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.net(x)


class LoopMlp(nn.Module):
    """Repeated-layer model that creates multi-pass op records."""

    def __init__(self) -> None:
        """Initialize the shared layer."""

        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two passes through the same layer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        for _ in range(2):
            x = torch.relu(self.linear(x))
        return x


class RaisingGradTrace:
    """Minimal non-torch-like object for backend gate coverage."""

    @property
    def backward_passes(self) -> Any:
        """Raise like non-torch trace backward accessors."""

        raise ValueError("jax traces do not support true backward capture")

    @property
    def saved_grad_ops(self) -> Any:
        """Raise like non-torch trace grad accessors."""

        raise ValueError("jax traces do not expose op-level saved gradients")


def test_lineage_walks_ancestors_and_reports_bad_lookup() -> None:
    """lineage walks graph labels and reports lookup failures without raising."""

    torch.manual_seed(0)
    trace = tl.trace(TinyMlp(), torch.randn(2, 4))

    result = tl.debug.lineage(trace, trace.output_layers[0], max_depth=2)
    labels = [node[0] for node in result.nodes]

    assert result.start_label == "output_1:1"
    assert "linear_2_3:1" in labels
    assert all(depth <= 2 for _, depth, _, _, _ in result.nodes)

    missing = tl.debug.lineage(trace, "missing")
    assert missing.nodes == []
    assert "unavailable" in missing.message


def test_compare_matches_diverges_and_skips_unsaved() -> None:
    """compare aligns pass-qualified ops and handles unsaved activations."""

    torch.manual_seed(1)
    model_a = TinyMlp()
    model_b = TinyMlp()
    model_b.load_state_dict(model_a.state_dict())
    x = torch.randn(2, 4)
    trace_a = tl.trace(model_a, x)
    trace_b = tl.trace(model_b, x + 0.01)
    unsaved = tl.trace(model_b, x, layers_to_save="none")

    same = tl.debug.compare(trace_a, trace_a)
    diff = tl.debug.compare(trace_a, trace_b)
    skipped = tl.debug.compare(trace_a, unsaved)

    assert same.attrs["matched"] > 0
    assert diff.attrs["value_diverged"] > 0
    assert skipped.attrs["activation_unavailable"] > 0
    assert any("unsaved" in reason for reason in skipped["reason"])


def test_compare_handles_recurrent_pass_records_without_layer_accessors() -> None:
    """compare uses op records directly for repeated layers."""

    torch.manual_seed(2)
    trace = tl.trace(LoopMlp(), torch.randn(1, 4))

    frame = tl.debug.compare(trace, trace)

    assert frame.attrs["matched"] > 0
    assert any(op.endswith(":2") for op in frame["op"])


def test_compare_defaults_are_locked() -> None:
    """compare exposes the specified tolerance defaults."""

    params = signature(tl.debug.compare).parameters

    assert params["rtol"].default == 1e-5
    assert params["atol"].default == 1e-8


def test_dead_neurons_reports_rows_and_skip_reasons() -> None:
    """dead_neurons reports dense floating tensors and invalid dims."""

    torch.manual_seed(3)
    trace = tl.trace(TinyCnn(), torch.randn(1, 1, 5, 5))

    frame = tl.debug.dead_neurons(trace, dim=1)
    invalid = tl.debug.dead_neurons(trace, dim=99)

    assert {"op", "total_units", "dead_count", "dead_frac", "sample_dead_idx"}.issubset(
        frame.columns
    )
    assert frame.attrs["skipped"] == 0
    assert invalid.attrs["skipped"] > 0
    assert set(invalid["reason"]) == {"invalid-dim"}
    assert "insufficient sample" in invalid.attrs["note"]


def test_gradient_flow_audit_empty_torch_only_and_defaults() -> None:
    """gradient_flow_audit handles no-backward and non-torch-like traces."""

    torch.manual_seed(4)
    trace = tl.trace(TinyMlp(), torch.randn(2, 4), save_grads=True)

    no_backward = tl.debug.gradient_flow_audit(trace)
    non_torch = tl.debug.gradient_flow_audit(RaisingGradTrace())  # type: ignore[arg-type]
    params = signature(tl.debug.gradient_flow_audit).parameters

    assert "re-trace backward_ready=True" in no_backward.attrs["message"]
    assert non_torch.attrs["message"] == "torch-only"
    assert params["vanishing_threshold"].default == 1e-7
    assert params["exploding_threshold"].default == 1e4


def test_gradient_flow_audit_reports_saved_gradients() -> None:
    """gradient_flow_audit reports saved gradients after log_backward."""

    torch.manual_seed(5)
    x = torch.randn(2, 4, requires_grad=True)
    trace = tl.trace(TinyMlp(), x, save_grads=True)
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss)

    frame = tl.debug.gradient_flow_audit(trace)

    assert len(frame) > 0
    assert frame.attrs["bwd"] == 1
    assert {"vanishing", "exploding", "dead", "unavailable"}.issubset(frame.attrs)


def test_recompute_candidates_ranks_budget_and_excludes_zero_flops() -> None:
    """recompute_candidates ranks candidates and separates zero-FLOP ops."""

    torch.manual_seed(6)
    trace = tl.trace(TinyCnn(), torch.randn(1, 1, 5, 5))
    flatten_trace = tl.trace(nn.Flatten(), torch.randn(1, 2, 2))

    frame = tl.debug.recompute_candidates(trace, budget_gb=1e-9)
    zero = tl.debug.recompute_candidates(flatten_trace)

    assert len(frame) > 0
    assert bool(frame["suggested"].any())
    assert frame.attrs["total_freeable"] > 0
    assert zero.attrs["excluded_nonpositive_flops_forward_count"] > 0
    assert len(zero) == 0
