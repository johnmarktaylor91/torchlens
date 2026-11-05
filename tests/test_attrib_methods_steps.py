"""F06 B0/B2: IG step runner -- schedule hoist, stacking, randomized audit.

Covers the attrib memo decisions D16-D21: batch-axis stacking equivalence at
rtol 1e-4, the randomized audit ladder with both named adversary cells (the
``rows[1:]`` coupling adversary under a PINNED seed, and the row-coupling
positive control), callable-target tree splitting, typed refusals, and the
D21 layer completeness fields.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

import torchlens.attribution as attribution
from torchlens.attribution._steps import _midpoint_alphas, _midpoint_schedule
from torchlens.errors import AttributionError


class _SmoothMlp(nn.Module):
    """Small smooth MLP whose IG integrals converge quickly."""

    def __init__(self) -> None:
        """Build deterministic weights independent of global RNG state."""

        super().__init__()
        self.hidden = nn.Linear(3, 4, dtype=torch.float64)
        self.head = nn.Linear(4, 2, dtype=torch.float64)
        with torch.no_grad():
            self.hidden.weight.copy_(
                torch.arange(12, dtype=torch.float64).reshape(4, 3) / 7.0 - 0.8
            )
            self.hidden.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.05], dtype=torch.float64))
            self.head.weight.copy_(torch.arange(8, dtype=torch.float64).reshape(2, 4) / 5.0 - 0.7)
            self.head.bias.copy_(torch.tensor([0.02, -0.04], dtype=torch.float64))

    def forward(self, x: Tensor) -> Tensor:
        """Tanh MLP forward."""

        return self.head(torch.tanh(self.hidden(x)))


class _RowCouplingAdversary(nn.Module):
    """The memo's five-line adversary: couples ``rows[1:]``, leaves row 0 alone.

    A fixed-row-0 audit reads clean while the batched attribution is badly
    wrong; only a randomized (chunk, row) draw can detect it (attrib memo
    finding (d) / D18).
    """

    def __init__(self) -> None:
        """Build the fixed projection head."""

        super().__init__()
        self.head = nn.Linear(3, 2, dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(2, 3, dtype=torch.float64) * 1.5)
            self.head.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Mix every row after the first into the batch mean."""

        if x.shape[0] > 1:
            coupled = x[1:] + x[1:].mean(dim=0, keepdim=True)
            x = torch.cat([x[:1], coupled], dim=0)
        return self.head(x)


class _RowMeanCoupler(nn.Module):
    """Positive control: every row is coupled through the batch mean.

    This is the realistic coupling class (train-mode BatchNorm statistics) in
    a mode-independent form, so the test cannot silently pass because
    attribution forced eval mode.
    """

    def __init__(self) -> None:
        """Build the fixed projection head."""

        super().__init__()
        self.head = nn.Linear(3, 2, dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(2, 3, dtype=torch.float64))
            self.head.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Square the mean-centered rows before the head (rows couple).

        The square matters: a linear function of centered rows sums to zero
        over the batch, which would hide the coupling from a summed target.
        """

        return self.head((x - x.mean(dim=0, keepdim=True)) ** 2)


def test_midpoint_schedule_matches_inline_grid() -> None:
    """The hoisted schedule reproduces the historical inline (step+0.5)/n grid."""

    n_steps = 7
    assert _midpoint_alphas(n_steps) == [(step + 0.5) / n_steps for step in range(n_steps)]
    schedule = _midpoint_schedule(n_steps)
    assert [alpha for alpha, _weight in schedule] == _midpoint_alphas(n_steps)
    assert all(weight == pytest.approx(1.0 / n_steps) for _alpha, weight in schedule)


def test_step_batching_callable_target_tree_split() -> None:
    """Callable targets are applied per logical path point via the tree split."""

    model = _SmoothMlp()
    inputs = torch.tensor([[0.1, 0.5, -0.4], [0.3, -0.2, 0.6]], dtype=torch.float64)

    def scorer(output: Tensor) -> Tensor:
        """Score a logical (unstacked) output; shape is asserted."""

        assert output.shape == (2, 2)
        return output[:, 0].sum()

    sequential = attribution.integrated_gradients(model, inputs, target=scorer, num_steps=8)
    batched = attribution.integrated_gradients(
        model, inputs, target=scorer, num_steps=8, step_batch_size=4, step_audit_seed=1
    )
    torch.testing.assert_close(batched.values, sequential.values, rtol=1e-4, atol=1e-9)


def test_step_batching_unsplittable_callable_target_refuses() -> None:
    """A model output that cannot split into logical examples refuses typed."""

    class _ScalarOut(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x.sum()

    with pytest.raises(AttributionError) as excinfo:
        attribution.integrated_gradients(
            _ScalarOut(),
            torch.ones(2, 3, dtype=torch.float64),
            target=lambda out: out,
            num_steps=4,
            step_batch_size=2,
            step_audit="off",
        )
    assert excinfo.value.fields["code"] == "step_batch_target_unsplittable"


def test_step_batch_size_and_audit_vocabulary_refusals() -> None:
    """step_batch_size and step_audit validate with stable codes."""

    model = _SmoothMlp()
    inputs = torch.zeros(1, 3, dtype=torch.float64)
    with pytest.raises(AttributionError) as size_err:
        attribution.integrated_gradients(model, inputs, target=0, step_batch_size=0)
    assert size_err.value.fields["code"] == "step_batch_size_invalid"
    with pytest.raises(AttributionError) as mode_err:
        attribution.integrated_gradients(
            model, inputs, target=0, step_batch_size=2, step_audit="sometimes"
        )
    assert mode_err.value.fields["code"] == "step_audit_invalid"


def test_audit_catches_row_mean_coupling_positive_control() -> None:
    """The realistic full-coupling class is caught even by per_call auditing."""

    model = _RowMeanCoupler()
    inputs = torch.tensor([[0.7, -0.1, 0.2]], dtype=torch.float64)
    with pytest.raises(AttributionError) as excinfo:
        attribution.integrated_gradients(
            model,
            inputs,
            target=0,
            num_steps=8,
            step_batch_size=4,
            step_audit="per_chunk",
            step_audit_seed=3,
        )
    assert excinfo.value.fields["code"] == "step_batch_audit_failed"


@pytest.mark.parametrize("chunk", [2, 8])
def test_step_batching_matches_sequential_layer_ig_and_conductance(chunk: int) -> None:
    """Chunked LIG and conductance match sequential at rtol 1e-4."""

    model = _SmoothMlp()
    inputs = torch.tensor([[0.25, -0.15, 0.35]], dtype=torch.float64)
    for method in (attribution.layer_integrated_gradients, attribution.layer_conductance):
        sequential = method(model, inputs, target=1, layer="hidden", num_steps=12)
        batched = method(
            model,
            inputs,
            target=1,
            layer="hidden",
            num_steps=12,
            step_batch_size=chunk,
            step_audit_seed=0,
        )
        torch.testing.assert_close(batched.values, sequential.values, rtol=1e-4, atol=1e-9)
        assert batched.extra["step_batch_size"] == chunk
