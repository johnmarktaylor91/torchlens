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
from torchlens.attribution import AttributionError
from torchlens.attribution._steps import _midpoint_alphas, _midpoint_schedule

pytestmark = pytest.mark.smoke


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


@pytest.mark.parametrize("chunk", [2, 8])
def test_step_batching_matches_sequential_input_ig(chunk: int) -> None:
    """Chunked input IG matches sequential at the memo's rtol 1e-4."""

    model = _SmoothMlp()
    inputs = torch.tensor([[0.4, -0.3, 0.2]], dtype=torch.float64)
    sequential = attribution.integrated_gradients(model, inputs, target=1, n_steps=12)
    batched = attribution.integrated_gradients(
        model, inputs, target=1, n_steps=12, step_batch_size=chunk, step_audit_seed=0
    )
    torch.testing.assert_close(batched.values, sequential.values, rtol=1e-4, atol=1e-9)
    assert batched.extra["step_batch_size"] == chunk
    assert batched.extra["path_evaluations_logical"] == 12
    # Physical calls: endpoints + ceil(12/chunk) chunks + 1 audit recompute.
    assert batched.extra["physical_forward_calls"] < sequential.extra["physical_forward_calls"]
    audit = batched.extra["step_audit"]
    assert audit["mode"] == "per_call"
    assert audit["seed"] == 0
    assert len(audit["audited_pairs"]) == 1
    assert audit["sampled_test_not_proof"] is True
    assert audit["worst_deviation"] <= audit["rtol"]


def test_step_batching_callable_target_tree_split() -> None:
    """Callable targets are applied per logical path point via the tree split."""

    model = _SmoothMlp()
    inputs = torch.tensor([[0.1, 0.5, -0.4], [0.3, -0.2, 0.6]], dtype=torch.float64)

    def scorer(output: Tensor) -> Tensor:
        """Score a logical (unstacked) output; shape is asserted."""

        assert output.shape == (2, 2)
        return output[:, 0].sum()

    sequential = attribution.integrated_gradients(model, inputs, target=scorer, n_steps=8)
    batched = attribution.integrated_gradients(
        model, inputs, target=scorer, n_steps=8, step_batch_size=4, step_audit_seed=1
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
            n_steps=4,
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


def test_audit_catches_rows_tail_coupling_adversary() -> None:
    """The rows[1:] adversary is caught by a randomized draw (pinned seed).

    The seed is pinned so the drawn row is a coupled one; a fixed-row-0 audit
    is silent on this adversary by construction (memo finding (d)).
    """

    model = _RowCouplingAdversary()
    inputs = torch.tensor([[0.5, -0.4, 0.3]], dtype=torch.float64)
    # Find a seed whose drawn row is NOT the first stacked row, then assert
    # detection under that pinned seed -- deterministic by construction.
    detecting_seed = None
    for seed in range(20):
        probe = attribution._steps._StepAuditor("per_call", 1, [6], seed=seed)
        row = probe.row_for_chunk(0)
        if row is not None and row != 0:
            detecting_seed = seed
            break
    assert detecting_seed is not None
    with pytest.raises(AttributionError) as excinfo:
        attribution.integrated_gradients(
            model,
            inputs,
            target=0,
            n_steps=6,
            step_batch_size=6,
            step_audit_seed=detecting_seed,
        )
    assert excinfo.value.fields["code"] == "step_batch_audit_failed"
    assert "step_batch_size=1" in str(excinfo.value)


def test_audit_catches_row_mean_coupling_positive_control() -> None:
    """The realistic full-coupling class is caught even by per_call auditing."""

    model = _RowMeanCoupler()
    inputs = torch.tensor([[0.7, -0.1, 0.2]], dtype=torch.float64)
    with pytest.raises(AttributionError) as excinfo:
        attribution.integrated_gradients(
            model,
            inputs,
            target=0,
            n_steps=8,
            step_batch_size=4,
            step_audit="per_chunk",
            step_audit_seed=3,
        )
    assert excinfo.value.fields["code"] == "step_batch_audit_failed"


def test_audit_off_is_disclosed_not_silent() -> None:
    """step_audit='off' is an explicit expert choice and rides the extra."""

    model = _SmoothMlp()
    inputs = torch.tensor([[0.2, 0.1, -0.3]], dtype=torch.float64)
    result = attribution.integrated_gradients(
        model, inputs, target=0, n_steps=4, step_batch_size=2, step_audit="off"
    )
    assert result.extra["step_audit"]["mode"] == "off"
    assert result.extra["step_audit"]["audited_pairs"] == []


@pytest.mark.parametrize("chunk", [2, 8])
def test_step_batching_matches_sequential_layer_ig_and_conductance(chunk: int) -> None:
    """Chunked LIG and conductance match sequential at rtol 1e-4."""

    model = _SmoothMlp()
    inputs = torch.tensor([[0.25, -0.15, 0.35]], dtype=torch.float64)
    for method in (attribution.layer_integrated_gradients, attribution.layer_conductance):
        sequential = method(model, inputs, target=1, layer="hidden", n_steps=12)
        batched = method(
            model,
            inputs,
            target=1,
            layer="hidden",
            n_steps=12,
            step_batch_size=chunk,
            step_audit_seed=0,
        )
        torch.testing.assert_close(batched.values, sequential.values, rtol=1e-4, atol=1e-9)
        assert batched.extra["step_batch_size"] == chunk


def test_layer_completeness_fields_present_with_caveat() -> None:
    """D21: LIG and conductance expose the completeness fields + caveat."""

    model = _SmoothMlp()
    inputs = torch.tensor([[0.6, -0.2, 0.4]], dtype=torch.float64)
    for method in (attribution.layer_integrated_gradients, attribution.layer_conductance):
        result = method(model, inputs, target=0, layer="hidden", n_steps=64)
        for key in (
            "attribution_sum",
            "target_delta",
            "completeness_residual",
            "residual_rel",
            "target_delta_abs",
            "completeness_caveat",
        ):
            assert key in result.extra, (method, key)
        # ``hidden`` IS a bottleneck of this Sequential-shaped MLP, so the
        # layer-space residual is small at n=64.
        assert result.extra["residual_rel"] < 0.02
        assert result.extra["target_delta_abs"] == pytest.approx(
            float(result.extra["target_delta"].abs()), rel=1e-12
        )


def test_input_ig_residual_rel_and_delta_disclosure() -> None:
    """D27: input IG carries residual_rel and |target_delta| beside the residual."""

    model = _SmoothMlp()
    inputs = torch.tensor([[0.45, -0.3, 0.15]], dtype=torch.float64)
    result = attribution.integrated_gradients(model, inputs, target=1, n_steps=128)
    assert result.extra["residual_rel"] < 1e-3
    assert result.extra["target_delta_abs"] > 0.0
    residual = float(result.extra["completeness_residual"].abs())
    assert result.extra["residual_rel"] == pytest.approx(
        residual / result.extra["target_delta_abs"], rel=1e-9
    )
