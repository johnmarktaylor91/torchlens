"""Golden tests for torch replay-validation decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation.status import ValidationReplayStatus


GOLDEN_PATH = Path(__file__).with_name("validation_decisions.json")


class TinyFeedForward(nn.Module):
    """Small feed-forward model with representative elementwise and linear ops."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a feed-forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return torch.relu(self.linear(x)).sum(dim=1)


class TinyBatchNorm(nn.Module):
    """BatchNorm model that exercises registered buffer capture."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.bn.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a BatchNorm-backed pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.bn(x) + self.bn.running_mean


class TinyRecurrent(nn.Module):
    """Small recurrent model with repeated module calls."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.cell = nn.GRUCell(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three recurrent steps.

        Parameters
        ----------
        x:
            Input sequence with shape ``(steps, batch, features)``.

        Returns
        -------
        torch.Tensor
            Final hidden state.
        """

        h = torch.zeros(x.shape[1], 3, dtype=x.dtype)
        for step in range(x.shape[0]):
            h = self.cell(x[step], h)
        return h


class TinyIntervention(nn.Module):
    """Small model used for validating edited traces."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a pass with an editable ReLU site.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return torch.sigmoid(torch.relu(self.linear(x)))


class TinyEmptyLike(nn.Module):
    """Model with an uninitialized allocation followed by deterministic write."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run an ``empty_like`` allocation and zero it.

        Parameters
        ----------
        x:
            Input tensor used as the allocation template.

        Returns
        -------
        torch.Tensor
            Zeroed tensor.
        """

        y = torch.empty_like(x)
        y.zero_()
        return y


class TinyAddRelu(nn.Module):
    """Small model used for corruption and partial-save golden cases."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a replayable add followed by ReLU.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU output.
        """

        return torch.relu(x + 1)


class TinyAddMul(nn.Module):
    """Small model with a selectively saved downstream multiplication."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run an add followed by a multiplication.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Multiplied output tensor.
        """

        return (x + 1) * 2


class TinyCholesky(nn.Module):
    """Model whose perturbed parent can make replay execution invalid."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a Cholesky factorization.

        Parameters
        ----------
        x:
            Positive-definite input matrix.

        Returns
        -------
        torch.Tensor
            Cholesky factor.
        """

        return torch.linalg.cholesky(x)


class TinySwampedAdd(nn.Module):
    """Tiny additive model whose perturbation can be hidden by fp32 spacing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add a large fp32 tensor.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Swamped additive output.
        """

        return x + torch.full_like(x, 1.0e8)


class TinyMultiplyByZero(nn.Module):
    """Tiny model for the generic invariant-output probe."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply by a zero tensor.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Zero-valued output.
        """

        return x * torch.zeros_like(x)


def _trace_output(trace: Any) -> torch.Tensor:
    """Return the first tensor output saved on a trace.

    Parameters
    ----------
    trace:
        TorchLens trace.

    Returns
    -------
    torch.Tensor
        Detached output tensor.
    """

    return trace[trace.output_layers[0]].out.detach().clone()


def _first_op_with_func(trace: Any, func_name: str) -> Any:
    """Return the first op whose captured function name matches.

    Parameters
    ----------
    trace:
        TorchLens trace.
    func_name:
        Function name to locate.

    Returns
    -------
    Any
        Matching operation.
    """

    return next(layer for layer in trace.layer_list if layer.func_name == func_name)


def _wrong_add(input_tensor: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
    """Return an intentionally wrong add replay result.

    Parameters
    ----------
    input_tensor:
        First add operand from saved replay args.
    *_args:
        Ignored positional operands.
    **_kwargs:
        Ignored keyword operands.

    Returns
    -------
    torch.Tensor
        Zero tensor with the same shape as ``input_tensor``.
    """

    return torch.zeros_like(input_tensor)


def _save_only_mul(ctx: Any) -> bool:
    """Select only multiplication ops during predicate capture.

    Parameters
    ----------
    ctx:
        Predicate record context.

    Returns
    -------
    bool
        True when the op is a multiplication.
    """

    return ctx.func_name in {"__mul__", "mul"}


def _status_for_trace(trace: Any, outputs: list[torch.Tensor]) -> ValidationReplayStatus:
    """Validate a trace and return its cached status.

    Parameters
    ----------
    trace:
        TorchLens trace to validate.
    outputs:
        Ground-truth output tensors.

    Returns
    -------
    ValidationReplayStatus
        Cached torch validation status.
    """

    result = trace.validate_forward_pass(outputs)
    if isinstance(result, ValidationReplayStatus):
        return result
    return trace.validation_replay_status


def _seeded_status_for_trace(
    seed: int,
    trace: Any,
    outputs: list[torch.Tensor],
) -> ValidationReplayStatus:
    """Validate a trace after pinning perturbation RNG.

    Parameters
    ----------
    seed:
        RNG seed for replay perturbations.
    trace:
        TorchLens trace to validate.
    outputs:
        Ground-truth output tensors.

    Returns
    -------
    ValidationReplayStatus
        Cached torch validation status.
    """

    torch.manual_seed(seed)
    return _status_for_trace(trace, outputs)


def _case_summary(status: ValidationReplayStatus) -> dict[str, Any]:
    """Build the JSON-stable golden payload for one case.

    Parameters
    ----------
    status:
        Validation status to summarize.

    Returns
    -------
    dict[str, Any]
        Golden payload.
    """

    return {
        "state": status.state,
        "replayed_node_count": status.replayed_node_count,
        "unverified_node_count": status.unverified_node_count,
        "failed_node_count": status.failed_node_count,
        "unverified_reason_counts": dict(status.unverified_reason_counts),
        "exempted_reason_counts": dict(status.exempted_reason_counts),
        "decisions": list(status.decisions),
    }


def build_validation_decision_snapshot() -> dict[str, Any]:
    """Build the complete validation decision snapshot.

    Returns
    -------
    dict[str, Any]
        Snapshot keyed by stable case name.
    """

    torch.manual_seed(11)
    ff = TinyFeedForward().eval()
    x_ff = torch.randn(2, 4)
    full_capture = CaptureOptions(layers_to_save="all", save_arg_values=True)
    ff_trace = tl.trace(ff, x_ff, capture=full_capture)

    torch.manual_seed(12)
    bn = TinyBatchNorm().eval()
    x_bn = torch.randn(3, 4)
    bn_trace = tl.trace(bn, x_bn, capture=full_capture)

    torch.manual_seed(13)
    recurrent = TinyRecurrent().eval()
    x_recurrent = torch.randn(3, 2, 3)
    recurrent_trace = tl.trace(
        recurrent,
        x_recurrent,
        capture=CaptureOptions(
            layers_to_save="all",
            save_arg_values=True,
            random_seed=13,
        ),
    )

    torch.manual_seed(14)
    intervention = TinyIntervention().eval()
    x_intervention = torch.randn(2, 3)
    clean = tl.trace(intervention, x_intervention, capture=full_capture)
    edited = clean.fork("zero_relu")
    relu_pass = next(layer for layer in edited.layer_list if layer.func_name == "relu")
    edited.set(tl.func("relu"), torch.zeros_like(relu_pass.out), confirm_mutation=True)
    edited.run(intervention, x_intervention)

    torch.manual_seed(15)
    empty_like = TinyEmptyLike().eval()
    x_empty = torch.randn(2, 3)
    empty_trace = tl.trace(empty_like, x_empty, capture=full_capture)

    torch.manual_seed(16)
    corrupted = TinyAddRelu().eval()
    x_corrupted = torch.randn(2, 3)
    corrupted_trace = tl.trace(corrupted, x_corrupted, capture=full_capture)
    _first_op_with_func(corrupted_trace, "__add__").func = _wrong_add

    torch.manual_seed(17)
    partial_save = TinyAddRelu().eval()
    x_partial = torch.randn(2, 3)
    partial_trace = tl.trace(
        partial_save,
        x_partial,
        capture=CaptureOptions(layers_to_save="all", save_arg_values=False),
    )

    torch.manual_seed(19)
    selective_save = TinyAddMul().eval()
    x_selective = torch.randn(2, 3)
    selective_trace = tl.trace(
        selective_save,
        x_selective,
        save=_save_only_mul,
        capture=CaptureOptions(save_arg_values=True),
    )

    torch.manual_seed(18)
    cholesky = TinyCholesky().eval()
    x_cholesky = torch.eye(3).unsqueeze(0) * 2
    cholesky_trace = tl.trace(
        cholesky,
        x_cholesky,
        capture=full_capture,
    )

    torch.manual_seed(20)
    swamped = TinySwampedAdd().eval()
    x_swamped = torch.tensor([10000.0, 10001.0], dtype=torch.float32)
    swamped_trace = tl.trace(
        swamped,
        x_swamped,
        capture=full_capture,
    )

    torch.manual_seed(21)
    multiply_zero = TinyMultiplyByZero().eval()
    x_multiply_zero = torch.randn(2, 3)
    multiply_zero_trace = tl.trace(
        multiply_zero,
        x_multiply_zero,
        capture=full_capture,
    )

    return {
        "tiny_feed_forward": _case_summary(
            _seeded_status_for_trace(101, ff_trace, [_trace_output(ff_trace)])
        ),
        "tiny_batch_norm": _case_summary(
            _seeded_status_for_trace(102, bn_trace, [_trace_output(bn_trace)])
        ),
        "tiny_recurrent": _case_summary(
            _seeded_status_for_trace(103, recurrent_trace, [_trace_output(recurrent_trace)])
        ),
        "tiny_intervention": _case_summary(
            _seeded_status_for_trace(104, edited, [_trace_output(edited)])
        ),
        "tiny_empty_like": _case_summary(
            _seeded_status_for_trace(105, empty_trace, [_trace_output(empty_trace)])
        ),
        "tiny_corrupted_replay": _case_summary(
            _seeded_status_for_trace(106, corrupted_trace, [_trace_output(corrupted_trace)])
        ),
        "tiny_partial_save": _case_summary(
            _seeded_status_for_trace(107, partial_trace, [_trace_output(partial_trace)])
        ),
        "tiny_selective_save": _case_summary(
            _seeded_status_for_trace(109, selective_trace, [_trace_output(selective_trace)])
        ),
        "tiny_cholesky": _case_summary(
            _seeded_status_for_trace(108, cholesky_trace, [_trace_output(cholesky_trace)])
        ),
        "tiny_swamped_add": _case_summary(
            _seeded_status_for_trace(110, swamped_trace, [_trace_output(swamped_trace)])
        ),
        "tiny_multiply_zero": _case_summary(
            _seeded_status_for_trace(
                111,
                multiply_zero_trace,
                [_trace_output(multiply_zero_trace)],
            )
        ),
    }


def test_validation_decision_snapshot_matches_golden() -> None:
    """Ensure torch validation decisions stay behavior-locked."""

    expected = json.loads(GOLDEN_PATH.read_text())
    assert build_validation_decision_snapshot() == expected


def test_validation_decision_snapshot_covers_required_categories() -> None:
    """Ensure the golden zoo covers every S1 decision category."""

    snapshot = build_validation_decision_snapshot()
    decisions = [decision for case in snapshot.values() for decision in case["decisions"]]
    reason_decisions = {(decision["decision"], decision["reason"]) for decision in decisions}

    assert any(decision["decision"] == "validated" for decision in decisions)
    assert ("failed", "replay_mismatch") in reason_decisions
    assert ("exempted", "uninitialized_by_design") in reason_decisions
    assert ("exempted", "functionless_source_or_boundary") in reason_decisions
    assert ("exempted", "intentional_intervention_replacement") in reason_decisions
    assert ("exempted", "not_saved_by_user") in reason_decisions
    assert ("exempted", "ulp_swamped_perturbation") in reason_decisions
    assert ("exempted", "multiplicative_zero_annihilator") in reason_decisions
    assert ("exempted", "generic_invariant_output_probe") not in reason_decisions
    assert ("unverified", "missing_saved_args") in reason_decisions
    assert ("unverified", "perturbation_execution_exception") in reason_decisions


def test_validation_decision_snapshot_detects_mutation() -> None:
    """Ensure a changed decision stream is not equal to the golden payload."""

    snapshot = build_validation_decision_snapshot()
    mutated = json.loads(json.dumps(snapshot))
    first_case = next(iter(mutated.values()))
    first_case["decisions"][0]["reason"] = "mutated_reason"

    assert mutated != snapshot
