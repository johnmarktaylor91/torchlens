"""Regression tests for validation exemption hardening."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.validation import core
from torchlens.validation import backward as backward_validation
from torchlens.validation.exemptions import (
    SKIP_VALIDATION_ENTIRELY,
    _binary_extrema_nonperturbed_arg_dominates,
    perturbed_layer_at_structural_position,
)
from torchlens.validation.status import ValidationReplayStatus


def _fake_layer(**kwargs: Any) -> Any:
    """Build a minimal layer-like object for exemption unit tests.

    Parameters
    ----------
    **kwargs:
        Attributes to install on the fake object.

    Returns
    -------
    Any
        Layer-like namespace.
    """

    return SimpleNamespace(**kwargs)


def test_structural_arg_exemption_uses_parent_position_not_equal_value() -> None:
    """Identical tensor values must not prove a parent occupies a structural slot."""

    layer = _fake_layer(
        saved_args=(torch.tensor([1, 2]), torch.tensor([1, 2])),
        parent_arg_positions={"args": {1: "index_parent"}, "kwargs": {}},
    )

    assert not perturbed_layer_at_structural_position(
        None,  # type: ignore[arg-type]
        layer,
        ["value_parent"],
        {1},
    )
    assert perturbed_layer_at_structural_position(
        None,  # type: ignore[arg-type]
        layer,
        ["index_parent"],
        {1},
    )


def test_full_is_not_inplace_rng_arg_logging_exemption() -> None:
    """A deterministic ``full`` parent must not use the in-place RNG carve-out."""

    parent = _fake_layer(
        layer_label="full_1_1",
        label="full_1_1:1",
        func_name="full",
        out=torch.full((2,), 3.0),
        out_versions_by_child={},
    )
    child = _fake_layer(
        layer_label="add_1_2",
        label="add_1_2:1",
        parent_arg_positions={"args": {0: "full_1_1"}, "kwargs": {}},
        parents=["full_1_1"],
    )
    trace = {"full_1_1": parent}

    result = core._check_arglocs_correct_for_arg(  # noqa: SLF001
        trace,  # type: ignore[arg-type]
        child,
        parent,
        "args",
        0,
        torch.zeros(2),
    )
    assert result.decision == "failed"


def test_binary_extrema_requires_actual_nonperturbed_dominance() -> None:
    """Equal output alone must not exempt binary extrema perturbation."""

    layer = _fake_layer(parent_arg_positions={"args": {0: "lhs", 1: "rhs"}, "kwargs": {}})
    args = (torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0]))

    assert not _binary_extrema_nonperturbed_arg_dominates("maximum", args, layer, ["lhs"])
    assert not _binary_extrema_nonperturbed_arg_dominates("maximum", args, layer, ["rhs"])


def test_magnitude_ratio_shortcut_removed_from_posthoc_exemptions() -> None:
    """The old ``other_mag / perturbed_mag > 100`` predicate must stay removed."""

    source = inspect.getsource(core.posthoc_perturb_check)

    assert "other_mag" not in source
    assert "perturbed_mag" not in source


class DetachedParamModel(nn.Module):
    """Model whose parameter is deliberately disconnected from the loss."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(3))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an output disconnected from parameters.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor depending only on input.
        """

        return self.relu(x * 2)


class OneHotModel(nn.Module):
    """Model that consumes integer class indices through ``one_hot``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return float one-hot encodings.

        Parameters
        ----------
        x:
            Integer class-index tensor.

        Returns
        -------
        torch.Tensor
            One-hot tensor.
        """

        return torch.nn.functional.one_hot(x, num_classes=4).float()


class EmptyLikeModel(nn.Module):
    """Model using uninitialized memory followed by a deterministic write."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a zeroed tensor allocated with ``empty_like``.

        Parameters
        ----------
        x:
            Input tensor used as the allocation template.

        Returns
        -------
        torch.Tensor
            Zero-valued tensor with the same shape as ``x``.
        """

        y = torch.empty_like(x)
        y.zero_()
        return y


class AddReluModel(nn.Module):
    """Small model with a replayable computational add op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ReLU of an add.

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


class AddMulModel(nn.Module):
    """Small model with a selectively saved downstream multiplication."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an add followed by a multiplication.

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


class CholeskyModel(nn.Module):
    """Model whose perturbation can make a valid replay input invalid."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a Cholesky factorization.

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


class PackedSequenceStyleModel(nn.Module):
    """Packed-sequence model with structural lengths metadata."""

    def __init__(self) -> None:
        """Initialize recurrent and projection layers."""

        super().__init__()
        self.lstm = nn.LSTM(8, 4, batch_first=False)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run pack, LSTM, unpack, and projection.

        Parameters
        ----------
        x:
            Input tensor with shape ``(seq_len, batch, features)``.

        Returns
        -------
        torch.Tensor
            Projected final padded timestep.
        """

        lengths = torch.tensor([5, 3, 2])
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=True)
        output, _state = self.lstm(packed)
        padded, _lens = nn.utils.rnn.pad_packed_sequence(output)
        return self.fc(padded[-1])


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


def _first_op_with_func(trace: Any, func_name: str) -> Any:
    """Return the first op in a trace with a matching function name.

    Parameters
    ----------
    trace:
        TorchLens trace.
    func_name:
        Captured function name to find.

    Returns
    -------
    Any
        Matching op.
    """

    return next(layer for layer in trace.layer_list if layer.func_name == func_name)


def _first_output(trace: Any) -> torch.Tensor:
    """Return a detached copy of the trace output.

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


def test_backward_validation_zero_param_grads_is_not_pass() -> None:
    """Backward validation must not pass when no parameter grads are checked."""

    model = DetachedParamModel()

    assert not backward_validation.validate_backward_pass(
        model,
        torch.randn(2, 3),
        random_seed=5,
        validate_metadata=False,
    )


def test_backward_validation_zero_param_grads_still_runs_layer_grad_validation() -> None:
    """Layer-grad validation should run when parameter grads are empty."""

    model = DetachedParamModel()

    assert backward_validation.validate_backward_pass(
        model,
        torch.randn(2, 3),
        random_seed=5,
        validate_metadata=False,
        validate_layer_grads=True,
    )


def test_one_hot_index_perturbation_uses_valid_alternate_class() -> None:
    """One-hot index validation should perturb within ``num_classes``."""

    trace = tl.trace(
        OneHotModel(),
        torch.tensor([1]),
        layers_to_save="all",
        save_arg_values=True,
    )

    assert trace.validate_forward_pass([torch.tensor([[0.0, 1.0, 0.0, 0.0]])])


def test_skip_validation_registry_entries_have_justifications() -> None:
    """Every uninitialized-memory replay exemption must carry a proof string."""

    assert SKIP_VALIDATION_ENTIRELY
    assert all(justification for justification in SKIP_VALIDATION_ENTIRELY.values())


def test_empty_like_is_justified_exempted_not_unverified() -> None:
    """Uninitialized-memory ops should pass as justified design exemptions."""

    trace = tl.trace(
        EmptyLikeModel(),
        torch.randn(2, 3),
        layers_to_save="all",
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    status = trace.validation_replay_status
    assert status.state == "passed"
    assert status.unverified_node_count == 0
    assert status.exempted_reason_counts["uninitialized_by_design"] >= 1
    decisions = [
        decision
        for decision in status.decisions
        if decision.get("reason") == "uninitialized_by_design"
    ]
    assert decisions
    assert all(decision.get("justification") for decision in decisions)


def test_functionless_computational_op_fails_loudly() -> None:
    """A lost callable on a computational op must not be source-exempted."""

    trace = tl.trace(
        AddReluModel(),
        torch.randn(2, 3),
        layers_to_save="all",
        save_arg_values=True,
    )
    _first_op_with_func(trace, "__add__").func = None

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is False
    status = trace.validation_replay_status
    assert status.state == "failed"
    assert any(
        decision["decision"] == "failed" and decision["reason"] == "functionless_computational_op"
        for decision in status.decisions
    )


def test_missing_saved_args_yields_reason_coded_unverified() -> None:
    """Missing saved args should produce status-visible unverified decisions."""

    trace = tl.trace(
        AddReluModel(),
        torch.randn(2, 3),
        layers_to_save="all",
        save_arg_values=False,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert isinstance(result, ValidationReplayStatus)
    assert result.state == "unverified"
    assert result.unverified_reason_counts["missing_saved_args"] >= 1


def test_selective_save_unchecked_surface_is_exempted_not_saved_by_user() -> None:
    """Predicate-excluded replay data should be a by-design exemption."""

    trace = tl.trace(
        AddMulModel(),
        torch.randn(2, 3),
        save=_save_only_mul,
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    status = trace.validation_replay_status
    assert status.state == "passed"
    assert status.unverified_node_count == 0
    assert status.exempted_reason_counts["not_saved_by_user"] >= 1
    decisions = [
        decision for decision in status.decisions if decision.get("reason") == "not_saved_by_user"
    ]
    assert decisions
    assert all(decision.get("justification") for decision in decisions)


def test_selective_save_checkable_mismatch_still_fails() -> None:
    """A retained selective-save payload mismatch must still fail validation."""

    model = AddMulModel()
    x = torch.randn(2, 3)
    trace = tl.trace(model, x, save=_save_only_mul, save_arg_values=True)
    mul_op = _first_op_with_func(trace, "__mul__")
    mul_op._internal_set("out", torch.zeros_like(mul_op.out))  # noqa: SLF001

    result = trace.validate_forward_pass([model(x).detach().clone()], validate_metadata=False)

    assert result is False
    status = trace.validation_replay_status
    assert status.state == "failed"
    assert any(decision["reason"] == "arg_logging_mismatch" for decision in status.decisions)


def test_missing_parent_payload_yields_reason_coded_unverified() -> None:
    """Missing parent payload should be surfaced as unverified, not an exception."""

    trace = tl.trace(
        AddReluModel(),
        torch.randn(2, 3),
        layers_to_save="all",
        save_arg_values=True,
    )
    add_op = _first_op_with_func(trace, "__add__")
    trace[add_op.parents[0]]._internal_set("out", None)  # noqa: SLF001

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert isinstance(result, ValidationReplayStatus)
    assert result.state == "unverified"
    assert result.unverified_reason_counts["missing_saved_parent_payload"] >= 1


def test_replay_mismatch_with_missing_nonperturbed_parent_still_fails() -> None:
    """Saved args must still let ordinary replay catch a real mismatch."""

    trace = tl.trace(
        AddReluModel(),
        torch.randn(2, 3),
        layers_to_save="all",
        save_arg_values=True,
    )
    add_op = _first_op_with_func(trace, "__add__")
    trace[add_op.parents[0]]._internal_set("out", None)  # noqa: SLF001

    def wrong_add(input_tensor: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        """Return an intentionally wrong add result for replay testing.

        Parameters
        ----------
        input_tensor:
            First add operand from the replayed saved args.
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

    add_op.func = wrong_add

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is False
    status = trace.validation_replay_status
    assert status.state == "failed"
    assert status.unverified_reason_counts["missing_saved_parent_payload"] >= 1
    assert any(decision["reason"] == "replay_mismatch" for decision in status.decisions)


def test_perturbation_exception_yields_reason_coded_unverified() -> None:
    """Invalid perturbed inputs should be unverified rather than exempted."""

    trace = tl.trace(
        CholeskyModel(),
        torch.eye(3).unsqueeze(0) * 2,
        layers_to_save="all",
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert isinstance(result, ValidationReplayStatus)
    assert result.state == "unverified"
    assert result.unverified_reason_counts["perturbation_execution_exception"] >= 1


def test_fully_saved_vanilla_model_has_zero_unverified_decisions() -> None:
    """Healthy full-save traces should not produce unverified decisions."""

    trace = tl.trace(
        AddReluModel(),
        torch.randn(2, 3),
        layers_to_save="all",
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    assert trace.validation_replay_status.unverified_node_count == 0


def test_packed_sequence_structural_trace_passes() -> None:
    """Packed-sequence structural metadata should not leave validation unverified."""

    model = PackedSequenceStyleModel()
    trace = tl.trace(
        model,
        torch.rand(5, 3, 8),
        layers_to_save="all",
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    status = trace.validation_replay_status
    assert status.state == "passed"
    assert status.unverified_node_count == 0


def test_validation_status_cache_invalidated_after_same_shape_rerun() -> None:
    """Rerunning a trace should clear cached replay-validation status."""

    model = AddReluModel()
    trace = tl.trace(model, torch.ones(2, 3), layers_to_save="all", save_arg_values=True)
    trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)
    old_status = trace.validation_replay_status

    trace.run(model, torch.ones(2, 3) * 2)
    new_status = trace.validation_replay_status

    assert new_status is not old_status
    assert new_status.state == "available"


def test_validation_status_cache_invalidated_on_fork() -> None:
    """Forks should not inherit a completed validation status."""

    trace = tl.trace(
        AddReluModel(),
        torch.randn(2, 3),
        layers_to_save="all",
        save_arg_values=True,
    )
    trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    fork = trace.fork("status_check")

    assert fork.validation_replay_status.state == "available"
