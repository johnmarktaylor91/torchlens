"""Regression tests for validation exemption hardening."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import core
from torchlens.validation import backward as backward_validation
from torchlens.validation.diagnostics import get_validation_failure
from torchlens.validation.invariants import (
    MetadataInvariantError,
    _check_backend_neutral_graph_topology,
    _check_pass_count_consistency,
    check_metadata_invariants,
)
from torchlens.validation.exemptions import (
    SKIP_VALIDATION_ENTIRELY,
    _binary_extrema_nonperturbed_arg_dominates,
    _check_getitem_exempt,
    _check_scatter_exempt,
    _check_setitem_exempt,
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


def test_getitem_exemption_uses_parent_position_not_equal_value() -> None:
    """Equal data/index values do not make the data parent structural."""

    layer = _fake_layer(
        saved_args=(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
        parent_arg_positions={"args": {0: "data_parent", 1: "index_parent"}, "kwargs": {}},
    )

    assert not _check_getitem_exempt(None, layer, ["data_parent"])  # type: ignore[arg-type]
    assert _check_getitem_exempt(None, layer, ["index_parent"])  # type: ignore[arg-type]


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


def test_reduction_depth_reads_conv_keyword_weight() -> None:
    """Band-C eligibility must read convolution weights passed by keyword."""

    layer = _fake_layer(
        func_name="conv2d",
        saved_args=(torch.randn(1, 3, 8, 8),),
        saved_kwargs={"weight": torch.randn(16, 3, 5, 5)},
    )

    assert core._op_reduction_depth(layer) == 75  # noqa: SLF001


def test_reduction_depth_reads_matmul_keyword_operands() -> None:
    """Band-C eligibility must read contraction operands passed by keyword."""

    layer = _fake_layer(
        func_name="mm",
        saved_args=(),
        saved_kwargs={"input": torch.randn(4, 128), "mat2": torch.randn(128, 5)},
    )

    assert core._op_reduction_depth(layer) == 128  # noqa: SLF001


def test_reduction_depth_withholds_shallow_late_lenience() -> None:
    """Elementwise or shallow ops stay ineligible regardless of graph position."""

    layer = _fake_layer(
        func_name="__add__",
        saved_args=(torch.randn(4), torch.randn(4)),
        saved_kwargs={},
        step_index=250,
    )

    assert core._op_reduction_depth(layer) == 1  # noqa: SLF001


class PartialSetitemDestinationModel(nn.Module):
    """Model that partially overwrites an all-zero setitem destination."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a partially overwritten scratch tensor."""

        buffer = torch.zeros(4, 2)
        buffer[:1] = x[:1]
        return buffer


def test_setitem_blank_destination_partial_overwrite_is_not_exempt() -> None:
    """Blank destination values do not prove setitem perturbation insensitivity."""

    trace = tl.trace(
        PartialSetitemDestinationModel(),
        torch.randn(4, 2),
        layers_to_save="all",
        save_arg_values=True,
    )
    setitem_op = next(op for op in trace.layer_list if op.func_name == "__setitem__")
    destination_label = setitem_op.parent_arg_positions["args"][0]

    assert not _check_setitem_exempt(trace, setitem_op, [destination_label])


class DuplicateIndexSetitemDestinationModel(nn.Module):
    """Model whose duplicate advanced indices leave destination values live."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overwrite one destination element twice and leave another untouched."""

        buffer = torch.zeros(2)
        buffer[[0, 0]] = x[:2]
        return buffer


def test_setitem_duplicate_advanced_index_destination_is_not_exempt() -> None:
    """Duplicate advanced indices do not prove full destination overwrite."""

    trace = tl.trace(
        DuplicateIndexSetitemDestinationModel(),
        torch.tensor([3.0, 4.0]),
        layers_to_save="all",
        save_arg_values=True,
    )
    setitem_op = next(op for op in trace.layer_list if op.func_name == "__setitem__")
    destination_label = setitem_op.parent_arg_positions["args"][0]

    assert not _check_setitem_exempt(trace, setitem_op, [destination_label])


def test_scatter_exemption_uses_destination_position_not_equal_value() -> None:
    """Equal source/destination values do not make the source parent structural."""

    class EqualValuedParentTrace:
        """Minimal trace resolving parent outputs by label."""

        def __getitem__(self, label: str) -> Any:
            """Return a fake parent with an equal-valued output."""

            del label
            return _fake_layer(out=torch.zeros(3))

    layer = _fake_layer(
        saved_args=(torch.zeros(3), 0, torch.arange(3), torch.zeros(3)),
        saved_kwargs={},
        parent_arg_positions={"args": {0: "dest_parent", 3: "src_parent"}, "kwargs": {}},
    )
    trace = EqualValuedParentTrace()

    assert not _check_scatter_exempt(trace, layer, ["src_parent"])  # type: ignore[arg-type]
    assert _check_scatter_exempt(trace, layer, ["dest_parent"])  # type: ignore[arg-type]


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


class SwampedAddModel(nn.Module):
    """Model with an additive parent perturbation below fp32 output spacing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add a large same-dtype tensor that swamps small x perturbations.

        Parameters
        ----------
        x:
            Floating point input tensor.

        Returns
        -------
        torch.Tensor
            Additive output with large representable spacing.
        """

        return x + torch.full_like(x, 1.0e8)


class SimilarMagnitudeAddModel(nn.Module):
    """Model whose add parent perturbation remains representable."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add a similar-magnitude tensor.

        Parameters
        ----------
        x:
            Floating point input tensor.

        Returns
        -------
        torch.Tensor
            Additive output whose spacing should not hide perturbations.
        """

        return x + torch.full_like(x, 10.0)


class SaturatingSignExpModel(nn.Module):
    """Model whose sign output is constant over the captured exp range."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sign(exp(x)).

        Parameters
        ----------
        x:
            Floating point input tensor.

        Returns
        -------
        torch.Tensor
            Positive sign output.
        """

        return torch.sign(torch.exp(x))


class QuantizedCeilSigmoidModel(nn.Module):
    """Model whose ceil output is constant over the captured sigmoid range."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ceil(sigmoid(x)).

        Parameters
        ----------
        x:
            Floating point input tensor.

        Returns
        -------
        torch.Tensor
            Unit-valued ceil output.
        """

        return torch.ceil(torch.sigmoid(x))


class ExpOverflowModel(nn.Module):
    """Model with legitimate exponential overflow."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return exp(x).

        Parameters
        ----------
        x:
            Large floating point input tensor.

        Returns
        -------
        torch.Tensor
            Exponential output that may overflow to infinity.
        """

        return torch.exp(x)


class InfTimesFiniteModel(nn.Module):
    """Model multiplying an infinite operand by a finite operand."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an all-inf product through finite dataflow parents.

        Parameters
        ----------
        x:
            Floating point input tensor.

        Returns
        -------
        torch.Tensor
            All-infinite product tensor.
        """

        infinite = x * 0 + float("inf")
        finite = x * 0 + 2.0
        return infinite * finite


class BigSquareOverflowModel(nn.Module):
    """Model whose square overflows to infinity."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Square a very large finite tensor.

        Parameters
        ----------
        x:
            Floating point input tensor.

        Returns
        -------
        torch.Tensor
            All-infinite squared tensor.
        """

        big = x + 1.0e30
        return big * big


class GetitemMaxModel(nn.Module):
    """Selection model with data values at the fp32 finite maximum."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a slice of ``x``.

        Parameters
        ----------
        x:
            Floating point input tensor.

        Returns
        -------
        torch.Tensor
            First selected element.
        """

        return x[:1]


class UniqueMaxModel(nn.Module):
    """Unique-value model with data values at the fp32 finite maximum."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unique values from ``x``.

        Parameters
        ----------
        x:
            Floating point input tensor.

        Returns
        -------
        torch.Tensor
            Unique values.
        """

        return torch.unique(x)


class MultiplyByZeroModel(nn.Module):
    """Model where a parent is provably annihilated by another parent."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply by a zero tensor.

        Parameters
        ----------
        x:
            Floating point input tensor.

        Returns
        -------
        torch.Tensor
            Zero output independent of ``x``.
        """

        return x * torch.zeros_like(x)


class LoopOutputBookkeepingModel(nn.Module):
    """Small loop model that exercises output bookkeeping perturbation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a parameter-free loop.

        Parameters
        ----------
        x:
            Positive floating point input.

        Returns
        -------
        torch.Tensor
            Loop output.
        """

        y = x + 2.0
        for _ in range(3):
            y = torch.log(y)
            y = torch.sin(y)
        return y + 3.0


class CrossEntropyKwargModel(nn.Module):
    """Model passing a structural target tensor by keyword."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy with keyword target.

        Parameters
        ----------
        logits:
            Class logits.
        target:
            Integer class labels.

        Returns
        -------
        torch.Tensor
            Cross entropy loss.
        """

        return torch.nn.functional.cross_entropy(input=logits, target=target)


class BufferOwnerModel(nn.Module):
    """BatchNorm model with registered buffers under a child module."""

    def __init__(self) -> None:
        """Initialize the BatchNorm model."""

        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.bn.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run BatchNorm and consume a registered buffer outside the owner.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            BatchNorm output plus running mean.
        """

        return self.bn(x) + self.bn.running_mean


class TrainingInstanceNormModel(nn.Module):
    """InstanceNorm model with training-mode running-stat buffers."""

    def __init__(self) -> None:
        """Initialize the InstanceNorm model."""

        super().__init__()
        self.norm = nn.InstanceNorm1d(3, track_running_stats=True)
        self.norm.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run training-mode InstanceNorm.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """

        return self.norm(x)


class ScalarMaskedFillAllSelectedModel(nn.Module):
    """In-place scalar masked-fill model with a fully selected mask."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fill all positions of a clone with a Python scalar.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Zero-filled clone.
        """

        y = x.clone()
        mask = torch.ones(1, 1, x.shape[-1], dtype=torch.bool, device=x.device)
        return y.masked_fill_(mask, 0.0)


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


def _install_constant_replay(trace: Any, func_name: str) -> None:
    """Replace the last matching op replay callable with a constant replay.

    Parameters
    ----------
    trace:
        TorchLens trace to corrupt.
    func_name:
        Function name identifying the op to corrupt.

    Returns
    -------
    None
        ``trace`` is mutated in place.
    """

    op = [layer for layer in trace.layer_list if layer.func_name == func_name][-1]
    saved_output = op.out.detach().clone()

    def constant_replay(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        """Return the saved output while ignoring replay inputs.

        Parameters
        ----------
        *_args:
            Ignored positional replay arguments.
        **_kwargs:
            Ignored keyword replay arguments.

        Returns
        -------
        torch.Tensor
            Saved output clone.
        """

        return saved_output.detach().clone()

    op.func = constant_replay


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


def test_swamped_fp32_add_uses_ulp_predicate() -> None:
    """Swamped fp32 add should pass only through the ULP spacing predicate."""

    x = torch.tensor([10000.0, 10001.0], dtype=torch.float32)
    trace = tl.trace(
        SwampedAddModel(),
        x,
        layers_to_save="all",
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    status = trace.validation_replay_status
    assert status.exempted_reason_counts["ulp_swamped_perturbation"] >= 1
    assert status.failed_node_count == 0


def test_similar_magnitude_influential_add_is_not_ulp_exempted() -> None:
    """A representable add perturbation must not receive the ULP exemption."""

    x = torch.tensor([10.0, 11.0], dtype=torch.float32)
    trace = tl.trace(
        SimilarMagnitudeAddModel(),
        x,
        layers_to_save="all",
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    assert "ulp_swamped_perturbation" not in trace.validation_replay_status.exempted_reason_counts


def test_boundary_crossing_validates_piecewise_constant_ops() -> None:
    """Piecewise-constant ops should validate through boundary-crossing candidates."""

    cases = (
        (SaturatingSignExpModel(), torch.tensor([1.0, 2.0], dtype=torch.float32)),
        (QuantizedCeilSigmoidModel(), torch.tensor([1.0, 2.0], dtype=torch.float32)),
    )
    for model, x in cases:
        trace = tl.trace(
            model,
            x,
            layers_to_save="all",
            save_arg_values=True,
        )

        result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

        assert result is True
        assert trace.validation_replay_status.failed_node_count == 0
        assert (
            "locally_constant_by_construction"
            not in trace.validation_replay_status.exempted_reason_counts
        )


def test_instance_norm_running_stats_are_training_update_targets() -> None:
    """Training InstanceNorm running stats should use the normalization proof."""

    trace = tl.trace(
        TrainingInstanceNormModel(),
        torch.randn(2, 3, 4),
        layers_to_save="all",
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    status = trace.validation_replay_status
    assert status.failed_node_count == 0
    assert status.exempted_reason_counts["pre_perturbation_exemption"] >= 1


def test_scalar_masked_fill_all_selected_input_parent_is_structural() -> None:
    """A fully selected scalar ``masked_fill_`` proves the input parent irrelevant."""

    trace = tl.trace(
        ScalarMaskedFillAllSelectedModel(),
        torch.randn(1, 2, 10),
        layers_to_save="all",
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    status = trace.validation_replay_status
    assert status.failed_node_count == 0
    assert status.exempted_reason_counts["pre_perturbation_exemption"] >= 1


def test_corrupted_piecewise_constant_wrong_edges_still_fail() -> None:
    """Wrong replay edges for quantizing ops must not receive sampling exemptions."""

    cases = (
        (SaturatingSignExpModel(), torch.tensor([1.0, 2.0], dtype=torch.float32), "sign"),
        (QuantizedCeilSigmoidModel(), torch.tensor([1.0, 2.0], dtype=torch.float32), "ceil"),
        (InfTimesFiniteModel(), torch.tensor([1.0, 2.0], dtype=torch.float32), "__mul__"),
    )
    for model, x, func_name in cases:
        trace = tl.trace(
            model,
            x,
            layers_to_save="all",
            save_arg_values=True,
        )
        _install_constant_replay(trace, func_name)

        result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

        assert result is False
        assert trace.validation_replay_status.failed_node_count >= 1
        assert any(
            decision["decision"] == "failed" and decision["reason"] == "perturbation_insensitive"
            for decision in trace.validation_replay_status.decisions
        )


def test_all_inf_legitimate_ops_validate_or_exempt_with_proof() -> None:
    """All-inf legitimate ops should not false-fail after boundary perturbation."""

    cases = (
        (ExpOverflowModel(), torch.full((2,), 100.0, dtype=torch.float32)),
        (ExpOverflowModel(), torch.full((2,), 20.0, dtype=torch.float16)),
        (InfTimesFiniteModel(), torch.tensor([1.0, 2.0], dtype=torch.float32)),
        (BigSquareOverflowModel(), torch.tensor([1.0, 2.0], dtype=torch.float32)),
    )
    for model, x in cases:
        trace = tl.trace(
            model,
            x,
            layers_to_save="all",
            save_arg_values=True,
        )

        result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

        assert result is True
        assert trace.validation_replay_status.failed_node_count == 0


def test_max_finite_selection_data_parents_perturb_distinctly() -> None:
    """Selection data perturbations should not no-op at finite dtype maxima."""

    x = torch.full((2,), torch.finfo(torch.float32).max, dtype=torch.float32)
    for model in (GetitemMaxModel(), UniqueMaxModel()):
        trace = tl.trace(
            model,
            x,
            layers_to_save="all",
            save_arg_values=True,
        )

        result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

        assert result is True
        assert trace.validation_replay_status.failed_node_count == 0


def test_corrupted_swamped_add_replay_fails_without_generic_probe_exemption() -> None:
    """A constant replay callable must fail even if generic probes match."""

    trace = tl.trace(
        SwampedAddModel(),
        torch.tensor([10000.0, 10001.0], dtype=torch.float32),
        layers_to_save="all",
        save_arg_values=True,
    )
    _install_constant_replay(trace, "__add__")

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is False
    status = trace.validation_replay_status
    assert "generic_invariant_output_probe" not in status.exempted_reason_counts
    assert any(
        decision["decision"] == "failed" and decision["reason"] == "perturbation_insensitive"
        for decision in status.decisions
    )
    failure = get_validation_failure(trace)
    assert failure is not None
    assert failure.extra["generic_invariant_probe_matched"] is True


def test_multiplicative_zero_annihilator_uses_structural_proof() -> None:
    """Multiplication by a saved zero operand should use a structural proof."""

    trace = tl.trace(
        MultiplyByZeroModel(),
        torch.randn(2, 3),
        layers_to_save="all",
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    assert (
        trace.validation_replay_status.exempted_reason_counts["multiplicative_zero_annihilator"]
        >= 1
    )
    decisions = [
        decision
        for decision in trace.validation_replay_status.decisions
        if decision.get("reason") == "multiplicative_zero_annihilator"
    ]
    assert decisions
    assert all(decision.get("justification") for decision in decisions)


def test_generic_probe_does_not_exempt_influential_parent() -> None:
    """The generic diagnostic probe must not exempt a genuinely influential add parent."""

    trace = tl.trace(
        SimilarMagnitudeAddModel(),
        torch.tensor([10.0, 11.0], dtype=torch.float32),
        layers_to_save="all",
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    assert (
        "generic_invariant_output_probe"
        not in trace.validation_replay_status.exempted_reason_counts
    )


def test_output_bookkeeping_projection_is_structural() -> None:
    """Loop outputs with NaNs should use meaningful perturbation candidates."""

    trace = tl.trace(
        LoopOutputBookkeepingModel(),
        torch.full((2, 3), 1.5),
        layers_to_save="all",
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    assert trace.validation_replay_status.failed_node_count == 0


def test_structural_arg_exemption_covers_keyword_parent_positions() -> None:
    """Structural arg exemptions should use kwarg parent-position metadata."""

    logits = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.4, 0.7]], dtype=torch.float32)
    target = torch.tensor([0, 2], dtype=torch.long)
    trace = tl.trace(
        CrossEntropyKwargModel(),
        logits,
        input_kwargs={"target": target},
        layers_to_save="all",
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    assert trace.validation_replay_status.exempted_reason_counts["pre_perturbation_exemption"] >= 1


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


def test_selective_save_interior_gap_is_unverified() -> None:
    """A selected op behind an unsaved parent must not become a clean pass."""

    trace = tl.trace(
        AddMulModel(),
        torch.randn(2, 3),
        save=_save_only_mul,
        save_arg_values=True,
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert isinstance(result, ValidationReplayStatus)
    assert result.state == "unverified"
    status = trace.validation_replay_status
    assert status.unverified_reason_counts["missing_saved_parent_payload"] >= 1
    assert "not_saved_by_user" not in status.exempted_reason_counts
    computational_ops = [op for op in trace.layer_list if op.func is not None]
    assert computational_ops
    assert all(op.has_saved_args and op.saved_args is not None for op in computational_ops)


def test_not_saved_by_user_requires_exact_negative_predicate_decision() -> None:
    """Only an exact per-op negative predicate decision proves user exclusion."""

    trace = tl.trace(
        AddMulModel(),
        torch.randn(2, 3),
        save=_save_only_mul,
        save_arg_values=False,
    )
    add_op = _first_op_with_func(trace, "__add__")
    mul_op = _first_op_with_func(trace, "__mul__")

    add_proof = core._not_saved_by_user_justification(  # noqa: SLF001
        trace,
        add_op,
        "missing_saved_args",
    )

    assert add_proof is not None
    assert "predicate_save_out=False" in add_proof
    assert (
        core._not_saved_by_user_justification(  # noqa: SLF001
            trace,
            mul_op,
            "missing_saved_args",
        )
        is None
    )


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
    trace.layer_dict_all_keys[add_op.parents[0]]._internal_set("out", None)  # noqa: SLF001

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
    trace.layer_dict_all_keys[add_op.parents[0]]._internal_set("out", None)  # noqa: SLF001

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

    torch.manual_seed(108)
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


def test_buffer_semantic_ownership_invariant_fires_on_wrong_module_stack() -> None:
    """Buffer source nodes must claim the owner or an active consumer module."""

    trace = tl.trace(
        BufferOwnerModel(),
        torch.randn(3, 4),
        layers_to_save="all",
        save_arg_values=True,
    )
    buffer_op = next(layer for layer in trace.layer_list if layer.is_buffer)
    buffer_op._internal_set("modules", ["self:1"])  # noqa: SLF001
    buffer_op._internal_set("module", "self:1")  # noqa: SLF001

    with pytest.raises(MetadataInvariantError, match="buffer_xrefs"):
        check_metadata_invariants(trace)


def test_backend_neutral_graph_topology_invariant_fires_on_asymmetric_edge() -> None:
    """Non-torch graph topology should catch parent/child asymmetry."""

    trace = tl.trace(
        AddReluModel(),
        torch.randn(2, 3),
        layers_to_save="all",
        save_arg_values=True,
    )
    child = _first_op_with_func(trace, "relu")
    removed_parent = child.parents[0]
    child._internal_set("parents", [])  # noqa: SLF001

    with pytest.raises(MetadataInvariantError, match="backend_neutral_graph_topology"):
        _check_backend_neutral_graph_topology(trace)

    assert removed_parent


def test_pass_count_consistency_invariant_fires_on_op_count_mismatch() -> None:
    """Layer aggregate pass counts must agree with contained op records."""

    trace = tl.trace(
        AddReluModel(),
        torch.randn(2, 3),
        layers_to_save="all",
        save_arg_values=True,
    )
    layer = _first_op_with_func(trace, "__add__")
    layer._internal_set("num_passes", layer.num_passes + 1)  # noqa: SLF001

    with pytest.raises(MetadataInvariantError, match="pass_count_consistency"):
        _check_pass_count_consistency(trace)
