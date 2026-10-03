"""Regression tests for validation exemption hardening."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import backward as backward_validation, core
from torchlens.validation.diagnostics import get_validation_failure
from torchlens.validation.exemptions import (
    SKIP_VALIDATION_ENTIRELY,
    _binary_extrema_nonperturbed_arg_dominates,
    _check_getitem_exempt,
    _check_norm_running_stat_exempt,
    _check_scatter_exempt,
    _check_scatter_or_index_domain_exempt,
    _check_setitem_exempt,
    _check_zipped_sibling_exempt,
    _posthoc_overwrite_decision,
    _scatter_index_fully_overwrites_dim,
    perturbed_layer_at_structural_position,
)
from torchlens.validation.invariants import (
    MetadataInvariantError,
    _check_backend_neutral_graph_topology,
    _check_branching_invariants,
    _check_layer_pass_to_layer_log_xrefs,
    _check_non_torch_backward_inert,
    _check_non_torch_primitive_op_inert,
    _check_pass_count_consistency,
    _check_receptive_field_metadata_invariants,
    _check_site_key_invariants,
    check_metadata_invariants,
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    setitem_op = next(op for op in trace.layer_list if op.func_name == "__setitem__")
    destination_label = setitem_op.parent_arg_positions["args"][0]

    assert not _check_setitem_exempt(trace, setitem_op, [destination_label])


def test_posthoc_setitem_duplicate_indices_are_not_full_overwrite() -> None:
    """Posthoc overwrite logic delegates to the duplicate-index guard."""

    destination = torch.zeros(4)
    layer = _fake_layer(
        func_name="__setitem__",
        parent_arg_positions={"args": {0: "destination"}, "kwargs": {}},
    )
    decision = _posthoc_overwrite_decision(
        layer,
        ["destination"],
        (destination, torch.tensor([0, 0, 0, 0]), torch.full((4,), 9.0)),
    )
    assert not decision.exempt


def test_posthoc_scalar_partial_setitem_is_not_full_overwrite() -> None:
    """A scalar write to one cell leaves the rest of the destination live."""

    destination = torch.zeros(4, 4)
    layer = _fake_layer(
        func_name="__setitem__",
        parent_arg_positions={"args": {0: "destination"}, "kwargs": {}},
    )
    decision = _posthoc_overwrite_decision(
        layer,
        ["destination"],
        (destination, (0, 0), 5.0),
    )
    assert not decision.exempt


def test_scatter_partial_other_dimensions_are_not_full_overwrite() -> None:
    """Covering scatter dim columns cannot excuse untouched destination rows."""

    destination = torch.zeros(4, 4)
    index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    assert not _scatter_index_fully_overwrites_dim(destination, 1, index)


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
    """Model where a parent is provably annihilated by another parent.

    Integer dtype on purpose: float multiply-by-zero is now HONESTLY validated
    through the signed-zero exact tier (the product's zero signs carry the
    perturbed parent's signs, so the edge is provably live and no exemption is
    consulted). Integers have no signed zero, so the perturbed replay stays
    bit-identical and the structural annihilator proof remains the only
    legitimate rescue -- exactly the path this test pins.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply an integer cast by a zero tensor.

        Parameters
        ----------
        x:
            Floating point input tensor.

        Returns
        -------
        torch.Tensor
            Integer zero output independent of ``x``.
        """

        cast = x.to(torch.int64)
        return cast * torch.zeros_like(cast)


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

    with pytest.warns(RuntimeWarning, match="zero parameter gradients"):
        assert not backward_validation.validate_backward_pass(
            model,
            torch.randn(2, 3),
            random_seed=5,
            validate_metadata=False,
        )


def test_backward_validation_zero_param_grads_still_runs_layer_grad_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layer-grad validation runs when parameter grads are empty -- and still fails.

    The layer-grad oracle IS exercised (asserted directly by counting the
    comparison call rather than inferring it from the return value), but it
    cannot turn an unverifiable parameter-gradient census into a pass. This
    used to assert True, which directly contradicted the sibling
    ``test_backward_validation_zero_param_grads_is_not_pass`` -- the two tests
    call the same code path because ``validate_layer_grads`` defaults to True,
    so one of them was necessarily red.
    """

    model = DetachedParamModel()
    calls = 0

    from torchlens.validation import _layer_grad_report

    original_comparison = _layer_grad_report._compare_module_output_grads

    def count_original_comparison(*args: Any, **kwargs: Any) -> Any:
        """Count and delegate without recursing through the monkeypatch."""

        nonlocal calls
        calls += 1
        return original_comparison(*args, **kwargs)

    monkeypatch.setattr(
        _layer_grad_report, "_compare_module_output_grads", count_original_comparison
    )
    with pytest.warns(RuntimeWarning, match="zero parameter gradients"):
        assert not backward_validation.validate_backward_pass(
            model,
            torch.randn(2, 3),
            random_seed=5,
            validate_metadata=False,
            validate_layer_grads=True,
        )
    assert calls == 1


class _NaNLossModel(nn.Module):
    """Model whose output is entirely NaN, making every gradient NaN."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an all-NaN output.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            All-NaN tensor of the linear output's shape.
        """

        return self.lin(x) * float("nan")


class _ZeroLossModel(nn.Module):
    """Model whose output is annihilated, making every gradient exactly zero."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an all-zero output.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            All-zero tensor of the linear output's shape.
        """

        return self.lin(x) * 0.0


def test_setitem_overwrite_proof_rejects_negative_index_aliasing() -> None:
    """Negative advanced indices must be normalized before the coverage proof.

    ``dest[tensor([0, -2])] = repl`` on a length-2 dim addresses element 0
    TWICE: raw values 0 and -2 are distinct to ``torch.unique`` but alias the
    same position, so element 1 survives with its prior value while the proof
    counted a full overwrite (deephunt finding H1). The positional proof
    indexes an identity-position tensor with the saved index, so torch's own
    indexing semantics normalize negatives, slices, and masks exactly.
    """

    from torchlens.validation.exemptions import _setitem_destination_coverage_is_total

    dest = torch.tensor([10.0, 20.0])
    aliasing = (dest, torch.tensor([0, -2]), torch.tensor([1.0, 2.0]))
    assert not _setitem_destination_coverage_is_total(dest, aliasing)

    # Positive controls: genuine full single-coverage stays exempt, including
    # through negative spellings that cover distinct positions.
    plain = (dest, torch.tensor([0, 1]), torch.tensor([1.0, 2.0]))
    assert _setitem_destination_coverage_is_total(dest, plain)
    negative_full = (dest, torch.tensor([1, -2]), torch.tensor([1.0, 2.0]))
    assert _setitem_destination_coverage_is_total(dest, negative_full)


def test_index_put_overwrite_proof_rejects_negative_index_aliasing() -> None:
    """The ``index_put`` full-overwrite proof has the same negative-index hole.

    ``torch.index_put(dest, (tensor([0, -2]),), values)`` on a length-2 dim
    passes value-uniqueness and total-coverage while element 1 survives
    (deephunt finding H2).
    """

    from torchlens.validation.exemptions import _index_put_destination_is_fully_overwritten

    dest = torch.tensor([10.0, 20.0])
    aliasing_layer = _fake_layer(
        saved_args=(dest, (torch.tensor([0, -2]),), torch.tensor([1.0, 2.0])),
        saved_kwargs={},
    )
    assert not _index_put_destination_is_fully_overwritten(dest, aliasing_layer)

    plain_layer = _fake_layer(
        saved_args=(dest, (torch.tensor([0, 1]),), torch.tensor([1.0, 2.0])),
        saved_kwargs={},
    )
    assert _index_put_destination_is_fully_overwritten(dest, plain_layer)
    negative_full_layer = _fake_layer(
        saved_args=(dest, (torch.tensor([1, -2]),), torch.tensor([1.0, 2.0])),
        saved_kwargs={},
    )
    assert _index_put_destination_is_fully_overwritten(dest, negative_full_layer)


@pytest.mark.smoke
def test_equivalence_symmetry_catches_suffixed_in_module_group_corruption() -> None:
    """Symmetric group corruption on a suffixed in-module layer must FAIL.

    Per-op ``equivalence_class`` carries the module suffix appended at op
    creation, while ``trace.op_equivalence_classes`` keys are pre-suffix, so
    the key-based group lookup missed for EVERY parameterized in-module layer
    and the group-symmetry comparison silently skipped: corrupting
    ``equivalent_ops`` identically on all passes of a shared Linear (and its
    Layer, keeping the pass-agreement checks satisfied) passed the full
    invariant suite while the same corruption on an unsuffixed op was caught
    instantly (deephunt finding H4). Membership lookup restores the invariant.
    """

    class SharedLinearModel(nn.Module):
        """Model applying one Linear twice so its passes form a group."""

        def __init__(self) -> None:
            """Initialize the model."""

            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the shared linear twice.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Twice-transformed output.
            """

            return self.lin(self.lin(x))

    model = SharedLinearModel().eval()
    trace = tl.trace(model, torch.randn(2, 4))
    check_metadata_invariants(trace)

    linear_ops = [op for op in trace.layer_list if op.func_name == "linear"]
    assert len(linear_ops) == 2
    # The vacuousness precondition: the suffixed per-op key misses the
    # pre-suffix trace-dict keys, while membership still resolves the group.
    assert all(op.equivalence_class not in trace.op_equivalence_classes for op in linear_ops), (
        "suffix mismatch precondition gone -- update or retire this regression test"
    )
    expected_group = {op.label for op in linear_ops}
    assert any(group == expected_group for group in trace.op_equivalence_classes.values())

    wrong_group = {linear_ops[0].label}
    for op in linear_ops:
        op.equivalent_ops = set(wrong_group)
    layer = trace.layer_logs[linear_ops[0].layer_label]
    layer.equivalent_ops = set(wrong_group)

    with pytest.raises(MetadataInvariantError, match="equivalent_ops"):
        check_metadata_invariants(trace)


def test_wiped_grad_fn_registry_on_detached_trace_fails_invariants() -> None:
    """An empty grad-fn registry beside backward evidence must FAIL, not skip.

    The whole backward invariant family was gated on ``grad_fn_logs`` being
    non-empty, and the capture-event journal is never serialized: on a
    detached trace (pickle restore, fork, bundle load) the event-flow
    reconciliation early-returns, so wiping ``grad_fn_logs`` (with
    ``backward_pass_logs``, ``num_backward_passes``, ``grad_fn_order``, and
    dangling per-layer backpointers all left in place) silently skipped
    registry, backpointer, saved-grad, density, topology, and domain checks
    (deephunt finding H5). The gate now refuses when backward-projection
    evidence exists without a registry.
    """

    from torchlens.options import CaptureOptions

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    trace = tl.trace(
        model,
        torch.randn(2, 4),
        capture=CaptureOptions(layers_to_save="all", save_grads="all"),
    )
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    check_metadata_invariants(trace)
    assert trace.backward_pass_logs, "backward evidence precondition"
    assert trace.num_backward_passes >= 1
    assert trace.grad_fn_logs, "registry precondition"

    # The journal is a run-scoped live-capture fact: pickle restore, fork, and
    # bundle load all produce journal-less traces, which is where the
    # event-flow reconciliation cannot catch the wipe.
    trace._capture_events = None
    check_metadata_invariants(trace)

    trace.grad_fn_logs = {}
    with pytest.raises(MetadataInvariantError, match="grad_fn"):
        check_metadata_invariants(trace)


def test_forward_only_trace_keeps_empty_backward_registry_valid() -> None:
    """A trace that never ran backward legitimately has an empty registry.

    Guards the H5 refusal against over-firing: forward-time
    ``grad_fn_object_id`` stamps are NOT backward evidence, and a plain
    forward-only capture (with or without its journal) must keep passing with
    every backward field empty.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    trace = tl.trace(model, torch.randn(2, 4))
    assert not trace.grad_fn_logs
    check_metadata_invariants(trace)
    trace._capture_events = None
    check_metadata_invariants(trace)


@pytest.mark.smoke
def test_forged_placeholder_still_fails_func_call_id_after_save_load(
    tmp_path: Any,
) -> None:
    """A plain-capture placeholder must STILL fail after save/load (locked rule).

    ``op_has_genuine_replacement_evidence`` returned True unconditionally for
    ``_loaded_from_bundle`` traces (journal edit records are live-capture
    facts and never serialize), so a placeholder that FAILED the
    ``func_call_id_consistency`` invariant live PASSED it after a save/load
    round trip -- a laundering channel that contradicted the locked
    2026-06-02 rule (deephunt finding M3). The save path now stamps the live
    corroboration verdict into persisted op annotations and the loaded arm
    requires it.
    """

    from torchlens.validation.invariants import check_func_call_id_invariant

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    trace = tl.trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    relu_op = next(op for op in trace.layer_list if op.func_name == "relu")
    object.__setattr__(relu_op, "func", None)
    object.__setattr__(relu_op, "func_name", "intervention_replacement")
    object.__setattr__(relu_op, "intervention_replaced", True)
    object.__setattr__(relu_op, "func_call_id", None)

    with pytest.raises(MetadataInvariantError, match="func_call_id"):
        check_func_call_id_invariant(trace)

    path = str(tmp_path / "forged.tlspec")
    tl.save(trace, path)
    loaded = tl.load(path)
    assert getattr(loaded, "_loaded_from_bundle", False) is True
    with pytest.raises(MetadataInvariantError, match="func_call_id"):
        check_func_call_id_invariant(loaded)


def test_genuine_intervention_replacement_survives_save_load(tmp_path: Any) -> None:
    """A genuinely corroborated replacement keeps its exemption after load.

    Guards the M3 fail-closed change against over-firing: the save-time stamp
    carries the live verdict, so a live-fire intervention capture (hook-minted
    ``replaced=True`` FireRecord plus armed spec) still passes the
    ``func_call_id_consistency`` invariant after a save/load round trip.
    """

    from torchlens.validation.invariants import check_func_call_id_invariant

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(
        model,
        torch.randn(3, 4),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    replaced_ops = [op for op in trace.layer_list if getattr(op, "intervention_replaced", False)]
    assert replaced_ops, "zero_ablate must stamp its site"
    check_func_call_id_invariant(trace)

    path = str(tmp_path / "genuine.tlspec")
    tl.save(trace, path)
    loaded = tl.load(path)
    assert getattr(loaded, "_loaded_from_bundle", False) is True
    check_func_call_id_invariant(loaded)
    loaded_replaced = [
        op for op in loaded.layer_list if getattr(op, "intervention_replaced", False)
    ]
    assert loaded_replaced
    for op in loaded_replaced:
        stamp = (getattr(op, "annotations", None) or {}).get("replacement_evidence_v1")
        assert isinstance(stamp, dict) and stamp.get("corroborated") is True


def test_to_tensor_template_exemption_refuses_perturbed_data_source() -> None:
    """``to(other)`` only excuses the TEMPLATE parent, never the data SOURCE.

    The posthoc blanket fired on func name + template-tensor presence without
    consulting ``layers_to_perturb``: perturbing the SOURCE (``args[0]``) of a
    ``to(other)`` cast and seeing an unchanged output is exactly a
    dropped-substitution capture bug, yet it was excused as
    ``type_template_output`` (deephunt finding H3). The template parent
    (``args[1]`` / ``other=``) is genuinely structural -- only its
    dtype/device flow into the output -- and keeps the exemption, mirroring
    the F2 ``*_like`` template-slot tightening.
    """

    from torchlens.validation.exemptions import _posthoc_structural_output_decision

    layer = _fake_layer(
        func_name="to",
        saved_args=(torch.randn(2, 2), torch.zeros(2, 2, dtype=torch.float64)),
        saved_kwargs={},
        parent_arg_positions={
            "args": {0: "source_parent", 1: "template_parent"},
            "kwargs": {},
        },
        out=torch.randn(2, 2).to(torch.float64),
        dtype=torch.float64,
    )

    source_decision = _posthoc_structural_output_decision(
        layer, layer.saved_args, ["source_parent"]
    )
    assert not source_decision.exempt

    template_decision = _posthoc_structural_output_decision(
        layer, layer.saved_args, ["template_parent"]
    )
    assert template_decision.exempt
    assert template_decision.reason == "type_template_output"

    # Missing position metadata fails closed, like the *_like template proof.
    unmapped = _fake_layer(
        func_name="to",
        saved_args=layer.saved_args,
        saved_kwargs={},
        parent_arg_positions={"args": {}, "kwargs": {}},
        out=layer.out,
        dtype=torch.float64,
    )
    assert not _posthoc_structural_output_decision(
        unmapped, unmapped.saved_args, ["source_parent"]
    ).exempt


def test_func_name_none_string_cannot_launder_missing_func_call_id() -> None:
    """The literal ``"none"`` func_name must not exempt a computational op.

    ``_is_func_call_id_exempt`` ended with a bare string set
    (``{"input","output","buffer","none"}``) with no cross-check against the
    ``is_input``/``is_output``/``is_buffer``/``is_internal_source`` flags, so
    a traced relu whose ``func_call_id`` was nulled and ``func_name`` rewritten
    to ``"none"`` (func still callable, no special flags) passed the full
    invariant suite (deephunt finding M2). The sentinel is only legitimate on
    flagged bookkeeping ops (outputs) and functionless internal sources.
    """

    from torchlens.validation.invariants import check_func_call_id_invariant

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    trace = tl.trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    check_metadata_invariants(trace)

    relu_op = next(op for op in trace.layer_list if op.func_name == "relu")
    object.__setattr__(relu_op, "func_call_id", None)
    object.__setattr__(relu_op, "func_name", "none")
    assert callable(relu_op.func)
    assert not relu_op.is_internal_source

    with pytest.raises(MetadataInvariantError):
        check_func_call_id_invariant(trace)
    with pytest.raises(MetadataInvariantError):
        check_metadata_invariants(trace)


class _BareBernoulliModel(nn.Module):
    """Model drawing an in-place bernoulli mask from computed probabilities."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw a mask from sigmoid probabilities and scale it.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scaled drawn mask.
        """

        mask = torch.sigmoid(x).bernoulli_()
        return mask * 2.0


class _ExplicitPBernoulliModel(nn.Module):
    """Model drawing into a destination with an explicit probability tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overwrite a scratch destination with draws from sigmoid(x).

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scaled drawn mask.
        """

        probabilities = torch.sigmoid(x)
        destination = x * 0.5
        mask = destination.bernoulli_(probabilities)
        return mask * 2.0


def test_bernoulli_parent_replay_exemption_requires_snapshot_proof() -> None:
    """A replay mismatch under a bernoulli_ parent needs a snapshot proof.

    ANY replay mismatch on an op with ANY ``bernoulli_`` parent was exempted
    wholesale -- including a mismatch caused by a corrupted recorded func on
    the CHILD (deephunt finding M1): swapping the recorded func of
    ``mask * 2.0`` to ``torch.add`` flipped the decision from
    ``failed:replay_mismatch`` to ``exempted:parent_inplace_rng_bernoulli``.
    The exemption now requires re-feeding the child's own saved-arg snapshots
    at the bernoulli-parent slots to reproduce the saved output; a corrupted
    child func cannot pass that proof.
    """

    torch.manual_seed(0)
    trace = tl.trace(
        _BareBernoulliModel().eval(),
        torch.randn(4, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    mul_op = next(op for op in trace.layer_list if op.func_name == "__mul__")
    bernoulli_parents = [
        parent for parent in mul_op.parents if trace[parent].func_name == "bernoulli_"
    ]
    assert bernoulli_parents, "mul must be a child of the in-place bernoulli"

    mul_op.func = torch.add
    result = core._check_whether_func_on_saved_parents_yields_saved_tensor(  # noqa: SLF001
        trace, mul_op.label, perturb=False
    )
    assert result.decision == "failed"


def test_bernoulli_arg_logging_case2_requires_binary_draw_shape() -> None:
    """Case 2 of arg logging only excuses a genuine in-place re-draw shape.

    The companion blanket validated ANY parent-logged-but-value-mismatched
    arg whenever the parent was named ``bernoulli_`` (deephunt M1). The
    legitimate mutation shape is an in-place RE-DRAW: both the child's
    snapshot and the parent's current out are same-shape, same-dtype 0/1
    draws. Arbitrary corrupted values must fall through to the Case 3
    failure.
    """

    corrupted_parent = _fake_layer(
        layer_label="bernoulli__1_1",
        label="bernoulli__1_1:1",
        func_name="bernoulli_",
        out=torch.tensor([0.3, 0.7]),
        out_versions_by_child={},
    )
    child = _fake_layer(
        layer_label="mul_1_2",
        label="mul_1_2:1",
        parent_arg_positions={"args": {0: "bernoulli__1_1"}, "kwargs": {}},
        parents=["bernoulli__1_1"],
    )
    trace = {"bernoulli__1_1": corrupted_parent}
    result = core._check_arglocs_correct_for_arg(  # noqa: SLF001
        trace,  # type: ignore[arg-type]
        child,
        corrupted_parent,
        "args",
        0,
        torch.tensor([9.0, 9.0]),
    )
    assert result.decision == "failed"

    genuine_parent = _fake_layer(
        layer_label="bernoulli__1_1",
        label="bernoulli__1_1:1",
        func_name="bernoulli_",
        out=torch.tensor([0.0, 1.0]),
        out_versions_by_child={},
    )
    trace = {"bernoulli__1_1": genuine_parent}
    result = core._check_arglocs_correct_for_arg(  # noqa: SLF001
        trace,  # type: ignore[arg-type]
        child,
        genuine_parent,
        "args",
        0,
        torch.tensor([1.0, 0.0]),
    )
    assert result.decision == "validated"


class _OutOfPlaceBernoulliModel(nn.Module):
    """Model with a genuine values-as-probabilities bernoulli edge."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw a mask from sigmoid probabilities out of place.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scaled drawn mask.
        """

        return torch.bernoulli(torch.sigmoid(x)) * 2.0


def test_bernoulli_models_pass_forward_validation() -> None:
    """Real bernoulli models must validate end-to-end (L17 false-FAIL fix).

    Any model containing a bernoulli draw failed forward validation at the
    bernoulli op's own parent edge with ``perturbation_insensitive``
    (deephunt L17). Two distinct root causes, both fixed:

    - A genuine probability edge (out-of-place ``torch.bernoulli(x)``, or
      ``.bernoulli_(p)``'s ``p``) replays identical samples under restored
      RNG for small in-domain perturbations. The bernoulli-aware perturbation
      now forces complement-of-saved-draw probabilities (the deterministic
      extremes), which provably flip every drawn element when the edge is
      live -- a genuinely dropped edge still replays unchanged and fails.
    - ``bernoulli_``'s destination edge carries NO values at all: the bare
      form fills with Bernoulli(0.5) draws IGNORING self's values (verified
      empirically -- ``zeros.bernoulli_()`` produces ones), and the explicit
      form overwrites with Bernoulli(p) draws. That edge is a pure
      shape/dtype/device template and earns the posthoc template exemption.
    """

    from torchlens.validation import validate_forward_pass

    torch.manual_seed(0)
    assert validate_forward_pass(_BareBernoulliModel().eval(), torch.randn(4, 4)) is True
    torch.manual_seed(0)
    assert validate_forward_pass(_ExplicitPBernoulliModel().eval(), torch.randn(4, 4)) is True
    torch.manual_seed(0)
    assert validate_forward_pass(_OutOfPlaceBernoulliModel().eval(), torch.randn(4, 4)) is True


def test_backward_validation_all_nan_grads_is_not_pass() -> None:
    """An all-NaN stock gradient census must be unverifiable, never PASS.

    A NaN loss makes every stock AND candidate gradient all-NaN, so every
    ``equal_nan=True`` comparison passes vacuously with ZERO numeric detection
    power -- the exact degenerate-evidence shape the ABORTED_NONFINITE doctrine
    refuses elsewhere. Before the degenerate-evidence guard this reported
    ``True`` (deephunt finding H6).
    """

    model = _NaNLossModel().eval()

    with pytest.warns(RuntimeWarning, match="degenerate"):
        assert not backward_validation.validate_backward_pass(
            model,
            torch.randn(2, 4),
            random_seed=7,
            validate_metadata=False,
        )


def test_backward_validation_all_zero_grads_is_not_pass() -> None:
    """An all-zero stock gradient census must be unverifiable, never PASS.

    A ``* 0.0`` loss zeroes every gradient buffer, so a capture that filled its
    gradient records with zeros -- a classic bug shape -- is indistinguishable
    from a correct one; the comparison has zero detection power (deephunt
    finding M12, companion to H6).
    """

    model = _ZeroLossModel().eval()

    with pytest.warns(RuntimeWarning, match="degenerate"):
        assert not backward_validation.validate_backward_pass(
            model,
            torch.randn(2, 4),
            random_seed=7,
            validate_metadata=False,
        )


def test_backward_validation_mixed_nan_and_zero_grads_is_not_pass() -> None:
    """A MIXED all-NaN + all-zero census must be unverifiable, never PASS.

    The two degeneracy arms used to be tracked as independent whole-census
    totals, so one all-NaN gradient killed the all-zero arm and one all-zero
    gradient killed the all-nonfinite arm -- a census with ZERO finite-nonzero
    elements (no detection power at all) returned ``None`` and validation
    passed vacuously (b4-fable round-2 probe; defect in the a2680381 guard).
    The decisive predicate is per-element finite-AND-nonzero existence.
    """

    class MixedDegenerateModel(nn.Module):
        """Model whose census is one all-NaN grad plus one all-zero grad."""

        def __init__(self) -> None:
            """Initialize the two degenerate-gradient parameters."""

            super().__init__()
            self.nan_param = nn.Parameter(torch.tensor(1.0))
            self.zero_param = nn.Parameter(torch.tensor(1.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return ``x`` plus a NaN-grad term and a zero-grad term.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Input with one NaN-gradient and one zero-gradient
                contribution.
            """

            nan_term = (self.nan_param * 0.0) * torch.tensor(float("inf"))
            zero_term = (self.zero_param * x.sum()) * 0.0
            return x + nan_term + zero_term

    model = MixedDegenerateModel().eval()

    with pytest.warns(RuntimeWarning, match="degenerate"):
        assert not backward_validation.validate_backward_pass(
            model,
            torch.randn(2, 4),
            random_seed=7,
            validate_metadata=False,
            # The bare two-parameter module has no submodule outputs, so the
            # layer-grad report is empty; disable it to reach the parameter
            # census guard under test.
            validate_layer_grads=False,
        )


def test_stock_param_grad_degeneracy_mixed_census_is_degenerate() -> None:
    """Unit pin: the mixed census classifies degenerate, not real-power."""

    census = {
        "a": torch.full((3,), float("nan")),
        "b": torch.zeros(3),
    }
    verdict = backward_validation._stock_param_grad_degeneracy(census)
    assert verdict == "mixed-nonfinite-zero"

    # Any finite nonzero element anywhere restores detection power.
    census["c"] = torch.tensor([0.0, 1e-30, 0.0])
    assert backward_validation._stock_param_grad_degeneracy(census) is None


def test_backward_validation_partial_nan_grads_still_pass() -> None:
    """A PARTIALLY NaN census keeps its detection power and still passes.

    The degenerate-evidence guard is TOTAL-degeneracy only: finite nonzero
    gradients elsewhere in the census retain real comparison power, and the
    NaN-pattern-agreement doctrine (``equal_nan=True``) continues to govern the
    NaN positions. Guards the guard against over-firing (the L15 pressure that
    historically breeds disarm-style exemptions).
    """

    class PartialNaNModel(nn.Module):
        """Model with one NaN-gradient parameter and one healthy linear."""

        def __init__(self) -> None:
            """Initialize the model."""

            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the linear output plus a NaN-gradient term.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Linear output with a NaN contribution on ``scale`` only.
            """

            nan_term = (self.scale * 0.0) * torch.tensor(float("inf"))
            return self.lin(x) + nan_term

    model = PartialNaNModel().eval()
    assert (
        backward_validation.validate_backward_pass(
            model,
            torch.randn(2, 4),
            random_seed=7,
            validate_metadata=False,
        )
        is True
    )


def test_one_hot_index_perturbation_uses_valid_alternate_class() -> None:
    """One-hot index validation should perturb within ``num_classes``."""

    trace = tl.trace(
        OneHotModel(),
        torch.tensor([1]),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )

    assert trace.validate_forward_pass([torch.tensor([[0.0, 1.0, 0.0, 0.0]])])


def test_swamped_fp32_add_uses_ulp_predicate() -> None:
    """Swamped fp32 add should pass only through the ULP spacing predicate."""

    x = torch.tensor([10000.0, 10001.0], dtype=torch.float32)
    trace = tl.trace(
        SwampedAddModel(),
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
            capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
            capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
            capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
            capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
        )

        result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

        assert result is True
        assert trace.validation_replay_status.failed_node_count == 0


def test_corrupted_swamped_add_replay_fails_without_reexecuting_diagnostic_probes() -> None:
    """A constant replay callable fails without extra diagnostic executions."""

    trace = tl.trace(
        SwampedAddModel(),
        torch.tensor([10000.0, 10001.0], dtype=torch.float32),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
    assert set(failure.extra) == {"perturbed_parents"}
    assert "generic_invariant_probe_matched" not in failure.extra


def test_multiplicative_zero_annihilator_uses_structural_proof() -> None:
    """Multiplication by a saved zero operand should use a structural proof."""

    trace = tl.trace(
        MultiplyByZeroModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    assert trace.validation_replay_status.failed_node_count == 0


def test_structural_arg_exemption_covers_keyword_parent_positions() -> None:
    """Keyword-spelled index-class parents are sensitivity-VERIFIED, not exempted.

    F2 tightening: the ``cross_entropy`` target blanket was removed from the
    structural registries, so a keyword-spelled target edge must now be
    exercised by the in-domain rotation (which reads kwarg parent-position
    metadata) and register genuine sensitivity instead of a blanket
    ``pre_perturbation_exemption``.
    """

    logits = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.4, 0.7]], dtype=torch.float32)
    target = torch.tensor([0, 2], dtype=torch.long)
    trace = tl.trace(
        CrossEntropyKwargModel(),
        logits,
        input_kwargs={"target": target},
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    cross_entropy_op = next(op for op in trace.layer_list if op.func_name == "cross_entropy")
    assert cross_entropy_op.parent_arg_positions["kwargs"]["target"]
    perturbation_decisions = [
        decision
        for decision in trace.validation_replay_status.decisions
        if decision.get("op_label") == cross_entropy_op.label
        and decision.get("phase") == "perturbation"
    ]
    # Both keyword-spelled parent edges (input= logits and target= labels) are
    # perturbed and register real sensitivity -- no blanket exemption remains.
    assert len(perturbation_decisions) >= 2, trace.validation_replay_status.decisions
    assert all(
        decision["decision"] == "validated" and decision["reason"] == "perturbation_changed"
        for decision in perturbation_decisions
    ), perturbation_decisions


def test_skip_validation_registry_entries_have_justifications() -> None:
    """Every uninitialized-memory replay exemption must carry a proof string."""

    assert SKIP_VALIDATION_ENTIRELY
    assert all(justification for justification in SKIP_VALIDATION_ENTIRELY.values())


def test_empty_like_is_justified_exempted_not_unverified() -> None:
    """Uninitialized-memory ops should pass as justified design exemptions."""

    trace = tl.trace(
        EmptyLikeModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=False),
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
        capture=tl.options.CaptureOptions(save_arg_values=True),
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
        capture=tl.options.CaptureOptions(save_arg_values=False),
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
    trace = tl.trace(
        model, x, save=_save_only_mul, capture=tl.options.CaptureOptions(save_arg_values=True)
    )
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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


class StepInvalidNarrowModel(nn.Module):
    """Model whose control parent is invalid under EVERY perturbation.

    ``narrow(0, start, x.shape[0])`` with ``start == 0`` admits NO other valid
    start value: the wide random draw, ``step_up`` (+1), and ``step_down`` (-1,
    which wraps to ``size - 1``) all overrun the dimension and raise -- on any
    seed, by construction. r29 MED: the r28 step-retry made the former
    ``CholeskyModel`` vehicle soundly ``validated`` (a +-1-ULP step keeps the
    input positive-definite and changes the output), silently emptying the
    ``perturbation_execution_exception`` tripwire category across every armed
    vehicle; this model keeps the category reachable, retry included. The
    start index is a model INPUT (input ops carry no perturbation check of
    their own), so no upstream op can add a seed-dependent side decision.
    """

    def forward(self, x: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        """Narrow the full length of ``x`` from a traced zero start index.

        Parameters
        ----------
        x:
            One-dimensional input.
        start:
            Zero-dimensional long start index; must be 0.

        Returns
        -------
        torch.Tensor
            The narrowed (full-length) input, scaled.
        """

        return x.narrow(0, start, x.shape[0]) * 1.0


def test_perturbation_exception_yields_reason_coded_unverified() -> None:
    """Invalid perturbed inputs should be unverified rather than exempted."""

    trace = tl.trace(
        StepInvalidNarrowModel(),
        [torch.tensor([-2.0, 0.5, 1.5, 2.5]), torch.tensor(0)],
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )

    torch.manual_seed(108)
    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert isinstance(result, ValidationReplayStatus)
    assert result.state == "unverified"
    assert result.unverified_reason_counts["perturbation_execution_exception"] >= 1


def test_step_retry_soundly_validates_domain_constrained_perturbation() -> None:
    """The r28 step retry converts the old cholesky vehicle into evidence.

    A wide random perturbation of the ``cholesky`` parent leaves the
    positive-definite domain and raises; the +-1-ULP step retry stays inside it
    and CHANGES the output, so the edge is now proven real
    (``validated``/``perturbation_changed``) instead of reason-coded
    ``unverified``. Pinned so the retry benefit cannot silently regress; the
    ``perturbation_execution_exception`` category itself stays armed through
    :class:`StepInvalidNarrowModel` above.
    """

    trace = tl.trace(
        CholeskyModel(),
        torch.eye(3).unsqueeze(0) * 2,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )

    torch.manual_seed(108)
    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    status = trace.validation_replay_status
    assert status.state == "passed"
    assert not status.unverified_reason_counts


def test_fully_saved_vanilla_model_has_zero_unverified_decisions() -> None:
    """Healthy full-save traces should not produce unverified decisions."""

    trace = tl.trace(
        AddReluModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )

    result = trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    assert result is True
    status = trace.validation_replay_status
    assert status.state == "passed"
    assert status.unverified_node_count == 0


def test_validation_status_cache_invalidated_after_same_shape_rerun() -> None:
    """Rerunning a trace should clear cached replay-validation status."""

    model = AddReluModel()
    trace = tl.trace(
        model,
        torch.ones(2, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    trace.validate_forward_pass([_first_output(trace)], validate_metadata=False)

    fork = trace.fork("status_check")

    assert fork.validation_replay_status.state == "available"


def test_buffer_semantic_ownership_invariant_fires_on_wrong_module_stack() -> None:
    """Buffer source nodes must claim the owner or an active consumer module.

    Two redundant tripwires cover this plant: op_log_fields'
    module_call_stack<->modules coherence check (runs first in the contract
    order, and the plant rewrites modules without the stack) and
    buffer_xrefs' semantic-ownership check. The plant must refuse either
    way — the test pins the refusal and the planted module claim, not which
    redundant layer wins the race.
    """

    trace = tl.trace(
        BufferOwnerModel(),
        torch.randn(3, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    buffer_op = next(layer for layer in trace.layer_list if layer.is_buffer)
    buffer_op._internal_set("modules", ["self:1"])  # noqa: SLF001
    buffer_op._internal_set("module", "self:1")  # noqa: SLF001

    with pytest.raises(MetadataInvariantError, match="module"):
        check_metadata_invariants(trace)


def test_backend_neutral_graph_topology_invariant_fires_on_asymmetric_edge() -> None:
    """Non-torch graph topology should catch parent/child asymmetry."""

    trace = tl.trace(
        AddReluModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    layer = _first_op_with_func(trace, "__add__")
    layer._internal_set("num_passes", layer.num_passes + 1)  # noqa: SLF001

    with pytest.raises(MetadataInvariantError, match="pass_count_consistency"):
        _check_pass_count_consistency(trace)


# ---------------------------------------------------------------------------
# Mutation-margin arming: registry survivors W4 (killing tests, not CI repair;
# each kills the whole-function "return None" disarm mutant_driver.py plants
# on the named METADATA_INVARIANT_CONTRACTS entry). Each check is a duck-typed
# pure function of its argument, so a synthetic SimpleNamespace fake proves
# the real violation path without needing a real capture to coincidentally
# reach it.
# ---------------------------------------------------------------------------


def test_branching_invariant_fires_on_stale_is_branching_flag() -> None:
    """A trace claiming ``is_branching`` with no multi-child layer must raise."""

    fake_trace = SimpleNamespace(
        is_branching=True,
        layer_list=[SimpleNamespace(children=[]), SimpleNamespace(children=["only_child"])],
    )

    with pytest.raises(MetadataInvariantError, match="branching_invariants"):
        _check_branching_invariants(fake_trace)  # type: ignore[arg-type]


def test_layer_pass_layer_log_xrefs_fires_on_label_key_mismatch() -> None:
    """A layer_logs entry whose key disagrees with its own label must raise."""

    mismatched_layer = SimpleNamespace(
        layer_label="real_label_1_1",
        num_passes=1,
        ops={1: SimpleNamespace(pass_index=1, layer_label="real_label_1_1")},
    )
    fake_trace = SimpleNamespace(layer_logs={"stale_key_1_1": mismatched_layer})

    with pytest.raises(MetadataInvariantError, match="layer_pass_layer_log_xrefs"):
        _check_layer_pass_to_layer_log_xrefs(fake_trace)  # type: ignore[arg-type]


def test_layer_pass_layer_log_xrefs_fires_on_ops_key_mismatch() -> None:
    """Arm 1: a layer's ``ops`` keys must match ``range(1, num_passes + 1)``.

    (M1 raise-arm campaign: ``layer_pass_layer_log_xrefs#a01`` survivor --
    the label-mismatch killer above trips arm 0 first, leaving this arm
    unexercised.)
    """

    matching_label = "real_label_1_1"
    layer_with_wrong_keys = SimpleNamespace(
        layer_label=matching_label,
        num_passes=1,
        ops={2: SimpleNamespace(pass_index=2, layer_label=matching_label)},
    )
    fake_trace = SimpleNamespace(layer_logs={matching_label: layer_with_wrong_keys})

    with pytest.raises(MetadataInvariantError, match="layer_pass_layer_log_xrefs"):
        _check_layer_pass_to_layer_log_xrefs(fake_trace)  # type: ignore[arg-type]


def test_pass_count_consistency_fires_on_ops_key_mismatch() -> None:
    """Arm 0: a layer's ``ops`` keys must match ``range(1, num_passes + 1)``.

    (M1 raise-arm campaign: ``pass_count_consistency#a00`` survivor. The
    real-capture killer bumping ``layer.num_passes`` on an Op pulled from
    ``trace.layer_list`` does NOT reach the ``Layer`` record this checker
    actually reads from ``trace.layer_logs`` -- empirically proven: it kept
    this arm a SURVIVOR. A duck-typed fake of the exact object the checker
    consumes avoids that real-capture aliasing gap.
    """

    fake_trace = SimpleNamespace(
        layer_logs={
            "layer_1_1": SimpleNamespace(
                ops={1: SimpleNamespace(pass_index=1, num_passes=2)},
                num_passes=2,
            ),
        },
    )

    with pytest.raises(MetadataInvariantError, match="pass_count_consistency"):
        _check_pass_count_consistency(fake_trace)  # type: ignore[arg-type]


def test_non_torch_backward_inert_fires_on_populated_backward_flag() -> None:
    """A non-torch trace declaring ``has_backward_pass`` must raise."""

    fake_trace = SimpleNamespace(has_backward_pass=True)

    with pytest.raises(MetadataInvariantError, match="non_torch_backward_inert"):
        _check_non_torch_backward_inert(fake_trace)  # type: ignore[arg-type]


def test_non_torch_backward_inert_fires_on_populated_grad_fn_logs() -> None:
    """Arm 1: ``grad_fn_logs`` alone (no other backward field) must raise.

    Mutation-margin arming (M1 raise-arm campaign, run 36309580288):
    ``non_torch_backward_inert#a01`` survived because the whole-function
    killer above only ever trips the first ``has_backward_pass`` arm; every
    later arm needs its own scenario where every EARLIER arm stays silent.
    """

    fake_trace = SimpleNamespace(has_backward_pass=False, grad_fn_logs={"relu_1_1": object()})

    with pytest.raises(MetadataInvariantError, match="non_torch_backward_inert"):
        _check_non_torch_backward_inert(fake_trace)  # type: ignore[arg-type]


def test_non_torch_backward_inert_fires_on_nonzero_backward_pass_count() -> None:
    """Arm 5 (last): a nonzero ``num_backward_passes`` alone must raise.

    (M1 raise-arm campaign: ``non_torch_backward_inert#a05`` survivor.)
    """

    fake_trace = SimpleNamespace(
        has_backward_pass=False,
        grad_fn_logs=None,
        grad_fn_order=None,
        backward_pass_logs=None,
        backward_root_grad_fn_object_ids=None,
        num_backward_passes=2,
    )

    with pytest.raises(MetadataInvariantError, match="non_torch_backward_inert"):
        _check_non_torch_backward_inert(fake_trace)  # type: ignore[arg-type]


def test_non_torch_primitive_op_inert_fires_on_populated_profile() -> None:
    """A non-torch trace carrying a primitive-op profile must raise."""

    fake_trace = SimpleNamespace(_primitive_op_profile=object())

    with pytest.raises(MetadataInvariantError, match="non_torch_primitive_op_inert"):
        _check_non_torch_primitive_op_inert(fake_trace)  # type: ignore[arg-type]


def test_site_key_invariants_fires_on_malformed_key() -> None:
    """A retained op's non-``None`` site_key must be well-formed and prefixed."""

    fake_trace = SimpleNamespace(
        layer_list=[SimpleNamespace(site_key="not-a-real-site-key", label="op_1_1")],
    )

    with pytest.raises(MetadataInvariantError, match="site_key_invariants"):
        _check_site_key_invariants(fake_trace)  # type: ignore[arg-type]


def test_receptive_field_metadata_invariant_wraps_geometry_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A geometric-contract violation must surface as MetadataInvariantError.

    The registry entry's own job (independent of the real RF geometry math,
    which is exercised elsewhere) is the ReceptiveFieldError ->
    MetadataInvariantError translation; a whole-function ``return None``
    mutant skips the call entirely, so forcing the inner checker to raise
    proves the translation still runs.
    """

    from torchlens.receptive_field._errors import ReceptiveFieldError

    def _raise_geometry_violation(trace: Any) -> None:
        del trace
        raise ReceptiveFieldError("synthetic geometry violation")

    monkeypatch.setattr(
        "torchlens.receptive_field._validation.check_geometric_metadata_invariants",
        _raise_geometry_violation,
    )

    with pytest.raises(MetadataInvariantError, match="synthetic geometry violation"):
        _check_receptive_field_metadata_invariants(SimpleNamespace())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Mutation-margin arming: exempt survivors W3 (X11, X12, X14 -- the
# ``return True`` exempt-everything disarm). X02/X05/X06/X07 already have
# direct killers (test_bug_fixes_phase14.py, and the scatter test above);
# these three custom exemption gates had none.
# ---------------------------------------------------------------------------


def test_norm_running_stat_exemption_uses_position_and_training_flag() -> None:
    """Only running_mean/running_var in TRAINING mode are exempt."""

    layer = _fake_layer(
        saved_args=(
            torch.zeros(4),  # 0: input
            torch.ones(4),  # 1: weight
            torch.zeros(4),  # 2: bias
            torch.zeros(4),  # 3: running_mean
            torch.ones(4),  # 4: running_var
            True,  # 5: training
        ),
        parent_arg_positions={
            "args": {
                0: "input",
                1: "weight",
                2: "bias",
                3: "running_mean",
                4: "running_var",
            },
            "kwargs": {},
        },
    )

    assert _check_norm_running_stat_exempt(None, layer, ["running_mean"])  # type: ignore[arg-type]
    assert not _check_norm_running_stat_exempt(None, layer, ["input"])  # type: ignore[arg-type]

    eval_layer = _fake_layer(
        saved_args=layer.saved_args[:5] + (False,),
        parent_arg_positions=layer.parent_arg_positions,
    )
    assert not _check_norm_running_stat_exempt(  # type: ignore[arg-type]
        None, eval_layer, ["running_mean"]
    )


def test_scatter_or_index_domain_exemption_uses_destination_position() -> None:
    """The combined scatter/index-domain gate must stay strict for the src parent.

    Mirrors ``test_scatter_exemption_uses_destination_position_not_equal_value``
    but calls the wrapping gate X12 actually plants on, which a bare
    ``_check_scatter_exempt`` killer cannot reach.
    """

    class EqualValuedParentTrace:
        """Minimal trace resolving parent outputs by label."""

        def __getitem__(self, label: str) -> Any:
            del label
            return _fake_layer(out=torch.zeros(3))

    layer = _fake_layer(
        saved_args=(torch.zeros(3), 0, torch.arange(3), torch.zeros(3)),
        saved_kwargs={},
        parent_arg_positions={"args": {0: "dest_parent", 3: "src_parent"}, "kwargs": {}},
    )
    trace = EqualValuedParentTrace()

    assert not _check_scatter_or_index_domain_exempt(  # type: ignore[arg-type]
        trace, layer, ["src_parent"]
    )
    assert _check_scatter_or_index_domain_exempt(  # type: ignore[arg-type]
        trace, layer, ["dest_parent"]
    )


def test_zipped_sibling_exemption_stays_strict_on_own_output_index() -> None:
    """A perturbed parent at this output's OWN zipped index must stay strict."""

    layer = _fake_layer(
        multi_output_index=0,
        parent_arg_positions={"args": {0: "input_0", 1: "input_1"}, "kwargs": {}},
    )

    assert _check_zipped_sibling_exempt(None, layer, ["input_1"])  # type: ignore[arg-type]
    assert not _check_zipped_sibling_exempt(None, layer, ["input_0"])  # type: ignore[arg-type]
