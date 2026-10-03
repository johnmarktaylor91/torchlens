"""Regression tests for Phase 14 bug fixes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens.capture.arg_positions import FUNC_ARG_SPECS, extract_tensors_and_params
from torchlens.data_classes.func_call_location import FuncCallLocation
from torchlens.options import SaveOptions
from torchlens.utils.hashing import make_short_barcode_from_input
from torchlens.utils.tensor_utils import tensor_nanequal
from torchlens.validation import (
    MetadataInvariantError,
    check_metadata_invariants,
    validate_forward_pass,
)
from torchlens.validation.core import _check_arglocs_correct_for_arg
from torchlens.validation.exemptions import _check_interpolate_exempt, _check_lstm_exempt
from torchlens.visualization._render_common import GRADIENT_ARROW_COLOR


def _sample_func_for_location(x: torch.Tensor) -> torch.Tensor:
    """Return ``x`` unchanged for call-location metadata tests.

    Parameters
    ----------
    x:
        Input tensor.

    Returns
    -------
    torch.Tensor
        The original tensor.
    """

    return x


class _AlternatingBranchBlock(nn.Module):
    """Shared block whose branch arm alternates across recurrent ops."""

    def __init__(self) -> None:
        """Initialize the shared linear layer."""

        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, call_index: int) -> torch.Tensor:
        """Run one alternating branch step.

        Parameters
        ----------
        x:
            Input tensor.
        call_index:
            One-indexed recurrent pass number.

        Returns
        -------
        torch.Tensor
            Shared linear output.
        """

        branch_marker = (x.mean() * 0) + (1.0 if call_index % 2 == 1 else -1.0)
        if branch_marker > 0:
            y = self.shared(x)
        else:
            y = self.shared(x)
        return y


class _AlternatingRecurrentModel(nn.Module):
    """Recurrent conditional model used for rolled aggregate regressions."""

    def __init__(self) -> None:
        """Initialize the recurrent block."""

        super().__init__()
        self.block = _AlternatingBranchBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run four recurrent ops.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final recurrent output.
        """

        for call_index in range(1, 5):
            x = self.block(x, call_index)
        return x


class _MutatingErrorModel(nn.Module):
    """Model that mutates state before raising during validation."""

    def __init__(self) -> None:
        """Initialize the mutable buffer."""

        super().__init__()
        self.register_buffer("counter", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate state and raise.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Unreachable output.
        """

        self.counter.add_(1)
        raise RuntimeError("intentional validation failure")


class _IdentityOutputModel(nn.Module):
    """Model that returns its input directly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The original tensor.
        """

        return x


class _ViewMutationOutputTransformModel(nn.Module):
    """Return a base tensor after mutating one of its views."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate a two-row view and return its six-row base tensor."""

        base = x + 1
        base[1:3].zero_()
        return base


class _NewEmptyFilledModel(nn.Module):
    """Model with deterministic output after an uninitialized allocation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Allocate with ``new_empty`` and overwrite every element.

        Parameters
        ----------
        x:
            Tensor providing shape, dtype, and device.

        Returns
        -------
        torch.Tensor
            Fully initialized tensor derived from ``new_empty``.
        """

        out = x.new_empty(x.shape)
        return out.fill_(3.0)


class _WhereEqualBranchesModel(nn.Module):
    """Model whose ``where`` condition selects between equal tensors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``where`` with identical true and false branches.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The branch tensor selected by an irrelevant condition.
        """

        branch = x + 1.0
        condition = x[:, :1] > 0
        return torch.where(condition, branch, branch)


class _FunctionalParameterViewModel(nn.Module):
    """Model using a differentiable tensor view as a functional weight."""

    def __init__(self) -> None:
        """Initialize one registered parameter."""

        super().__init__()
        self.weight = nn.Parameter(torch.eye(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``F.linear`` with a tensor view of a registered parameter.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Functional linear output.
        """

        return F.linear(x, self.weight.t())


class _ResidualRecurrentModel(nn.Module):
    """Small recurrent model that produces grads in rolled mode."""

    def __init__(self) -> None:
        """Initialize the shared projection."""

        super().__init__()
        self.proj = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two recurrent residual steps.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scalar model output.
        """

        for _ in range(2):
            x = torch.relu(self.proj(x))
        return x.sum()


def test_bfloat16_tolerance_allows_dtype_scale_replay_drift() -> None:
    """BFLOAT16 replay tolerance matches dtype precision."""

    saved = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    replayed = saved + torch.tensor([0.003, -0.004], dtype=torch.bfloat16)
    mismatched = saved + torch.tensor([0.1, 0.1], dtype=torch.bfloat16)

    assert tensor_nanequal(saved, replayed, allow_tolerance=True)
    assert not tensor_nanequal(saved, mismatched, allow_tolerance=True)


def test_func_call_location_does_not_retain_frame_function_object() -> None:
    """FuncCallLocation snapshots metadata and releases live function refs."""

    loc = FuncCallLocation(
        file=__file__,
        line_number=1,
        func_name="_sample_func_for_location",
        _frame_func_obj=_sample_func_for_location,
    )

    assert loc._frame_func_obj is None
    assert "x" in (loc.func_signature or "")
    assert "Return ``x`` unchanged" in (loc.func_docstring or "")


def test_arg_specs_extract_common_keyword_tensor_arguments() -> None:
    """Static ArgSpecs extract tensors passed through common kwargs."""

    x = torch.randn(2, 3)
    weight = nn.Parameter(torch.randn(4, 3))
    bias = nn.Parameter(torch.randn(4))

    tensors, params = extract_tensors_and_params(
        FUNC_ARG_SPECS["linear"],
        (),
        {"input": x, "weight": weight, "bias": bias},
    )
    assert tensors == [x]
    assert params == [weight, bias]

    cat_tensors, _ = extract_tensors_and_params(
        FUNC_ARG_SPECS["cat"],
        (),
        {"tensors": [x, x + 1]},
    )
    assert len(cat_tensors) == 2

    where_tensors, _ = extract_tensors_and_params(
        FUNC_ARG_SPECS["where"],
        (),
        {"condition": x > 0, "input": x, "other": x - 1},
    )
    assert len(where_tensors) == 3


def test_validate_forward_pass_handles_new_empty_followed_by_full_overwrite() -> None:
    """Validation replay skips only the uninitialized allocation itself."""

    model = _NewEmptyFilledModel()
    x = torch.randn(2, 3)

    assert validate_forward_pass(model, x, validate_metadata=True) is True


def test_validate_forward_pass_handles_where_equal_branch_condition() -> None:
    """Perturbation skips an irrelevant ``where`` condition with equal branches."""

    model = _WhereEqualBranchesModel()
    x = torch.tensor([[1.0, -2.0], [-3.0, 4.0]])

    assert validate_forward_pass(model, x, validate_metadata=True) is True


def test_functional_parameter_view_does_not_inflate_param_counts() -> None:
    """Differentiable tensor views are not counted as separate parameters."""

    model = _FunctionalParameterViewModel()
    trace = tl.trace(
        model, torch.randn(2, 4), capture=tl.options.CaptureOptions(save_arg_values=True)
    )

    assert trace.num_params == model.weight.numel()
    assert trace.num_params_trainable == model.weight.numel()
    assert trace.num_params_frozen == 0
    assert trace.num_params == trace.num_params_trainable + trace.num_params_frozen
    assert check_metadata_invariants(trace) is True


@pytest.mark.parametrize("save_raw_activations", [True, False])
def test_view_mutation_output_recomputes_transformed_payload_metadata(
    *, save_raw_activations: bool
) -> None:
    """Describe transformed synthetic outputs from the actual returned base tensor."""

    input_value = torch.arange(18.0).reshape(6, 3)
    expected_raw = input_value + 1
    expected_raw[1:3].zero_()
    expected_transformed = expected_raw + 10
    trace = tl.trace(
        _ViewMutationOutputTransformModel(),
        input_value,
        save=SaveOptions(
            activation_transform=lambda value: value + 10,
            save_raw_activations=save_raw_activations,
        ),
    )
    output_op = trace[trace.output_layers[0]]

    assert output_op.shape == (6, 3)
    assert output_op.dtype == expected_raw.dtype
    assert output_op.activation_memory == expected_raw.nelement() * expected_raw.element_size()
    if save_raw_activations:
        assert torch.equal(output_op.out, expected_raw)
    else:
        assert output_op.out is None
    assert torch.equal(output_op.transformed_out, expected_transformed)
    assert output_op.transformed_out_shape == (6, 3)
    assert output_op.transformed_out_dtype == expected_transformed.dtype
    assert output_op.transformed_activation_memory == (
        expected_transformed.nelement() * expected_transformed.element_size()
    )
    assert check_metadata_invariants(trace) is True


def test_conditional_then_children_merge_across_multipass_layerlog() -> None:
    """Rolled LayerLogs expose THEN and ELSE child views from all ops."""

    trace = tl.trace(
        _AlternatingRecurrentModel(),
        torch.ones(1, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_code_context=True),
    )
    conditional_id = trace.conditional_records[0].id
    parent_layer = next(
        layer
        for layer in trace.layer_logs.values()
        if conditional_id in layer.conditional_arm_children
        and "then" in layer.conditional_arm_children[conditional_id]
        and "else" in layer.conditional_arm_children[conditional_id]
    )

    assert parent_layer.conditional_then_children
    assert parent_layer.conditional_else_children
    assert check_metadata_invariants(trace) is True


def test_conditional_then_invariant_catches_derived_view_corruption() -> None:
    """Metadata invariants reject stale THEN child projections."""

    trace = tl.trace(
        _AlternatingRecurrentModel(),
        torch.ones(1, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_code_context=True),
    )
    parent_layer = next(
        layer for layer in trace.layer_logs.values() if layer.conditional_then_children
    )
    parent_layer.conditional_then_children = []

    with pytest.raises(MetadataInvariantError, match="conditional_then_children"):
        check_metadata_invariants(trace)


def test_short_barcode_uses_stable_sha256_prefix() -> None:
    """Deterministic barcodes are SHA-256 prefixes of the type-tagged encoding.

    The pre-fcd3172e encoding (``str()`` joined by a raw NUL byte) is the
    collision bug that commit fixed — ``1`` vs ``"1"`` and ``["a\\x00b"]`` vs
    ``["a", "b"]`` hashed identically — so this pins the current type-tagged
    JSON construction and the collision cases the fix exists to keep distinct.
    """

    payload = ["ab", "c", 123]
    expected = hashlib.sha256(
        json.dumps(
            [[type(x).__name__, repr(x)] for x in payload],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    assert make_short_barcode_from_input(payload, barcode_len=16) == expected[:16]

    # The collisions the type-tagged encoding exists to prevent stay distinct.
    assert make_short_barcode_from_input([1]) != make_short_barcode_from_input(["1"])
    assert make_short_barcode_from_input(["a\x00b"]) != make_short_barcode_from_input(["a", "b"])


def test_validate_forward_pass_restores_state_after_exception() -> None:
    """validate_forward_pass restores module state when direct forward raises."""

    model = _MutatingErrorModel()

    with pytest.raises(RuntimeError, match="intentional validation failure"):
        validate_forward_pass(model, torch.ones(1))

    assert torch.equal(model.counter, torch.zeros(1))


def test_validation_arglocs_allow_same_parent_tensor_in_multiple_slots() -> None:
    """Arg-location checks do not fail on duplicate same-parent argument values."""

    parent = SimpleNamespace(
        layer_label="input_1",
        out=torch.tensor([2.0]),
        out_versions_by_child={},
        func_name="input",
    )
    target = SimpleNamespace(
        layer_label="add_1_1",
        parents=["input_1"],
        parent_arg_positions={"args": {0: "input_1"}, "kwargs": {}},
    )
    trace = {"input_1": parent}

    assert _check_arglocs_correct_for_arg(
        trace,
        target,  # type: ignore[arg-type]
        parent,  # type: ignore[arg-type]
        "args",
        1,
        parent.out,
    )


def test_lstm_exemption_only_treats_hidden_state_as_structural() -> None:
    """LSTM exemption uses hidden-state position despite equal data content."""

    h = torch.zeros(1, 2, 3)
    c = torch.ones(1, 2, 3)
    weight = torch.randn(12, 3)
    layer = SimpleNamespace(
        saved_args=(h.clone(), (h, c), [weight]),
        parent_arg_positions={
            "args": {0: "equal_data", 1: "hidden", 2: "weight"},
            "kwargs": {},
        },
    )
    trace = {
        "equal_data": SimpleNamespace(out=h.clone()),
        "hidden": SimpleNamespace(out=h),
        "weight": SimpleNamespace(out=weight),
    }

    assert _check_lstm_exempt(trace, layer, ["hidden"])  # type: ignore[arg-type]
    assert not _check_lstm_exempt(trace, layer, ["equal_data"])  # type: ignore[arg-type]
    assert not _check_lstm_exempt(trace, layer, ["weight"])  # type: ignore[arg-type]


def test_lstm_exemption_resolves_real_nested_hidden_state_positions() -> None:
    """Nested ``(1, i)`` hidden-state keys still resolve to positional slot 1.

    A real ``lstm(input, (h0, c0))`` capture registers its hidden-state parents
    under TUPLE keys ``(1, 0)`` / ``(1, 1)``, never a bare ``1``; matching only
    bare integer keys silently makes this exemption unreachable.
    """

    h = torch.zeros(1, 2, 3)
    c = torch.ones(1, 2, 3)
    layer = SimpleNamespace(
        saved_args=(torch.randn(4, 2, 3), (h, c)),
        parent_arg_positions={
            "args": {0: "data", (1, 0): "h0", (1, 1): "c0"},
            "kwargs": {},
        },
    )
    trace = {
        "data": SimpleNamespace(out=torch.randn(4, 2, 3)),
        "h0": SimpleNamespace(out=h),
        "c0": SimpleNamespace(out=c),
    }

    assert _check_lstm_exempt(trace, layer, ["h0"])  # type: ignore[arg-type]
    assert _check_lstm_exempt(trace, layer, ["c0"])  # type: ignore[arg-type]
    assert not _check_lstm_exempt(trace, layer, ["data"])  # type: ignore[arg-type]


def test_interpolate_exemption_uses_scale_factor_position_not_content() -> None:
    """Equal-valued input and scale tensors are disambiguated structurally."""

    equal_value = torch.tensor(2.0)
    layer = SimpleNamespace(
        saved_args=(equal_value, None, equal_value.clone()),
        saved_kwargs={},
        parent_arg_positions={
            "args": {0: "equal_data", 2: "scale_factor"},
            "kwargs": {},
        },
    )
    trace = {
        "equal_data": SimpleNamespace(out=equal_value.clone()),
        "scale_factor": SimpleNamespace(out=equal_value.clone()),
    }

    assert _check_interpolate_exempt(trace, layer, ["scale_factor"])  # type: ignore[arg-type]
    assert not _check_interpolate_exempt(trace, layer, ["equal_data"])  # type: ignore[arg-type]


def test_validate_forward_pass_handles_identity_output_layer() -> None:
    """Validation perturbs through synthetic output identity nodes."""

    assert validate_forward_pass(_IdentityOutputModel(), torch.randn(2, 3), random_seed=1)


def test_rolled_forward_graph_supports_grad_arrows(tmp_path: Path) -> None:
    """Rolled forward graphs render grad arrows after backward capture."""

    trace = tl.trace(
        _ResidualRecurrentModel(),
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
    )
    trace[trace.output_layers[0]].out.backward()
    dot = trace.draw(
        vis_mode="rolled",
        vis_outpath=str(tmp_path / "rolled_grad"),
        vis_fileformat="dot",
        vis_save_only=True,
    )

    assert GRADIENT_ARROW_COLOR in dot
