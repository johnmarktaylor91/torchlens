"""Regression tests for Phase 14 bug fixes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture.arg_positions import FUNC_ARG_SPECS, extract_tensors_and_params
from torchlens.data_classes.func_call_location import FuncCallLocation
from torchlens.utils.hashing import make_short_barcode_from_input
from torchlens.utils.tensor_utils import tensor_nanequal
from torchlens.validation import (
    MetadataInvariantError,
    check_metadata_invariants,
    validate_forward_pass,
)
from torchlens.validation.core import _check_arglocs_correct_for_arg
from torchlens.validation.exemptions import _check_lstm_exempt
from torchlens.visualization.rendering import GRADIENT_ARROW_COLOR


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
    """Shared block whose branch arm alternates across recurrent passes."""

    def __init__(self) -> None:
        """Initialize the shared linear layer."""

        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, pass_num: int) -> torch.Tensor:
        """Run one alternating branch step.

        Parameters
        ----------
        x:
            Input tensor.
        pass_num:
            One-indexed recurrent pass number.

        Returns
        -------
        torch.Tensor
            Shared linear output.
        """

        branch_marker = (x.mean() * 0) + (1.0 if pass_num % 2 == 1 else -1.0)
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
        """Run four recurrent passes.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final recurrent output.
        """

        for pass_num in range(1, 5):
            x = self.block(x, pass_num)
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


class _ResidualRecurrentModel(nn.Module):
    """Small recurrent model that produces gradients in rolled mode."""

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


def test_conditional_then_children_merge_across_multipass_layerlog() -> None:
    """Rolled LayerLogs expose THEN and ELSE child views from all passes."""

    model_log = tl.log_forward_pass(
        _AlternatingRecurrentModel(),
        torch.ones(1, 4),
        layers_to_save="all",
        save_source_context=True,
    )
    conditional_id = model_log.conditional_events[0].id
    parent_layer = next(
        layer
        for layer in model_log.layer_logs.values()
        if conditional_id in layer.cond_branch_children_by_cond
        and "then" in layer.cond_branch_children_by_cond[conditional_id]
        and "else" in layer.cond_branch_children_by_cond[conditional_id]
    )

    assert parent_layer.cond_branch_then_children
    assert parent_layer.cond_branch_else_children
    assert check_metadata_invariants(model_log) is True


def test_conditional_then_invariant_catches_derived_view_corruption() -> None:
    """Metadata invariants reject stale THEN child projections."""

    model_log = tl.log_forward_pass(
        _AlternatingRecurrentModel(),
        torch.ones(1, 4),
        layers_to_save="all",
        save_source_context=True,
    )
    parent_layer = next(
        layer for layer in model_log.layer_logs.values() if layer.cond_branch_then_children
    )
    parent_layer.cond_branch_then_children = []

    with pytest.raises(MetadataInvariantError, match="cond_branch_then_children"):
        check_metadata_invariants(model_log)


def test_short_barcode_uses_stable_sha256_prefix() -> None:
    """Deterministic barcodes are stable SHA-256 prefixes."""

    payload = ["ab", "c", 123]
    expected = hashlib.sha256("\x00".join(str(x) for x in payload).encode("utf-8")).hexdigest()

    assert make_short_barcode_from_input(payload, barcode_len=16) == expected[:16]


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
        activation=torch.tensor([2.0]),
        children_tensor_versions={},
        func_name="input",
    )
    target = SimpleNamespace(
        layer_label="add_1_1",
        parent_layers=["input_1"],
        parent_layer_arg_locs={"args": {0: "input_1"}, "kwargs": {}},
    )
    model_log = {"input_1": parent}

    assert _check_arglocs_correct_for_arg(
        model_log,
        target,  # type: ignore[arg-type]
        parent,  # type: ignore[arg-type]
        "args",
        1,
        parent.activation,
    )


def test_lstm_exemption_only_treats_hidden_state_as_structural() -> None:
    """LSTM perturbation exemption handles hidden tuples but not params."""

    h = torch.zeros(1, 2, 3)
    c = torch.ones(1, 2, 3)
    weight = torch.randn(12, 3)
    layer = SimpleNamespace(captured_args=(torch.randn(4, 2, 3), (h, c), [weight]))
    hidden_log = {"hidden": SimpleNamespace(activation=c)}
    weight_log = {"weight": SimpleNamespace(activation=weight)}

    assert _check_lstm_exempt(hidden_log, layer, ["hidden"])  # type: ignore[arg-type]
    assert not _check_lstm_exempt(weight_log, layer, ["weight"])  # type: ignore[arg-type]


@pytest.mark.smoke
def test_validate_forward_pass_handles_identity_output_layer() -> None:
    """Validation perturbs through synthetic output identity nodes."""

    assert validate_forward_pass(_IdentityOutputModel(), torch.randn(2, 3), random_seed=1)


def test_rolled_forward_graph_supports_gradient_arrows(tmp_path: Path) -> None:
    """Rolled forward graphs render gradient arrows after backward capture."""

    model_log = tl.log_forward_pass(
        _ResidualRecurrentModel(),
        torch.randn(2, 3, requires_grad=True),
        gradients_to_save="all",
    )
    model_log[model_log.output_layers[0]].activation.backward()
    dot = model_log.render_graph(
        vis_mode="rolled",
        vis_outpath=str(tmp_path / "rolled_grad"),
        vis_fileformat="dot",
        vis_save_only=True,
    )

    assert GRADIENT_ARROW_COLOR in dot
