"""Layer-to-layer empirical receptive-field regression tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field._errors import ReceptiveFieldUnavailableError
from torchlens.receptive_field._gradient import gradient_for_unit


def _trace_chain(*, save: object | None = None) -> object:
    """Capture a positive two-convolution chain with retained autograd state.

    Parameters
    ----------
    save:
        Optional TorchLens activation selector used for selective-retention tests.

    Returns
    -------
    object
        Captured trace with a source and target convolution operation.
    """

    model = nn.Sequential(
        nn.Conv2d(1, 1, 3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(1, 1, 3, padding=1, bias=False),
    )
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[2].weight.fill_(1.0)
    inputs = torch.ones(1, 1, 5, 5, requires_grad=True)
    kwargs: dict[str, object] = {"save_mode": "reference"}
    if save is not None:
        kwargs["save"] = save
    return tl.trace(
        model,
        inputs,
        capture=tl.options.CaptureOptions(backward_ready=True),
        **kwargs,
    )


def _conv_endpoints(trace: object) -> tuple[object, object]:
    """Return the earlier source convolution and later target convolution.

    Parameters
    ----------
    trace:
        Captured two-convolution trace.

    Returns
    -------
    tuple[object, object]
        Source and target operations in execution order.
    """

    convolutions = [op for op in trace.layer_list if op.func_name == "conv2d"]  # type: ignore[attr-defined]
    assert len(convolutions) == 2
    return convolutions[0], convolutions[1]


def test_layer_to_layer_gradient_reads_retained_source_activation() -> None:
    """Differentiate a target unit with respect to an earlier saved activation."""

    trace = _trace_chain()
    source, target = _conv_endpoints(trace)

    result = gradient_for_unit(target, (0, 0, 2, 2), source=source)

    assert result.io_role == source.label
    assert result.grad.shape == source.out.shape
    assert result.support_ranges == ((0, 1), (0, 1), (1, 4), (1, 4))
    assert result.support_mask.sum().item() == 9


def test_unretained_layer_source_names_both_real_label_selectors() -> None:
    """Reject an unretained source with the escrowed real-selector recipe."""

    trace = _trace_chain(save=tl.label("conv2d_2"))
    source, target = _conv_endpoints(trace)
    expected_recipe = (
        "tl.trace(model, x, backward_ready=True, save="
        f"tl.label({source.layer_label_short!r}) | tl.label({target.layer_label_short!r}))"
    )

    with pytest.raises(
        ReceptiveFieldUnavailableError, match="no retained activation payload"
    ) as exc:
        gradient_for_unit(target, (0, 0, 2, 2), source=source)

    assert expected_recipe in str(exc.value)


class _DetachedPath(nn.Module):
    """Captured structural path whose source-to-target autograd edge is detached."""

    def __init__(self) -> None:
        """Create the differentiable parameter needed to retain target autograd state."""

        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return an output structurally derived from a detached input.

        Parameters
        ----------
        inputs:
            Input activation retained as the layer-to-layer source.

        Returns
        -------
        torch.Tensor
            Target activation whose autograd path omits ``inputs``.
        """

        return inputs.detach() + self.bias


def test_reachable_but_none_layer_source_gradient_raises_typed_error() -> None:
    """Treat a structurally reachable layer source returning ``None`` as unavailable."""

    inputs = torch.randn(1, 3, requires_grad=True)
    trace = tl.trace(
        _DetachedPath(),
        inputs,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    source = next(op for op in trace.layer_list if op.is_input)
    target = next(op for op in reversed(trace.layer_list) if not op.is_output)

    with pytest.raises(ReceptiveFieldUnavailableError, match="reachable.*autograd returned no"):
        gradient_for_unit(target, (0, 0), source=source)


def test_input_source_matches_existing_input_gradient() -> None:
    """Preserve the legacy input gradient exactly when it is passed as ``source``."""

    trace = _trace_chain()
    _, target = _conv_endpoints(trace)
    input_op = next(op for op in trace.layer_list if op.is_input)

    existing = gradient_for_unit(target, (0, 0, 2, 2), input=input_op, retain_graph=True)
    source_result = gradient_for_unit(target, (0, 0, 2, 2), source=input_op, retain_graph=True)

    assert source_result.op_label == existing.op_label
    assert source_result.io_role == existing.io_role
    assert source_result.unit == existing.unit
    assert torch.equal(source_result.grad, existing.grad)
    assert torch.equal(source_result.support_mask, existing.support_mask)
    assert source_result.support_ranges == existing.support_ranges
    assert source_result.spatial_support_mask is not None
    assert existing.spatial_support_mask is not None
    assert torch.equal(source_result.spatial_support_mask, existing.spatial_support_mask)
    assert source_result.batch_support == existing.batch_support
    assert source_result.cross_batch_influence == existing.cross_batch_influence
