"""Tests for Phase 6 NaN/Inf debugging affordances."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import CaptureError
from torchlens.partial import PartialModelLog


class NonFiniteFixture(nn.Module):
    """Tiny module that produces NaN at a known division op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor containing NaN via zero divided by zero.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor with NaN values at the division operation.
        """

        zeros = x - x
        return zeros / zeros


class ForwardErrorFixture(nn.Module):
    """Tiny module that fails after one logged operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raise after producing one intermediate tensor.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            This function always raises before returning.

        Raises
        ------
        RuntimeError
            Always raised after the logged add operation.
        """

        _ = x + 1
        raise RuntimeError("fixture failure")


def _input_tensor() -> torch.Tensor:
    """Return a deterministic fixture input.

    Returns
    -------
    torch.Tensor
        Float input tensor for the NaN fixture.
    """

    return torch.ones(1, 2)


def test_first_nonfinite_locates_nan_op() -> None:
    """ModelLog.first_nonfinite locates the known NaN-producing op."""

    model_log = tl.log_forward_pass(NonFiniteFixture(), _input_tensor())
    answer = model_log.first_nonfinite()
    assert "First non-finite" in answer
    assert "__truediv__" in answer
    assert "shape=(1, 2)" in answer
    assert "dtype=torch.float32" in answer


def test_print_model_log_shows_inline_nan_flags(capsys: pytest.CaptureFixture[str]) -> None:
    """Printing a ModelLog surfaces the inline NaN/Inf debugging line."""

    model_log = tl.log_forward_pass(NonFiniteFixture(), _input_tensor())
    print(model_log)
    captured = capsys.readouterr()
    assert "NaN/Inf" in captured.out
    assert "__truediv__" in captured.out
    assert "shape=(1, 2)" in captured.out


def test_raise_on_nan_raises_capture_error_with_partial_graph() -> None:
    """raise_on_nan stops at the first non-finite tensor and renders partial DOT."""

    with pytest.raises(CaptureError) as exc_info:
        tl.log_forward_pass(
            NonFiniteFixture(),
            _input_tensor(),
            capture=tl.options.CaptureOptions(raise_on_nan=True),
        )
    error = exc_info.value
    assert "op='__truediv__'" in str(error)
    assert "shape=(1, 2)" in str(error)
    assert error.fields["op"] == "__truediv__"
    assert error.fields["shape"] == (1, 2)

    partial = tl.partial.from_failed_capture(error)
    graph = partial.render_graph()
    assert "digraph torchlens_partial" in graph
    assert "__truediv__" in graph
    assert "shape=(1, 2)" in graph


def test_partial_model_log_constructible_from_failed_capture() -> None:
    """PartialModelLog can be recovered from the failed capture exception."""

    with pytest.raises(CaptureError) as exc_info:
        tl.log_forward_pass(
            NonFiniteFixture(),
            _input_tensor(),
            capture=tl.options.CaptureOptions(raise_on_nan=True),
        )
    partial = tl.partial.from_failed_capture(exc_info.value)
    assert isinstance(partial, PartialModelLog)
    assert partial is exc_info.value.partial_log
    assert len(partial.raw_layers) >= 1
    assert "__truediv__" in partial.first_nonfinite()


def test_partial_model_log_attached_to_generic_forward_exception() -> None:
    """Failed non-NaN captures also attach a PartialModelLog."""

    with pytest.raises(RuntimeError) as exc_info:
        tl.log_forward_pass(ForwardErrorFixture(), _input_tensor())
    partial = tl.partial.from_failed_capture(exc_info.value)
    assert isinstance(partial, PartialModelLog)
    graph = partial.render_graph()
    assert "__add__" in graph
    assert "fixture failure" in graph
