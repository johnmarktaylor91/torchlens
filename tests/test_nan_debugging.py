"""Tests for Phase 6 NaN/Inf debugging affordances."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import CaptureError, TraceCaptureError
from torchlens.types import PartialTrace


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


def test_partial_trace_constructible_after_failed_capture() -> None:
    """PartialTrace can be recovered from the failed capture exception."""

    with pytest.raises(CaptureError) as exc_info:
        tl.trace(
            NonFiniteFixture(),
            _input_tensor(),
            capture=tl.options.CaptureOptions(on_nonfinite="raise"),
        )
    partial = exc_info.value.partial_trace
    assert isinstance(partial, PartialTrace)
    assert partial is exc_info.value.partial_trace
    assert len(partial.raw_layers) >= 1
    assert "__truediv__" in partial.first_nonfinite()


def test_partial_trace_attached_to_generic_forward_exception() -> None:
    """Failed non-NaN captures also attach a PartialTrace."""

    with pytest.raises(TraceCaptureError) as exc_info:
        tl.trace(ForwardErrorFixture(), _input_tensor())
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    partial = exc_info.value.partial_trace
    assert isinstance(partial, PartialTrace)
    graph = partial.draw().dot
    assert "__add__" in graph
    assert "fixture failure" in graph


def test_find_nan_accepts_explicit_capture_options() -> None:
    """find_nan preserves only explicit capture fields while enabling its diagnostic gate."""

    result = tl.debug.find_nan(
        NonFiniteFixture(),
        _input_tensor(),
        capture=tl.options.CaptureOptions(inference_only=True),
    )

    assert result.found is True
    assert result.scope == "first non-finite tensor"


def test_first_nonfinite_rescans_after_saved_activation_changes() -> None:
    """The memoized scan must never serve a stale clean verdict."""

    trace = tl.trace(nn.Sequential(nn.Linear(2, 2)), _input_tensor())
    assert trace.first_nonfinite().startswith("No non-finite")

    with torch.no_grad():
        trace._layer_list[-1].out.mul_(float("nan"))
    in_place_answer = trace.first_nonfinite()
    assert "First non-finite" in in_place_answer
    assert "No NaN or Inf" not in trace.explain()

    replaced = trace._layer_list[0]
    replaced.out = torch.full_like(replaced.out, float("inf"))
    replaced_answer = trace.first_nonfinite()
    assert "First non-finite" in replaced_answer
    assert str(replaced.layer_label) in replaced_answer
