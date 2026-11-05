"""A vmap call is a recorded boundary, and validation checks that boundary.

TorchLens records a ``torch.vmap`` / ``torch.func.vmap`` call as one boundary
op and runs the transform's interior with logging paused, so the completeness
witness does not census the interior by design: an op inside the vectorized
function is part of the boundary, and replaying the boundary re-runs it. The
check that matters is the boundary itself: a correct capture validates True,
and a boundary whose recorded output is wrong must fail validation.

Generic-mode ``associative_scan`` on CPU takes the same path. Its eager entry
never dispatches ``associative_scan_op``; it slices and concatenates with
ordinary ops (captured one by one) and applies the combine function through
``torch.vmap`` (captured as vmap boundaries), so a wrong combine output must
fail validation the same way.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from _validation_capture import _capture
from test_completeness_hop_rules import _CLEAN_MUL, _associative_scan, _Forward
from torch import nn

import torchlens as tl
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.validation import last_validation_failure

_Forwards = Callable[[], Callable[[torch.Tensor], torch.Tensor]]


def _torch_vmap_forward() -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward that maps a row function with ``torch.vmap``.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    return lambda x: torch.vmap(lambda row: row.sin() * 2.0)(x) + x


def _func_vmap_forward() -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward that maps a row function with ``torch.func.vmap``.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    return lambda x: torch.func.vmap(lambda row: row.sin() * 2.0)(x) + x


def _associative_scan_forward() -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward running generic-mode ``associative_scan`` on CPU.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function (skips when this torch has no ``associative_scan``).
    """

    return _associative_scan(_CLEAN_MUL)


_VMAP_PATHS: dict[str, _Forwards] = {
    "torch.vmap": _torch_vmap_forward,
    "torch.func.vmap": _func_vmap_forward,
    "associative_scan": _associative_scan_forward,
}


def _vmap_ops(trace: Any) -> list[Any]:
    """Return the trace's vmap boundary ops.

    Parameters
    ----------
    trace:
        Completed capture.

    Returns
    -------
    list[Any]
        Ops recorded as a vmap transform boundary.
    """

    return [op for op in trace.ops if op.func_name == "vmap"]


def _validate(model: nn.Module, x: torch.Tensor) -> bool:
    """Validate one forward with the capture's own warnings silenced.

    Parameters
    ----------
    model:
        Model to validate.
    x:
        Input tensor.

    Returns
    -------
    bool
        The public validation verdict.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validate(model, x, scope="forward", validate_metadata=True))


@pytest.mark.parametrize("path", sorted(_VMAP_PATHS))
def test_vmap_call_is_a_recorded_boundary_that_validates(path: str) -> None:
    """The vmap call is one boundary op; its interior is not logged or censused."""

    model = _Forward(_VMAP_PATHS[path]())
    x = torch.randn(3, 3)
    wrap_torch(completeness_witness=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x)
    try:
        func_names = [op.func_name for op in trace.ops]
        diagnostics = list(trace.completeness_diagnostics)
        verified = trace.provenance.verified
    finally:
        trace.cleanup()
    assert "vmap" in func_names
    assert "__mul__" not in func_names and "sin" not in func_names, func_names
    assert diagnostics == []
    assert verified is True
    assert _validate(model, x) is True


def _shift_saved_output(op: Any) -> None:
    """Shift a boundary op's saved output by one, in place.

    Parameters
    ----------
    op:
        Recorded vmap boundary op.
    """

    with torch.no_grad():
        op.out.add_(1.0)


def _shift_replay(op: Any) -> None:
    """Make a boundary op's recorded callable return its true output plus one.

    Parameters
    ----------
    op:
        Recorded vmap boundary op.
    """

    recorded = op.func

    def shifted(*args: Any, **kwargs: Any) -> Any:
        """Run the recorded boundary, then shift its output."""

        return recorded(*args, **kwargs) + 1.0

    object.__setattr__(op, "func", shifted)


@pytest.mark.parametrize("path", sorted(_VMAP_PATHS))
def test_wrong_vmap_boundary_saved_output_fails_validation(path: str) -> None:
    """A vmap boundary whose saved output is wrong fails validation."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace, ground_truth = _capture(_Forward(_VMAP_PATHS[path]()), torch.randn(3, 3))
        try:
            boundaries = _vmap_ops(trace)
            for op in boundaries:
                _shift_saved_output(op)
            verdict = trace.validate_forward_pass(ground_truth)
        finally:
            trace.cleanup()
    assert boundaries
    assert verdict is False
    assert last_validation_failure() is not None


@pytest.mark.parametrize("path", sorted(_VMAP_PATHS))
def test_vmap_boundary_replay_is_compared_to_its_saved_output(path: str) -> None:
    """A boundary whose replay disagrees with its saved output fails, naming the boundary."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace, ground_truth = _capture(_Forward(_VMAP_PATHS[path]()), torch.randn(3, 3))
        try:
            boundaries = _vmap_ops(trace)
            for op in boundaries:
                _shift_replay(op)
            verdict = trace.validate_forward_pass(ground_truth)
        finally:
            trace.cleanup()
    failure = last_validation_failure()
    assert boundaries
    assert verdict is False
    assert failure is not None and failure.func_name == "vmap", failure
