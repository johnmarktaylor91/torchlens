"""Anti-recurrence tests for the non-torch tensor-display crash (cert10 follow-on).

Round-10 fix ``d2bfca03`` correctly made ``finalize_single_pass_trace`` flip
every preview-backend (MLX/tinygrad/TF/JAX/Paddle) op's ``_tracing_finished``
to ``True``. That newly routes finalized preview-backend ops with a *saved*
activation into ``Op._str_after_pass`` -> ``Op._tensor_contents_str_helper``
-> ``torchlens.utils.display.tensor_stats_summary``, which was written for
``torch.Tensor`` only (``tensor.device``, ``torch.isnan``, ``.to(torch.float64)``,
``.detach()``, ``.requires_grad``, ...). Calling ``str(op)``/``repr(op)`` on
such an op raised ``AttributeError`` (e.g. a numpy/preview-backend array has
no ``.device``/``.detach()``/``.requires_grad``).

This test module covers:

(a) A backend-agnostic unit test that ``tensor_stats_summary`` handles a
    non-torch array-like without raising, reporting shape/dtype.
(b) An end-to-end test (gated behind ``importorskip``) tracing a tiny MLX or
    tinygrad model with a saved activation, then calling ``str(op)``/
    ``repr(op)`` on a finalized op that has a saved (non-torch) activation.
(c) A torch control: ``tensor_stats_summary(torch.randn(...))`` on a
    deterministic tensor must still produce the exact original full-stats
    string -- the torch code path is untouched by this fix.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.utils.display import tensor_stats_summary


class _FakeArray:
    """Minimal duck-typed non-torch array: exposes only ``.shape``/``.dtype``.

    Deliberately has none of ``torch.Tensor``'s surface (no ``.device``,
    ``.detach()``, ``.requires_grad``, ``.to()``, ...) to prove the fix relies
    only on the near-universal ``.shape``/``.dtype`` duck-typing contract
    documented in ``tensor_stats_summary``, not on any specific array
    library's API.
    """

    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.dtype = dtype


class TestTensorStatsSummaryNonTorch:
    """(a) Backend-agnostic: tensor_stats_summary must never raise on non-torch arrays."""

    def test_numpy_array_does_not_raise(self) -> None:
        """A numpy array (no .device/.detach()/grad_fn) must produce a safe summary."""

        arr = np.random.randn(3, 4)
        result = tensor_stats_summary(arr)
        assert isinstance(result, str)
        assert "3" in result and "4" in result
        assert "float64" in result

    def test_numpy_array_reports_shape_and_dtype(self) -> None:
        """The non-torch summary must surface shape and dtype, not just avoid crashing."""

        arr = np.zeros((2, 5), dtype=np.int32)
        result = tensor_stats_summary(arr)
        assert "2" in result
        assert "5" in result
        assert "int32" in result

    def test_duck_typed_fake_array_does_not_raise(self) -> None:
        """A minimal object exposing only .shape/.dtype (no numpy/torch) must not raise."""

        arr = _FakeArray(shape=(7, 9), dtype="fake_dtype")
        result = tensor_stats_summary(arr)
        assert isinstance(result, str)
        assert "7" in result and "9" in result
        assert "fake_dtype" in result

    def test_object_with_no_shape_or_dtype_does_not_raise(self) -> None:
        """Even a bare object with neither .shape nor .dtype must not raise."""

        result = tensor_stats_summary(object())
        assert isinstance(result, str)

    def test_numpy_array_omits_torch_only_stats(self) -> None:
        """No torch-only numeric stats (mean/std/min/max/nan/inf) are fabricated."""

        arr = np.random.randn(3, 4)
        result = tensor_stats_summary(arr)
        for torch_only_field in ("mean=", "std=", "min=", "max=", "nan=", "inf="):
            assert torch_only_field not in result


class TestOpStrDisplayNonTorch:
    """(a-continued) Op.__str__/repr must not crash when self.out is a non-torch array."""

    def test_op_str_with_numpy_out_does_not_raise(self) -> None:
        """Simulate a finalized op whose .out is a numpy array (not torch.Tensor)."""

        model = nn.Linear(3, 5)
        x = torch.randn(2, 3)
        log = tl.trace(model, x, save=lambda op: True)
        assert log.num_ops > 0

        op = None
        for op_label in log.op_labels:
            candidate = log[op_label]
            if candidate.out is not None:
                op = candidate
                break
        assert op is not None, "expected at least one op with a saved activation"

        # Swap in a numpy array standing in for a preview-backend array, and
        # force the finalized ("after pass") display branch, mirroring the
        # state a preview-backend op is in after d2bfca03's per-op
        # _tracing_finished flip.
        original_out = op.out
        original_finished = op._tracing_finished
        op.out = np.asarray(original_out.detach().numpy())
        op._tracing_finished = True
        try:
            s = str(op)
            r = repr(op)
        finally:
            op.out = original_out
            op._tracing_finished = original_finished

        assert isinstance(s, str)
        assert isinstance(r, str)
        # The shape/dtype summary line must still be present.
        for dim in original_out.shape:
            assert str(dim) in s


mlx = pytest.importorskip("mlx", reason="optional MLX preview backend not installed")


class _MlxLinear:
    """Tiny MLX linear model used for the end-to-end saved-activation check."""

    def __init__(self) -> None:
        import mlx.nn as mlx_nn

        self.l1 = mlx_nn.Linear(4, 8)

    def __call__(self, x: "mlx.core.array") -> "mlx.core.array":  # noqa: F821
        return self.l1(x)


@pytest.mark.optional
def test_mlx_finalized_op_str_with_saved_activation_does_not_raise() -> None:
    """(b) End-to-end: MLX op with a saved activation must not crash on str()/repr().

    Before this fix, a finalized preview-backend op whose ``out`` is a saved
    non-torch array raised ``AttributeError`` inside
    ``tensor_stats_summary``/``Op._tensor_contents_str_helper`` (e.g.
    ``'array' object has no attribute 'device'``).
    """

    import mlx.core as mx

    model = _MlxLinear()
    x = mx.random.normal((2, 4))
    log = tl.trace(model, x, save=tl.func("linear"))

    saved_ops = [log[label] for label in log.op_labels if log[label].out is not None]
    assert saved_ops, "expected at least one op with a saved MLX activation"
    for op in saved_ops:
        assert op._tracing_finished is True
        # Must not raise -- this is the exact crash class this fix closes.
        s = str(op)
        r = repr(op)
        assert isinstance(s, str)
        assert isinstance(r, str)


class TestTensorStatsSummaryTorchControl:
    """(c) Torch control: the torch.Tensor path must be byte-identical to before the fix."""

    def test_torch_full_stats_output_unchanged(self) -> None:
        """A deterministic tensor's full-stats summary string must match exactly.

        Covers mean/std/min/max/nan/inf/neg/zero all being exercised, so a
        regression narrowing the torch branch (e.g. accidentally routing a
        real torch.Tensor into the non-torch shape/dtype-only summary) would
        be caught by a shrunken string missing these fields.
        """

        tensor = torch.tensor(
            [
                [1.0, -2.0, 0.0, float("nan")],
                [3.0, float("inf"), -0.0, 2.5],
            ],
            dtype=torch.float32,
        )
        result = tensor_stats_summary(tensor)

        expected = (
            "Tensor[2, 4] float32 cpu mean=0.75 std=1.677 "
            "min=-2 max=3 nan=12.5% inf=12.5% neg=12.5% zero=25% "
            "[⚠ 12.5% NaN] [⚠ 12.5% Inf]"
        )
        assert result == expected

    def test_torch_empty_tensor_unchanged(self) -> None:
        """The empty-tensor short-circuit branch must still fire for torch tensors."""

        tensor = torch.empty(0)
        result = tensor_stats_summary(tensor)
        assert result == "Tensor[0] float32 cpu empty"

    def test_torch_simple_tensor_regression(self) -> None:
        """A plain finite tensor with no NaN/Inf must produce the plain stats line."""

        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = tensor_stats_summary(tensor)
        assert result == (
            "Tensor[2, 2] float32 cpu mean=2.5 std=1.118 min=1 max=4 nan=0% inf=0% neg=0% zero=0%"
        )
