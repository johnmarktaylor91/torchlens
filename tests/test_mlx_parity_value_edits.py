"""Value-dependent edit and stop conditions on MLX match torch on a twin model.

A value-dependent condition is a ``tl.where`` predicate or a bare callable in
``tl.when`` or ``stop_after``. MLX evaluates lazily, so the predicate context's
value fields (``is_scalar_bool``, ``bool_value``) are filled only when such a
condition is present, and only a 0-d boolean output is evaluated at the op.
Plain and static-label captures evaluate nothing during the forward.

Each test runs the same condition on a torch model and an MLX model with
identical weights, compares the edited outputs, and checks that the MLX trace
validates.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_mlx, pytest.mark.rare]

mlx = pytest.importorskip("mlx")
torch = pytest.importorskip("torch")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends.mlx import MLXBackend  # noqa: E402
from torchlens.errors import InvalidArgumentError, TorchLensWarning  # noqa: E402


class _MLXFlagNet(nn.Module):
    """MLX twin: relu hidden layer gated by a scalar boolean of its own sum."""

    def __init__(self) -> None:
        """Initialize deterministic positive weights."""

        super().__init__()
        self.l1 = nn.Linear(4, 3)
        self.l2 = nn.Linear(3, 2)
        self.l1.weight = mx.full((3, 4), 0.5)
        self.l1.bias = mx.full((3,), 0.1)
        self.l2.weight = mx.full((2, 3), 0.25)
        self.l2.bias = mx.full((2,), -0.2)

    def __call__(self, x: mx.array) -> mx.array:
        """Run the gated forward."""

        h = nn.relu(self.l1(x))
        flag = h.sum() > 0
        return self.l2(mx.where(flag, h, -h))


class _TorchFlagNet(torch.nn.Module):
    """Torch twin of ``_MLXFlagNet`` with the same weights."""

    def __init__(self) -> None:
        """Initialize the same deterministic weights."""

        super().__init__()
        self.l1 = torch.nn.Linear(4, 3)
        self.l2 = torch.nn.Linear(3, 2)
        with torch.no_grad():
            self.l1.weight.fill_(0.5)
            self.l1.bias.fill_(0.1)
            self.l2.weight.fill_(0.25)
            self.l2.bias.fill_(-0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the gated forward."""

        h = torch.relu(self.l1(x))
        flag = h.sum() > 0
        return self.l2(torch.where(flag, h, -h))


def _final(trace: Any) -> np.ndarray:
    """Return the trace's final output as a float64 numpy array."""

    out = trace[trace.output_layers[0]].out
    if isinstance(out, torch.Tensor):
        return out.detach().double().numpy()
    return np.asarray(out, dtype=np.float64)


def _both(**kwargs_by_backend: Any) -> tuple[Any, Any]:
    """Trace the torch and MLX twins with per-backend keyword arguments."""

    torch_trace = tl.trace(_TorchFlagNet(), torch.ones(1, 4), **kwargs_by_backend["torch"])
    mlx_trace = tl.trace(_MLXFlagNet(), mx.ones((1, 4)), backend="mlx", **kwargs_by_backend["mlx"])
    return torch_trace, mlx_trace


def _linear_count(trace: Any) -> int:
    """Count captured linear ops."""

    return sum(op.type == "linear" for op in trace._layer_list)


def test_metadata_predicate_edit_matches_torch_and_validates() -> None:
    """A tl.where predicate over layer type and shape edits the same op on both."""

    def condition(ctx: Any) -> bool:
        return ctx.layer_type == "relu" and tuple(ctx.shape) == (1, 3)

    edit = tl.when(tl.where(condition), tl.scale(0.5))
    torch_trace, mlx_trace = _both(torch={"intervene": edit}, mlx={"intervene": edit})
    baseline = _final(tl.trace(_MLXFlagNet(), mx.ones((1, 4)), backend="mlx"))

    np.testing.assert_allclose(_final(mlx_trace), _final(torch_trace), rtol=1e-5, atol=1e-6)
    assert not np.allclose(_final(mlx_trace), baseline)
    assert [op.type for op in mlx_trace._layer_list if op.is_edited] == ["relu"]
    assert MLXBackend().validate_trace(mlx_trace).passed


def test_scalar_bool_value_edit_matches_torch_and_validates() -> None:
    """A bare callable reading bool_value flips the gate on both backends."""

    def gate_is_true(ctx: Any) -> bool:
        return bool(ctx.is_scalar_bool and ctx.bool_value is True)

    torch_trace, mlx_trace = _both(
        torch={"intervene": tl.when(gate_is_true, tl.replace_with(torch.tensor(False)))},
        mlx={"intervene": tl.when(gate_is_true, tl.replace_with(mx.array(False)))},
    )
    baseline = _final(tl.trace(_MLXFlagNet(), mx.ones((1, 4)), backend="mlx"))

    np.testing.assert_allclose(_final(mlx_trace), _final(torch_trace), rtol=1e-5, atol=1e-6)
    assert not np.allclose(_final(mlx_trace), baseline)
    edited = [op for op in mlx_trace._layer_list if op.is_edited]
    assert len(edited) == 1
    assert tuple(edited[0].shape) == ()
    assert MLXBackend().validate_trace(mlx_trace).passed


def test_scalar_bool_condition_that_reads_false_does_not_fire() -> None:
    """bool_value is the real value: a condition on False matches nothing on either."""

    def gate_is_false(ctx: Any) -> bool:
        return bool(ctx.is_scalar_bool and ctx.bool_value is False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # zero-match notice
        torch_trace, mlx_trace = _both(
            torch={"intervene": tl.when(gate_is_false, tl.zero_ablate())},
            mlx={"intervene": tl.when(gate_is_false, tl.zero_ablate())},
        )
    assert not any(op.is_edited for op in mlx_trace._layer_list)
    assert not any(op.is_edited for op in torch_trace._layer_list)
    np.testing.assert_allclose(_final(mlx_trace), _final(torch_trace), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("wrap", [False, True], ids=["callable", "tl_where"])
def test_value_dependent_stop_matches_torch(wrap: bool) -> None:
    """stop_after on the scalar boolean halts before the second linear on both."""

    def at_gate(ctx: Any) -> bool:
        return bool(ctx.is_scalar_bool)

    site = tl.where(at_gate) if wrap else at_gate
    torch_trace, mlx_trace = _both(torch={"stop_after": site}, mlx={"stop_after": site})

    assert torch_trace.is_stopped is True
    assert mlx_trace.is_stopped is True
    assert _linear_count(mlx_trace) == _linear_count(torch_trace) == 1
    assert MLXBackend().validate_trace(mlx_trace).passed


def test_unfired_callable_stop_warns_like_torch() -> None:
    """An exploratory callable stop that never fires warns and records the site."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(
            _MLXFlagNet(), mx.ones((1, 4)), backend="mlx", stop_after=lambda ctx: False
        )
    assert trace.is_stopped is False
    codes = [
        getattr(entry.message, "fields", {}).get("code")
        for entry in caught
        if isinstance(entry.message, TorchLensWarning)
    ]
    assert "stop_after_never_fired_callable" in codes
    assert trace.annotations["unmatched_capture_selectors"][-1]["slot"] == "stop_after"


def test_unfired_tl_where_stop_refuses_like_torch() -> None:
    """A selector-shaped stop that never fires refuses with torch's code."""

    with pytest.raises(InvalidArgumentError) as info:
        tl.trace(
            _MLXFlagNet(),
            mx.ones((1, 4)),
            backend="mlx",
            stop_after=tl.where(lambda ctx: False),
        )
    assert info.value.fields["code"] == "stop_after_never_fired"


def test_requires_grad_field_stays_unavailable_on_mlx() -> None:
    """MLX arrays carry no gradient flag, so reading it refuses instead of guessing."""

    from torchlens.errors import MLXValueUnavailableError

    with pytest.raises(Exception) as info:
        tl.trace(
            _MLXFlagNet(),
            mx.ones((1, 4)),
            backend="mlx",
            intervene=tl.when(lambda ctx: bool(ctx.tensor_requires_grad), tl.zero_ablate()),
        )
    chain = [info.value, info.value.__cause__, info.value.__context__]
    assert any(isinstance(err, MLXValueUnavailableError) for err in chain)


def test_static_and_plain_captures_read_no_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a value-dependent condition the value reader never runs."""

    from torchlens.backends.mlx import interventions

    calls: list[int] = []
    real = interventions._value_record_context

    def _spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(interventions, "_value_record_context", _spy)
    model, x = _MLXFlagNet(), mx.ones((1, 4))
    tl.trace(model, x, backend="mlx")
    tl.trace(model, x, backend="mlx", intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
    tl.trace(model, x, backend="mlx", stop_after=tl.func("relu"))
    assert calls == []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # zero-match notice
        tl.trace(model, x, backend="mlx", intervene=tl.when(lambda ctx: False, tl.zero_ablate()))
    assert calls
