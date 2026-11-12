"""Value-dependent edit and stop conditions on MLX match torch on a twin model.

A value-dependent condition is a ``tl.where`` predicate or a bare callable in
``tl.when`` or ``stop_after``. The condition sees the same context on both
backends: metadata fields are real, and the value fields ``is_scalar_bool`` and
``bool_value`` are ``None`` during the forward, as on torch's live capture path.
MLX evaluates nothing at the op for such a condition.

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


def test_value_fields_read_none_on_both_backends() -> None:
    """is_scalar_bool and bool_value are None at every scalar op, on both backends."""

    seen: dict[str, list[tuple[Any, Any]]] = {"torch": [], "mlx": []}

    def recorder(name: str) -> Any:
        def condition(ctx: Any) -> bool:
            if tuple(ctx.shape) == ():
                seen[name].append((ctx.is_scalar_bool, ctx.bool_value))
            return bool(ctx.is_scalar_bool and ctx.bool_value is True)

        return condition

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # zero-match notice
        torch_trace, mlx_trace = _both(
            torch={"intervene": tl.when(recorder("torch"), tl.zero_ablate())},
            mlx={"intervene": tl.when(recorder("mlx"), tl.zero_ablate())},
        )
    assert seen["torch"] and set(seen["torch"]) == {(None, None)}
    assert seen["mlx"] and set(seen["mlx"]) == {(None, None)}
    assert not any(op.is_edited for op in torch_trace._layer_list)
    assert not any(op.is_edited for op in mlx_trace._layer_list)
    np.testing.assert_allclose(_final(mlx_trace), _final(torch_trace), rtol=1e-5, atol=1e-6)
    assert MLXBackend().validate_trace(mlx_trace).passed


@pytest.mark.parametrize("wrap", [False, True], ids=["callable", "tl_where"])
def test_value_dependent_stop_matches_torch(wrap: bool) -> None:
    """A metadata stop condition halts after the relu, before the second linear, on both."""

    def at_relu(ctx: Any) -> bool:
        return ctx.layer_type == "relu"

    site = tl.where(at_relu) if wrap else at_relu
    torch_trace, mlx_trace = _both(torch={"stop_after": site}, mlx={"stop_after": site})

    assert torch_trace.is_stopped is True
    assert mlx_trace.is_stopped is True
    assert _linear_count(mlx_trace) == _linear_count(torch_trace) == 1
    assert MLXBackend().validate_trace(mlx_trace).passed


def _warning_codes(caught: list[Any]) -> list[Any]:
    """Return the TorchLens warning codes among caught warnings."""

    return [
        getattr(entry.message, "fields", {}).get("code")
        for entry in caught
        if isinstance(entry.message, TorchLensWarning)
    ]


def test_unfired_callable_stop_warns_like_torch() -> None:
    """A callable stop on a value field never fires: both backends warn and record it."""

    def at_true_gate(ctx: Any) -> bool:
        return bool(ctx.is_scalar_bool)

    traces = {}
    for name, model, x, kwargs in (
        ("torch", _TorchFlagNet(), torch.ones(1, 4), {}),
        ("mlx", _MLXFlagNet(), mx.ones((1, 4)), {"backend": "mlx"}),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            traces[name] = tl.trace(model, x, stop_after=at_true_gate, **kwargs)
        assert "stop_after_never_fired_callable" in _warning_codes(caught), name
    for trace in traces.values():
        assert trace.is_stopped is False
        assert trace.annotations["unmatched_capture_selectors"][-1]["slot"] == "stop_after"


def test_unfired_tl_where_stop_refuses_like_torch() -> None:
    """A selector-shaped stop that never fires refuses with the same code on both."""

    site = tl.where(lambda ctx: bool(ctx.is_scalar_bool))
    codes = []
    for model, x, kwargs in (
        (_TorchFlagNet(), torch.ones(1, 4), {}),
        (_MLXFlagNet(), mx.ones((1, 4)), {"backend": "mlx"}),
    ):
        with pytest.raises(InvalidArgumentError) as info:
            tl.trace(model, x, stop_after=site, **kwargs)
        codes.append(info.value.fields["code"])
    assert codes == ["stop_after_never_fired", "stop_after_never_fired"]


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


def test_static_and_plain_captures_build_no_value_contexts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a value-dependent condition no value context is built."""

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
