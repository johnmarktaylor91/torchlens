"""tinygrad edit helpers, finished-trace edits and derived gradients of an edited program.

Every built-in helper the tinygrad backend applies (the MLX set: ``zero_ablate``,
``scale``, ``add``, ``clamp``, ``mean_ablate``, ``replace_with``) must give the
same edited capture as the torch backend on a twin model and validate; the
same helpers must work on a finished trace (``trace.fork().do``) and give what
a live edit gives. Derived gradients of an edited capture must be tinygrad's
gradients of the edited program, computed here by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

import torchlens as tl
from torchlens.backends.tinygrad import GradOptions, TinygradBackend
from torchlens.errors import BackendUnsupportedError

tinygrad = pytest.importorskip("tinygrad")
Tensor = tinygrad.Tensor

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.rare]

_X = np.asarray([[1.0, -0.5, 0.25, 2.0], [-1.5, 0.75, 1.0, -0.25]], dtype=np.float32)
_W1 = np.asarray(
    [[0.5, -0.25, 0.75], [0.25, 0.5, -0.5], [-0.75, 0.25, 0.5], [0.5, 0.75, 0.25]],
    dtype=np.float32,
)
_B1 = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
_W2 = np.asarray([[1.0, -0.5], [0.5, 0.25], [-0.25, 0.75]], dtype=np.float32)
_REPLACEMENT = np.full((2, 3), 0.5, dtype=np.float32)


class _TinyMLP:
    """Two-layer tinygrad MLP with fixed weights and one relu to edit."""

    def __init__(self) -> None:
        """Hold the fixed weights as realized tinygrad tensors."""

        self.w1 = Tensor(_W1).realize()
        self.b1 = Tensor(_B1).realize()
        self.w2 = Tensor(_W2).realize()

    def __call__(self, x: Any) -> Any:
        """Return ``relu(x @ w1 + b1) @ w2``."""

        return self.hidden(x).matmul(self.w2)

    def hidden(self, x: Any) -> Any:
        """Return ``relu(x @ w1 + b1)``."""

        return (x.matmul(self.w1) + self.b1).relu()


class _TorchTwin(torch.nn.Module):
    """The torch twin of :class:`_TinyMLP`."""

    def __init__(self) -> None:
        """Hold the same weights as torch parameters."""

        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(_W1))
        self.b1 = torch.nn.Parameter(torch.tensor(_B1))
        self.w2 = torch.nn.Parameter(torch.tensor(_W2))

    def forward(self, x: Any) -> Any:
        """Return ``relu(x @ w1 + b1) @ w2``."""

        return torch.matmul(torch.relu(torch.matmul(x, self.w1) + self.b1), self.w2)


def _np(value: Any) -> np.ndarray:
    """A payload from either backend as a numpy array."""

    if hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value.numpy())


def _relu_op(trace: Any) -> Any:
    """The one relu op (tinygrad records relu as ``where``)."""

    ops = [op for op in trace._layer_list if op.func_name in {"relu", "where"}]
    assert len(ops) == 1, [op.func_name for op in trace._layer_list]
    return ops[0]


def _final_output(trace: Any) -> np.ndarray:
    """The payload of the trace's only output op."""

    return _np(trace[trace.output_layers[0]].out)


def _helpers(backend: str) -> list[tuple[str, Any]]:
    """``(id, helper)`` for every helper the tinygrad backend applies."""

    native = Tensor(_REPLACEMENT) if backend == "tinygrad" else torch.tensor(_REPLACEMENT)
    return [
        ("zero_ablate", tl.zero_ablate()),
        ("scale", tl.scale(0.5)),
        ("add", tl.add(1.0)),
        ("clamp", tl.clamp(low=0.0, high=0.5)),
        ("mean_ablate_all", tl.mean_ablate(dim=None)),
        ("mean_ablate_axis", tl.mean_ablate(dim=-1)),
        ("replace_with", tl.replace_with(native)),
    ]


_HELPER_IDS = [name for name, _helper in _helpers("torch")]


def _helper(name: str, backend: str) -> Any:
    """The named helper built for one backend."""

    return dict(_helpers(backend))[name]


def _live(name: str) -> Any:
    """A live tinygrad capture with the named helper at the relu."""

    return tl.trace(
        _TinyMLP(),
        Tensor(_X),
        backend="tinygrad",
        intervene=tl.when(tl.func("where"), _helper(name, "tinygrad")),
    )


@pytest.mark.parametrize("name", _HELPER_IDS)
def test_live_helper_matches_torch_twin_and_validates(name: str) -> None:
    """Each helper's live capture equals torch's edited capture and validates."""

    edited = _live(name)
    torch_edited = tl.trace(
        _TorchTwin(),
        torch.tensor(_X),
        intervene=tl.when(tl.func("relu"), _helper(name, "torch")),
    )
    np.testing.assert_allclose(
        _np(_relu_op(edited).out), _np(_relu_op(torch_edited).out), atol=1e-6
    )
    np.testing.assert_allclose(_final_output(edited), _final_output(torch_edited), atol=1e-5)
    assert _relu_op(edited).is_edited is True
    assert TinygradBackend().validate_trace(edited) is True


@pytest.mark.parametrize("name", _HELPER_IDS)
def test_fork_helper_equals_live_helper_and_passes_replay(name: str) -> None:
    """A helper applied to a finished trace equals the same helper applied live."""

    from torchlens.intervention._backend_replay import verify_backend_replay

    clean = tl.trace(_TinyMLP(), Tensor(_X), backend="tinygrad")
    edited = clean.fork().do(_relu_op(clean).label, _helper(name, "tinygrad"))
    live = _live(name)
    np.testing.assert_array_equal(_np(_relu_op(edited).out), _np(_relu_op(live).out))
    np.testing.assert_array_equal(_final_output(edited), _final_output(live))
    assert verify_backend_replay(edited) == []
    assert edited.is_pristine is False


def test_fork_unported_helper_refuses_typed() -> None:
    """A helper outside the tinygrad table refuses on a finished trace, naming the helper."""

    clean = tl.trace(_TinyMLP(), Tensor(_X), backend="tinygrad")
    with pytest.raises(BackendUnsupportedError, match="clip_norm") as info:
        clean.fork().do(_relu_op(clean).label, tl.clip_norm(1.0))
    assert info.value.code == "interventions_unsupported"


def test_mean_ablate_population_source_refuses_typed() -> None:
    """A non-tinygrad mean source refuses before the capture."""

    with pytest.raises(BackendUnsupportedError, match="mean_ablate") as info:
        tl.trace(
            _TinyMLP(),
            Tensor(_X),
            backend="tinygrad",
            intervene=tl.when(tl.func("where"), tl.mean_ablate(torch.ones(3), dim=None)),
        )
    assert info.value.code == "interventions_unsupported"


def _hand_gradients(edit: Callable[[Any], Any]) -> tuple[np.ndarray, np.ndarray]:
    """tinygrad's input and relu gradients of the edited program, computed by hand."""

    model = _TinyMLP()
    x = Tensor(_X)
    hidden = edit(model.hidden(x))
    loss = hidden.matmul(model.w2).sum()
    loss.backward()
    return _np(x.grad), _np(hidden.grad)


_GRAD_EDITS: list[tuple[str, Any, Callable[[Any], Any]]] = [
    ("scale", tl.scale(0.5), lambda h: h * 0.5),
    ("add", tl.add(1.0), lambda h: h + 1.0),
    ("zero_ablate", tl.zero_ablate(), lambda h: h * 0.0),
]


@pytest.mark.parametrize(
    ("helper", "by_hand"),
    [pytest.param(helper, by_hand, id=name) for name, helper, by_hand in _GRAD_EDITS],
)
def test_derived_gradients_are_gradients_of_the_edited_program(
    helper: Any, by_hand: Callable[[Any], Any]
) -> None:
    """Input and intermediate derived gradients equal the edited program's, not the model's."""

    trace = tl.trace(
        _TinyMLP(),
        Tensor(_X),
        backend="tinygrad",
        intervene=tl.when(tl.func("where"), helper),
        capture=tl.options.CaptureOptions(
            grad_options=GradOptions(
                loss_fn=lambda out: out.sum(), input_grad_argnums=(0,), intermediate_grads=True
            )
        ),
    )
    input_grad, relu_grad = _hand_gradients(by_hand)
    np.testing.assert_allclose(
        _np(trace.autograd.derived_grads["inputs.0"].grad), input_grad, atol=1e-6
    )
    relu = _relu_op(trace)
    np.testing.assert_allclose(
        _np(trace.autograd.intermediate_derived_grads[relu.label].grad), relu_grad, atol=1e-6
    )
    assert relu.is_edited is True
    assert TinygradBackend().validate_trace(trace) is True


def test_edited_gradients_differ_from_plain_gradients() -> None:
    """The gradient run is the edited program: a scaled relu halves the input gradient."""

    options = tl.options.CaptureOptions(
        grad_options=GradOptions(loss_fn=lambda out: out.sum(), input_grad_argnums=(0,))
    )
    plain = tl.trace(_TinyMLP(), Tensor(_X), backend="tinygrad", capture=options)
    edited = tl.trace(
        _TinyMLP(),
        Tensor(_X),
        backend="tinygrad",
        intervene=tl.when(tl.func("where"), tl.scale(0.5)),
        capture=options,
    )
    np.testing.assert_allclose(
        _np(edited.autograd.derived_grads["inputs.0"].grad),
        0.5 * _np(plain.autograd.derived_grads["inputs.0"].grad),
        atol=1e-6,
    )


def test_edited_gradients_leave_input_grad_untouched() -> None:
    """The bracketed gradient run restores the user's input ``.grad``."""

    x = Tensor(_X)
    tl.trace(
        _TinyMLP(),
        x,
        backend="tinygrad",
        intervene=tl.when(tl.func("where"), tl.scale(0.5)),
        capture=tl.options.CaptureOptions(
            grad_options=GradOptions(loss_fn=lambda out: out.sum(), input_grad_argnums=(0,))
        ),
    )
    assert x.grad is None


_CONDITIONS = [
    pytest.param(
        lambda ctx: ctx.func_name in {"relu", "where"} and tuple(ctx.shape) == (2, 3),
        id="callable",
    ),
    pytest.param(
        tl.where(lambda ctx: ctx.func_name in {"relu", "where"} and ctx.dtype is not None),
        id="where",
    ),
]


@pytest.mark.parametrize("condition", _CONDITIONS)
def test_value_condition_matches_torch_twin_and_validates(condition: Any) -> None:
    """A condition over the predicate context edits the op torch edits, and validates."""

    edited = tl.trace(
        _TinyMLP(), Tensor(_X), backend="tinygrad", intervene=tl.when(condition, tl.scale(0.5))
    )
    torch_edited = tl.trace(
        _TorchTwin(), torch.tensor(_X), intervene=tl.when(condition, tl.scale(0.5))
    )
    assert [op.func_name for op in edited._layer_list if op.is_edited] == ["where"]
    np.testing.assert_allclose(
        _np(_relu_op(edited).out), _np(_relu_op(torch_edited).out), atol=1e-6
    )
    np.testing.assert_allclose(_final_output(edited), _final_output(torch_edited), atol=1e-5)
    assert TinygradBackend().validate_trace(edited) is True


def test_value_condition_combined_with_static_selector() -> None:
    """``tl.func(...) & tl.where(...)`` evaluates the static leaf on the op."""

    condition = tl.func("where") & tl.where(lambda ctx: tuple(ctx.shape) == (2, 3))
    edited = tl.trace(
        _TinyMLP(), Tensor(_X), backend="tinygrad", intervene=tl.when(condition, tl.add(1.0))
    )
    assert _relu_op(edited).is_edited is True
    assert sum(op.is_edited for op in edited._layer_list) == 1


def test_value_condition_with_derived_gradients() -> None:
    """Derived gradients of a value-condition edit are the edited program's."""

    trace = tl.trace(
        _TinyMLP(),
        Tensor(_X),
        backend="tinygrad",
        intervene=tl.when(lambda ctx: ctx.func_name == "where", tl.scale(0.5)),
        capture=tl.options.CaptureOptions(
            grad_options=GradOptions(loss_fn=lambda out: out.sum(), input_grad_argnums=(0,))
        ),
    )
    input_grad, _relu_grad = _hand_gradients(lambda h: h * 0.5)
    np.testing.assert_allclose(
        _np(trace.autograd.derived_grads["inputs.0"].grad), input_grad, atol=1e-6
    )


def test_value_condition_matching_nothing_warns() -> None:
    """A condition that matches no op warns like a static selector and edits nothing."""

    with pytest.warns(UserWarning, match="matched zero sites"):
        trace = tl.trace(
            _TinyMLP(),
            Tensor(_X),
            backend="tinygrad",
            intervene=tl.when(lambda ctx: ctx.func_name == "conv2d", tl.zero_ablate()),
        )
    assert not any(op.is_edited for op in trace._layer_list)
