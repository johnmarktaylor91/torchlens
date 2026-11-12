"""tinygrad parity rows: static-selector edits, keyword inputs and source lines.

An edited tinygrad capture must equal what the torch backend produces for the
same edit on a twin model, and must pass replay validation. A plain capture
must stay realize-once and unchanged. Each op's ``code_context`` must point at
the user code that made it, outermost frame first, as on torch.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends.tinygrad import GradOptions, TinygradBackend
from torchlens.errors import BackendUnsupportedError

tinygrad = pytest.importorskip("tinygrad")
Tensor = tinygrad.Tensor
torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.rare]

_X = np.asarray([[1.0, -0.5, 0.25, 2.0], [-1.5, 0.75, 1.0, -0.25]], dtype=np.float32)
_W1 = np.asarray(
    [[0.5, -0.25, 0.75], [0.25, 0.5, -0.5], [-0.75, 0.25, 0.5], [0.5, 0.75, 0.25]],
    dtype=np.float32,
)
_B1 = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
_W2 = np.asarray([[1.0, -0.5], [0.5, 0.25], [-0.25, 0.75]], dtype=np.float32)


class _TinyMLP:
    """Two-layer tinygrad MLP with fixed weights and a relu to edit."""

    def __init__(self) -> None:
        """Hold the fixed weights as realized tinygrad tensors."""

        self.w1 = Tensor(_W1).realize()
        self.b1 = Tensor(_B1).realize()
        self.w2 = Tensor(_W2).realize()

    def __call__(self, x: Any) -> Any:
        """Return ``relu(x @ w1 + b1) @ w2``."""

        hidden = (x.matmul(self.w1) + self.b1).relu()
        return hidden.matmul(self.w2)


class _TorchTwin(torch.nn.Module):
    """The torch twin of :class:`_TinyMLP` with the same weights."""

    def __init__(self) -> None:
        """Hold the fixed weights as torch parameters."""

        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(_W1))
        self.b1 = torch.nn.Parameter(torch.tensor(_B1))
        self.w2 = torch.nn.Parameter(torch.tensor(_W2))

    def forward(self, x: Any) -> Any:
        """Return ``relu(x @ w1 + b1) @ w2``."""

        hidden = torch.relu(torch.matmul(x, self.w1) + self.b1)
        return torch.matmul(hidden, self.w2)


def _np(value: Any) -> np.ndarray:
    """A payload from either backend as a numpy array."""

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value.numpy())


def _relu_op(trace: Any) -> Any:
    """The one relu op of a trace (tinygrad records relu as ``where``)."""

    ops = [op for op in trace._layer_list if op.func_name in {"relu", "where"}]
    assert len(ops) == 1, [op.func_name for op in trace._layer_list]
    return ops[0]


def _final_output(trace: Any) -> np.ndarray:
    """The payload of the trace's only output op."""

    return _np(trace[trace.output_layers[0]].out)


def _edit_cases() -> list[tuple[str, Any, Callable[[np.ndarray], np.ndarray]]]:
    """``(id, edit, expected relu payload from the plain one)`` per supported edit."""

    replacement = np.full((2, 3), 0.5, dtype=np.float32)
    return [
        ("zero_ablate", tl.zero_ablate(), np.zeros_like),
        ("scale", tl.scale(0.5), lambda out: out * 0.5),
        ("add", tl.add(1.0), lambda out: out + 1.0),
        ("clamp", tl.clamp(low=0.0, high=0.5), lambda out: np.clip(out, 0.0, 0.5)),
        ("replace_with", "replace", lambda out: replacement),
    ]


def _backend_edit(edit: Any, backend: str) -> Any:
    """The edit for one backend; ``replace`` builds a backend-native replacement tensor."""

    if edit != "replace":
        return edit
    value = np.full((2, 3), 0.5, dtype=np.float32)
    return tl.replace_with(Tensor(value) if backend == "tinygrad" else torch.tensor(value))


@pytest.mark.parametrize(
    ("edit", "expected"),
    [pytest.param(edit, expected, id=name) for name, edit, expected in _edit_cases()],
)
def test_edited_capture_matches_torch_twin_and_validates(
    edit: Any, expected: Callable[[np.ndarray], np.ndarray]
) -> None:
    """The edited relu and the downstream output equal torch's; the trace validates."""

    model, twin = _TinyMLP(), _TorchTwin()
    plain = tl.trace(model, Tensor(_X), backend="tinygrad")
    edited = tl.trace(
        model,
        Tensor(_X),
        backend="tinygrad",
        intervene=tl.when(tl.func("where"), _backend_edit(edit, "tinygrad")),
    )
    torch_edited = tl.trace(
        twin,
        torch.tensor(_X),
        intervene=tl.when(tl.func("relu"), _backend_edit(edit, "torch")),
    )

    relu = _relu_op(edited)
    assert relu.is_edited is True
    assert relu.intervention_fired is True
    np.testing.assert_allclose(_np(relu.out), expected(_np(_relu_op(plain).out)), atol=1e-6)
    np.testing.assert_allclose(_np(relu.out), _np(_relu_op(torch_edited).out), atol=1e-5)
    np.testing.assert_allclose(_final_output(edited), _final_output(torch_edited), atol=1e-5)
    assert [op.label for op in edited._layer_list] == [op.label for op in plain._layer_list]
    assert [op.label for op in edited._layer_list if op.is_edited] == [relu.label]
    assert TinygradBackend().validate_trace(edited) is True


def test_callable_transform_edit_matches_torch_twin() -> None:
    """A user callable transform edits the matched op on both backends alike."""

    def _negate(out: Any, *, ctx: object) -> Any:
        """Return the negated activation."""

        del ctx
        return out * -1.0

    edited = tl.trace(
        _TinyMLP(), Tensor(_X), backend="tinygrad", intervene=tl.when(tl.func("where"), _negate)
    )
    torch_edited = tl.trace(
        _TorchTwin(), torch.tensor(_X), intervene=tl.when(tl.func("relu"), _negate)
    )

    np.testing.assert_allclose(_np(_relu_op(edited).out), _np(_relu_op(torch_edited).out))
    np.testing.assert_allclose(_final_output(edited), _final_output(torch_edited), atol=1e-5)
    assert TinygradBackend().validate_trace(edited) is True


def test_edit_hides_its_own_ops_and_keeps_the_site_parents() -> None:
    """The edit's ops are not recorded: the edited op keeps the plain op's parents."""

    plain = tl.trace(_TinyMLP(), Tensor(_X), backend="tinygrad")
    edited = tl.trace(
        _TinyMLP(),
        Tensor(_X),
        backend="tinygrad",
        intervene=tl.when(tl.func("where"), tl.scale(3.0)),
    )

    assert len(edited._layer_list) == len(plain._layer_list)
    assert [op.func_name for op in edited._layer_list] == [op.func_name for op in plain._layer_list]
    assert list(_relu_op(edited).parents) == list(_relu_op(plain).parents)


def test_stripping_the_edit_declaration_fails_validation() -> None:
    """An edited payload presented as a plain captured value FAILS replay validation."""

    trace = tl.trace(
        _TinyMLP(),
        Tensor(_X),
        backend="tinygrad",
        intervene=tl.when(tl.func("where"), tl.scale(0.5)),
    )
    captures = trace.tinygrad_uop_captures
    index = next(i for i, capture in enumerate(captures) if capture.edit is not None)
    captures[index] = dataclasses.replace(captures[index], edit=None)

    assert TinygradBackend().validate_trace(trace) is not True


def test_edited_capture_realizes_once_per_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain capture realizes once; an edited one realizes once in each of its two runs."""

    from torchlens.backends._protocols.tinygrad import TinygradProtocol

    calls: list[int] = []
    original_once = TinygradProtocol.realize_once

    def counting_once(self: Any, values: Any) -> None:
        calls.append(len(values))
        original_once(self, values)

    monkeypatch.setattr(TinygradProtocol, "realize_once", counting_once)
    tl.trace(_TinyMLP(), Tensor(_X), backend="tinygrad")
    assert len(calls) == 1
    calls.clear()
    tl.trace(
        _TinyMLP(),
        Tensor(_X),
        backend="tinygrad",
        intervene=tl.when(tl.func("where"), tl.zero_ablate()),
    )
    assert len(calls) == 2


def test_plain_capture_is_unchanged_by_an_earlier_edited_capture() -> None:
    """Payloads of a plain capture are the same before and after an edited capture."""

    model = _TinyMLP()
    before = [_np(op.out) for op in tl.trace(model, Tensor(_X), backend="tinygrad")._layer_list]
    tl.trace(
        model, Tensor(_X), backend="tinygrad", intervene=tl.when(tl.func("where"), tl.add(2.0))
    )
    after = tl.trace(model, Tensor(_X), backend="tinygrad")

    assert len(before) == len(after._layer_list)
    for expected, op in zip(before, after._layer_list):
        np.testing.assert_array_equal(_np(op.out), expected)
        assert op.is_edited is False
    assert TinygradBackend().validate_trace(after) is True


def test_edit_on_repeated_identical_calls_edits_every_call() -> None:
    """tinygrad interns identical UOps; an edit at that op applies at every call returning it."""

    def twice(x: Any) -> Any:
        """Two identical relu calls summed."""

        return x.relu() + x.relu()

    edited = tl.trace(
        twice, Tensor(_X), backend="tinygrad", intervene=tl.when(tl.func("where"), tl.zero_ablate())
    )

    np.testing.assert_allclose(_final_output(edited), np.zeros_like(_X))
    assert TinygradBackend().validate_trace(edited) is True


def test_zero_matches_warns_and_returns_the_plain_capture() -> None:
    """A selector matching nothing warns like torch and returns the plain trace."""

    with pytest.warns(UserWarning, match="matched zero sites"):
        trace = tl.trace(
            _TinyMLP(),
            Tensor(_X),
            backend="tinygrad",
            intervene=tl.when(tl.func("conv2d"), tl.zero_ablate()),
        )
    assert not any(op.is_edited for op in trace._layer_list)


def test_value_dependent_edit_refuses_typed() -> None:
    """A value-dependent predicate stays refused (lazy forward has no values yet)."""

    with pytest.raises(BackendUnsupportedError, match="value-dependent") as info:
        tl.trace(
            _TinyMLP(),
            Tensor(_X),
            backend="tinygrad",
            intervene=tl.when(lambda op: op.out is not None, tl.zero_ablate()),
        )
    assert info.value.code == "interventions_unsupported"


def test_stop_after_refuses_typed() -> None:
    """stop_after stays refused on tinygrad."""

    with pytest.raises(BackendUnsupportedError, match="stop_after") as info:
        tl.trace(_TinyMLP(), Tensor(_X), backend="tinygrad", stop_after=tl.func("where"))
    assert info.value.code == "interventions_unsupported"


def test_grad_options_with_edit_refuses_typed() -> None:
    """Derived gradients cannot be combined with an edit."""

    with pytest.raises(BackendUnsupportedError, match="grad_options") as info:
        TinygradBackend().capture_trace(
            lambda x: (x * x).sum(),
            Tensor([1.0, 2.0]),
            grad_options=GradOptions(input_grad_argnums=(0,)),
            intervene=tl.when(tl.func("mul"), tl.zero_ablate()),
        )
    assert info.value.code == "intervention_with_derived_grads_unsupported"


def test_shape_changing_edit_refuses_typed() -> None:
    """An edit must keep the edited op's shape and dtype."""

    def _sum(out: Any, *, ctx: object) -> Any:
        """Return a scalar in place of the activation."""

        del ctx
        return out.sum()

    with pytest.raises(BackendUnsupportedError, match="shape and dtype"):
        tl.trace(
            _TinyMLP(), Tensor(_X), backend="tinygrad", intervene=tl.when(tl.func("where"), _sum)
        )


def test_lazy_random_weights_refuse_an_edit() -> None:
    """Unrealized random weights would be drawn twice by the two runs; refuse typed."""

    from tinygrad import nn

    Tensor.manual_seed(0)
    layer = nn.Linear(4, 3)
    with pytest.raises(BackendUnsupportedError, match="random") as info:
        tl.trace(
            lambda value: layer(value).relu(),
            Tensor(_X),
            backend="tinygrad",
            intervene=tl.when(tl.func("where"), tl.zero_ablate()),
        )
    assert info.value.code == "rng_state_in_graph"


def test_flag_false_refuses_the_edit_surface() -> None:
    """The capability table stays load-bearing: can_intervene=False refuses."""

    spec = tl.backends.spec("tinygrad")
    object.__setattr__(spec.capabilities, "can_intervene", False)
    try:
        with pytest.raises(BackendUnsupportedError, match="can_intervene=False"):
            tl.trace(
                _TinyMLP(),
                Tensor(_X),
                backend="tinygrad",
                intervene=tl.when(tl.func("where"), tl.zero_ablate()),
            )
    finally:
        object.__setattr__(spec.capabilities, "can_intervene", True)


def _keyword_block(x: Any, *, scale: Any, shift: Any = None) -> Any:
    """A forward taking tensor keyword inputs."""

    out = (x * scale).relu()
    return out + shift if shift is not None else out


@pytest.mark.parametrize("route", ["input_kwargs", "forward_kwargs"])
def test_keyword_inputs_capture_and_validate(route: str) -> None:
    """Tensor keyword inputs become trace inputs and the trace validates."""

    x, scale, shift = (
        Tensor(_X),
        Tensor(np.full(4, 2.0, np.float32)),
        Tensor(np.ones(4, np.float32)),
    )
    kwargs = {"scale": scale, "shift": shift}
    if route == "input_kwargs":
        trace = tl.trace(_keyword_block, x, backend="tinygrad", input_kwargs=kwargs)
    else:
        trace = tl.trace(_keyword_block, x, backend="tinygrad", **kwargs)

    inputs = [op for op in trace._layer_list if op.is_input]
    assert len(inputs) == 3
    expected = _keyword_block(Tensor(_X), scale=scale, shift=shift).numpy()
    np.testing.assert_allclose(_final_output(trace), expected)
    assert TinygradBackend().validate_trace(trace) is True


def _source_frames(trace: Any, func_name: str) -> list[Any]:
    """The source frames of the first op named ``func_name``."""

    op = next(op for op in trace._layer_list if op.func_name == func_name)
    return list(op.source.frames)


def test_source_lines_point_at_user_code() -> None:
    """Each op's code_context runs from the trace call site down to the model line."""

    model = _TinyMLP()
    call_line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
    trace = tl.trace(model, Tensor(_X), backend="tinygrad")

    frames = _source_frames(trace, "where")
    assert frames, "the relu op has no code_context"
    here = str(Path(__file__).resolve())
    assert all(str(Path(frame.file).resolve()) == here for frame in frames)
    assert frames[0].line_number == call_line
    model_line = inspect.getsourcelines(_TinyMLP.__call__)[1] + 3
    assert frames[-1].line_number == model_line
    assert frames[-1].func_name == "__call__"
    assert "relu" in frames[-1].call_line


def test_source_lines_match_the_torch_frame_shape() -> None:
    """tinygrad and torch both list the call site first and the model line last."""

    tg = _source_frames(tl.trace(_TinyMLP(), Tensor(_X), backend="tinygrad"), "where")
    pt = _source_frames(tl.trace(_TorchTwin(), torch.tensor(_X)), "relu")

    assert tg[0].func_name == pt[0].func_name == "test_source_lines_match_the_torch_frame_shape"
    assert {tg[-1].func_name, pt[-1].func_name} == {"__call__", "forward"}


def test_save_source_context_snapshots_source() -> None:
    """save_source_context=True is accepted and the lines carry their source text."""

    trace = tl.trace(
        _TinyMLP(),
        Tensor(_X),
        backend="tinygrad",
        capture=tl.options.CaptureOptions(save_source_context=True),
    )
    frames = _source_frames(trace, "where")
    assert frames and "relu" in frames[-1].call_line
