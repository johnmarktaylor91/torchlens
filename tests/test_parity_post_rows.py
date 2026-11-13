"""Backend-neutral post-processing matches torch on a twin MLP (rows S2, S3, S4, X1).

Every non-torch backend captures the same two-layer MLP with identical weights.
The value-dependent and lookback ``save=`` predicates, the save-shaping options
and the op-repr value statistics must keep, shape and report exactly what torch
does on its twin. Payloads are compared through numpy fingerprints because op
labels and function names differ between frameworks.

tinygrad lowers ``relu`` to ``where(cmplt(zeros, x), x, zeros)`` and a matmul to
reshape/expand/mul/sum, so its graph is not a structural twin of torch's: it
holds extra hidden-shaped payloads (the comparison mask, the expanded zeros)
and no output-shaped op has a hidden-shaped parent. Its cells compare against
torch's rule applied to tinygrad's own full-save graph, computed here from
public op fields, and still require every torch payload the lowered graph can
hold.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens._data_classes._value_state import payload_facts, value_words
from torchlens.intervention.errors import MultiMatchWarning
from torchlens.options import CaptureOptions

_X = np.arange(8, dtype=np.float32).reshape(2, 4) / 8.0 - 0.3
_W1 = (np.arange(20, dtype=np.float32).reshape(5, 4) / 10.0 - 1.0).astype(np.float32)
_W2 = (np.arange(15, dtype=np.float32).reshape(3, 5) / 7.0 - 1.0).astype(np.float32)
_HIDDEN = (2, 5)
_OUTPUT = (2, 3)
_MODEL_SHAPES = frozenset({_HIDDEN, _OUTPUT})


def _torch_trace(**kwargs: Any) -> Any:
    """Capture the reference MLP on torch."""

    import torch

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 5, bias=False), torch.nn.ReLU(), torch.nn.Linear(5, 3, bias=False)
    ).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(_W1))
        model[2].weight.copy_(torch.from_numpy(_W2))
    return tl.trace(model, torch.from_numpy(_X), **kwargs)


def _jax_trace(**kwargs: Any) -> Any:
    """Capture the MLP as a JAX function with explicit weight leaves."""

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def model(x: Any, w1: Any, w2: Any) -> Any:
        """Two dense layers with a rectifier between them."""

        return jnp.matmul(jax.nn.relu(jnp.matmul(x, w1)), w2)

    inputs = (jnp.asarray(_X), jnp.asarray(_W1.T), jnp.asarray(_W2.T))
    return tl.trace(model, inputs, backend="jax", **kwargs)


def _tf_trace(**kwargs: Any) -> Any:
    """Capture the MLP as a Keras model."""

    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential(
        [
            tf.keras.Input((4,)),
            tf.keras.layers.Dense(5, use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(3, use_bias=False),
        ]
    )
    model.layers[0].set_weights([_W1.T])
    model.layers[2].set_weights([_W2.T])
    return tl.trace(model, tf.constant(_X), backend="tf", **kwargs)


def _mlx_trace(**kwargs: Any) -> Any:
    """Capture the MLP as an MLX module."""

    if sys.platform != "darwin" and os.environ.get("TORCHLENS_RUN_UNSTABLE_MLX_TESTS") != "1":
        pytest.skip("MLX off macOS runs only with TORCHLENS_RUN_UNSTABLE_MLX_TESTS=1")
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class Net(nn.Module):
        """Two bias-free linear layers around a rectifier."""

        def __init__(self) -> None:
            """Install the reference weights."""

            super().__init__()
            self.fc1 = nn.Linear(4, 5, bias=False)
            self.fc2 = nn.Linear(5, 3, bias=False)
            self.fc1.weight = mx.array(_W1)
            self.fc2.weight = mx.array(_W2)

        def __call__(self, x: Any) -> Any:
            """Return the MLP output."""

            return self.fc2(nn.relu(self.fc1(x)))

    model = Net()
    mx.eval(model.parameters())
    return tl.trace(model, mx.array(_X), backend="mlx", **kwargs)


def _paddle_trace(**kwargs: Any) -> Any:
    """Capture the MLP as a Paddle layer stack."""

    paddle = pytest.importorskip("paddle")
    fc1 = paddle.nn.Linear(4, 5, bias_attr=False)
    fc2 = paddle.nn.Linear(5, 3, bias_attr=False)
    fc1.weight.set_value(paddle.to_tensor(_W1.T))
    fc2.weight.set_value(paddle.to_tensor(_W2.T))
    model = paddle.nn.Sequential(fc1, paddle.nn.ReLU(), fc2)
    model.eval()
    return tl.trace(model, paddle.to_tensor(_X), backend="paddle", **kwargs)


def _tinygrad_trace(**kwargs: Any) -> Any:
    """Capture the MLP as a tinygrad object."""

    tinygrad = pytest.importorskip("tinygrad")
    nn = pytest.importorskip("tinygrad.nn")

    class Net:
        """Two bias-free linear layers around a rectifier."""

        def __init__(self) -> None:
            """Install the reference weights."""

            self.fc1 = nn.Linear(4, 5, bias=False)
            self.fc2 = nn.Linear(5, 3, bias=False)
            self.fc1.weight = tinygrad.Tensor(_W1).realize()
            self.fc2.weight = tinygrad.Tensor(_W2).realize()

        def __call__(self, x: Any) -> Any:
            """Return the MLP output."""

            return self.fc2(self.fc1(x).relu())

    return tl.trace(Net(), tinygrad.Tensor(_X).realize(), backend="tinygrad", **kwargs)


#: Backends whose graphs lower torch's ops into several primitives (module docstring).
_LOWERED = (_tinygrad_trace,)

_STRUCTURAL_TWINS = [
    pytest.param(_jax_trace, marks=pytest.mark.backend_jax, id="jax"),
    pytest.param(_tf_trace, marks=pytest.mark.tf_backend, id="tf"),
    pytest.param(_mlx_trace, marks=pytest.mark.backend_mlx, id="mlx"),
    pytest.param(_paddle_trace, marks=pytest.mark.backend_paddle, id="paddle"),
]
_CAPTURES = [
    *_STRUCTURAL_TWINS,
    pytest.param(_tinygrad_trace, marks=pytest.mark.backend_tinygrad, id="tinygrad"),
]


def _host(value: Any) -> np.ndarray:
    """Return a payload as a float64 numpy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu") and not hasattr(value, "numpy"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _fingerprint(value: Any) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return a hashable shape and rounded-value fingerprint for one payload."""

    array = _host(value)
    return tuple(array.shape), tuple(np.round(array.ravel(), 4).tolist())


def _model_payloads(trace: Any, field: str = "out", ops: Any = None) -> list[Any]:
    """Return every public model-activation payload in ``field`` (of ``ops`` when given).

    Only the twin's activation shapes count: frameworks that record weight reads
    as ops (TensorFlow variable reads) have payloads torch's parameters never do.
    """

    payloads = []
    for op in trace._layer_list if ops is None else ops:
        if op.is_input or not op.has_saved_activation:
            continue
        value = getattr(op, field, None)
        if value is not None and _fingerprint(value)[0] in _MODEL_SHAPES:
            payloads.append(value)
    return payloads


def _kept(trace: Any, field: str = "out") -> set[tuple[tuple[int, ...], tuple[float, ...]]]:
    """Return the fingerprints of every public model-activation payload in ``field``."""

    return {_fingerprint(value) for value in _model_payloads(trace, field)}


def _assert_validates(trace: Any) -> None:
    """Replay-validate a filtered or shaped twin trace (hidden raw copies included)."""

    status = trace.validate_forward_pass([])
    # MLX returns a ValidationStatus record; the other backends return a bool.
    assert getattr(status, "passed", status) is True


def _op_kind(op: Any) -> str:
    """Return the capture-event kind torch's predicate history distinguishes."""

    for kind in ("output", "input", "buffer"):
        if bool(getattr(op, f"is_{kind}", False)):
            return kind
    return "op"


def _parent_ops(trace: Any) -> dict[int, list[Any]]:
    """Map each op (by identity) to its parent ops, resolving labels with or without pass."""

    by_label: dict[str, Any] = {}
    for op in trace._layer_list:
        for label in (str(op.label), str(op.label).rsplit(":", 1)[0]):
            by_label.setdefault(label, op)
    return {id(op): [by_label[str(p)] for p in op.parents or ()] for op in trace._layer_list}


def _is_hidden(op: Any) -> bool:
    """Op-side twin of ``_hidden_shape``."""

    return tuple(op.shape or ()) == _HIDDEN


def _lowered_keep(trace: Any, window: int, successor: Callable[[Any], bool] | None) -> list[Any]:
    """Apply torch's lookback rules to a finished full-save graph, from public op fields.

    The history holds the last ``window`` op-kind events. ``tl.preceded_by(P)``
    (``successor`` is ``None``) keeps an op when a parent inside the window
    matches ``P`` (any windowed op when it has no parents);
    ``candidate & tl.followed_by(S)`` keeps each windowed parent matching the
    candidate of every op-kind event matching ``S``. ``P`` and the candidate
    are the hidden-shape predicate.
    """

    parents = _parent_ops(trace)
    history: list[Any] = []
    keep: list[Any] = []
    for op in trace._layer_list:
        recent = history[-window:]
        windowed = [parent for parent in parents[id(op)] if any(parent is r for r in recent)]
        if successor is None:
            pool = windowed if parents[id(op)] else recent
            if any(_is_hidden(prior) for prior in pool):
                keep.append(op)
        elif _op_kind(op) == "op" and successor(op):
            keep.extend(parent for parent in windowed if _is_hidden(parent))
        if _op_kind(op) == "op":
            history.append(op)
    return keep


def _run(capture_fn: Callable[..., Any], monkeypatch: Any, /, **kwargs: Any) -> tuple[Any, Any]:
    """Capture the backend twin and the torch reference with the same options."""

    monkeypatch.setenv("DEV", "PYTHON")
    return capture_fn(**kwargs), _torch_trace(**kwargs)


def _hidden_shape(ctx: Any) -> bool:
    """Value-dependent predicate: the op output has the hidden-layer shape."""

    return tuple(ctx.shape or ()) == _HIDDEN


def _output_shape(ctx: Any) -> bool:
    """Value-dependent predicate: the op output has the model-output shape."""

    return tuple(ctx.shape or ()) == _OUTPUT


@pytest.mark.rare
@pytest.mark.backend_parity
@pytest.mark.parametrize("capture", _CAPTURES)
@pytest.mark.parametrize(
    "save",
    [
        pytest.param(_hidden_shape, id="callable"),
        pytest.param(tl.where(_hidden_shape), id="where"),
    ],
)
def test_s2_value_dependent_save_keeps_what_torch_keeps(
    capture: Callable[..., Any], save: Any, monkeypatch: Any
) -> None:
    """S2: a value-dependent ``save=`` predicate keeps the torch twin's payloads."""

    trace, reference = _run(capture, monkeypatch, save=save)
    expected = _kept(reference)
    assert expected, "torch twin kept nothing; the predicate does not exercise S2"
    if capture in _LOWERED:
        full = capture()
        own = {_fingerprint(v) for v in _model_payloads(full) if _fingerprint(v)[0] == _HIDDEN}
        assert expected <= own, "the lowered graph lost a payload torch keeps"
        expected = own
    assert _kept(trace) == expected
    _assert_validates(trace)


@pytest.mark.rare
@pytest.mark.backend_parity
@pytest.mark.parametrize("capture", _STRUCTURAL_TWINS)
@pytest.mark.parametrize(
    ("save", "capture_options"),
    [
        pytest.param(tl.preceded_by(tl.where(_hidden_shape)), None, id="preceded_by"),
        pytest.param(
            tl.preceded_by(tl.where(_hidden_shape)),
            CaptureOptions(lookback=2),
            id="preceded_by_lookback",
        ),
        pytest.param(
            tl.where(_hidden_shape) & tl.followed_by(tl.where(_output_shape)),
            CaptureOptions(lookback=4, lookback_payload_policy="detached_raw"),
            id="followed_by",
        ),
    ],
)
def test_s3_lookback_selectors_keep_what_torch_keeps(
    capture: Callable[..., Any], save: Any, capture_options: Any, monkeypatch: Any
) -> None:
    """S3: ``tl.preceded_by`` / ``tl.followed_by`` windows match the torch twin."""

    kwargs: dict[str, Any] = {"save": save}
    if capture_options is not None:
        kwargs["capture"] = capture_options
    trace, reference = _run(capture, monkeypatch, **kwargs)
    expected = _kept(reference)
    assert expected, "torch twin kept nothing; the selector does not exercise S3"
    assert _kept(trace) == expected
    _assert_validates(trace)


def _rank3(ctx: Any) -> bool:
    """tinygrad successor: the rank-3 reshape its matmul starts with."""

    return len(tuple(ctx.shape or ())) == 3


@pytest.mark.rare
@pytest.mark.backend_parity
@pytest.mark.backend_tinygrad
@pytest.mark.parametrize(
    ("save", "window", "successor"),
    [
        pytest.param(tl.preceded_by(tl.where(_hidden_shape)), 8, None, id="preceded_by"),
        pytest.param(tl.preceded_by(tl.where(_hidden_shape)), 2, None, id="preceded_by_lookback"),
        pytest.param(
            tl.where(_hidden_shape) & tl.followed_by(tl.where(_rank3)),
            4,
            _rank3,
            id="followed_by",
        ),
    ],
)
def test_s3_lookback_selectors_on_the_lowered_tinygrad_graph(
    save: Any, window: int, successor: Any, monkeypatch: Any
) -> None:
    """S3 on tinygrad: torch's lookback rules applied to tinygrad's own lowered graph.

    tinygrad's matmul is reshape/expand/mul/sum, so no output-shaped op has a
    hidden-shaped parent and torch's ``preceded_by`` cannot select the model
    output there; the successor for ``followed_by`` is the matmul's first
    reshape, whose parent is the rectified hidden activation torch keeps.
    """

    monkeypatch.setenv("DEV", "PYTHON")
    kwargs: dict[str, Any] = {"save": save}
    if window != 8:  # 8 is torch's history when lookback is left at 0
        policy = "detached_raw" if successor is not None else "metadata_only"
        kwargs["capture"] = CaptureOptions(lookback=window, lookback_payload_policy=policy)
    trace = _tinygrad_trace(**kwargs)
    full = _tinygrad_trace()
    keep = _lowered_keep(full, window, successor)
    expected = {_fingerprint(value) for value in _model_payloads(full, ops=keep)}
    parents = _parent_ops(full)
    assert not any(
        tuple(op.shape or ()) == _OUTPUT and any(_is_hidden(p) for p in parents[id(op)])
        for op in full._layer_list
    ), "an output-shaped op has a hidden-shaped parent; the torch twin applies again"
    relu = _fingerprint(np.maximum(_X @ _W1.T, 0.0))
    assert relu in expected, "the lowered graph must still keep torch's rectified payload"
    assert _kept(trace) == expected
    _assert_validates(trace)


def _double(tensor: Any) -> Any:
    """Activation transform used by the shaping twins."""

    return tensor * 2


@pytest.mark.rare
@pytest.mark.backend_parity
@pytest.mark.parametrize("capture", _CAPTURES)
@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"activation_transform": _double}, id="transform"),
        pytest.param(
            {"activation_transform": _double, "save_raw_activations": False},
            id="transform_no_raw",
        ),
        pytest.param({"output_device": "cpu"}, id="output_device"),
        pytest.param({"detach_saved_activations": True}, id="detach"),
    ],
)
def test_s4_save_shaping_matches_torch(
    capture: Callable[..., Any], options: dict[str, Any], monkeypatch: Any
) -> None:
    """S4: save shaping yields the torch twin's raw and transformed payloads."""

    trace, reference = _run(capture, monkeypatch, capture=CaptureOptions(**options))
    expected = {field: _kept(reference, field) for field in ("out", "transformed_out")}
    if capture in _LOWERED:
        expected = _lowered_shaping(capture(), options, expected)
    for field, prints in expected.items():
        assert _kept(trace, field) == prints, field
    assert trace._options_save_raw_activations == reference._options_save_raw_activations
    _assert_validates(trace)


def _lowered_shaping(full: Any, options: dict[str, Any], torch_sets: dict[str, Any]) -> Any:
    """Return the raw and transformed sets torch's save path gives a lowered full graph.

    Raw payloads stay public unless a transform runs without
    ``save_raw_activations``; the transform runs on every saved payload. Every
    payload torch keeps must be among them.
    """

    transform = options.get("activation_transform")
    raw = _model_payloads(full)
    expected = {
        "out": set()
        if transform is not None and options.get("save_raw_activations") is False
        else {_fingerprint(v) for v in raw},
        "transformed_out": set()
        if transform is None
        else {_fingerprint(transform(_host(v))) for v in raw},
    }
    for field, prints in torch_sets.items():
        assert prints <= expected[field], f"the lowered graph lost a torch {field} payload"
    return expected


@pytest.mark.rare
@pytest.mark.backend_parity
@pytest.mark.parametrize("capture", _CAPTURES)
def test_s4_layer_visualizers_render_the_torch_payloads(
    capture: Callable[..., Any], monkeypatch: Any, tmp_path: Any
) -> None:
    """S4: ``layer_visualizers`` render the same payloads as on the torch twin."""

    seen: dict[str, list[Any]] = {"twin": [], "torch": []}

    def visualizer(sink: list[Any]) -> Callable[..., Any]:
        """Return a visualizer that records each payload fingerprint."""

        def render(tensor: Any, *, layer_label: Any = None) -> str:
            sink.append(_fingerprint(tensor))
            return "<p>ok</p>"

        return render

    monkeypatch.setenv("DEV", "PYTHON")
    monkeypatch.chdir(tmp_path)
    selector = tl.where(lambda op: True)
    # A selector over every op resolves many sites; the resolver discloses that.
    with pytest.warns(MultiMatchWarning):
        twin_trace = capture(
            capture=CaptureOptions(layer_visualizers={selector: visualizer(seen["twin"])})
        )
    with pytest.warns(MultiMatchWarning):
        _torch_trace(
            capture=CaptureOptions(layer_visualizers={selector: visualizer(seen["torch"])})
        )
    twin = {print_ for print_ in seen["twin"] if print_[0] in _MODEL_SHAPES}
    reference = {print_ for print_ in seen["torch"] if print_[0] in _MODEL_SHAPES}
    assert reference
    if capture in _LOWERED:
        assert reference <= twin
        reference = _kept(twin_trace)
    assert twin == reference
    rendered = [op for op in twin_trace._layer_list if getattr(op, "visualizer_path", None)]
    assert rendered and all(os.path.exists(op.visualizer_path) for op in rendered)


@pytest.mark.rare
@pytest.mark.backend_parity
@pytest.mark.parametrize("capture", _CAPTURES)
def test_s4_save_args_keeps_typed_refusal(capture: Callable[..., Any], monkeypatch: Any) -> None:
    """S4: ``save_args`` stays a typed refusal (non-torch captures record no call args)."""

    from torchlens.errors import BackendUnsupportedError

    monkeypatch.setenv("DEV", "PYTHON")
    with pytest.raises(BackendUnsupportedError):
        capture(capture=CaptureOptions(save_args=True))


def _last_output_shaped(trace: Any) -> Any:
    """Return the last saved op carrying the model-output activation."""

    return [
        op
        for op in trace._layer_list
        if op.has_saved_activation and tuple(op.shape or ()) == _OUTPUT and op.out is not None
    ][-1]


@pytest.mark.rare
@pytest.mark.backend_parity
@pytest.mark.parametrize("capture", _CAPTURES)
def test_x1_op_repr_value_statistics_match_torch(
    capture: Callable[..., Any], monkeypatch: Any
) -> None:
    """X1: op reprs carry torch's value statistics, computed through the payload codec."""

    trace, reference = _run(capture, monkeypatch)
    twin_op = _last_output_shaped(trace)
    torch_op = _last_output_shaped(reference)
    twin_stats = payload_facts(twin_op)[1]
    torch_stats = payload_facts(torch_op)[1]
    assert twin_stats is not None
    assert value_words(twin_stats) == value_words(torch_stats)
    assert "preview backend" not in repr(twin_op)
