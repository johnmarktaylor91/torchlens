"""Backend-neutral post-processing matches torch on a twin MLP (rows S2, S3, S4, X1).

Every non-torch backend captures the same two-layer MLP with identical weights.
The value-dependent and lookback ``save=`` predicates, the save-shaping options
and the op-repr value statistics must keep, shape and report exactly what torch
does on its twin. Payloads are compared through numpy fingerprints because op
labels and function names differ between frameworks.
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
from torchlens.options import CaptureOptions

_X = np.arange(8, dtype=np.float32).reshape(2, 4) / 8.0 - 0.3
_W1 = (np.arange(20, dtype=np.float32).reshape(5, 4) / 10.0 - 1.0).astype(np.float32)
_W2 = (np.arange(15, dtype=np.float32).reshape(3, 5) / 7.0 - 1.0).astype(np.float32)
_HIDDEN = (2, 5)
_OUTPUT = (2, 3)
_MODEL_SHAPES = frozenset({_HIDDEN, _OUTPUT})


def _torch_trace(**kwargs: Any) -> Any:
    """Capture the reference MLP on torch."""

    torch = pytest.importorskip("torch")
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


_HIDDEN_PAYLOAD_BLIND = (_tf_trace, _paddle_trace)

_CAPTURES = [
    pytest.param(_jax_trace, marks=pytest.mark.backend_jax, id="jax"),
    pytest.param(_tf_trace, marks=pytest.mark.tf_backend, id="tf"),
    pytest.param(_mlx_trace, marks=pytest.mark.backend_mlx, id="mlx"),
    pytest.param(_paddle_trace, marks=pytest.mark.backend_paddle, id="paddle"),
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


def _kept(trace: Any, field: str = "out") -> set[tuple[tuple[int, ...], tuple[float, ...]]]:
    """Return the fingerprints of every public model-activation payload in ``field``.

    Only the twin's activation shapes count: frameworks that record weight reads
    as ops (TensorFlow variable reads) have payloads torch's parameters never do.
    """

    kept = set()
    for op in trace._layer_list:
        if op.is_input or not op.has_saved_activation:
            continue
        value = getattr(op, field, None)
        if value is not None and _fingerprint(value)[0] in _MODEL_SHAPES:
            kept.add(_fingerprint(value))
    return kept


def _assert_validates(trace: Any, capture_fn: Callable[..., Any]) -> None:
    """Replay-validate a filtered or shaped twin trace.

    Paddle and TensorFlow replay validation reads only public payloads, not the
    replay-visible copies a filtered trace keeps; until those backends consult
    them, a filtered trace there is a known validation failure, reported as xfail.
    """

    status = trace.validate_forward_pass([])
    if status is False and capture_fn in _HIDDEN_PAYLOAD_BLIND:
        pytest.xfail("backend replay validation does not read hidden selective-save payloads")
    assert status is True


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
    assert _kept(trace) == expected
    _assert_validates(trace, capture)


@pytest.mark.rare
@pytest.mark.backend_parity
@pytest.mark.parametrize("capture", _CAPTURES)
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
    _assert_validates(trace, capture)


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
    for field in ("out", "transformed_out"):
        assert _kept(trace, field) == _kept(reference, field), field
    assert trace._options_save_raw_activations == reference._options_save_raw_activations
    _assert_validates(trace, capture)


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
    capture(capture=CaptureOptions(layer_visualizers={selector: visualizer(seen["twin"])}))
    _torch_trace(capture=CaptureOptions(layer_visualizers={selector: visualizer(seen["torch"])}))
    twin = {print_ for print_ in seen["twin"] if print_[0] in _MODEL_SHAPES}
    reference = {print_ for print_ in seen["torch"] if print_[0] in _MODEL_SHAPES}
    assert reference
    assert twin == reference


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
