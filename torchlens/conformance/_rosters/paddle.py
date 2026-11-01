"""Paddle conformance roster: the probe's models plus Conv1D/Conv3D rows and two
pretrained ``paddle.vision`` families.

Every ``config_built`` row fixes its weights from NumPy arrays defined here, so
``expected_intermediates`` is computed with NumPy from the row's own
definition, never from a capture. ``expected_op_kinds`` lists every op kind
(``Op.func_name``) one forward performs, in execution order, including the
``constant`` ops the constants rule records for closure tensors.

Ecosystem rows need downloaded weights and run only under
``TORCHLENS_CONFORMANCE_ECOSYSTEM=1`` (the suite reports them ``not_run``
otherwise). Paddle is imported only inside builders.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .._adapters import RosterModel
from . import ecosystem_enabled

#: Roster groups (SPEC 4.4).
GROUP_CONFIG_BUILT = "config_built"
GROUP_ECOSYSTEM = "ecosystem"

_X24 = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 4)
_W44 = np.linspace(-0.5, 0.5, 16, dtype=np.float32).reshape(4, 4)
_B4 = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
_IMG = np.linspace(-1.0, 1.0, 36, dtype=np.float32).reshape(1, 1, 6, 6)
_VOL = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(1, 1, 4, 4, 4)
_SEQ = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(1, 2, 8)


def _param(shape: tuple[int, ...], scale: float = 0.3) -> np.ndarray:
    """A deterministic weight array of ``shape``."""

    count = int(np.prod(shape))
    return (np.sin(np.arange(count, dtype=np.float32) + 1.0) * scale).reshape(shape)


# ---------------------------------------------------------------- NumPy oracles


def _relu(value: np.ndarray) -> np.ndarray:
    """NumPy reference for the rectified linear unit."""

    return np.maximum(value, 0.0)


def _softmax(value: np.ndarray) -> np.ndarray:
    """NumPy reference for a numerically stable softmax along ``axis``."""

    shifted = np.exp(value - value.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def _gelu(value: np.ndarray) -> np.ndarray:
    """NumPy reference for the exact (erf) GELU."""

    erf = np.vectorize(math.erf)
    return (0.5 * value * (1.0 + erf(value / math.sqrt(2.0)))).astype(np.float32)


def _layer_norm(value: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """NumPy reference for layer normalisation without affine parameters."""

    mean = value.mean(axis=-1, keepdims=True)
    var = value.var(axis=-1, keepdims=True)
    return (value - mean) / np.sqrt(var + eps)


def _conv(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, padding: int) -> np.ndarray:
    """N-d cross-correlation, stride 1, zero padding (NC* layout)."""

    spatial = x.ndim - 2
    padded = np.pad(x, [(0, 0), (0, 0)] + [(padding, padding)] * spatial)
    window_shape: tuple[int, ...] = (1, 1, *(int(k) for k in weight.shape[2:]))
    # windows: (N, C, *out_positions, 1, 1, *kernel); the two unit window axes are summed away.
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape)
    positions, offsets = "pqr"[:spatial], "xyz"[:spatial]
    out = np.einsum(
        f"nc{positions}uv{offsets},oc{offsets}->no{positions}",
        windows.astype(np.float64),
        weight.astype(np.float64),
    )
    return (out + bias.reshape((1, -1) + (1,) * spatial)).astype(np.float32)


def _max_pool2d(value: np.ndarray, size: int) -> np.ndarray:
    """NumPy reference for a non-overlapping ``size`` x ``size`` max pool."""

    n, c, h, w = value.shape
    return value.reshape(n, c, h // size, size, w // size, size).max(axis=(3, 5))


# ------------------------------------------------------------------- row helper


def _row(**fields: Any) -> RosterModel:
    """Build a ``RosterModel``, passing only the fields it declares.

    The roster contract fields (``expected_intermediates``, ``has_constants``,
    ``random_init``, ``group``) arrive with the suite-completion packet; until
    then they are carried here and dropped at construction.
    """

    declared = {item.name for item in dataclasses.fields(RosterModel)}
    return RosterModel(**{key: value for key, value in fields.items() if key in declared})


def _set_linear(layer: Any, weight: np.ndarray, bias: np.ndarray) -> None:
    """Load fixed NumPy weights into a ``paddle.nn.Linear`` (weight is [in, out])."""

    import paddle

    layer.weight.set_value(paddle.to_tensor(weight))
    layer.bias.set_value(paddle.to_tensor(bias))


# ------------------------------------------------------------- config_built rows


def _fn_chain_branch() -> tuple[Any, Any]:
    """The fn_chain_branch row: add, multiply and relu as a plain function."""

    import paddle

    def chain(v: Any) -> Any:
        """Add one, multiply by the input and rectify."""

        a = v + 1.0
        m = a * v
        return paddle.nn.functional.relu(m)

    return chain, paddle.to_tensor(_X24)


def _fn_multi_output() -> tuple[Any, Any]:
    """The fn_multi_output row: a function returning a sum and a scaled copy."""

    import paddle

    def multi(v: Any) -> Any:
        """Return the input's sum and twice the input."""

        return v.sum(), v * 2.0

    return multi, paddle.to_tensor(_X24)


def _fn_loop3() -> tuple[Any, Any]:
    """The fn_loop3 row: three matmul, add, relu steps with closure weights."""

    import paddle

    weight = paddle.to_tensor(_W44)
    bias = paddle.to_tensor(_B4)

    def loop(v: Any) -> Any:
        """Apply the shared affine-relu step three times."""

        for _ in range(3):
            v = paddle.nn.functional.relu(paddle.matmul(v, weight) + bias)
        return v

    return loop, paddle.to_tensor(_X24)


def _loop3_intermediates() -> dict[str, list[np.ndarray]]:
    """Expected intermediates of the fn_loop3 row, one list per op kind."""

    value = _X24
    matmuls: list[np.ndarray] = []
    relus: list[np.ndarray] = []
    for _ in range(3):
        matmuls.append(value @ _W44)
        value = _relu(matmuls[-1] + _B4)
        relus.append(value)
    return {"matmul": matmuls, "functional.relu": relus}


def _fn_norm_softmax() -> tuple[Any, Any]:
    """The fn_norm_softmax row: mean-variance normalisation then softmax."""

    import paddle

    def norm_softmax(v: Any) -> Any:
        """Normalise over the last axis and apply softmax."""

        mu = paddle.mean(v, axis=-1, keepdim=True)
        var = paddle.var(v, axis=-1, unbiased=False, keepdim=True)
        n = (v - mu) / paddle.sqrt(var + 1e-5)
        return paddle.nn.functional.softmax(n, axis=-1)

    return norm_softmax, paddle.to_tensor(_X24)


def _fn_reshape_transpose() -> tuple[Any, Any]:
    """The fn_reshape_transpose row: reshape, transpose and a matmul with ones."""

    import paddle

    def reshape_t(v: Any) -> Any:
        """Reshape to 4x2, transpose and multiply by a column of ones."""

        return paddle.matmul(
            paddle.transpose(paddle.reshape(v, [4, 2]), [1, 0]), paddle.ones([4, 1])
        )

    return reshape_t, paddle.to_tensor(_X24)


def _fn_constants_planted() -> tuple[Any, Any]:
    """The fn_constants_planted row: a closure weight and a literal ``full`` constant."""

    import paddle

    weight = paddle.to_tensor(_W44)

    def planted(v: Any) -> Any:
        """Multiply by the closure weight and scale by a literal constant."""

        return paddle.matmul(v, weight) * paddle.full([1, 4], 0.5, dtype="float32")

    return planted, paddle.to_tensor(_X24)


_MLP_W1, _MLP_B1 = _param((4, 3)), _param((3,), 0.1)
_MLP_W2, _MLP_B2 = _param((3, 2), 0.4), _param((2,), 0.05)


def _mod_mlp() -> tuple[Any, Any]:
    """The mod_mlp row: a two-layer ``Sequential`` perceptron with fixed weights."""

    import paddle

    model = paddle.nn.Sequential(paddle.nn.Linear(4, 3), paddle.nn.ReLU(), paddle.nn.Linear(3, 2))
    _set_linear(model[0], _MLP_W1, _MLP_B1)
    _set_linear(model[2], _MLP_W2, _MLP_B2)
    return model, paddle.to_tensor(_X24)


_SHARED_W, _SHARED_B = _param((4, 4)), _param((4,), 0.1)


def _mod_shared_twice() -> tuple[Any, Any]:
    """The mod_shared_twice row: one linear layer applied twice with a relu between."""

    import paddle

    class Shared(paddle.nn.Layer):
        """One linear layer applied twice."""

        def __init__(self) -> None:
            super().__init__()
            self.lin = paddle.nn.Linear(4, 4)

        def forward(self, v: Any) -> Any:
            """Apply the linear layer, relu, then the same layer again."""

            return self.lin(paddle.nn.functional.relu(self.lin(v)))

    model = Shared()
    _set_linear(model.lin, _SHARED_W, _SHARED_B)
    return model, paddle.to_tensor(_X24)


_CNN_W, _CNN_B = _param((2, 1, 3, 3)), _param((2,), 0.1)
_CNN_FC_W, _CNN_FC_B = _param((18, 3), 0.2), _param((3,), 0.1)


def _mod_cnn() -> tuple[Any, Any]:
    """The mod_cnn row: conv, relu, max-pool, flatten and a linear head."""

    import paddle

    class CNN(paddle.nn.Layer):
        """Conv, relu, max-pool, flatten, linear."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = paddle.nn.Conv2D(1, 2, 3, padding=1)
            self.pool = paddle.nn.MaxPool2D(2)
            self.fc = paddle.nn.Linear(18, 3)

        def forward(self, v: Any) -> Any:
            """Convolve, rectify, pool, flatten and project."""

            h = self.pool(paddle.nn.functional.relu(self.conv(v)))
            return self.fc(paddle.flatten(h, 1))

    model = CNN()
    model.conv.weight.set_value(paddle.to_tensor(_CNN_W))
    model.conv.bias.set_value(paddle.to_tensor(_CNN_B))
    _set_linear(model.fc, _CNN_FC_W, _CNN_FC_B)
    return model, paddle.to_tensor(_IMG)


def _cnn_intermediates() -> dict[str, np.ndarray]:
    """Expected intermediates of the mod_cnn row."""

    conv = _conv(_IMG, _CNN_W, _CNN_B, padding=1)
    pooled = _max_pool2d(_relu(conv), 2)
    return {
        "conv2d": conv,
        "functional.relu": _relu(conv),
        "functional.linear": pooled.reshape(1, -1) @ _CNN_FC_W + _CNN_FC_B,
    }


_NEST_W, _NEST_B = _param((4, 4)), _param((4,), 0.1)


def _mod_nested_residual() -> tuple[Any, Any]:
    """The mod_nested_residual row: a residual gelu block followed by layer norm."""

    import paddle

    class Nested(paddle.nn.Layer):
        """Residual gelu block followed by a layer norm."""

        def __init__(self) -> None:
            super().__init__()
            self.block = paddle.nn.Sequential(paddle.nn.Linear(4, 4), paddle.nn.GELU())
            self.ln = paddle.nn.LayerNorm(4)

        def forward(self, v: Any) -> Any:
            """Add the block's output to the input and normalise."""

            return self.ln(v + self.block(v))

    model = Nested()
    _set_linear(model.block[0], _NEST_W, _NEST_B)
    return model, paddle.to_tensor(_X24)


def _nested_intermediates() -> dict[str, np.ndarray]:
    """Expected intermediates of the mod_nested_residual row."""

    hidden = _gelu(_X24 @ _NEST_W + _NEST_B)
    return {
        "functional.gelu": hidden,
        "functional.layer_norm": _layer_norm(_X24 + hidden).astype(np.float32),
    }


_CONV1_W, _CONV1_B = _param((3, 2, 3)), _param((3,), 0.1)
_CONV3_W, _CONV3_B = _param((2, 1, 3, 3, 3), 0.2), _param((2,), 0.1)


def _conv_layer_model(rank: int) -> tuple[Any, Any]:
    """A ``rank``-dimensional convolution layer with fixed weights and its input."""

    import paddle

    layer: Any
    weight: np.ndarray
    bias: np.ndarray
    value: np.ndarray
    if rank == 1:
        layer, weight, bias, value = paddle.nn.Conv1D(2, 3, 3, padding=1), _CONV1_W, _CONV1_B, _SEQ
    else:
        layer, weight, bias, value = paddle.nn.Conv3D(1, 2, 3, padding=1), _CONV3_W, _CONV3_B, _VOL
    layer.weight.set_value(paddle.to_tensor(weight))
    layer.bias.set_value(paddle.to_tensor(bias))
    model = paddle.nn.Sequential(layer, paddle.nn.ReLU())
    return model, paddle.to_tensor(value)


@dataclasses.dataclass(frozen=True)
class PaddleRowSpec:
    """One config_built row with its full roster contract.

    Parameters
    ----------
    family:
        Row name.
    builder:
        Zero-arg factory returning ``(model, example_input)``.
    kinds:
        Expected op kinds in execution order.
    intermediates:
        ``func_name -> array | list[array]`` computed with NumPy.
    has_constants:
        Whether the forward reads a closure constant or a literal producer.
    """

    family: str
    builder: Callable[[], tuple[Any, Any]]
    kinds: tuple[str, ...]
    intermediates: dict[str, Any]
    has_constants: bool


def config_built_specs() -> tuple[PaddleRowSpec, ...]:
    """The config_built rows with their contract fields (also read by tests).

    Returns
    -------
    tuple[PaddleRowSpec, ...]
        Rows in a stable order.
    """

    a = _X24 + 1.0
    mu = _X24.mean(-1, keepdims=True)
    normed = (_X24 - mu) / np.sqrt(_X24.var(-1, keepdims=True) + 1e-5)
    loop_kinds = ("matmul", "tensor.__add__", "functional.relu")
    specs: tuple[
        tuple[str, Callable[[], tuple[Any, Any]], tuple[str, ...], dict[str, Any], bool], ...
    ] = (
        (
            "fn_chain_branch",
            _fn_chain_branch,
            ("input", "tensor.__add__", "tensor.__mul__", "functional.relu"),
            {"functional.relu": _relu(a * _X24)},
            False,
        ),
        (
            "fn_multi_output",
            _fn_multi_output,
            ("input", "tensor.sum", "tensor.__mul__"),
            {"tensor.sum": np.asarray(_X24.sum(), dtype=np.float32), "tensor.__mul__": _X24 * 2},
            False,
        ),
        (
            "fn_loop3",
            _fn_loop3,
            ("input", "constant", "matmul", "constant", "tensor.__add__", "functional.relu")
            + loop_kinds * 2,
            _loop3_intermediates(),
            True,
        ),
        (
            "fn_norm_softmax",
            _fn_norm_softmax,
            (
                "input",
                "mean",
                "var",
                "tensor.__sub__",
                "tensor.__add__",
                "sqrt",
                "tensor.__truediv__",
                "functional.softmax",
            ),
            {"functional.softmax": _softmax(normed).astype(np.float32)},
            False,
        ),
        (
            "fn_reshape_transpose",
            _fn_reshape_transpose,
            ("input", "reshape", "transpose", "ones", "matmul"),
            {"matmul": _X24.reshape(4, 2).T @ np.ones((4, 1), np.float32)},
            True,
        ),
        (
            "fn_constants_planted",
            _fn_constants_planted,
            ("input", "constant", "matmul", "full", "tensor.__mul__"),
            {"matmul": _X24 @ _W44, "tensor.__mul__": (_X24 @ _W44) * 0.5},
            True,
        ),
        (
            "mod_mlp",
            _mod_mlp,
            ("input", "functional.linear", "functional.relu", "functional.linear"),
            {"functional.relu": _relu(_X24 @ _MLP_W1 + _MLP_B1)},
            False,
        ),
        (
            "mod_shared_twice",
            _mod_shared_twice,
            ("input", "functional.linear", "functional.relu", "functional.linear"),
            {"functional.relu": _relu(_X24 @ _SHARED_W + _SHARED_B)},
            False,
        ),
        (
            "mod_cnn",
            _mod_cnn,
            (
                "input",
                "conv2d",
                "functional.relu",
                "functional.max_pool2d",
                "flatten",
                "functional.linear",
            ),
            _cnn_intermediates(),
            False,
        ),
        (
            "mod_nested_residual",
            _mod_nested_residual,
            (
                "input",
                "functional.linear",
                "functional.gelu",
                "tensor.__add__",
                "functional.layer_norm",
            ),
            _nested_intermediates(),
            False,
        ),
        (
            "mod_conv1d",
            lambda: _conv_layer_model(1),
            ("input", "conv1d", "functional.relu"),
            {"conv1d": _conv(_SEQ, _CONV1_W, _CONV1_B, padding=1)},
            False,
        ),
        (
            "mod_conv3d",
            lambda: _conv_layer_model(3),
            ("input", "conv3d", "functional.relu"),
            {"conv3d": _conv(_VOL, _CONV3_W, _CONV3_B, padding=1)},
            False,
        ),
    )
    return tuple(PaddleRowSpec(*spec) for spec in specs)


def _config_built_rows() -> tuple[RosterModel, ...]:
    """The config_built rows as ``RosterModel`` records."""

    return tuple(
        _row(
            family=spec.family,
            realism="config_built",
            builder=spec.builder,
            expected_op_kinds=spec.kinds,
            expected_intermediates=spec.intermediates,
            has_constants=spec.has_constants,
            random_init=False,
            group=GROUP_CONFIG_BUILT,
        )
        for spec in config_built_specs()
    )


# ---------------------------------------------------------------- ecosystem rows

#: Pretrained ``paddle.vision`` checkpoints (FORKS F-I: ships with Paddle, no new
#: dependency). Evidence is Paddle's own download URL and MD5 for the weights.
ECOSYSTEM_CHECKPOINTS: dict[str, str] = {
    "resnet18": (
        "paddle.vision.models.resnet18(pretrained=True): "
        "https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams "
        "md5 cf548f46534aa3560945be4b95cd11c4 (paddlepaddle 3.3.1)"
    ),
    "mobilenet_v2": (
        "paddle.vision.models.mobilenet_v2(pretrained=True): "
        "https://paddle-hapi.bj.bcebos.com/models/mobilenet_v2_x1.0.pdparams "
        "md5 0340af0a901346c8d46f4529882fb63d (paddlepaddle 3.3.1)"
    ),
}


def _vision_model(name: str) -> Callable[[], tuple[Any, Any]]:
    """A builder loading the pretrained ``paddle.vision`` model ``name``."""

    def build() -> tuple[Any, Any]:
        """Load the pretrained model in eval mode with a fixed synthetic image."""

        import paddle
        from paddle.vision import models

        model = getattr(models, name)(pretrained=True)
        model.eval()
        image = np.linspace(-1.0, 1.0, 3 * 64 * 64, dtype=np.float32).reshape(1, 3, 64, 64)
        return model, paddle.to_tensor(image)

    return build


def _ecosystem_rows() -> tuple[RosterModel, ...]:
    """The pretrained ecosystem rows as ``RosterModel`` records."""

    return tuple(
        _row(
            family=name,
            realism="pretrained",
            builder=_vision_model(name),
            checkpoint_evidence=evidence,
            group=GROUP_ECOSYSTEM,
        )
        for name, evidence in ECOSYSTEM_CHECKPOINTS.items()
    )


def roster(include_ecosystem: bool | None = None) -> tuple[RosterModel, ...]:
    """Every Paddle roster row: ``config_built`` first, then ``ecosystem``.

    Parameters
    ----------
    include_ecosystem:
        Whether the ecosystem rows are included; ``None`` (the default)
        includes them when ``TORCHLENS_CONFORMANCE_ECOSYSTEM=1``.

    Returns
    -------
    tuple[RosterModel, ...]
        Rows in a stable order.
    """

    if include_ecosystem is None:
        include_ecosystem = ecosystem_enabled()
    rows = _config_built_rows()
    return rows + _ecosystem_rows() if include_ecosystem else rows
