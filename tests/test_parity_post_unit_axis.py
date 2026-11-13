"""Recorded preview layouts preserve dead-channel counts across backends."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.debug._graph import _feature_dim, dead_neurons_rows
from torchlens.debug._unit_axis_layout import backend_unit_axis

_BACKENDS = ("mlx", "tf", "jax", "paddle", "tinygrad")
_SHAPE = (2, 3, 4, 5)
_JAX_REPR = (
    "{'window_strides': (1, 1), 'dimension_numbers': "
    "ConvDimensionNumbers(lhs_spec=(0, 3, 1, 2), rhs_spec=(3, 2, 0, 1), "
    "out_spec=(0, 3, 1, 2)), 'precision': None}"
)


@dataclass
class _Trace:
    """Resolve fake op labels using the same lookup door as a completed trace."""

    _provenance_backend: str
    ops: dict[str, Any]
    modules: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, label: str) -> Any:
        """Return the fake op at a pass-qualified label."""

        return self.ops[label]


def _op(name: str = "conv2d", label: str = "conv:1", **metadata: Any) -> Any:
    """Build a metadata-only op with a spatial activation shape."""

    fields = {
        "func_name": name,
        "label": label,
        "shape": _SHAPE,
        "parents": [],
        "children": [],
        "annotations": {},
        "non_tensor_kwargs": {},
    }
    return SimpleNamespace(**{**fields, **metadata})


@pytest.mark.parametrize(
    ("backend", "name", "metadata", "axis"),
    [
        ("mlx", "conv2d", {}, 3),
        ("mlx", "conv_general", {}, 3),
        ("mlx", "max_pool2d", {}, 3),
        ("mlx", "conv_transpose2d", {}, 3),
        ("tinygrad", "conv2d", {}, 1),
        ("tinygrad", "avg_pool2d", {}, 1),
        ("tf", "Conv2D", {"annotations": {"tf_attrs": {"data_format": "b'NHWC'"}}}, 3),
        ("tf", "MaxPool", {"annotations": {"tf_attrs": {"data_format": "b'NCHW'"}}}, 1),
        (
            "tf",
            "Conv2D",
            {"annotations": {"tf_static_attrs": {"data_format": "b'NHWC'"}}},
            3,
        ),
        (
            "tf",
            "DepthwiseConv2dNative",
            {"annotations": {"tf_attrs": {"0": "'data_format'", "1": "b'NCHW'"}}},
            1,
        ),
        ("tf", "conv2d", {"saved_kwargs": {"data_format": "channels_last"}}, 3),
        ("tf", "conv2d", {"non_tensor_kwargs": {"data_format": "channels_first"}}, 1),
        ("paddle", "conv2d", {"non_tensor_kwargs": {"data_format": "NCHW"}}, 1),
        ("paddle", "max_pool2d", {"non_tensor_kwargs": {"data_format": "NHWC"}}, 3),
        ("paddle", "conv2d", {"saved_kwargs": {"data_format": "NHWC"}}, 3),
        (
            "jax",
            "conv_general_dilated",
            {"non_tensor_kwargs": {"dimension_numbers": SimpleNamespace(out_spec=(0, 2, 1, 3))}},
            2,
        ),
        (
            "jax",
            "conv_general_dilated",
            {"non_tensor_kwargs": {"dimension_numbers": ("NHWC", "HWIO", "NHWC")}},
            3,
        ),
        (
            "jax",
            "conv_general_dilated",
            {"annotations": {"jax_params": {"dimension_numbers": ("NCHW", "OIHW", "NCHW")}}},
            1,
        ),
        ("jax", "conv_general_dilated", {"annotations": {"jax_params": _JAX_REPR}}, 3),
        (
            "jax",
            "conv_general_dilated",
            {"annotations": {"jax_params": "{'dimension_numbers': ('NHWC', 'HWIO', 'NCHW')}"}},
            1,
        ),
    ],
)
def test_recorded_layout_axis(backend: str, name: str, metadata: dict, axis: int) -> None:
    """Read the exact call/annotation carriers produced by preview captures."""

    op = _op(name, **metadata)
    trace = _Trace(backend, {op.label: op})
    memo: dict[str, int] = {}
    out = SimpleNamespace(ndim=4, shape=_SHAPE)
    assert backend_unit_axis(trace, op, out, memo) == axis
    assert memo == {op.label: axis}
    assert _feature_dim(trace, op, out, None, {}) == axis


@pytest.mark.parametrize("backend", _BACKENDS)
def test_elementwise_followers_inherit_and_memoize(backend: str) -> None:
    """Follow same-shaped parents through multiple elementwise operations."""

    conv = _op(
        "conv_general_dilated" if backend == "jax" else "conv2d",
        non_tensor_kwargs={"data_format": "NHWC"},
        annotations={"jax_params": _JAX_REPR},
    )
    relu = _op("relu", "relu:1", parents=[conv.label])
    added = _op("add", "add:1", parents=[relu.label])
    trace = _Trace(backend, {op.label: op for op in (conv, relu, added)})
    memo: dict[str, int] = {}
    axis = 1 if backend == "tinygrad" else 3
    out = SimpleNamespace(ndim=4, shape=_SHAPE)
    assert backend_unit_axis(trace, added, out, memo) == axis
    assert memo == {op.label: axis for op in (conv, relu, added)}
    conv.annotations = {}
    conv.non_tensor_kwargs = {}
    assert backend_unit_axis(trace, relu, out, memo) == axis


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unknown_spatial_layout_stays_unavailable(backend: str) -> None:
    """Unknown operations never acquire an axis from rank or backend identity."""

    op = _op("unknown", non_tensor_kwargs={"data_format": "NHWC"})
    trace = _Trace(backend, {op.label: op})
    out = SimpleNamespace(ndim=4, shape=_SHAPE)
    memo: dict[str, int] = {}
    assert backend_unit_axis(trace, op, out, memo) is None
    assert not memo
    assert _feature_dim(trace, op, out, None, {}) is None
    assert _feature_dim(trace, op, out, -1, {}) == 3


@pytest.mark.parametrize(
    ("backend", "name", "metadata"),
    [
        ("tf", "Conv2D", {}),
        ("paddle", "conv2d", {}),
        ("jax", "conv_general_dilated", {}),
        ("tf", "Conv2D", {"annotations": {"tf_attrs": {"data_format": "NHWC_VECT_C"}}}),
        ("paddle", "conv2d", {"non_tensor_kwargs": {"data_format": "unknown"}}),
        ("jax", "conv_general_dilated", {"annotations": {"jax_params": "unreadable"}}),
        (
            "jax",
            "conv_general_dilated",
            {"non_tensor_kwargs": {"dimension_numbers": SimpleNamespace(out_spec=(0, 3, 3, 2))}},
        ),
    ],
)
def test_missing_or_invalid_layout_does_not_inherit(
    backend: str, name: str, metadata: dict
) -> None:
    """A spatial op with unknown layout stops inheritance from its input."""

    parent = _op("conv2d", "parent:1", non_tensor_kwargs={"data_format": "NHWC"})
    op = _op(name, parents=[parent.label], **metadata)
    trace = _Trace(backend, {parent.label: parent, op.label: op})
    assert backend_unit_axis(trace, op, SimpleNamespace(ndim=4, shape=_SHAPE), {}) is None


def test_parent_shape_boundary_and_cycle_are_bounded() -> None:
    """Neither a shape change nor an ungrounded parent cycle proves a layout."""

    conv = _op(shape=(2, 4, 5, 3))
    relu = _op("relu", "relu:1", parents=[conv.label])
    trace = _Trace("mlx", {op.label: op for op in (conv, relu)})
    out = SimpleNamespace(ndim=4, shape=_SHAPE)
    assert backend_unit_axis(trace, relu, out, {}) is None
    relu.parents = [relu.label]
    assert backend_unit_axis(trace, relu, out, {}) is None


def test_lowered_relu_inherits_through_any_same_shaped_parent() -> None:
    """tinygrad's relu (``where(cmplt(zeros, x), x, zeros)``) reaches the conv via a later parent."""

    conv = _op("reshape", "reshape_8_17:3", output_of_modules=["conv"])
    scalar = _op("reshape", "reshape_1_2:1", shape=(1, 1, 1, 1))
    zeros = _op("expand", "expand_1_3:1", parents=[scalar.label])
    mask = _op("cmplt", "cmplt_1_20:1", parents=[zeros.label, conv.label])
    relu = _op("where", "where_1_21:1", parents=[mask.label, conv.label, zeros.label])
    trace = _Trace(
        "tinygrad",
        {op.label: op for op in (conv, scalar, zeros, mask, relu)},
        {"conv": SimpleNamespace(class_qualname="tinygrad.nn.Conv2d")},
    )
    memo: dict[str, int] = {}
    out = SimpleNamespace(ndim=4, shape=_SHAPE)
    assert backend_unit_axis(trace, relu, out, memo) == 1
    assert memo == {conv.label: 1, mask.label: 1, relu.label: 1}
    assert backend_unit_axis(trace, zeros, out, {}) is None


def test_conflicting_same_shaped_ancestors_leave_the_axis_unknown() -> None:
    """Two same-shaped parents evidencing different channel axes prove no axis."""

    first = _op("Conv2D", "first:1", annotations={"tf_attrs": {"data_format": "b'NHWC'"}})
    second = _op("Conv2D", "second:1", annotations={"tf_attrs": {"data_format": "b'NCHW'"}})
    added = _op("AddV2", "add:1", parents=[first.label, second.label])
    trace = _Trace("tf", {op.label: op for op in (first, second, added)})
    out = SimpleNamespace(ndim=4, shape=_SHAPE)
    assert backend_unit_axis(trace, added, out, {}) is None
    second.annotations = {"tf_attrs": {"data_format": "b'NHWC'"}}
    assert backend_unit_axis(trace, added, out, {}) == 3


def test_tinygrad_lowered_conv_uses_recorded_module_return() -> None:
    """Lowered UOp names retain a conv axis through recorded module identity."""

    conv = _op("reshape", output_of_modules=["conv"])
    trace = _Trace(
        "tinygrad",
        {conv.label: conv},
        {"conv": SimpleNamespace(class_qualname="tinygrad.nn.Conv2d")},
    )
    out = SimpleNamespace(ndim=4, shape=_SHAPE)
    assert backend_unit_axis(trace, conv, out, {}) == 1
    conv.output_of_modules = []
    assert backend_unit_axis(trace, conv, out, {}) is None


@pytest.mark.parametrize(("shape", "axis"), [((3,), None), ((2, 3), 1), ((2, 5, 3), 2)])
def test_fixed_layout_across_ranks(shape: tuple[int, ...], axis: int | None) -> None:
    """Dense and scalar conventions remain intact alongside channels-last conv1d."""

    op = _op("conv1d", shape=shape)
    trace = _Trace("mlx", {op.label: op})
    assert backend_unit_axis(trace, op, SimpleNamespace(ndim=len(shape), shape=shape), {}) == axis


@pytest.mark.parametrize(
    ("backend", "name", "shape", "metadata", "axis"),
    [
        ("mlx", "max_pool3d", (2, 4, 5, 6, 3), {}, 4),
        ("tinygrad", "conv2d", (2, 3, 5), {}, 1),
        ("tf", "Conv3D", (2, 3, 4, 5, 6), {"non_tensor_kwargs": {"data_format": "NCDHW"}}, 1),
        ("tf", "conv1d", (2, 5, 3), {"non_tensor_kwargs": {"data_format": "NWC"}}, 2),
        ("paddle", "conv3d", (2, 4, 5, 6, 3), {"non_tensor_kwargs": {"data_format": "NDHWC"}}, 4),
        ("paddle", "conv2d", (2, 3, 4, 5), {"non_tensor_kwargs": {"data_format": "NWC"}}, None),
        (
            "jax",
            "conv_general_dilated",
            (2, 4, 5, 6, 3),
            {"non_tensor_kwargs": {"dimension_numbers": SimpleNamespace(out_spec=(0, 4, 1, 2, 3))}},
            4,
        ),
    ],
)
def test_layout_formats_validate_activation_rank(
    backend: str, name: str, shape: tuple[int, ...], metadata: dict, axis: int | None
) -> None:
    """Layouts cover one/three spatial dimensions and reject mismatched format ranks."""

    op = _op(name, shape=shape, **metadata)
    trace = _Trace(backend, {op.label: op})
    assert backend_unit_axis(trace, op, SimpleNamespace(ndim=len(shape), shape=shape), {}) == axis


def _weights() -> np.ndarray:
    """Return three positive/zero convolution kernels, with channel one dead."""

    return np.array([1.0, 0.0, 2.0], dtype=np.float32).reshape(3, 1, 1, 1)


def _torch_trace() -> Any:
    """Capture the reference NCHW convolution followed by ReLU."""

    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 3, 1, bias=False), torch.nn.ReLU()).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(_weights()))
    return tl.trace(model, torch.ones(2, 1, 4, 5))


def _mlx_trace() -> Any:
    """Capture an identical channels-last MLX convolution and ReLU."""

    if sys.platform != "darwin" and os.environ.get("TORCHLENS_RUN_UNSTABLE_MLX_TESTS") != "1":
        pytest.skip("MLX off macOS runs only with TORCHLENS_RUN_UNSTABLE_MLX_TESTS=1")
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class Net(nn.Module):
        """Return the rectified convolution output."""

        def __init__(self) -> None:
            """Install fixed OHWI weights without bias."""

            super().__init__()
            self.conv = nn.Conv2d(1, 3, 1, bias=False)
            self.conv.weight = mx.array(_weights())

        def __call__(self, x: Any) -> Any:
            """Evaluate the convolution and its elementwise follower."""

            return nn.relu(self.conv(x))

    model = Net()
    mx.eval(model.parameters())
    return tl.trace(model, mx.ones((2, 4, 5, 1)), backend="mlx")


def _tf_trace() -> Any:
    """Capture the same convolution and ReLU using TensorFlow/Keras."""

    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential(
        [
            tf.keras.Input((4, 5, 1)),
            tf.keras.layers.Conv2D(
                3,
                1,
                use_bias=False,
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.Constant(_weights().transpose(2, 3, 1, 0)),
            ),
            tf.keras.layers.ReLU(),
        ]
    )
    return tl.trace(model, tf.ones((2, 4, 5, 1)), backend="tf")


def _jax_trace() -> Any:
    """Capture lax convolution with explicit output dimensions and a ReLU."""

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def model(x: Any, weight: Any) -> Any:
        """Apply an NHWC convolution and rectification."""

        return jax.nn.relu(
            jax.lax.conv_general_dilated(
                x, weight, (1, 1), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")
            )
        )

    weight = jnp.asarray(_weights().transpose(2, 3, 1, 0))
    return tl.trace(model, (jnp.ones((2, 4, 5, 1)), weight), backend="jax")


def _paddle_trace() -> Any:
    """Capture an NCHW Paddle layer carrying its explicit data format."""

    paddle = pytest.importorskip("paddle")
    conv = paddle.nn.Conv2D(1, 3, 1, bias_attr=False, data_format="NCHW")
    conv.weight.set_value(paddle.to_tensor(_weights()))
    model = paddle.nn.Sequential(conv, paddle.nn.ReLU())
    model.eval()
    return tl.trace(model, paddle.ones([2, 1, 4, 5]), backend="paddle")


def _tinygrad_trace() -> Any:
    """Capture a module-owned tinygrad convolution followed by rectification."""

    tinygrad = pytest.importorskip("tinygrad")
    nn = pytest.importorskip("tinygrad.nn")

    class Net:
        """Keep the convolution's module-return identity in the lowered trace."""

        def __init__(self) -> None:
            """Install the reference OIHW weights."""

            self.conv = nn.Conv2d(1, 3, 1, bias=False)
            self.conv.weight = tinygrad.Tensor(_weights()).realize()

        def __call__(self, x: Any) -> Any:
            """Return a rectified NCHW activation."""

            return self.conv(x).relu()

    return tl.trace(Net(), tinygrad.Tensor.ones(2, 1, 4, 5).realize(), backend="tinygrad")


@pytest.mark.rare
@pytest.mark.backend_parity
@pytest.mark.parametrize(
    "capture",
    [
        pytest.param(_mlx_trace, marks=pytest.mark.backend_mlx, id="mlx"),
        pytest.param(_tf_trace, marks=pytest.mark.tf_backend, id="tf"),
        pytest.param(_jax_trace, marks=pytest.mark.backend_jax, id="jax"),
        pytest.param(_paddle_trace, marks=pytest.mark.backend_paddle, id="paddle"),
        pytest.param(_tinygrad_trace, marks=pytest.mark.backend_tinygrad, id="tinygrad"),
    ],
)
def test_real_conv_relu_dead_channels_match_torch(capture: Any, monkeypatch: Any) -> None:
    """Count three channels and exactly the constructed dead channel on every backend."""

    monkeypatch.setenv("DEV", "PYTHON")
    preview = capture()
    reference = _torch_trace()
    preview_rows, _ = dead_neurons_rows(preview)
    reference_rows, _ = dead_neurons_rows(reference)
    assert preview_rows and reference_rows
    fields = ("total_units", "dead_count", "sample_dead_idx")
    expected = (3, 1, [1])
    assert tuple(reference_rows[-1][field] for field in fields) == expected
    assert tuple(preview_rows[-1][field] for field in fields) == expected
    assert preview_rows[-1]["reason"] == ""
