"""Paddle keyword inputs, op source lines and transposed convolution, as in torch.

* Keyword tensor inputs (``input_kwargs``) become input layers and validate.
* Every recorded op carries its user source frames (``layer.source.frames``),
  outermost first: the call site, the root's frame, then the frame that ran
  the op last. Input layers have none, and ``save_source_context`` is accepted.
* ``Conv1DTranspose``/``Conv2DTranspose``/``Conv3DTranspose`` record one
  transposed-convolution op inside the layer's module and validate.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends.paddle.backend import PaddleBackend  # noqa: E402
from torchlens.options import CaptureOptions  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]


class _KwargModel(paddle.nn.Layer):
    """Linear layer with a keyword tensor bias and a keyword scalar scale."""

    def __init__(self) -> None:
        """Build the linear layer."""

        super().__init__()
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x: Any, bias: Any = None, scale: float = 1.0) -> Any:
        """Apply the layer, scale, and add the keyword bias."""

        hidden = self.fc(x) * scale
        return hidden + bias


def _x() -> Any:
    """Return a deterministic ``[2, 4]`` input."""

    return paddle.arange(8, dtype="float32").reshape([2, 4]) / 8.0


def _kwargs() -> dict[str, Any]:
    """Return the keyword inputs: one tensor, one scalar."""

    return {"bias": paddle.full([4], 0.5), "scale": 3.0}


def _validates(model: Any, trace: Any, args: Any, **kwargs: Any) -> None:
    """Assert the trace validates and the public validate re-run agrees."""

    assert PaddleBackend().validate_trace(trace) is True
    assert tl.validate(model, args, backend="paddle", **kwargs) is True


def test_paddle_keyword_tensor_input_captures_and_validates() -> None:
    """A keyword tensor is an input layer feeding the add, and validation passes."""

    paddle.seed(0)
    model = _KwargModel()
    trace = tl.trace(model, _x(), input_kwargs=_kwargs(), backend="paddle")

    inputs = [layer for layer in trace.layers if layer.layer_type == "input"]
    assert len(inputs) == 2
    add = next(layer for layer in trace.layers if layer.layer_type == "add")
    assert set(add.parents) & {layer.label for layer in inputs}
    _validates(model, trace, _x(), input_kwargs=_kwargs())


def test_paddle_keyword_only_tensor_inputs_validate() -> None:
    """A forward that takes every tensor by keyword captures and validates."""

    def combine(*, left: Any, right: Any) -> Any:
        """Multiply and add two keyword tensors."""

        return left * right + left

    kwargs = {"left": _x(), "right": _x() + 1.0}
    trace = tl.trace(combine, (), input_kwargs=kwargs, backend="paddle")

    assert len([layer for layer in trace.layers if layer.layer_type == "input"]) == 2
    _validates(combine, trace, (), input_kwargs=kwargs)


def _line_of(func: Any, text: str) -> int:
    """Return the file line number of the first source line containing ``text``."""

    lines, start = inspect.getsourcelines(func)
    return start + next(i for i, line in enumerate(lines) if text in line)


def test_paddle_op_source_frames_match_torch_shape() -> None:
    """The multiply's frames end at the user's line and begin at the call site."""

    paddle.seed(0)
    trace = tl.trace(_KwargModel(), _x(), input_kwargs=_kwargs(), backend="paddle")

    mul = next(layer for layer in trace.layers if layer.layer_type == "mul")
    frames = list(mul.source.frames)
    assert len(frames) >= 2
    assert frames[-1].file == __file__
    assert frames[-1].func_name == "forward"
    assert frames[-1].line_number == _line_of(_KwargModel.forward, "* scale")
    assert frames[0].func_name == "test_paddle_op_source_frames_match_torch_shape"
    assert all("layers.py" not in frame.file for frame in frames)
    for layer in trace.layers:
        if layer.layer_type == "input":
            assert list(layer.source.frames) == []


def test_paddle_layer_op_keeps_framework_forward_frame() -> None:
    """The linear op ends in Paddle's ``Linear.forward``, as torch keeps ``nn.Linear``'s."""

    trace = tl.trace(_KwargModel(), _x(), input_kwargs=_kwargs(), backend="paddle")

    linear = next(layer for layer in trace.layers if layer.layer_type == "linear")
    frames = list(linear.source.frames)
    assert frames[-1].func_name == "forward"
    assert frames[-1].file.endswith("common.py")
    assert any(f.file == __file__ and f.func_name == "forward" for f in frames)


def _tanh_root(x: Any) -> Any:
    """Function root: one op."""

    return paddle.tanh(x)


def test_paddle_function_root_source_frames_and_saved_context() -> None:
    """A function root records its op line; ``save_source_context`` loads the text."""

    options = CaptureOptions(save_source_context=True)
    trace = tl.trace(_tanh_root, _x(), backend="paddle", capture=options)

    tanh = next(layer for layer in trace.layers if layer.layer_type == "tanh")
    frames = list(tanh.source.frames)
    assert frames[-1].func_name == "_tanh_root"
    assert frames[-1].line_number == _line_of(_tanh_root, "paddle.tanh")
    assert "paddle.tanh" in frames[-1].call_line
    assert frames[0].func_name == "test_paddle_function_root_source_frames_and_saved_context"


class _Up(paddle.nn.Layer):
    """One transposed convolution followed by a ReLU."""

    def __init__(self, dims: int) -> None:
        """Build the transposed convolution of ``dims`` spatial dimensions."""

        super().__init__()
        layer_cls = {
            1: paddle.nn.Conv1DTranspose,
            2: paddle.nn.Conv2DTranspose,
            3: paddle.nn.Conv3DTranspose,
        }[dims]
        self.up = layer_cls(3, 2, 3, stride=2, padding=1)

    def forward(self, x: Any) -> Any:
        """Upsample, then rectify."""

        return paddle.nn.functional.relu(self.up(x))


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_paddle_transposed_conv_groups_at_layer_level(dims: int) -> None:
    """The layer records exactly one transposed-convolution op inside ``up``."""

    paddle.seed(0)
    model = _Up(dims)
    x = paddle.rand([1, 3, *([4] * dims)])
    trace = tl.trace(model, x, backend="paddle")

    op_name = f"conv{dims}d_transpose"
    in_up = [trace.layers[label] for label in trace[tl.in_module("up")].labels]
    assert [layer.layer_type for layer in in_up] == [op_name]
    assert list(in_up[0].parents) and "up:1" in in_up[0].modules
    relu = next(layer for layer in trace.layers if layer.layer_type == "relu")
    assert list(relu.parents) == [in_up[0].label]
    assert {"up.weight", "up.bias"} <= {param.address for param in trace.params}
    _validates(model, trace, x)


def test_paddle_functional_transposed_conv_with_output_size_validates() -> None:
    """A direct ``conv2d_transpose`` call with ``output_size`` is one validated op."""

    paddle.seed(0)
    weight = paddle.rand([3, 2, 3, 3])

    def upsample(x: Any) -> Any:
        """Transposed convolution to an explicit output size."""

        return paddle.nn.functional.conv2d_transpose(
            x, weight, stride=2, padding=1, output_size=[8, 8]
        )

    x = paddle.rand([1, 3, 4, 4])
    trace = tl.trace(upsample, x, backend="paddle")

    conv = next(layer for layer in trace.layers if layer.layer_type == "conv2d_transpose")
    assert list(conv.out.shape) == [1, 2, 8, 8]
    _validates(upsample, trace, x)
