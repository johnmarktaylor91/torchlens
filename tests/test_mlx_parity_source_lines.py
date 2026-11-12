"""MLX operations record the user source lines that ran them, like torch.

``op.source.frames`` is ordered outermost to innermost: the first frame is the
call site (the ``tl.trace`` line) and the last frame is the line that ran the
operation. These tests pin both ends on MLX for a module model, a nested
module, and a plain-function model, and check that ``save_source_context`` and
the context line count are honored, and that file names and line numbers
survive a save and load.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.backend_mlx, pytest.mark.rare]

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends.mlx import MLXBackend  # noqa: E402
from torchlens.options import CaptureOptions  # noqa: E402

_THIS_FILE = str(Path(__file__).resolve())


class _Block(nn.Module):
    """Inner module whose relu line is the expected innermost frame."""

    def __init__(self) -> None:
        """Initialize one deterministic linear layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.lin.weight = mx.full((4, 4), 0.25)
        self.lin.bias = mx.full((4,), 0.1)

    def __call__(self, x: mx.array) -> mx.array:
        """Run the block."""

        return nn.relu(self.lin(x))  # BLOCK_RELU


class _Net(nn.Module):
    """Outer module with a nested block and a direct elementwise op."""

    def __init__(self) -> None:
        """Initialize the nested block."""

        super().__init__()
        self.block = _Block()

    def __call__(self, x: mx.array) -> mx.array:
        """Run the model."""

        h = self.block(x)  # NET_BLOCK
        return mx.tanh(h * 2.0)  # NET_TANH


def _function_model(x: mx.array) -> mx.array:
    """Plain-function model."""

    return mx.sigmoid(x + 1.0)  # FN_SIGMOID


def _line_of(marker: str) -> int:
    """Return the line number of the source line carrying ``marker``."""

    for number, text in enumerate(Path(__file__).read_text().splitlines(), start=1):
        if text.rstrip().endswith(f"# {marker}"):
            return number
    raise AssertionError(f"marker {marker} not found")


def _op(trace: Any, layer_type: str) -> Any:
    """Return the first non-input op of ``layer_type``."""

    return next(op for op in trace._layer_list if not op.is_input and op.type == layer_type)


def _same_file(path: str) -> bool:
    """Whether ``path`` names this test file."""

    return str(Path(path).resolve()) == _THIS_FILE


def test_mlx_innermost_frame_is_the_user_line_that_ran_the_op() -> None:
    """The last frame names the user line; the first names the tl.trace call site."""

    trace = tl.trace(_Net(), mx.ones((1, 4)), backend="mlx")
    call_site_line = inspect.currentframe().f_lineno - 1  # the tl.trace line above

    relu = _op(trace, "relu").source.frames
    assert _same_file(relu[-1].file)
    assert relu[-1].line_number == _line_of("BLOCK_RELU")
    assert relu[-1].func_name == "__call__"
    assert relu[-1].func_qualname.endswith("_Block.__call__")

    assert _same_file(relu[0].file)
    assert relu[0].line_number == call_site_line
    assert relu[0].func_name == "test_mlx_innermost_frame_is_the_user_line_that_ran_the_op"

    # The nested call chain sits between: the outer model's block line.
    user_lines = [frame.line_number for frame in relu if _same_file(frame.file)]
    assert _line_of("NET_BLOCK") in user_lines

    tanh = _op(trace, "tanh").source.frames
    assert tanh[-1].line_number == _line_of("NET_TANH")
    assert _line_of("BLOCK_RELU") not in [frame.line_number for frame in tanh]


def test_mlx_plain_function_model_records_its_lines() -> None:
    """A plain-function model is its own innermost frame."""

    trace = tl.trace(_function_model, mx.ones((2, 3)), backend="mlx")
    frames = _op(trace, "sigmoid").source.frames
    assert _same_file(frames[-1].file)
    assert frames[-1].line_number == _line_of("FN_SIGMOID")
    assert frames[-1].func_name == "_function_model"


def test_mlx_source_context_text_and_line_count() -> None:
    """save_source_context snapshots the text; source_context_lines sizes it.

    As on torch, ``source_context_lines`` counts lines on each side of the call
    line, so 3 keeps a 7-line window.
    """

    trace = tl.trace(
        _Net(),
        mx.ones((1, 4)),
        backend="mlx",
        capture=CaptureOptions(save_source_context=True, source_context_lines=3),
    )
    frame = _op(trace, "tanh").source.frames[-1]
    assert frame.source_loading_enabled is True
    assert "mx.tanh(h * 2.0)" in frame.call_line
    assert frame.code_context is not None
    assert len(frame.code_context) == frame.num_context_lines == 2 * 3 + 1
    assert MLXBackend().validate_trace(trace).passed


def test_mlx_source_frames_survive_save_and_load(tmp_path: Path) -> None:
    """Saved bundles carry the frames: file and line survive the round trip."""

    trace = tl.trace(_Net(), mx.ones((1, 4)), backend="mlx")
    # A saved bundle keeps each frame's file name, not the absolute path.
    expected = [(Path(f.file).name, f.line_number) for f in _op(trace, "relu").source.frames]
    bundle = tmp_path / "mlx_source.tlspec"
    trace.save(bundle)
    loaded = tl.load(bundle)
    got = [(Path(f.file).name, f.line_number) for f in _op(loaded, "relu").source.frames]
    assert got == expected
