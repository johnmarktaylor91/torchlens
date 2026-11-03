"""The line of code that ran an op: frame order, lazy source text, and the op card.

Contract pinned here:

- ``op.code_context`` is ordered outermost to innermost, like a traceback: the
  call site first, the frame that ran the op LAST.
- Without ``save_source_context`` a capture stores no source text; the text is
  read from the file when asked for, and only while the file still has the
  size and modification time it had at capture. A changed or missing file
  shows a reason, never another line's text.
- ``save_source_context=True`` snapshots the text at capture.
- ``print(op)`` names the op's line (``file:line in function``) and its text.
"""

from __future__ import annotations

import importlib.util
import linecache
import os
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._site_key import operation_witness
from torchlens._data_classes.func_call_location import display_frame

_TOY_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn


    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x):
            y = self.linear(x)
            z = torch.relu(y)  # the relu line
            return z * 2
    """
)


def _load_toy(tmp_path: Path, name: str) -> tuple[ModuleType, Path]:
    """Write the toy model to ``tmp_path`` and import it as a fresh module."""

    path = tmp_path / f"{name}.py"
    path.write_text(_TOY_SOURCE)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path


def _line_of(path: Path, needle: str) -> int:
    """One-indexed line number of the first line containing ``needle``."""

    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(f"{needle!r} not in {path}")


@pytest.fixture
def toy(tmp_path: Path, request: pytest.FixtureRequest):
    """A toy model module written to a temp file, plus its path."""

    name = f"tl_lazy_source_{request.node.name.replace('[', '_').replace(']', '_')}"
    module, path = _load_toy(tmp_path, name)
    yield module, path
    sys.modules.pop(name, None)
    linecache.checkcache()


def test_innermost_frame_is_last_and_call_site_first(toy) -> None:
    module, path = toy
    trace = tl.trace(module.Toy(), torch.randn(2, 3))
    relu = trace["relu_1"]
    frames = list(relu.code_context)
    assert len(frames) >= 2
    assert frames[0].file == __file__  # the call site: this test's trace line
    assert frames[1].func_name == "forward"  # the outermost forward
    assert frames[-1].file == str(path)
    assert frames[-1].line_number == _line_of(path, "the relu line")
    assert operation_witness(relu) == (str(path), _line_of(path, "the relu line"))


def test_default_capture_reads_source_lazily(toy) -> None:
    module, path = toy
    trace = tl.trace(module.Toy(), torch.randn(2, 3))
    frame = trace["relu_1"].code_context[-1]
    assert frame.source_loading_enabled is False
    assert frame._call_line == ""  # nothing stored at capture
    assert frame.call_line == "z = torch.relu(y)  # the relu line"
    assert "--->  " in frame.code_context_labeled
    assert "z = torch.relu(y)" in repr(frame)


def test_op_card_shows_location_and_text(toy) -> None:
    module, path = toy
    trace = tl.trace(module.Toy(), torch.randn(2, 3))
    card = str(trace["relu_1"])
    line = _line_of(path, "the relu line")
    assert f"{path.name}:{line} in Toy.forward" in card
    assert "  code   z = torch.relu(y)  # the relu line" in card


def test_op_card_skips_torch_frames_for_module_ops(toy) -> None:
    module, path = toy
    trace = tl.trace(module.Toy(), torch.randn(2, 3))
    linear = trace["linear_1"]
    torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
    assert os.path.abspath(linear.code_context[-1].file).startswith(torch_dir)
    shown = display_frame(linear.code_context)
    assert shown is not None
    assert shown.file == str(path)
    assert shown.line_number == _line_of(path, "y = self.linear(x)")
    card = str(linear)
    assert "where  module linear:1, " in card
    assert "  code   y = self.linear(x)" in card


def test_edited_file_shows_reason_not_wrong_text(toy) -> None:
    module, path = toy
    trace = tl.trace(module.Toy(), torch.randn(2, 3))
    relu = trace["relu_1"]
    path.write_text("# a new first line shifts every line\n" + _TOY_SOURCE)
    frame = relu.code_context[-1]
    assert frame.call_line == ""
    assert frame.code_context is None
    assert repr(frame).endswith("code: source changed on disk since capture")
    assert "  code   (source changed on disk since capture)" in str(relu)


def test_missing_file_shows_reason(toy) -> None:
    module, path = toy
    trace = tl.trace(module.Toy(), torch.randn(2, 3))
    relu = trace["relu_1"]
    path.unlink()
    frame = relu.code_context[-1]
    assert frame.call_line == ""
    assert repr(frame).endswith("code: source file not found")
    assert "  code   (source file not found)" in str(relu)


def test_save_source_context_snapshots_at_capture(toy) -> None:
    module, path = toy
    trace = tl.trace(
        module.Toy(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(save_source_context=True),
    )
    relu = trace["relu_1"]
    path.write_text("# edited after capture\n" + _TOY_SOURCE)
    linecache.checkcache(str(path))
    frame = relu.code_context[-1]
    assert frame.source_loading_enabled is True
    assert frame.call_line == "z = torch.relu(y)  # the relu line"
    assert "  code   z = torch.relu(y)  # the relu line" in str(relu)


def test_default_capture_stores_no_source_text(toy) -> None:
    """The capture itself stays text-free: a fingerprint per frame, no stored text."""

    module, _path = toy
    trace = tl.trace(module.Toy(), torch.randn(2, 3))
    frames = [frame for op in trace.ops for frame in (op.code_context or ())]
    assert frames
    assert all(frame._linecache_entry is not None for frame in frames)
    assert all(frame._call_line == "" and frame._code_context is None for frame in frames)


def test_nested_model_frames(tmp_path: Path) -> None:
    """Nested modules: the last frame is the innermost forward's line."""

    class Inner(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x)  # inner line

    class Outer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = Inner()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x) + 1

    trace = tl.trace(Outer(), torch.randn(2, 3))
    frames = list(trace["sigmoid_1"].code_context)
    assert [f.func_name for f in frames][-2:] == ["forward", "forward"]
    assert frames[-1].call_line == "return torch.sigmoid(x)  # inner line"
    assert frames[0].file == __file__
