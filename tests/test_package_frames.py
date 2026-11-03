"""The frame filter ``torchlens._utils._package_frames.is_torchlens_code``.

Capture walks the live stack on every op and must skip its own frames; the
only identity a frame offers is ``co_filename``. These tests pin the verdict
for the three kinds of frame the filter meets (the user's own code, TorchLens's
code, and a user module whose path merely contains the word ``torchlens``) and
pin that the code-context walk, which reads the memoised verdicts inline on its
hot loop, agrees with the function for every frame it meets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import FrameType

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils import _package_frames, introspection
from torchlens._utils._package_frames import is_torchlens_code, normalized_filename


def _user_module_code(path: Path, source: str) -> object:
    """Compile ``source`` as if it were the module at ``path`` and return its code."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return compile(source, str(path), "exec")


def test_user_frame_is_not_torchlens_code() -> None:
    """The test's own frame (a user frame) is outside the package."""

    frame: FrameType = sys._getframe(0)
    assert not is_torchlens_code(frame.f_code)
    assert not normalized_filename(frame.f_code.co_filename).startswith(
        _package_frames.TORCHLENS_PACKAGE_PREFIX
    )


def test_torchlens_frame_is_torchlens_code() -> None:
    """A code object compiled from a file inside the package is TorchLens's."""

    assert is_torchlens_code(introspection._get_code_context.__code__)
    assert is_torchlens_code(tl.trace.__code__)
    assert is_torchlens_code(is_torchlens_code.__code__)


def test_user_module_whose_path_contains_torchlens_is_not_torchlens_code(tmp_path: Path) -> None:
    """A user file under ``.../my_torchlens_models/`` is a user frame, not the package."""

    for relative in (
        Path("my_torchlens_models") / "net.py",
        Path("torchlens") / "net.py",  # a user's own package named like ours
        Path("torchlens_contrib") / "model.py",  # a sibling install
        Path("projects") / "torchlens-experiments" / "run.py",
    ):
        code = _user_module_code(tmp_path / relative, "x = 1\n")
        assert "torchlens" in str(tmp_path / relative)
        assert not is_torchlens_code(code), relative  # type: ignore[arg-type]


def test_verdict_is_memoised_per_filename_and_stable() -> None:
    """The second answer for a filename is the cached first one."""

    code = sys._getframe(0).f_code
    first = is_torchlens_code(code)
    assert _package_frames._VERDICTS[code.co_filename] is first
    assert is_torchlens_code(code) is first


class _Inner(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) + 1


class _Outer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = _Inner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) * 2


@pytest.mark.smoke
def test_code_context_walk_agrees_with_the_frame_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The walk's inline verdict lookup keeps or drops exactly the frames the filter would."""

    seen: list[tuple[str, bool]] = []
    real = _package_frames.is_torchlens_code

    def spy(code):  # type: ignore[no-untyped-def]
        verdict = real(code)
        seen.append((code.co_filename, verdict))
        return verdict

    # Clear the memo so every filename the walk meets takes the function path once.
    monkeypatch.setattr(_package_frames, "_VERDICTS", {})
    monkeypatch.setattr(introspection, "_PACKAGE_VERDICTS", _package_frames._VERDICTS)
    monkeypatch.setattr(introspection, "is_torchlens_code", spy)
    trace = tl.trace(_Outer(), torch.ones(2, 3))
    package_prefix = _package_frames.TORCHLENS_PACKAGE_PREFIX
    for filename, verdict in seen:
        assert verdict is normalized_filename(filename).startswith(package_prefix), filename
    for op in trace.ops:
        for location in op.source.frames or ():
            assert not normalized_filename(str(location.file)).startswith(package_prefix), op.label
    assert any(
        str(location.file) == __file__ for op in trace.ops for location in op.source.frames or ()
    )
