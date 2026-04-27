"""Tests for TorchLens stack introspection helpers."""

import os
from types import FrameType
from typing import List, Optional

import pytest
import torch
from torch import nn

import torchlens
from torchlens.data_classes import FuncCallLocation
from torchlens.utils import introspection


class RecursiveStackModel(nn.Module):
    """Small module that creates a deep Python stack below ``forward``."""

    def __init__(self, depth: int) -> None:
        """Initialize the recursive model.

        Args:
            depth: Number of recursive calls to make below ``forward``.
        """
        super().__init__()
        self.depth = depth

    def forward(self, x: torch.Tensor) -> List[FuncCallLocation]:
        """Return the filtered function call stack from a recursive call.

        Args:
            x: Input tensor, included to exercise normal ``nn.Module`` calling.

        Returns:
            Filtered TorchLens function call stack.
        """
        return self._recurse(x, self.depth)

    def _recurse(self, x: torch.Tensor, depth: int) -> List[FuncCallLocation]:
        """Recurse until the stack is deep enough, then capture it.

        Args:
            x: Input tensor carried through recursion.
            depth: Remaining recursive depth.

        Returns:
            Filtered TorchLens function call stack.
        """
        if depth <= 0:
            return introspection._get_func_call_stack()
        return self._recurse(x, depth - 1)


def test_stack_metadata_helpers_run_only_for_surviving_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify expensive stack metadata helpers run after frame filtering.

    Args:
        monkeypatch: Pytest fixture for replacing helper functions.
    """
    col_offset_calls = []
    code_qualname_calls = []

    def fake_get_col_offset(frame: FrameType) -> Optional[int]:
        """Record column-offset calls without walking bytecode.

        Args:
            frame: Stack frame passed by ``_get_func_call_stack``.

        Returns:
            Dummy column offset.
        """
        col_offset_calls.append((frame.f_code.co_filename, frame.f_code.co_name))
        return 0

    def fake_get_code_qualname(frame: FrameType) -> Optional[str]:
        """Record qualname calls without inspecting code metadata.

        Args:
            frame: Stack frame passed by ``_get_func_call_stack``.

        Returns:
            Code object name for the frame.
        """
        code_qualname_calls.append((frame.f_code.co_filename, frame.f_code.co_name))
        return frame.f_code.co_name

    monkeypatch.setattr(introspection, "_get_col_offset", fake_get_col_offset)
    monkeypatch.setattr(introspection, "_get_code_qualname", fake_get_code_qualname)

    model = RecursiveStackModel(depth=12)
    stack = model(torch.ones(1))

    torchlens_pkg_dir = os.path.dirname(os.path.abspath(torchlens.__file__))
    surviving_frames = [(loc.file, loc.func_name) for loc in stack]

    assert len(stack) >= 13
    assert col_offset_calls == surviving_frames
    assert code_qualname_calls == surviving_frames
    assert len(col_offset_calls) == len(stack)
    assert len(code_qualname_calls) == len(stack)
    assert all(not filename.startswith(torchlens_pkg_dir) for filename, _ in col_offset_calls)


def test_disable_col_offset_skips_col_offset_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ``disable_col_offset`` bypasses bytecode column-offset inspection.

    Args:
        monkeypatch: Pytest fixture for replacing helper functions.
    """
    col_offset_calls = []
    code_qualname_calls = []

    def fake_get_col_offset(frame: FrameType) -> Optional[int]:
        """Record unexpected column-offset calls.

        Args:
            frame: Stack frame passed by ``_get_func_call_stack``.

        Returns:
            Dummy column offset.
        """
        col_offset_calls.append(frame)
        return 0

    def fake_get_code_qualname(frame: FrameType) -> Optional[str]:
        """Record qualname calls that should still occur.

        Args:
            frame: Stack frame passed by ``_get_func_call_stack``.

        Returns:
            Code object name for the frame.
        """
        code_qualname_calls.append(frame)
        return frame.f_code.co_name

    monkeypatch.setattr(introspection, "_get_col_offset", fake_get_col_offset)
    monkeypatch.setattr(introspection, "_get_code_qualname", fake_get_code_qualname)

    model = RecursiveStackModel(depth=12)

    class DisableOffsetModel(RecursiveStackModel):
        """Recursive model variant that disables column-offset extraction."""

        def _recurse(self, x: torch.Tensor, depth: int) -> List[FuncCallLocation]:
            """Recurse until the stack is deep enough, then capture it.

            Args:
                x: Input tensor carried through recursion.
                depth: Remaining recursive depth.

            Returns:
                Filtered TorchLens function call stack.
            """
            if depth <= 0:
                return introspection._get_func_call_stack(disable_col_offset=True)
            return self._recurse(x, depth - 1)

    stack = DisableOffsetModel(model.depth)(torch.ones(1))

    assert len(stack) >= 13
    assert col_offset_calls == []
    assert len(code_qualname_calls) == len(stack)
    assert all(loc.col_offset is None for loc in stack)
