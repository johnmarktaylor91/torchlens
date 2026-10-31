"""Right-to-left layout (U07.2) and the collision-safe bare ``draw()`` (forks.12).

U07.2 adds ``"rightleft"`` as the fourth layout direction: it maps to the
Graphviz ``rankdir=RL`` on the forward and backward entrypoints, and the
unknown-direction refusal teaches all four values.

forks.12: a bare ``trace.draw()`` in a plain script writes a file it never
overwrites and prints that file's path, even on a quiet trace. The shared
conftest fixture rewrites ``draw``'s default ``vis_outpath`` into ``tmp_path``
for every test; the bare-draw tests here put the shipped default back for the
duration of the call so they exercise the real default codepath.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._trace_viz import TraceVisualizationMixin
from torchlens._errors import InvalidArgumentError
from torchlens.visualization._render_utils import direction_to_rankdir


class _TwoLayer(nn.Module):
    """Small module-bearing model for direction and output-path checks."""

    def __init__(self) -> None:
        """Initialize the layers."""
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.fc2(torch.relu(self.fc1(x)))


def _trace() -> tl.Trace:
    """Return a small forward trace."""
    return tl.trace(_TwoLayer(), torch.randn(2, 3))


@contextmanager
def _shipped_draw_default() -> Iterator[None]:
    """Restore ``Trace.draw``'s shipped ``vis_outpath`` default for one block.

    The autouse conftest fixture swaps the default for a ``tmp_path`` file;
    the forks.12 contract is about the shipped default, so it is put back here
    and the caller ``chdir``s into its own ``tmp_path``.
    """

    func = TraceVisualizationMixin.draw
    saved = func.__defaults__
    positional = [
        p
        for p in inspect.signature(func).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    defaults = list(saved or ())
    first_defaulted = len(positional) - len(defaults)
    index = [p.name for p in positional].index("path") - first_defaulted
    defaults[index] = None
    func.__defaults__ = tuple(defaults)
    try:
        yield
    finally:
        func.__defaults__ = saved


def test_rightleft_maps_to_rankdir_rl() -> None:
    """The fourth direction maps to Graphviz ``RL``; the other three are unchanged."""
    assert direction_to_rankdir("rightleft") == "RL"
    assert direction_to_rankdir("leftright") == "LR"
    assert direction_to_rankdir("bottomup") == "BT"
    assert direction_to_rankdir("topdown") == "TB"


def test_rightleft_forward_draw_emits_rankdir_rl(tmp_path: Path) -> None:
    """A forward render with ``direction="rightleft"`` lays edges out right to left."""
    trace = _trace()
    try:
        dot = trace.draw(
            direction="rightleft",
            path=str(tmp_path / "rl"),
            vis_save_only=True,
            file_format="dot",
            vis_node_placement="dot",
        )
    finally:
        trace.cleanup()
    assert "rankdir=RL" in dot
    assert "rankdir=LR" not in dot


def test_rightleft_backward_draw_emits_rankdir_rl(tmp_path: Path) -> None:
    """``draw_backward(vis_direction="rightleft")`` emits ``rankdir=RL`` too."""
    trace = tl.trace(
        _TwoLayer(),
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
    )
    try:
        trace.log_backward(trace[trace.output_layers[0]].out.sum())
        dot = trace.draw_backward(
            vis_direction="rightleft",
            path=str(tmp_path / "bwd_rl"),
            vis_save_only=True,
            file_format="dot",
        )
    finally:
        trace.cleanup()
    assert "rankdir=RL" in dot


def test_unknown_direction_teaches_four_values() -> None:
    """A typo'd direction refuses typed and names every accepted value."""
    with pytest.raises(InvalidArgumentError) as excinfo:
        direction_to_rankdir("right-left")
    message = str(excinfo.value)
    for value in ("bottomup", "topdown", "leftright", "rightleft"):
        assert value in message
    assert excinfo.value.fields["code"] == "visualization_direction_invalid"


def test_two_bare_draws_write_two_files_and_print_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two bare ``draw()`` calls in one directory write two files and print both paths."""
    monkeypatch.chdir(tmp_path)
    trace = _trace()
    assert not trace.verbose
    try:
        with _shipped_draw_default():
            trace.draw(vis_save_only=True, file_format="svg")
            first_out = capsys.readouterr().out
            first_mtime = os.path.getmtime(tmp_path / "modelgraph.svg")
            trace.draw(vis_save_only=True, file_format="svg")
            second_out = capsys.readouterr().out
    finally:
        trace.cleanup()
    first = tmp_path / "modelgraph.svg"
    second = tmp_path / "modelgraph-2.svg"
    assert first.is_file() and first.stat().st_size > 0
    assert second.is_file() and second.stat().st_size > 0
    assert os.path.getmtime(first) == first_mtime, "the second bare draw overwrote the first"
    assert "TorchLens graph written to modelgraph.svg" in first_out
    assert "TorchLens graph written to modelgraph-2.svg" in second_out
    # The DOT source scratch file is cleaned up; only the two pictures remain.
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "modelgraph-2.svg",
        "modelgraph.svg",
    ]


def test_explicit_outpath_is_exact_and_quiet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An explicit ``vis_outpath`` is written exactly (overwriting) and stays quiet."""
    trace = _trace()
    target = tmp_path / "named"
    try:
        trace.draw(path=str(target), vis_save_only=True, file_format="svg")
        trace.draw(path=str(target), vis_save_only=True, file_format="svg")
    finally:
        trace.cleanup()
    assert (tmp_path / "named.svg").is_file()
    assert not (tmp_path / "named-2.svg").exists()
    assert "TorchLens graph written to" not in capsys.readouterr().out
