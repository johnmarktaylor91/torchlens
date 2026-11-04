"""Tests for single-module focus visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.layer import Layer


def _render_dot(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    """Render a Trace to DOT using a temporary SVG output path."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.draw(
        file_format="svg",
        path=str(tmp_path / "graph"),
        **kwargs,
    ).dot


def _boundary_node_count(dot_source: str, kind: str) -> int:
    """Count synthetic focus boundary node declarations."""

    return sum(
        1
        for line in dot_source.splitlines()
        if f"__module_focus_{kind}_" in line and " [label=" in line
    )


class _TwoBlockModel(nn.Module):
    """Small model with a named focusable block."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.block1 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return self.head(self.block1(self.stem(x)))


class _NestedFocusBlock(nn.Module):
    """Focusable module with an internal Sequential."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.inner = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.tail = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return self.tail(self.inner(x))


class _NestedBlock(nn.Module):
    """Model with an internal Sequential for collapse tests."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.block1 = _NestedFocusBlock()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return self.head(self.block1(self.stem(x)))


class _AddBlock(nn.Module):
    """Focusable block with two external upstream tensors."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Combine two tensors inside the block."""

        return self.relu(left + right)


def test_module_focus_invalid_address_raises(tmp_path: Path) -> None:
    """Unknown module address strings should fail clearly."""

    log = tl.trace(_TwoBlockModel(), torch.randn(1, 4))

    with pytest.raises(ValueError, match="nonexistent") as exc_info:
        _render_dot(log, tmp_path, module="nonexistent")
    assert exc_info.value.fields["code"] == "module_focus_not_found"


def test_module_focus_wrong_type_raises_module_focus_invalid(tmp_path: Path) -> None:
    """A non-Module, non-address focus value refuses with the typed code.

    r4 b6-opus R25 finding 2: ``module_focus_invalid`` was previously
    "provoked" only by this file's test FUNCTION NAMES; no test asserted the
    code was raised (and the name-adjacent test actually provokes
    ``module_focus_not_found``).
    """

    log = tl.trace(_TwoBlockModel(), torch.randn(1, 4))

    with pytest.raises(ValueError, match="module") as exc_info:
        _render_dot(log, tmp_path, module=42)
    assert exc_info.value.fields["code"] == "module_focus_invalid"


def test_module_focus_with_skip_and_collapse_fn(tmp_path: Path) -> None:
    """Module focus should compose with skip_fn and collapse_fn together."""

    log = tl.trace(_NestedBlock(), torch.randn(1, 4))

    def skip_fn(layer_log: Layer) -> bool:
        """Skip ReLU layers."""

        return layer_log.layer_type == "relu"

    def collapse_fn(module_log: Any) -> bool:
        """Collapse the internal tail module."""

        return module_log.address == "block1.tail"

    dot = _render_dot(log, tmp_path, module="block1", skip_fn=skip_fn, collapse_fn=collapse_fn)

    assert "relu_1_3 [" not in dot
    assert "@block1.tail" in dot


def test_modulelog_show_graph_method(tmp_path: Path) -> None:
    """Module.show_graph should match Trace.draw with module=self."""

    log = tl.trace(_TwoBlockModel(), torch.randn(1, 4))
    module_log = log.modules["block1"]
    direct = _render_dot(log, tmp_path / "direct", module=module_log)
    method = module_log.draw(
        file_format="svg",
        path=str(tmp_path / "method" / "graph"),
    ).dot

    assert method == direct


class _ListBlock(nn.Module):
    """Block with a nested MLP, held inside a never-called ``nn.ModuleList``."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.mlp = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return x + self.mlp(self.norm(x))


class _ModuleListModel(nn.Module):
    """GPT-style stack: ``blocks`` is a ModuleList that is iterated, never called."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([_ListBlock() for _ in range(3)])
        self.heads = nn.ModuleDict({"out": nn.Linear(4, 2)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.heads["out"](x)


@pytest.mark.parametrize(
    ("address", "title"),
    [
        ("blocks.1", "@blocks.1"),
        ("blocks.1.mlp", "@mlp"),
        ("blocks.1.mlp.0", "@mlp.0"),
        ("heads.out", "@out"),
    ],
)
@pytest.mark.parametrize("vis_mode", ["unrolled", "rolled"])
def test_module_focus_under_uncalled_container(
    tmp_path: Path, address: str, title: str, vis_mode: str
) -> None:
    """A module inside an uncalled ModuleList/ModuleDict focuses without a KeyError.

    The frame title (or, for a leaf module, its op's ``@name`` row) leaves off
    the scope line's prefix (``blocks.1``, ``heads``).
    """

    log = tl.trace(_ModuleListModel(), torch.randn(1, 4))
    assert "blocks" not in log.modules
    assert "heads" not in log.modules
    dot = _render_dot(log, tmp_path, module=address, rolled=vis_mode == "rolled")

    assert f"<B>{title}</B>" in dot or f">{title}</TD>" in dot
    assert f"@{address}" not in dot or address == title[1:]
    assert _boundary_node_count(dot, "input") >= 1
    assert _boundary_node_count(dot, "output") >= 1


def test_module_focus_under_uncalled_container_by_module_object(tmp_path: Path) -> None:
    """Passing the Module object for a ModuleList child renders like the address."""

    log = tl.trace(_ModuleListModel(), torch.randn(1, 4))
    dot = _render_dot(log, tmp_path, module=log.modules["blocks.1"])

    assert "@blocks.1" in dot
    # blocks.0 is outside the picture: it may name an input stub, never a drawn node.
    assert "cluster_blocks.0" not in dot
    for line in dot.splitlines():
        if "@blocks.0" in line:
            assert "__module_focus_" in line, line
