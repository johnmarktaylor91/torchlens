"""r-b4 R27-5: display/summary walkers render bounded markers, never crash.

``format_call_arg`` crashed with a raw ``RecursionError`` on an ordinary
container cycle (probe: ``x=[]; x.append(x)``), and the module-hierarchy /
container-repr walkers shared the same unbounded shape. Display paths are now
TOTAL: cycles render ``<cycle>``, over-deep nesting renders ``<max-depth>``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes._summary import DISPLAY_MAX_DEPTH, format_call_arg


def test_format_call_arg_renders_cycle_marker() -> None:
    """A self-referential captured argument renders a bounded cycle marker."""

    cyclic: list[object] = [1]
    cyclic.append(cyclic)
    assert format_call_arg(cyclic) == "[1, <cycle>]"

    cyclic_dict: dict[str, object] = {"a": 1}
    cyclic_dict["self"] = cyclic_dict
    assert format_call_arg(cyclic_dict) == "{a: 1, self: <cycle>}"


def test_format_call_arg_renders_depth_marker() -> None:
    """An over-deep captured argument renders a bounded depth marker."""

    deep: object = 1
    for _ in range(DISPLAY_MAX_DEPTH + 20):
        deep = [deep]
    rendered = format_call_arg(deep)
    assert "<max-depth>" in rendered


def test_format_call_arg_shared_substructure_not_marked_cyclic() -> None:
    """DAG reuse (one list under two slots) renders fully, never as a cycle."""

    shared = [1, 2]
    assert format_call_arg([shared, shared]) == "[[1, 2], [1, 2]]"


@pytest.mark.smoke
def test_format_call_arg_ordinary_values_unchanged() -> None:
    """The compact format for ordinary args is unchanged."""

    assert format_call_arg([1, "a", {"k": 2.5}]) == "[1, 'a', {k: 2.5}]"
    assert format_call_arg(torch.ones(2, 3)).startswith("Tensor(shape=(2, 3)")


@pytest.mark.smoke
def test_module_hierarchy_display_still_renders() -> None:
    """The module-hierarchy display path stays intact on a real trace."""

    from torchlens.data_classes.interface import _module_hierarchy_str

    model = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.ReLU(), nn.Linear(4, 2)))
    log = tl.trace(model, torch.ones(1, 4))
    hierarchy = _module_hierarchy_str(log)
    assert "1" in hierarchy  # child modules render
    assert "<cycle>" not in hierarchy and "<max-depth>" not in hierarchy
    assert "linear_1_1" in str(log)
