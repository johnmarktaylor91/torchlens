"""A06 capture-options truth: the chunked path drops NO capture option.

WALKTHROUGH list-A row 26 (second clause): ``chunk_size=`` silently dropped
``save_budget`` -- the chunk path rebuilds a recursive ``CaptureOptions``
from a hand-enumerated field list, so session knobs absent from that list
reset to their defaults on every chunked capture (the same include-list
disease the cache key had). The recursive constructor now covers every
``CaptureOptions`` field, and the AST tripwire here turns any future field
that is not consciously routed through the chunk path into a red test.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens.user_funcs as user_funcs
from torchlens._save_budget import SaveBudgetExceededError
from torchlens.options import CaptureOptions


class SmallNet(nn.Module):
    """fc -> relu."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _recursive_capture_options_keywords() -> set[str]:
    """Extract the keyword names of the chunk path's CaptureOptions rebuild."""

    tree = ast.parse(inspect.getsource(user_funcs))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "recursive_capture_options"
            and isinstance(node.value, ast.Call)
        ):
            return {kw.arg for kw in node.value.keywords if kw.arg is not None}
    raise AssertionError(
        "the chunk path's recursive_capture_options = CaptureOptions(...) "
        "assignment was not found in torchlens.user_funcs; if the chunk path "
        "was restructured, re-point this oracle at the new rebuild site"
    )


def test_chunk_path_rebuild_covers_every_capture_option_field() -> None:
    """Set-difference oracle: a new CaptureOptions field cannot silently drop.

    The chunk path rebuilds the options object, so every field must be
    consciously routed (forwarded, resolved, or reset with a reason visible
    in the construction itself). A field missing from the rebuild resets to
    its default on chunked captures only -- exactly how ``save_budget`` was
    lost. Adding a CaptureOptions field makes this red until the chunk path
    names it.
    """

    field_names = {f.name for f in dataclasses.fields(CaptureOptions) if not f.name.startswith("_")}
    rebuilt = _recursive_capture_options_keywords()
    dropped = field_names - rebuilt
    assert not dropped, (
        f"CaptureOptions fields silently dropped by the chunked rebuild: "
        f"{sorted(dropped)}; route each through recursive_capture_options in "
        "torchlens/user_funcs.py (forward the resolved value, or reset it "
        "explicitly with a comment saying why)"
    )
    unknown = rebuilt - field_names
    assert not unknown, f"chunked rebuild names non-fields: {sorted(unknown)}"


@pytest.mark.smoke
def test_chunked_capture_honors_save_budget() -> None:
    """A 1-byte budget must trip on the chunked path exactly as unchunked."""

    x = torch.randn(8, 4)
    with pytest.raises(SaveBudgetExceededError):
        tl.trace(SmallNet(), x, capture=CaptureOptions(save_budget=1))
    with pytest.raises(SaveBudgetExceededError):
        tl.trace(SmallNet(), x, chunk_size=4, capture=CaptureOptions(save_budget=1))


@pytest.mark.smoke
def test_chunked_capture_carries_peak_memory_knob() -> None:
    """measure_python_peak_memory survives the chunked rebuild."""

    x = torch.randn(8, 4)
    log = tl.trace(
        SmallNet(),
        x,
        chunk_size=4,
        capture=CaptureOptions(measure_python_peak_memory=True),
    )
    peak = log.forward_peak_memory
    assert peak is not None and peak > 0, (
        "measure_python_peak_memory=True on a chunked capture reported no "
        "tracemalloc peak; the knob was dropped by the recursive rebuild"
    )
