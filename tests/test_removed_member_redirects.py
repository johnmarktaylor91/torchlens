"""Removed public methods and attributes raise a typed redirect naming the replacement.

The 2026-10-01 shim removal deleted the alias members outright (remove-and-
rename, no alias), and lane F12 deleted ``VisualizationTheme.legend_items``.
A lookup of any of them must raise ``FacadeTeachingError`` (code
``facade_redirect``, ``AttributeError`` lineage) whose message names the
replacement, never a bare ``AttributeError``. Every other missing name keeps
failing plainly, and ``getattr`` with a default still degrades.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import FacadeTeachingError
from torchlens.options import VisualizationOptions
from torchlens.visualization.themes import THEME_PRESETS


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


@pytest.fixture(scope="module")
def owners():
    """One live instance of every class that lost public members."""

    first = tl.trace(_Toy(), torch.randn(1, 4))
    second = tl.trace(_Toy(), torch.randn(1, 4))
    try:
        yield {
            "Trace": first,
            "Bundle": tl.bundle({"a": first, "b": second}),
            "VisualizationOptions": VisualizationOptions(),
            "VisualizationTheme": THEME_PRESETS["torchlens"],
        }
    finally:
        first.cleanup()
        second.cleanup()


#: (owner class, removed member, live replacement member)
REMOVED_MEMBERS: list[tuple[str, str, str]] = [
    ("Trace", "replay", "push"),
    ("Trace", "replay_from", "push_from"),
    ("Trace", "rerun", "run"),
    ("Trace", "validate_saved_outs", "validate_forward_pass"),
    ("Trace", "conditional_then_entry_edges", "conditional_arm_entry_edges"),
    ("Trace", "conditional_elif_entry_edges", "conditional_arm_entry_edges"),
    ("Trace", "conditional_else_entry_edges", "conditional_arm_entry_edges"),
    ("Bundle", "replay", "push"),
    ("Bundle", "rerun", "run"),
    ("VisualizationOptions", "mode", "view"),
    ("VisualizationOptions", "max_module_depth", "depth"),
    ("VisualizationOptions", "layout_engine", "layout"),
    ("VisualizationOptions", "node_mode", "node_style"),
    ("VisualizationTheme", "legend_items", "semantic_palette"),
]


@pytest.mark.parametrize(("owner", "removed", "replacement"), REMOVED_MEMBERS)
def test_removed_member_raises_typed_redirect(
    owners: dict[str, Any], owner: str, removed: str, replacement: str
) -> None:
    """The removed spelling refuses typed and names a replacement that exists."""

    instance = owners[owner]
    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(instance, removed)
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert removed in str(excinfo.value)
    assert replacement in str(excinfo.value)
    assert getattr(instance, removed, "absent") == "absent"
    getattr(instance, replacement)  # the named replacement is live


@pytest.mark.parametrize("owner", ["Trace", "Bundle", "VisualizationOptions", "VisualizationTheme"])
def test_unknown_member_still_fails_plain(owners: dict[str, Any], owner: str) -> None:
    """Only removed public names redirect; any other miss stays a plain AttributeError."""

    with pytest.raises(AttributeError) as excinfo:
        getattr(owners[owner], "no_such_member_anywhere")
    assert not isinstance(excinfo.value, FacadeTeachingError)
