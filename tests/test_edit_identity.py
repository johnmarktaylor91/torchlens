"""The audit-row ``edit`` identity is always a string.

A replacement tensor exposes a ``name`` attribute that is ``None``. Audit rows
built from ``getattr(edit, "name", ...)`` therefore stored ``edit: None`` for a
value edit, and the load-time audit validator refused the saved artifact.
"""

from __future__ import annotations

import torch

import torchlens as tl
from torchlens._selection import _edit_identity


def test_value_edit_falls_back_to_the_given_identity() -> None:
    value = torch.zeros(2, 3)
    assert getattr(value, "name", "absent") is None
    assert _edit_identity(value, "value") == "value"


def test_named_edit_uses_its_name() -> None:
    edit = tl.zero_ablate()
    assert isinstance(edit.name, str)
    assert _edit_identity(edit, "value") == edit.name


def test_hook_callable_uses_its_function_name() -> None:
    def scale_hook(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        return out * 2

    assert _edit_identity(scale_hook, "value") == "scale_hook"


def test_non_string_name_is_ignored() -> None:
    class Odd:
        name = 7

    assert _edit_identity(Odd(), "fallback") == "fallback"
