"""``tl.Edit`` prints its public spelling (UI inventory, ``tl.Edit`` repr row)."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.intervention.types import Edit


@pytest.mark.smoke_cells(
    "test_builtin_edit_repr_is_the_public_constructor_call[edit1-tl.zero_ablate()]"
)
@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (tl.clip_norm(1.0), "tl.clip_norm(1.0)"),
        (tl.zero_ablate(), "tl.zero_ablate()"),
    ],
)
def test_builtin_edit_repr_is_the_public_constructor_call(edit: Edit, expected: str) -> None:
    """A builtin edit prints the call that builds it, and that call rebuilds it."""

    assert isinstance(edit, tl.Edit)
    assert repr(edit) == expected
    assert eval(repr(edit), {"tl": tl}) == edit


@pytest.mark.smoke
def test_tensor_arguments_print_by_shape_and_dtype() -> None:
    """A tensor argument prints as a readable placeholder, never its full values."""

    edit = tl.project_out(torch.ones(4))
    assert repr(edit) == "tl.project_out(<tensor float32 [4] on cpu>)"


def test_session_bound_edit_prints_as_tl_edit() -> None:
    """An edit no public call rebuilds prints as ``tl.Edit`` with its identity fields."""

    edit = Edit("custom_edit", portability="opaque_audit")
    assert repr(edit) == "tl.Edit('custom_edit', portability='opaque_audit')"


@pytest.mark.smoke
def test_no_edit_repr_prints_the_internal_class_name() -> None:
    """``Edit(`` never appears in an edit's printed form."""

    edits = [tl.zero_ablate(), tl.clip_norm(1.0), tl.steer(torch.zeros(3))]
    for edit in edits:
        assert "Edit(" not in repr(edit)
        assert repr(edit).startswith("tl.")
