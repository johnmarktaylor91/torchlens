"""``tl.Edit`` prints its public spelling (UI inventory, ``tl.Edit`` repr row)."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.intervention.types import Edit


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (tl.zero_grad(), "tl.zero_grad()"),
        (tl.scale_grad(2.0), "tl.scale_grad(2.0)"),
        (tl.clamp_grad(low=-1.0), "tl.clamp_grad(min=-1.0)"),
        (tl.zero_ablate(), "tl.zero_ablate()"),
        (tl.scale(0.5), "tl.scale(0.5)"),
        (tl.add_noise(0.1, seed=3), "tl.add_noise(0.1, seed=3)"),
        (tl.clamp(min=0.0, max=1.0), "tl.clamp(min=0.0, max=1.0)"),
    ],
)
def test_builtin_edit_repr_is_the_public_constructor_call(edit: Edit, expected: str) -> None:
    """A builtin edit prints the call that builds it, and that call rebuilds it."""

    assert isinstance(edit, tl.Edit)
    assert repr(edit) == expected
    assert eval(repr(edit), {"tl": tl}) == edit


def test_non_root_builtin_prints_its_public_namespace() -> None:
    """A builtin that is not a root name prints under ``tl.intervention``."""

    edit = tl.intervention.scramble_elements(seed=1)
    assert repr(edit).startswith("tl.intervention.scramble_elements(")


def test_tensor_arguments_print_by_shape_and_dtype() -> None:
    """A tensor argument prints as a readable placeholder, never its full values."""

    edit = tl.project_out(torch.ones(4))
    assert repr(edit) == "tl.project_out(<tensor shape=(4,) dtype=torch.float32>)"


def test_session_bound_edit_prints_as_tl_edit() -> None:
    """An edit no public call rebuilds prints as ``tl.Edit`` with its identity fields."""

    edit = Edit("custom_edit", kind="backward", portability="opaque_audit")
    assert repr(edit) == "tl.Edit('custom_edit', kind='backward', portability='opaque_audit')"


def test_no_edit_repr_prints_the_internal_class_name() -> None:
    """``Edit(`` never appears in an edit's printed form."""

    edits = [tl.zero_grad(), tl.hook_grad(lambda g, *, hook: g), tl.steer(torch.zeros(3))]
    for edit in edits:
        assert "Edit(" not in repr(edit)
        assert repr(edit).startswith("tl.")
