"""grind-r5 b7 R55: trace-entry validation, overrides membership, context unwind.

* ``tl.trace(model, q, k, v)`` put tensor ``k`` in ``input_kwargs`` and ``v``
  in the deprecated ``layers_to_save`` slot, then crashed DEEP with an
  ambient-import-dependent error that never named the mistake (this
  misdirected two round-1 review lanes).
* ``torch.overrides`` membership tables answered False for every wrapped
  function while wrappers were installed (the documented
  ``__torch_function__`` author checks).
* Overlapping ``active_intervention_context`` publications unwinding out of
  stack order re-published a DEAD context (two-thread probe).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._errors import ArgumentTypeError
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.intervention.runtime import active_intervention_context


class _TinyModel(nn.Module):
    """Two-input model used for the misspelling probes."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        out = self.lin(x)
        return out + y if y is not None else out


def test_tensor_input_kwargs_refuses_typed_naming_the_tuple_spelling() -> None:
    """The natural multi-input misspelling must refuse at entry, typed."""

    model = _TinyModel()
    q, k = torch.randn(1, 4), torch.randn(1, 4)
    with pytest.raises(ArgumentTypeError, match="tuple"):
        tl.trace(model, q, k)


def test_tensor_in_layers_to_save_slot_refuses_typed() -> None:
    """A third positional tensor lands in layers_to_save: name the mistake."""

    model = _TinyModel()
    q, k, v = torch.randn(1, 4), torch.randn(1, 4), torch.randn(1, 4)
    with pytest.raises(ArgumentTypeError, match="tuple"):
        tl.trace(model, q, {"y": k}, v)  # dict kwargs fine; tensor in 4th slot


def test_tuple_spelling_still_works() -> None:
    """The documented multi-input spelling stays green."""

    trace = tl.trace(_TinyModel(), (torch.randn(1, 4), torch.randn(1, 4)))
    assert trace.outcome.status.name == "COMPLETE"


@pytest.mark.smoke
def test_overrides_membership_resolves_wrappers_while_wrapped() -> None:
    """The documented author checks must answer True in both epochs."""

    wrap_torch()
    tl.trace(_TinyModel(), torch.randn(1, 4))  # ensure shims installed
    functional = torch.nn.functional

    assert functional.relu in torch.overrides.get_testing_overrides(), (
        "F.relu not a member of get_testing_overrides() while wrapped"
    )
    overridable = torch.overrides.get_overridable_functions()
    assert functional in overridable
    assert functional.relu in overridable[functional], (
        "F.relu not a member of get_overridable_functions()[F] while wrapped"
    )
    # Iterated CONTENTS stay original-keyed: no torchlens wrapper may appear
    # as a key (the census-gate invariant, also pinned in
    # tests/test_wrap_state_compat.py).
    assert all(
        id(fn) not in _state._decorated_to_orig for fn in torch.overrides.get_testing_overrides()
    )


@pytest.mark.smoke
def test_out_of_order_context_unwind_never_resurrects_a_dead_plan() -> None:
    """A non-top context exit splices out; the final exit restores the true
    pre-window state instead of re-publishing a dead context."""

    spec_a, plan_a = object(), object()
    spec_b, plan_b = object(), object()
    base_spec = _state._active_intervention_spec
    base_plan = _state._active_hook_plan

    context_a = active_intervention_context(intervention_spec=spec_a, hook_plan=plan_a)
    context_b = active_intervention_context(intervention_spec=spec_b, hook_plan=plan_b)
    context_a.__enter__()
    context_b.__enter__()

    context_a.__exit__(None, None, None)  # out of stack order
    assert _state._active_hook_plan is plan_b, "the LIVE context was clobbered"
    assert _state._active_intervention_spec is spec_b

    context_b.__exit__(None, None, None)
    assert _state._active_intervention_spec is base_spec, (
        "a dead context's spec was re-published after both exits"
    )
    assert _state._active_hook_plan is base_plan


def test_lifo_context_nesting_still_restores_exactly() -> None:
    """Ordinary nested publication keeps exact save/restore semantics."""

    base_spec = _state._active_intervention_spec
    base_plan = _state._active_hook_plan
    spec_a, plan_a = object(), object()
    spec_b, plan_b = object(), object()

    with active_intervention_context(intervention_spec=spec_a, hook_plan=plan_a):
        with active_intervention_context(intervention_spec=spec_b, hook_plan=plan_b):
            assert _state._active_hook_plan is plan_b
        assert _state._active_hook_plan is plan_a
        assert _state._active_intervention_spec is spec_a
    assert _state._active_intervention_spec is base_spec
    assert _state._active_hook_plan is base_plan
