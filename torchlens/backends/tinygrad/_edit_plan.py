"""Resolution of ``trace(intervene=tl.when(...))`` to a tinygrad edit.

This is the ``can_intervene`` capability implementation of the tinygrad
backend: it checks that the intervention is a static selector with a supported
action and returns the selector with a lazy ``tensor -> tensor`` edit
(the ``_helper_edits`` table: ``zero_ablate``, ``scale``, ``add``, ``clamp``,
``mean_ablate``, ``replace_with``; or a user transform). ``_edits`` applies the edit during capture.

Only static selectors are supported. A value-dependent predicate needs a
concrete value at predicate time, which a lazy tinygrad forward does not have
before its one realization; it refuses typed, as do ``stop_after=``,
module-output sites, gradient sites and helpers without a tinygrad application.
"""

from __future__ import annotations

from typing import Any

from ..._ir.selector_eval import first_selector_kind_outside
from ..._postprocess._selective_save import _STATIC_INTERVENTION_SELECTOR_KINDS
from ..._selection.selectors import BaseSelector
from ...intervention._types import Edit, InterventionDecision
from ...intervention.hooks import make_hook_context, normalize_hook, split_site_branches
from .._backend_protocol import refusal_code_for_capability
from ._edits import EDIT_REFUSAL, TinygradEdit, TinygradEditPlan, refuse_edit
from ._helper_edits import helper_application

__tl_layer__ = "L4"

_EDIT_SELECTOR_KINDS = _STATIC_INTERVENTION_SELECTOR_KINDS - frozenset({"module"})


def resolve_tinygrad_edit_plan(intervene: Any, halt: Any) -> TinygradEditPlan:
    """Validate ``trace(intervene=...)`` and resolve its tinygrad edit.

    Parameters
    ----------
    intervene
        Public ``trace(intervene=...)`` value.
    halt
        Public ``trace(stop_after=...)`` value.

    Returns
    -------
    TinygradEditPlan
        The static selector and its resolved edit.

    Raises
    ------
    BackendUnsupportedError
        For ``stop_after=`` and any edit outside the static-selector contract.
    """

    if halt is not None:
        raise refuse_edit(
            "trace(stop_after=...) is not supported; a tinygrad forward builds its whole "
            "lazy graph before anything runs. Capture the full forward instead.",
            code=EDIT_REFUSAL,
        )
    selector = getattr(intervene, "selector", None)
    decision = getattr(intervene, "decision", None)
    if not isinstance(selector, BaseSelector) or not isinstance(decision, InterventionDecision):
        raise refuse_edit(
            "trace(intervene=...) supports only tl.when(static_selector, edit). A "
            "value-dependent predicate needs a concrete value at predicate time, which a "
            "lazy tinygrad forward does not have before it is realized.",
            code=EDIT_REFUSAL,
        )
    if any(universe != "activation" for universe, _branch in split_site_branches(selector)):
        raise refuse_edit(
            f"the site {selector!r} addresses a gradient; tinygrad captures no backward pass.",
            code=refusal_code_for_capability("backward_capture"),
        )
    unsupported = first_selector_kind_outside(selector, allowed=_EDIT_SELECTOR_KINDS)
    if unsupported is not None:
        raise refuse_edit(
            "trace(intervene=...) supports the static selectors tl.func, tl.label, "
            "tl.label_contains, tl.in_module and their &, |, ~ combinations; "
            f"unsupported selector kind {unsupported!r}.",
            code=EDIT_REFUSAL,
        )
    return TinygradEditPlan(selector=selector, edit=_resolve_decision(decision))


def _resolve_decision(decision: InterventionDecision) -> TinygradEdit:
    """Resolve one ``tl.when`` decision to a tinygrad edit."""

    if decision.keep_grad or decision.isolate:
        raise refuse_edit(
            "edits do not support keep_grad or isolate (torch autograd semantics).",
            code=EDIT_REFUSAL,
        )
    if decision.action not in {"add_hook", "transform"} or decision.hook is None:
        raise refuse_edit(
            f"edits support helper and callable actions only, not {decision.action!r}.",
            code=EDIT_REFUSAL,
        )
    hook = decision.hook
    if isinstance(hook, Edit):
        return _resolve_helper(hook)
    normalized = normalize_hook(hook)
    identity = f"callable:{getattr(hook, '__qualname__', type(hook).__name__)}"

    def _apply(out: Any) -> Any:
        """Call the user transform with a post-forward hook context."""

        context = make_hook_context(name=identity, timing="post", signal="forward")
        return normalized(out, ctx=context)

    return TinygradEdit(identity, _apply)


def _resolve_helper(spec: Edit) -> TinygradEdit:
    """Resolve a built-in helper to its lazy tinygrad application (``_helper_edits``)."""

    return TinygradEdit(f"helper:{spec.name}", helper_application(spec))
