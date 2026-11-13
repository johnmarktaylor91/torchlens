"""Resolution of ``trace(intervene=tl.when(...))`` to a tinygrad edit.

This is the ``can_intervene`` capability implementation of the tinygrad
backend: it checks the intervention's condition and action and returns the
condition with a lazy ``tensor -> tensor`` edit (the ``_helper_edits`` table:
``zero_ablate``, ``scale``, ``add``, ``clamp``, ``mean_ablate``,
``replace_with``; or a user transform). ``_edits`` applies the edit during
capture.

A condition is a static selector (``tl.func``, ``tl.label``,
``tl.label_contains``, ``tl.in_module`` and their combinations), or a
condition that reads predicate contexts (a plain callable or a ``tl.where``
leaf), which ``_value_edits`` matches on the planning capture's contexts. A
``tl.grad(...)`` site edits that gradient in derived gradients
(``_grad_site_grads``). ``stop_after=``, module-output sites, sites that mix
activation and gradient branches and helpers without a tinygrad application
refuse typed.
"""

from __future__ import annotations

from typing import Any

from ..._capture.projections import _build_record_context
from ..._ir.selector_eval import first_selector_kind_outside
from ..._postprocess._selective_save import _STATIC_INTERVENTION_SELECTOR_KINDS
from ..._selection.selectors import BaseSelector
from ...intervention._types import Edit, InterventionDecision
from ...intervention.hooks import make_hook_context, normalize_hook, split_site_branches
from .._backend_protocol import refusal_code_for_capability
from ._edits import EDIT_REFUSAL, TinygradEdit, TinygradEditPlan, refuse_edit
from ._grad_site_grads import attach_grad_site_derived_grads
from ._helper_edits import helper_application
from ._value_edits import condition_matched_labels

__tl_layer__ = "L4"

_EDIT_SELECTOR_KINDS = _STATIC_INTERVENTION_SELECTOR_KINDS - frozenset({"module"})
_CONDITION_KINDS = _EDIT_SELECTOR_KINDS | frozenset({"predicate"})


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
        The condition and its resolved edit.

    Raises
    ------
    BackendUnsupportedError
        For ``stop_after=`` and any condition or edit outside the contract above.
    """

    if halt is not None:
        raise refuse_edit(
            "trace(stop_after=...) is not supported; a tinygrad forward builds its whole "
            "lazy graph before anything runs. Capture the full forward instead.",
            code=EDIT_REFUSAL,
        )
    selector = getattr(intervene, "selector", None)
    decision = getattr(intervene, "decision", None)
    if not callable(selector) or not isinstance(decision, InterventionDecision):
        raise refuse_edit(
            "trace(intervene=...) supports only tl.when(condition, edit) with one selector or "
            "callable condition.",
            code=EDIT_REFUSAL,
        )
    if isinstance(selector, BaseSelector) and _is_gradient_site(selector):
        edit = _resolve_decision(decision)
        return TinygradEditPlan(selector, edit, grad_site=attach_grad_site_derived_grads)
    reads_values = not isinstance(selector, BaseSelector) or _selector_reads_values(selector)
    return TinygradEditPlan(
        selector=selector,
        edit=_resolve_decision(decision),
        matcher=_match_on_contexts if reads_values else None,
    )


def _match_on_contexts(trace: Any, condition: Any) -> frozenset[str]:
    """Raw labels whose rebuilt predicate context satisfies ``condition`` (``_value_edits``)."""

    return condition_matched_labels(trace, condition, _build_record_context)


def _is_gradient_site(selector: BaseSelector) -> bool:
    """Whether every branch of ``selector`` addresses a gradient (a ``tl.grad`` site)."""

    return all(universe != "activation" for universe, _branch in split_site_branches(selector))


def _selector_reads_values(selector: BaseSelector) -> bool:
    """Check a selector condition; whether it has a leaf that reads values (``tl.where``)."""

    if any(universe != "activation" for universe, _branch in split_site_branches(selector)):
        raise refuse_edit(
            f"the site {selector!r} mixes activation and gradient branches; give each its "
            "own capture.",
            code=refusal_code_for_capability("backward_capture"),
        )
    unsupported = first_selector_kind_outside(selector, allowed=_CONDITION_KINDS)
    if unsupported is not None:
        raise refuse_edit(
            "trace(intervene=...) supports the static selectors tl.func, tl.label, "
            "tl.label_contains, tl.in_module, value conditions (tl.where or a callable) and "
            f"their &, |, ~ combinations; unsupported selector kind {unsupported!r}.",
            code=EDIT_REFUSAL,
        )
    return first_selector_kind_outside(selector, allowed=_EDIT_SELECTOR_KINDS) is not None


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
