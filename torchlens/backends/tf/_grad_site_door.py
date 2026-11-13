"""``Trace.do`` with a ``tl.grad`` site on a TensorFlow trace.

``intervention._grad_site_neutral.do_signal_plan_off_torch`` imports this
module by name and hands it the whole ``do()`` plan. TensorFlow has no
backward graph on a finished trace, so the door arms the gradient half in the
shared grad-site store (``armed_grad_sites(trace)`` lists it, and a backward
or reverse engine that computes gradients for the trace replays it) and
returns the same trace, as torch's ``set_only`` engine does.

Two shapes refuse with ``run_capability_unavailable`` before anything is
armed: a plan that also edits activations (TensorFlow edits a finished trace
on a fork, so the halves cannot share one trace; pass them as two calls) and
a ``loss=`` (TensorFlow computes gradients with ``grad_options=`` at capture).

Nothing imports this module statically, so it adds no import-cycle edge.
"""

from __future__ import annotations

from typing import Any

from ...errors import RunCapabilityUnavailableError
from ...intervention._grad_site_neutral import arm_grad_sites, build_grad_site_edits
from ._grad_sites import resolve_forward_labels, selector_operands, tf_grad_apply

_CODE = "run_capability_unavailable"
_STAGE = "tf_grad_site_door"


def do_signal_plan(trace: Any, plan: Any, request: Any) -> Any:
    """Arm the plan's ``tl.grad`` half on ``trace`` and return ``trace``.

    Parameters
    ----------
    trace:
        The TensorFlow trace ``do`` was called on.
    plan:
        The ``do()`` plan split into its activation and gradient halves.
    request:
        The ``do()`` request; only its ``loss`` is read.

    Raises
    ------
    RunCapabilityUnavailableError
        When the plan also edits activations or a loss is given.
    """

    if plan.forward is not None or getattr(request, "loss", None) is not None:
        raise RunCapabilityUnavailableError(
            "tl.grad edit sites are unavailable: on the tf backend Trace.do arms a "
            "tl.grad site on its own, without loss=; edit activations in a separate "
            "do() call, and compute edited gradients with grad_options= at capture",
            code=_CODE,
            detection_stage=_STAGE,
        )
    import tensorflow as tf  # deferred: the door runs only for a TensorFlow trace

    edits = []
    for index, (site, edit) in enumerate(_pairs(plan.backward)):
        edits.extend(
            build_grad_site_edits(
                site,
                tf_grad_apply(edit, tf),
                lambda inner: resolve_forward_labels(trace, inner),
                operands=selector_operands,
                name=f"tf_do_grad_{index}",
            )
        )
    arm_grad_sites(trace, tuple(edits))
    return trace


def _pairs(half: tuple[Any, Any]) -> list[tuple[Any, Any]]:
    """The ``(site, edit)`` pairs of one plan half (a pair, pair list or ``Recipe``)."""

    site, edit = half
    rules = getattr(site, "rules", None)
    if rules is not None:
        return [(rule.target, rule.edit) for rule in rules]
    if edit is None and isinstance(site, list):
        return list(site)
    return [(site, edit)]


__all__ = ["do_signal_plan"]
