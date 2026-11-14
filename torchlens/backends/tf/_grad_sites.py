"""TensorFlow ``tl.grad`` edit sites: arming and the gradient-replay tape edit.

A ``tl.when(tl.grad(selector), edit)`` clause edits the gradient flowing back
into the ops ``selector`` names. TensorFlow records no backward graph, so a
live capture keeps the clause on the trace instead of refusing it. Where the
backend computes gradients (the derived-gradient replay under a
``tf.GradientTape``), each named op's output is passed through an identity
whose ``tf.custom_gradient`` applies the edit to the incoming gradient: the
tape then propagates the edited gradient upstream, as torch's tensor hook does.
The same clauses are armed on the trace with the shared grad-site store, so
``armed_grad_sites(trace)`` lists them and the neutral reverse engine can
replay them.

This module imports nothing from the selector or hook packages (the import
cycle ratchet): the caller passes ``tl.when`` in, and composite selectors are
read through their ``selector_kind``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ._helper_hooks import tf_helper_hook

_STAGED = "_tf_grad_site_clauses"
"""Trace attribute holding the capture's ``tl.grad`` clauses, in clause order."""
_ARMED_KEYS = "_tf_grad_sites_armed"
"""Trace attribute holding ``(clause, (label, pass_index))`` for each site already armed."""


def grad_site_neutral() -> Any:
    """The backend-neutral gradient-site module (imported on first use).

    The TensorFlow modules reach it only through this function, so the backend
    tier keeps no eager import of the intervention tier.
    """

    from ...intervention import _grad_site_neutral  # deferred: L4 module, tl.grad clauses only

    return _grad_site_neutral


def stage_tf_grad_clauses(trace: Any, clauses: Sequence[Any]) -> None:
    """Keep a capture's ``tl.grad`` clauses on its trace (none: leave it unset)."""

    if clauses:
        trace.__dict__[_STAGED] = tuple(clauses)


def staged_tf_grad_clauses(trace: Any) -> tuple[Any, ...]:
    """Return the ``tl.grad`` clauses a capture kept on ``trace``."""

    return tuple(trace.__dict__.get(_STAGED, ()))


def selector_operands(target: Any) -> tuple[str | None, tuple[Any, ...]]:
    """Read a composite selector as ``(connective, children)``; a leaf is ``(None, ())``."""

    kind = getattr(target, "selector_kind", None)
    if kind in ("and", "or") and hasattr(target, "selectors"):
        return kind, tuple(target.selectors)
    if kind == "not" and hasattr(target, "selector"):
        return "not", (target.selector,)
    return None, ()


def tf_grad_apply(hook: Any, tf: Any) -> Callable[[Any, Any], Any]:
    """Return ``apply(grad, ctx) -> grad`` for a clause's helper or callable."""

    if callable(hook) and not (hasattr(hook, "name") and hasattr(hook, "kwargs")):
        return lambda grad, ctx: hook(grad, ctx=ctx)
    edit = tf_helper_hook(hook, tf)
    return lambda grad, ctx: edit(grad)


def arm_tf_grad_sites(trace: Any, tf: Any) -> None:
    """Arm the trace's staged ``tl.grad`` clauses in the shared grad-site store.

    Labels are the finished trace's op labels (the TensorFlow replay adapter's
    output-slot labels). Call it on a finalized trace. Idempotent per trace and
    site: a clause already armed on a label (and pass) is not armed again, so a
    capture that arms its clauses and a later derived-gradient replay that
    stages the same clauses arm each site once. After ``clear_grad_sites`` the
    clauses arm afresh.
    """

    if not staged_tf_grad_clauses(trace):
        return
    neutral = grad_site_neutral()
    armed = list(trace.__dict__.get(_ARMED_KEYS, ())) if neutral.armed_grad_sites(trace) else []
    edits: list[Any] = []
    for index, clause in enumerate(staged_tf_grad_clauses(trace)):
        for edit in neutral.build_grad_site_edits(
            clause.selector,
            tf_grad_apply(clause.decision.hook, tf),
            lambda inner: resolve_forward_labels(trace, inner),
            operands=selector_operands,
            name=f"tf_grad_{index}",
        ):
            key = (edit.label, edit.pass_index)
            if any(done is clause and done_key == key for done, done_key in armed):
                continue
            armed.append((clause, key))
            edits.append(edit)
    trace.__dict__[_ARMED_KEYS] = tuple(armed)
    if edits:
        neutral.arm_grad_sites(trace, tuple(edits))


def grad_replay_clauses(trace: Any, tf: Any, when: Callable[[Any, Any], Any]) -> list[Any]:
    """Activation clauses that apply the staged ``tl.grad`` edits on the replay tape.

    Each forward op a clause names gets an identity whose custom gradient is
    the edit. Sites restricted to a later backward pass never fire on the one
    pass the derived-gradient replay computes, so they add no clause.
    """

    clauses = []
    for clause in staged_tf_grad_clauses(trace):
        apply = tf_grad_apply(clause.decision.hook, tf)
        for inner, pass_index in grad_site_neutral().grad_site_targets(
            clause.selector, selector_operands
        ):
            if pass_index in (None, 1):
                clauses.append(when(inner, _tape_edit(apply, tf)))
    return clauses


def gradient_replay_intervene(
    trace: Any, tf: Any, rest: Any, grad: tuple[Any, ...], when: Callable[[Any, Any], Any]
) -> Any:
    """The replay's ``intervene``: ``rest`` plus the ``grad`` clauses as tape edits.

    The ``grad`` clauses are kept on and armed on the finished ``trace`` too.
    """

    stage_tf_grad_clauses(trace, grad)
    arm_tf_grad_sites(trace, tf)
    clauses = list(rest) if isinstance(rest, (list, tuple)) else [] if rest is None else [rest]
    clauses.extend(grad_replay_clauses(trace, tf, when))
    return clauses or None


def _tape_edit(apply: Callable[[Any, Any], Any], tf: Any) -> Callable[..., Any]:
    """An activation action: identity forward, ``apply`` on the incoming gradient."""

    def action(out: Any, *, ctx: Any = None) -> Any:
        @tf.custom_gradient
        def identity(value: Any) -> Any:
            def grad_fn(upstream: Any) -> Any:
                return apply(upstream, ctx)

            return tf.identity(value), grad_fn

        return identity(out)

    return action


def resolve_forward_labels(trace: Any, selector: Any) -> tuple[str, ...]:
    """Op labels a forward selector names on a finished trace, in trace order."""

    sites = trace.find_sites(selector, max_fanout=len(trace._layer_list) + 1)
    found = set(sites.labels())
    return tuple(
        op.label
        for op in trace._layer_list
        if op.label in found or getattr(op, "layer_label", None) in found
    )


__all__ = (
    "arm_tf_grad_sites",
    "grad_replay_clauses",
    "resolve_forward_labels",
    "gradient_replay_intervene",
    "selector_operands",
    "stage_tf_grad_clauses",
    "staged_tf_grad_clauses",
    "tf_grad_apply",
)
