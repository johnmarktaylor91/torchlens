"""The gradient contract for edits: ``keep_grad(edit)`` and its connectivity check.

An edit either keeps the autograd graph through the values it writes (``scale``,
``clamp``, ``steer``, a subspace component edit, ...) or writes values computed
without the site's graph (``zero_ablate``, ``replace_with``, ``patch_from``, ...),
which cuts every gradient path from the output back through the edited
elements to the parameters upstream of the site. The second kind is correct for
analysis and silent for training. ``keep_grad(edit)`` is how a caller says they
need the gradient: it returns the same edit with a fire-time check that raises
``edit_detached_from_graph`` when the edit writes values carrying no graph
while the site value had one (or when the site value has no graph to keep).
The matrix of which verb keeps the graph under which door is
``docs/gradient_contract.md``.

DOCUMENTED-UNSTABLE spelling.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from .._errors import InvalidArgumentError
from .types import Edit

__all__ = ["keep_grad"]

_GUIDE = "see docs/gradient_contract.md"
_ZERO_REMEDY = "tl.intervention.keep_grad(tl.scale(0.0)) writes zeros and keeps the graph"
_FIXED_REMEDY = (
    "tl.intervention.keep_grad(tl.intervention.compose(tl.scale(0.0), tl.add(value))) "
    "writes a fixed value and keeps the graph with a zero gradient"
)


def _site_label(hook: Any) -> str:
    """Return the user's label for the firing site."""

    layer_log = getattr(hook, "layer_log", None) or {}
    label = layer_log.get("layer_label") or layer_log.get("label")
    return str(label or getattr(hook, "name", None) or "<site>")


def _refusal(
    where: tuple[str, str], reason: str, why: str, remedy: str, *, code: str
) -> InvalidArgumentError:
    """Build the connectivity refusal for ``where`` = (verb, site label).

    ``code`` is spelled at each raise site so the refusal census sees it.
    """

    verb, site = where
    return InvalidArgumentError(
        f"keep_grad({verb}) at {site!r}: {why}",
        code=code,
        remedy=f"{remedy}; or drop keep_grad(...) to accept that no gradient flows ({_GUIDE})",
        argument="edit",
        verb=verb,
        site=site,
        reason=reason,
    )


def _detached_remedy(verb: str) -> str:
    """Return the graph-keeping spelling for a verb that cut the graph."""

    if verb == "zero_ablate":
        return _ZERO_REMEDY
    return _FIXED_REMEDY


def _check(out: Any, result: Any, hook: Any, verb: str) -> None:
    """Raise when ``result`` carries no graph although the caller asked for one."""

    if not isinstance(out, torch.Tensor) or not isinstance(result, torch.Tensor):
        return
    where = (verb, _site_label(hook))
    if not torch.is_grad_enabled():
        raise _refusal(
            where,
            "grad_disabled",
            "the edit fired with gradients disabled (torch.no_grad, inference_mode or "
            "an inference-only capture), so no gradient can flow",
            "run the forward with gradients enabled and do not pass inference_only=True",
            code="edit_detached_from_graph",
        )
    if not out.requires_grad:
        raise _refusal(
            where,
            "site_has_no_graph",
            "the site value carries no autograd graph (no parameter or input upstream "
            "requires grad, or the saved activations were detached)",
            "keep parameters trainable and capture with "
            "tl.options.CaptureOptions(detach_saved_activations=False)",
            code="edit_detached_from_graph",
        )
    if not result.requires_grad:
        raise _refusal(
            where,
            "edit_detached",
            "the edit wrote values computed without the site's graph, so no gradient "
            "reaches the parameters upstream of the edited elements",
            _detached_remedy(verb),
            code="edit_detached_from_graph",
        )


def _checked_factory(factory: Callable[[], Any], verb: str) -> Callable[[], Any]:
    """Wrap a hook factory with the connectivity check."""

    def checked() -> Callable[..., Any]:
        """Instantiate the inner hook and check what it writes."""

        inner = factory()

        def _hook(out: Any, *, hook: Any) -> Any:
            """Run the edit, then refuse a result that dropped the graph."""

            result = inner(out, hook=hook)
            _check(out, result, hook, verb)
            return result

        return _hook

    return checked


def keep_grad(edit: Any) -> Any:
    """Return ``edit`` with a check that the values it writes keep the autograd graph.

    Parameters
    ----------
    edit:
        A forward ``Edit`` (``tl.scale(2.0)``, ``tl.patch_from(source)``, ...)
        or a hook callable ``fn(out, *, hook)``.

    Returns
    -------
    Edit | Callable
        The same edit (a ``Edit`` stays an ``Edit`` with ``("keep_grad",
        True)`` in its metadata and ``portability="opaque_audit"``, because the
        check is session-time), which raises
        ``InvalidArgumentError`` / ``edit_detached_from_graph`` at fire time
        when the site value carries a graph and the written values do not,
        when the site value has no graph, or when gradients are disabled.

    Raises
    ------
    InvalidArgumentError
        ``keep_grad_edit_invalid`` for a backward-direction edit (it edits
        gradients, not a forward value) or an object that is not an edit.
    """

    if isinstance(edit, Edit) and dict(edit.metadata).get("keep_grad"):
        return edit
    if getattr(edit, "_torchlens_keep_grad", False):
        return edit
    if isinstance(edit, Edit):
        if edit.kind != "forward" or edit.direction == "backward" or edit.factory is None:
            raise InvalidArgumentError(
                f"keep_grad() got {edit.helper_name!r}, which is not a forward edit with a "
                "runtime hook; the check applies to the values a forward edit writes",
                code="keep_grad_edit_invalid",
                remedy=f"wrap a forward edit, e.g. tl.intervention.keep_grad(tl.scale(2.0)) "
                f"({_GUIDE})",
                argument="edit",
            )
        return dataclasses.replace(
            edit,
            factory=_checked_factory(edit.factory, edit.helper_name),
            portability="opaque_audit",
            metadata=tuple(edit.metadata) + (("keep_grad", True),),
        )
    if callable(edit):
        hook_fn = edit
        verb = getattr(edit, "__name__", "hook callable")

        def _checked_callable(out: Any, *, hook: Any) -> Any:
            """Run the user's hook, then refuse a result that dropped the graph."""

            result = hook_fn(out, hook=hook)
            _check(out, result, hook, verb)
            return result

        _checked_callable._torchlens_keep_grad = True  # type: ignore[attr-defined]
        return _checked_callable
    raise InvalidArgumentError(
        f"keep_grad() needs an edit or a hook callable; got {type(edit).__name__}",
        code="keep_grad_edit_invalid",
        remedy=f"wrap a forward edit, e.g. tl.intervention.keep_grad(tl.scale(2.0)) ({_GUIDE})",
        argument="edit",
    )
