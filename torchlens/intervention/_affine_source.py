"""``affine_source``: under a subspace, set the coordinates to ``weight @ x + bias``.

A learned low-rank intervention such as pyreft's LoReFT,
``h' = h + R (W h + b - R^T h)``, is in TorchLens terms
``do(tl.subspace(site, R.T, origin=...), affine_source(W, b))``: the subspace
names the rows ``R^T`` the edit acts through, and the new coordinates are an
affine map of the FULL site value ``h`` (not of the old coordinates). That is a
third kind of verb under a subspace, a *source* verb: the projection hands it
the site value ``x`` and it returns ``[..., k]`` coordinates, which the
projection writes back as ``x + Q^T (source(x) - Q x)``.

A ``[k, d]`` map has no meaning on the whole site, so outside a subspace the
verb refuses ``affine_source_needs_subspace`` (at ``do()`` for a selection
target, and at fire time for every other door). Malformed arguments refuse
``affine_source_invalid``. The verb is builtin: it saves inside the
``subspace_edit`` payload with ``weight`` and ``bias`` as tensor sidecars.

DOCUMENTED-UNSTABLE semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .._errors import InvalidArgumentError
from .helpers import _helper_spec
from .types import Edit

__all__ = ["SourceCall", "affine_source", "check_source_shape", "needs_subspace"]

AFFINE_SOURCE_NAME = "affine_source"
_GUIDE = "see the subspace section of docs/intervention_explainers.md"
_WORKING = (
    "fork.do(tl.subspace(site, rotation.T, origin=...), "
    "tl.intervention.affine_source(weight, bias=bias))"
)


def _invalid(
    argument: str, why: str, *, code: str = "affine_source_invalid"
) -> InvalidArgumentError:
    """Return the refusal for a malformed ``affine_source`` argument.

    ``code`` is spelled at each raise site so the refusal census sees it.
    """

    return InvalidArgumentError(
        f"affine_source {argument}= is malformed: {why}",
        code=code,
        remedy=(
            "pass weight as a finite real [k, d] tensor (k basis rows, d site features) "
            f"and bias as None or a finite real [k] tensor: {_WORKING} ({_GUIDE})"
        ),
        argument=argument,
    )


def _checked_tensor(value: Any, argument: str, ndim: int) -> torch.Tensor:
    """Return ``value`` if it is a finite, dense, real tensor of rank ``ndim``."""

    if not isinstance(value, torch.Tensor):
        raise _invalid(
            argument, f"expected a tensor, got {type(value).__name__}", code="affine_source_invalid"
        )
    if value.ndim != ndim or 0 in value.shape:
        raise _invalid(
            argument,
            f"expected a non-empty {ndim}-d tensor, got shape {tuple(value.shape)!r}",
            code="affine_source_invalid",
        )
    if value.layout != torch.strided or value.device.type == "meta":
        raise _invalid(
            argument,
            f"expected dense values, got {value.layout} on {value.device}",
            code="affine_source_invalid",
        )
    if not value.is_floating_point():
        raise _invalid(
            argument,
            f"expected a real float dtype, got {value.dtype}",
            code="affine_source_invalid",
        )
    if not bool(torch.isfinite(value).all()):
        raise _invalid(argument, "it contains NaN or Inf", code="affine_source_invalid")
    # A copy: a later in-place change to the caller's tensor never changes the edit.
    return value.detach().clone()


def needs_subspace(
    site: str, *, code: str = "affine_source_needs_subspace"
) -> InvalidArgumentError:
    """Return the refusal for ``affine_source`` applied without a subspace.

    ``code`` is always ``affine_source_needs_subspace``; raise sites spell it
    so the refusal census sees a coded site.
    """

    del code  # spelled at the raise site for the refusal census; always the same code
    return InvalidArgumentError(
        f"affine_source at {site!r} was applied to the whole site; its [k, d] weight "
        "maps the site value to k subspace coordinates, which only a subspace "
        "selection defines",
        code="affine_source_needs_subspace",
        remedy=f"name the basis the coordinates live on: {_WORKING} ({_GUIDE})",
        argument="target",
        site=site,
    )


def affine_source(weight: torch.Tensor, *, bias: torch.Tensor | None = None) -> Edit:
    """Create the source verb whose new subspace coordinates are ``weight @ x + bias``.

    Use it under a subspace selection: ``fork.do(tl.subspace(site, R.T,
    origin=...), tl.intervention.affine_source(W, bias=b))`` writes
    ``x + R (W x + b - R^T x)`` (LoReFT), where ``x`` is the full site value.
    The basis rows and ``weight`` must have the same shape ``[k, d]``.

    Parameters
    ----------
    weight:
        Finite real ``[k, d]`` tensor: row ``j`` maps the site value to the new
        coordinate ``j``.
    bias:
        Optional finite real ``[k]`` tensor added to the new coordinates.

    Returns
    -------
    Edit
        Builtin forward verb named ``affine_source``. Outside a subspace it
        refuses ``affine_source_needs_subspace``.

    Raises
    ------
    InvalidArgumentError
        ``affine_source_invalid`` for a malformed ``weight`` or ``bias``.
    """

    weight = _checked_tensor(weight, "weight", 2)
    if bias is not None:
        bias = _checked_tensor(bias, "bias", 1)
        if bias.shape[0] != weight.shape[0]:
            raise _invalid(
                "bias",
                f"expected [{weight.shape[0]}] to match weight's {weight.shape[0]} rows, "
                f"got {tuple(bias.shape)!r}",
                code="affine_source_invalid",
            )

    def factory() -> Callable[..., torch.Tensor]:
        """Return the hook; it computes only when a subspace projection calls it."""

        def _hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
            """Return ``out @ weight.T + bias``; refuse a whole-site call."""

            if not isinstance(hook, SourceCall):
                layer_log = getattr(hook, "layer_log", None) or {}
                raise needs_subspace(
                    str(layer_log.get("layer_label") or "<unknown site>"),
                    code="affine_source_needs_subspace",
                )
            coords = out @ weight.to(device=out.device, dtype=out.dtype).T
            if bias is None:
                return coords
            return coords + bias.to(device=out.device, dtype=out.dtype)

        return _hook

    return _helper_spec(
        AFFINE_SOURCE_NAME,
        args=(weight,),
        kwargs={"bias": bias},
        factory=factory,
        batch_independent=True,
    )


class SourceCall:
    """The hook context a subspace projection hands a source verb.

    It marks the call as coming from a projection (which moved the basis axis
    last) and forwards every attribute to the real :class:`HookContext`.
    """

    __slots__ = ("context",)

    def __init__(self, context: Any) -> None:
        self.context = context

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "context"), name)


def check_source_shape(edit: Edit, rows: torch.Tensor) -> None:
    """Refuse an ``affine_source`` whose weight is not shaped like the basis rows.

    Raises
    ------
    InvalidArgumentError
        ``affine_source_invalid`` when ``weight``'s shape differs from ``rows``.
    """

    weight = edit.args[0]
    if tuple(weight.shape) != tuple(rows.shape):
        raise _invalid(
            "weight",
            f"its shape {tuple(weight.shape)!r} differs from the subspace basis "
            f"{tuple(rows.shape)!r}; the weight maps the site's d features to the "
            "basis's k coordinates",
            code="affine_source_invalid",
        )
