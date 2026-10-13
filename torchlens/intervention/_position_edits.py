"""Token-position editing (U42.1 position editing, phase 2).

``at_positions(edit, positions, ...)`` restricts any same-shape edit to chosen
token positions of a ``(batch, tokens, ...)`` site value: the edit computes its
full replacement as usual, and only the selected positions take it (the others
keep the original value). Positions are physical indices (an int, a list of
ints, a slice, or a per-row ``(batch,)`` index tensor) or the padding-aware
``"first"`` / ``"last"`` real token of each row, resolved from an attention
mask under the same three-valued mask policy as token pooling (phase 1): pass
``mask=``, or assert an unpadded batch with ``assume_no_padding=True``;
neither refuses typed. DOCUMENTED-UNSTABLE spelling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .._errors import InvalidArgumentError
from .errors import HookValueError
from .hooks import HookContext, normalize_hook
from .types import HelperSpec

_NAMED = ("first", "last")


def _validate_positions(positions: Any) -> Any:
    """Return ``positions`` in a canonical form, refusing unsupported shapes."""

    if isinstance(positions, str):
        if positions in _NAMED:
            return positions
    elif isinstance(positions, slice):
        return positions
    elif isinstance(positions, int) and not isinstance(positions, bool):
        return (int(positions),)
    elif isinstance(positions, torch.Tensor):
        if positions.ndim == 1 and not positions.is_floating_point():
            return positions.detach().to(torch.int64).clone()
    elif isinstance(positions, (list, tuple)) and positions:
        if all(isinstance(item, int) and not isinstance(item, bool) for item in positions):
            return tuple(int(item) for item in positions)
    raise InvalidArgumentError(
        f"at_positions positions= must be an int, a non-empty list of ints, a slice, a "
        f"per-row (batch,) integer tensor, 'first' or 'last'; received {positions!r}",
        code="position_edit_invalid",
        remedy="pass e.g. positions=-1, positions=[0, 3], positions=slice(2, None), "
        "or positions='last' with mask=attention_mask",
        argument="positions",
    )


def _row_mask(
    positions: Any, out: torch.Tensor, axis: int, mask: torch.Tensor | None, assume: bool
) -> torch.Tensor:
    """Return the boolean ``(batch, tokens)`` selection for one site value."""

    batch, tokens = int(out.shape[0]), int(out.shape[axis])
    selected = torch.zeros(batch, tokens, dtype=torch.bool)
    if isinstance(positions, str):
        from ..transforms._context import TransformContext
        from ..transforms._pooling import _resolve_mask, _token_positions

        context = TransformContext(mask=mask) if mask is not None else None
        valid = _resolve_mask("at_positions", out, axis, context, assume)
        rows = _token_positions(positions, valid, out, axis).to(torch.int64).cpu()
        selected[torch.arange(batch), rows] = True
        return selected
    if isinstance(positions, slice):
        selected[:, positions] = True
        return selected
    if isinstance(positions, torch.Tensor):
        if int(positions.shape[0]) != batch:
            raise HookValueError(
                f"at_positions per-row index has {int(positions.shape[0])} rows; the site "
                f"value has batch {batch}",
                code="position_edit_out_of_range",
            )
        rows = positions
    else:
        rows = torch.tensor(positions, dtype=torch.int64)
    if bool(((rows < -tokens) | (rows >= tokens)).any()):
        raise HookValueError(
            f"at_positions positions {rows.tolist()} fall outside the {tokens} token "
            f"positions on axis {axis}",
            code="position_edit_out_of_range",
        )
    rows = rows % tokens
    if isinstance(positions, torch.Tensor):
        selected[torch.arange(batch), rows] = True
    else:
        selected[:, rows] = True
    return selected


def at_positions(
    edit: Any,
    positions: Any,
    *,
    mask: torch.Tensor | None = None,
    assume_no_padding: bool = False,
    axis: int = 1,
) -> HelperSpec:
    """Restrict an edit to chosen token positions of each batch row.

    Parameters
    ----------
    edit:
        Any same-shape forward edit (``tl.zero_ablate()``, ``tl.steer(...)``,
        ``tl.patch_from(...)``, or a hook callable taking ``(out, *, hook)``).
    positions:
        An int, a non-empty list of ints, a slice, a per-row ``(batch,)``
        integer tensor, or ``"first"`` / ``"last"`` (the first / last REAL token
        of each row, resolved from ``mask``).
    mask:
        ``(batch, tokens)`` attention mask for ``"first"`` / ``"last"``.
    assume_no_padding:
        Assert that every position is real, licensing physical first/last
        without a mask (recorded in the edit's identity).
    axis:
        Token axis of the site value (default 1, ``(batch, tokens, hidden)``).

    Returns
    -------
    HelperSpec
        The position-restricted edit.

    Raises
    ------
    InvalidArgumentError
        ``position_edit_invalid`` for unsupported positions or axis.
    """

    canonical = _validate_positions(positions)
    if isinstance(axis, bool) or not isinstance(axis, int) or axis == 0:
        raise InvalidArgumentError(
            f"at_positions axis= must be a non-batch int axis; received {axis!r}",
            code="position_edit_invalid",
            remedy="pass the token axis, e.g. axis=1 for (batch, tokens, hidden)",
            argument="axis",
        )
    inner = normalize_hook(edit, direction="forward")
    inner_name = getattr(edit, "helper_name", getattr(edit, "__qualname__", type(edit).__name__))

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook restricting the inner edit to the positions."""

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Apply the inner edit, keeping it only at the selected positions."""

            edited = inner(out, hook=hook)
            if not isinstance(edited, torch.Tensor) or edited.shape != out.shape:
                raise HookValueError(
                    f"at_positions needs a same-shape edit; {inner_name!r} returned "
                    f"{getattr(edited, 'shape', type(edited).__name__)} for a value of shape "
                    f"{tuple(out.shape)}",
                    code="position_edit_shape_mismatch",
                )
            token_axis = axis % out.ndim
            selected = _row_mask(canonical, out, token_axis, mask, assume_no_padding)
            view = [1] * out.ndim
            view[0], view[token_axis] = selected.shape[0], selected.shape[1]
            where = selected.reshape(view).to(out.device)
            return torch.where(where, edited.to(dtype=out.dtype), out)

        return _hook

    from .helpers import _helper_spec

    return _helper_spec(
        "at_positions",
        args=(edit,),
        kwargs={
            "positions": canonical,
            "mask": mask,
            "assume_no_padding": assume_no_padding,
            "axis": axis,
        },
        factory=factory,
        portability="opaque_audit",
    )
