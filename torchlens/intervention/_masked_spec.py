"""Portable element-masked builtin edits: the dense mask travels in the edit's keywords.

``do(selection, verb)`` with a builtin verb and an element mask stages the verb
under the engine scatter (``masked_edit._masked_factory``). The derived edit keeps
the verb's name and ``builtin`` portability and stores the dense bool ``mask`` and
its ``mask_sha256`` among its keywords, the same two keys a masked ``subspace_edit``
stores; savers write the mask through the tensor sidecar like any tensor argument.
On load :func:`split_stored_mask` takes the two keys off before the verb's
constructor runs and :func:`with_stored_mask` wraps the rebuilt verb with the same
scatter, so a reload applies bit-identically. A malformed mask, a mask without its
digest (or a digest without its mask) and a mask whose bytes no longer match its
digest refuse ``intervention_helper_arguments_invalid``. The digest detects
corruption and hand edits; it is not a signature. DOCUMENTED-UNSTABLE persisted form.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from .._errors import InvalidArgumentError
from .._selection.mask import _Mask
from .._selection.selection import _apply_invalid
from ._subspace_projection import SUBSPACE_EDIT_NAME, _mask_digest
from .masked_edit import _masked_factory
from .types import Edit

__all__ = ["MASK_KEYS", "masked_builtin_edit", "split_stored_mask", "with_stored_mask"]

#: The keywords a masked builtin edit stores beside the verb's own.
MASK_KEYS = ("mask", "mask_sha256")

_RESAVE = (
    "re-save the edit from the session that built it: fork.do(tl.units(site, mask), edit) "
    "then tl.io.save_intervention(fork, path, level='portable')"
)


def masked_builtin_edit(edit: Edit, mask: _Mask, site_label: str) -> Edit:
    """Return ``edit`` restricted to ``mask``, still a portable builtin.

    The stored keywords gain the dense bool ``mask`` and its ``mask_sha256``. A
    reloaded masked edit staged under another mask scatters through both, which is
    their intersection, so the intersection is the one mask stored.

    Parameters
    ----------
    edit:
        Builtin verb with a runtime factory.
    mask:
        The resolved site's element mask.
    site_label:
        Site label for apply-time refusals.

    Returns
    -------
    Edit
        The masked verb: same name and portability, scatter-wrapped factory.
    """

    dense = mask.to_dense().cpu().contiguous()
    stored = dict(edit.kwargs).get("mask")
    if isinstance(stored, torch.Tensor):
        if tuple(stored.shape) != tuple(dense.shape):
            raise _apply_invalid(
                "shape",
                f"edit {edit.name!r} already stores a mask of shape {tuple(stored.shape)!r}; "
                f"site {site_label!r} resolves a mask of shape {tuple(dense.shape)!r}.",
                site=site_label,
            )
        dense = dense & stored
    kwargs = tuple(item for item in edit.kwargs if item[0] not in MASK_KEYS)
    return dataclasses.replace(
        edit,
        kwargs=kwargs + (("mask", dense), ("mask_sha256", _mask_digest(dense))),
        factory=_masked_factory(edit.factory, mask, site_label),
    )


def split_stored_mask(
    name: str, kwargs: Mapping[str, Any]
) -> tuple[Mapping[str, Any], _Mask | None]:
    """Take a stored element mask off a saved builtin edit's keywords.

    Parameters
    ----------
    name:
        Builtin helper name. ``subspace_edit`` keeps its keywords: its own
        constructor checks its mask.
    kwargs:
        The saved keywords.

    Returns
    -------
    tuple[Mapping[str, Any], _Mask | None]
        The verb's own keywords and the checked mask, or the keywords unchanged
        and ``None`` when no mask is stored.

    Raises
    ------
    InvalidArgumentError
        ``intervention_helper_arguments_invalid`` for a malformed, partial or
        tampered stored mask.
    """

    if name == SUBSPACE_EDIT_NAME or not any(key in kwargs for key in MASK_KEYS):
        return kwargs, None
    own = {key: value for key, value in kwargs.items() if key not in MASK_KEYS}
    return own, _checked_mask(name, kwargs.get("mask"), kwargs.get("mask_sha256"))


def with_stored_mask(edit: Edit, mask: _Mask | None) -> Edit:
    """Wrap a rebuilt builtin verb with its stored mask (``None`` returns it unchanged)."""

    if mask is None:
        return edit
    return masked_builtin_edit(edit, mask, f"the saved {edit.name} edit's site")


def _refusal(name: str, field: str, why: str) -> InvalidArgumentError:
    """Return the refusal for one malformed stored mask field."""

    return InvalidArgumentError(
        f"The saved intervention's masked {name!r} edit has a malformed {field!r} keyword: "
        f"{why}, so the positions it would edit are not the ones recorded",
        code="intervention_helper_arguments_invalid",
        remedy=_RESAVE,
        argument=field,
    )


def _checked_mask(name: str, mask: Any, digest: Any) -> _Mask:
    """Return the stored element mask, refusing a malformed, partial or tampered one."""

    if mask is None:
        raise _refusal(name, "mask", "a mask digest was stored without its mask")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim < 1:
        kind = f"{mask.dtype} tensor" if isinstance(mask, torch.Tensor) else type(mask).__name__
        raise _refusal(name, "mask", f"expected a bool tensor, got {kind}")
    if mask.layout != torch.strided or mask.device.type == "meta":
        raise _refusal(name, "mask", f"expected a dense mask with data, got {mask.layout}")
    if not isinstance(digest, str) or not digest:
        raise _refusal(name, "mask_sha256", f"expected a non-empty string, got {digest!r}")
    dense = mask.detach().cpu().contiguous()
    if _mask_digest(dense) != digest:
        raise _refusal(name, "mask", f"it no longer matches its recorded sha256 {digest[:16]}...")
    return _Mask(shape=tuple(dense.shape), form="dense", dense=dense)
