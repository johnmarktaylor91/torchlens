"""Engine-owned masked-edit derivation for ``do(selection, edit)``.

The scatter contract lives here: a whole-site mask returns the edit's
behavior unchanged (recipe disclosure stamped); element masks wrap the hook
factory with the EDIT-THEN-SCATTER step on a fresh tensor (stored capture
truth is never written through). Consumers: the ACT selection do-plan
(``torchlens.selection.build_selection_do_plan``) and the edge/param
substitution doors. Moved out of ``torchlens/selection.py`` under the R43
file-size ratchet; semantics unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import torch

from .._errors import InvalidArgumentError
from ..selection import SiteEntry, _apply_invalid, _Mask
from .types import HelperSpec

__all__ = ["_derive_masked_edit", "_masked_factory", "_validate_edited"]


def _row_equivariance_facts(
    edit: HelperSpec, entry: SiteEntry, site_label: str
) -> tuple[tuple[str, Any], ...]:
    """Apply the D23 mask-equivariance law to a row-coherent element-masked edit.

    The v1 check is DRAW-INDEPENDENT (adjudicated over SOL's per-permutation
    rule, which would make legality depend on the realized seed and
    intermittently fail a seed sweep): every batch row must share the same
    non-batch mask. A shared token span across rows is a legitimate positional
    resampling ablation and passes; per-row-differing masks refuse typed,
    teaching the equal-mask-group remedy. The returned facts (mask digest +
    row-equivalence summary) ride the derived spec's persisted metadata NOW so
    the deferred mask-visible draw grouping needs no schema change.

    Parameters
    ----------
    edit:
        The row-coherent helper spec (metadata ``row_coherent`` +
        ``batch_axis``).
    entry:
        Resolved site entry carrying the element mask.
    site_label:
        Site label for refusal messages.

    Returns
    -------
    tuple[tuple[str, Any], ...]
        Metadata disclosure pairs for the derived spec.
    """

    metadata = dict(edit.metadata)
    axis = metadata.get("batch_axis")
    dense = entry._mask._dense_ro()
    if not isinstance(axis, int) or not -dense.ndim <= axis < dense.ndim:
        raise InvalidArgumentError(
            f"row-coherent edit {edit.helper_name!r} declares batch axis "
            f"{axis!r}, which is out of range for site {site_label!r}'s mask "
            f"of rank {dense.ndim} (shape {tuple(dense.shape)!r})",
            code="sampling_geometry_mismatch",
            remedy="pass the batch axis that exists on this site's geometry",
            argument="axis",
        )
    normalized_axis = axis % dense.ndim
    first = dense.select(normalized_axis, 0)
    rows_equal = all(
        torch.equal(dense.select(normalized_axis, index), first)
        for index in range(1, dense.shape[normalized_axis])
    )
    digest = hashlib.sha256(
        dense.detach().to("cpu", torch.uint8).contiguous().numpy().tobytes()
    ).hexdigest()[:16]
    if not rows_equal:
        raise InvalidArgumentError(
            f"row-coherent edit {edit.helper_name!r} at {site_label!r} received "
            "a partial mask whose selected elements DIFFER across batch rows: a "
            "whole-row exchange under a per-row mask would silently mix donor "
            "and subject values row by row. The v1 legality check is "
            "draw-independent (all rows must share one non-batch mask), so a "
            "seed sweep can never flip it",
            code="mask_not_row_equivariant",
            remedy="select the same positions in every row (an all-rows span "
            "such as tl.units masks broadcast over the batch axis), or split "
            "the batch into equal-mask groups and apply the edit per group",
            argument="selection",
        )
    return (
        ("selection_mask_digest", digest),
        ("selection_mask_rows_equal", True),
    )


def _validate_edited(edited: Any, out: torch.Tensor, site_label: str) -> torch.Tensor:
    """Validate the edit output against the site output (no-broadcast v1)."""

    if not isinstance(edited, torch.Tensor):
        raise _apply_invalid(
            "not_maskable",
            f"element-masked edit at {site_label!r} produced a non-tensor "
            f"({type(edited).__name__}); masked edits require tensor outputs.",
            site=site_label,
        )
    if tuple(edited.shape) != tuple(out.shape):
        try:
            torch.broadcast_shapes(tuple(edited.shape), tuple(out.shape))
            broadcastable = True
        except RuntimeError:
            broadcastable = False
        raise _apply_invalid(
            "broadcast" if broadcastable else "shape",
            f"edit output shape {tuple(edited.shape)!r} does not match site "
            f"{site_label!r} output shape {tuple(out.shape)!r}; no broadcasting in v1.",
            site=site_label,
        )
    if edited.dtype != out.dtype:
        raise _apply_invalid(
            "dtype",
            f"edit output dtype {edited.dtype} does not match site {site_label!r} "
            f"output dtype {out.dtype}.",
            site=site_label,
        )
    if edited.device != out.device:
        raise _apply_invalid(
            "device",
            f"edit output device {edited.device} does not match site {site_label!r} "
            f"output device {out.device}.",
            site=site_label,
        )
    return edited


def _masked_factory(inner_factory: Any, mask: _Mask, site_label: str) -> Any:
    """Wrap a helper hook factory with the engine-owned scatter step."""

    def factory() -> Any:
        """Instantiate the inner hook and wrap it with the mask scatter."""

        inner = inner_factory()

        def _masked_hook(out: Any, *, hook: Any) -> Any:
            """Run the inner edit, then scatter only masked elements into ``out``."""

            if not isinstance(out, torch.Tensor):
                raise _apply_invalid(
                    "not_maskable",
                    f"site {site_label!r} produced a non-tensor output at apply "
                    "time; element-masked edits address single-tensor outputs only.",
                    site=site_label,
                )
            if tuple(out.shape) != mask.shape:
                raise _apply_invalid(
                    "shape",
                    f"site {site_label!r} output shape {tuple(out.shape)!r} does not "
                    f"match the selection's recorded index space {mask.shape!r}.",
                    site=site_label,
                )
            edited = inner(out, hook=hook)
            edited = _validate_edited(edited, out, site_label)
            dense = mask._dense_ro().to(out.device)
            # EDIT-THEN-SCATTER on a fresh tensor; stored capture truth is
            # never written through.
            return torch.where(dense, edited, out)

        return _masked_hook

    return factory


def _derive_masked_edit(edit: Any, entry: SiteEntry, digest: str, site_label: str) -> Any:
    """Derive the per-site edit spec/hook under the mask contract.

    A whole-site mask short-circuits the scatter and returns the edit's
    behavior unchanged (only the recipe disclosure is stamped). Element
    masks wrap the hook factory with the engine scatter; the derived spec's
    factory is session-time (``FieldPolicy.DROP``) — the mask never enters
    a persisted KEEP field, and the recipe rides the DROP-gated
    ``selection_recipe`` family.
    """

    recipe = {
        "resolve_digest": digest,
        "site_key": repr(entry.site_key),
        "relation": entry.provenance.relation,
        "selected": entry.selected_count,
        "source": entry.provenance.source,
    }
    if isinstance(edit, HelperSpec):
        disclosure = (
            ("selection_digest", digest),
            ("selection_site", repr(entry.site_key)),
            ("selection_relation", entry.provenance.relation),
        )
        if entry._mask.form == "whole":
            return dataclasses.replace(
                edit,
                metadata=tuple(edit.metadata) + disclosure,
                selection_recipe=recipe,
            )
        if edit.factory is None:
            raise _apply_invalid(
                "not_maskable",
                f"edit {edit.helper_name!r} has no runtime factory to mask.",
                site=site_label,
            )
        if dict(edit.metadata).get("row_coherent"):
            disclosure = disclosure + _row_equivariance_facts(edit, entry, site_label)
        return dataclasses.replace(
            edit,
            factory=_masked_factory(edit.factory, entry._mask, site_label),
            # The derived spec cannot be re-executed from its persisted form
            # alone (the mask is session-time until the wave-3 bump).
            portability="opaque_audit",
            metadata=tuple(edit.metadata) + disclosure,
            selection_recipe=recipe,
        )
    if callable(edit):
        if entry._mask.form == "whole":
            return edit

        def _plain_factory() -> Any:
            """Adapt a bare hook callable to the masked-factory protocol."""

            def _adapter(out: Any, *, hook: Any) -> Any:
                """Forward to the user's hook callable unchanged."""

                return edit(out, hook=hook)

            return _adapter

        return _masked_factory(_plain_factory, entry._mask, site_label)()
    raise ValueError(
        "do(selection, edit) requires an Edit/HelperSpec, a hook callable, or a "
        f"replacement tensor; got {type(edit).__name__}."
    )
